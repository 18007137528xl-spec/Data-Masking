"""Where the vault key comes from.

An environment variable is fine for synthetic data and wrong for anything
else. It appears in process listings, container inspect output, CI logs, crash
dumps and orchestrator configuration, and it cannot be rotated without a
redeploy. In a cloud deployment the key must come from a managed key service.

The pattern here is **envelope encryption**: a Fernet data key is generated
once, encrypted under a KMS customer-managed key, and the *ciphertext* is what
gets stored next to the vault. The plaintext key exists only in process memory,
and obtaining it requires a `Decrypt` call the KMS logs and authorises. Losing
IAM access to the KMS key therefore locks the vault as effectively as losing
the key itself -- which is the point, and also why the KMS key must have
deletion protection and a documented recovery path.

Key URIs::

    env:DEIDKIT_VAULT_KEY               environment variable (dev only)
    file:/run/secrets/vault.key         mounted secret file
    awskms:<key-arn>?blob=<path>        AWS KMS envelope
    azurekv:<vault-url>/<secret-name>   Azure Key Vault secret
    gcpkms:<resource-name>?blob=<path>  Google Cloud KMS envelope

The cloud backends import their SDK lazily, so the package installs and runs
without any of them.
"""

from __future__ import annotations

import base64
import os
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from cryptography.fernet import Fernet


class KeyProviderError(RuntimeError):
    pass


class KeyProvider(ABC):
    """Supplies the Fernet key that opens a vault."""

    #: True when the key is protected by a managed key service.
    managed: bool = False

    @abstractmethod
    def get_key(self) -> bytes:
        """Return the plaintext Fernet key. Never log the return value."""

    @abstractmethod
    def describe(self) -> str:
        """Human-readable source, safe to write to a manifest or log."""

    def provision(self) -> str:
        """Create a new key at this location. Returns a description.

        Refuses to overwrite an existing key: doing so would orphan every
        surrogate already issued, and no subject in any published tier could
        be re-identified again.
        """
        raise KeyProviderError(
            f"{type(self).__name__} cannot provision keys; create the key in "
            "the key service and reference it by URI"
        )


# ----------------------------------------------------------------------
# development providers
# ----------------------------------------------------------------------
class EnvKeyProvider(KeyProvider):
    """Key from an environment variable. Development only."""

    managed = False

    def __init__(self, var: str = "DEIDKIT_VAULT_KEY") -> None:
        self.var = var

    def get_key(self) -> bytes:
        raw = os.environ.get(self.var)
        if not raw:
            raise KeyProviderError(f"{self.var} is not set")
        return raw.encode()

    def describe(self) -> str:
        return f"env:{self.var} (UNMANAGED -- development only)"


class FileKeyProvider(KeyProvider):
    """Key from a file, typically an orchestrator-mounted secret.

    Acceptable in production when the file is a tmpfs-mounted secret injected
    by the platform (Kubernetes secret, ECS secret, Docker secret) rather than
    a file living on a persistent volume beside the vault.
    """

    managed = False

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def get_key(self) -> bytes:
        if not self.path.exists():
            raise KeyProviderError(f"key file not found: {self.path}")
        mode = self.path.stat().st_mode & 0o077
        if mode:
            raise KeyProviderError(
                f"{self.path} is readable beyond its owner (mode bits {mode:o}). "
                "Refusing to read it: a vault key that other local accounts can "
                "read is not a vault key."
            )
        return self.path.read_bytes().strip()

    def describe(self) -> str:
        return f"file:{self.path}"

    def provision(self) -> str:
        if self.path.exists():
            raise KeyProviderError(
                f"{self.path} already exists. Refusing to overwrite: that would "
                "orphan every surrogate already issued."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(Fernet.generate_key())
        self.path.chmod(0o600)
        return f"wrote a new key to {self.path} (mode 600)"


# ----------------------------------------------------------------------
# managed providers -- envelope encryption
# ----------------------------------------------------------------------
class AwsKmsKeyProvider(KeyProvider):
    """Fernet key wrapped under an AWS KMS customer-managed key.

    The wrapped blob is stored at ``blob_path`` -- safe to keep beside the
    vault, since opening it requires ``kms:Decrypt`` on the CMK, which is
    authorised and logged in CloudTrail.
    """

    managed = True

    def __init__(
        self, key_id: str, blob_path: str | Path, region: str | None = None
    ) -> None:
        self.key_id = key_id
        self.blob_path = Path(blob_path)
        self.region = region

    def _client(self):
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover
            raise KeyProviderError(
                "AWS KMS support needs boto3: pip install 'deidkit[aws]'"
            ) from exc
        return boto3.client("kms", region_name=self.region) if self.region else boto3.client("kms")

    def get_key(self) -> bytes:
        if not self.blob_path.exists():
            raise KeyProviderError(
                f"wrapped key blob not found: {self.blob_path}. Provision one "
                "with 'deidkit keygen --key-uri ...' before the first run."
            )
        blob = self.blob_path.read_bytes()
        resp = self._client().decrypt(CiphertextBlob=blob, KeyId=self.key_id)
        return resp["Plaintext"]

    def describe(self) -> str:
        return f"awskms:{self.key_id} (blob at {self.blob_path})"

    def provision(self) -> str:
        if self.blob_path.exists():
            raise KeyProviderError(
                f"{self.blob_path} already exists. Refusing to overwrite: that "
                "would orphan every surrogate already issued."
            )
        plaintext = Fernet.generate_key()
        resp = self._client().encrypt(KeyId=self.key_id, Plaintext=plaintext)
        self.blob_path.parent.mkdir(parents=True, exist_ok=True)
        self.blob_path.write_bytes(resp["CiphertextBlob"])
        self.blob_path.chmod(0o600)
        return (
            f"generated a data key, wrapped it under {self.key_id}, and wrote "
            f"the blob to {self.blob_path}. The plaintext key was never written "
            "to disk."
        )


class AzureKeyVaultProvider(KeyProvider):
    """Fernet key held directly as an Azure Key Vault secret.

    Key Vault stores the secret itself rather than wrapping a local blob, so
    there is nothing on disk. Access is governed by the vault's RBAC and logged
    to the vault's diagnostic settings.
    """

    managed = True

    def __init__(self, vault_url: str, secret_name: str) -> None:
        self.vault_url = vault_url
        self.secret_name = secret_name

    def _client(self):
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient
        except ImportError as exc:  # pragma: no cover
            raise KeyProviderError(
                "Azure support needs the SDK: pip install 'deidkit[azure]'"
            ) from exc
        return SecretClient(
            vault_url=self.vault_url, credential=DefaultAzureCredential()
        )

    def get_key(self) -> bytes:
        secret = self._client().get_secret(self.secret_name)
        if not secret.value:
            raise KeyProviderError(f"secret {self.secret_name} is empty")
        return secret.value.encode()

    def describe(self) -> str:
        return f"azurekv:{self.vault_url}/{self.secret_name}"

    def provision(self) -> str:
        client = self._client()
        try:
            existing = client.get_secret(self.secret_name)
            if existing.value:
                raise KeyProviderError(
                    f"secret {self.secret_name} already has a value. Refusing to "
                    "overwrite: that would orphan every surrogate already issued."
                )
        except KeyProviderError:
            raise
        except Exception:
            pass  # not found is the expected path
        client.set_secret(self.secret_name, Fernet.generate_key().decode())
        return f"created secret {self.secret_name} in {self.vault_url}"


class GcpKmsKeyProvider(KeyProvider):
    """Fernet key wrapped under a Google Cloud KMS key."""

    managed = True

    def __init__(self, resource_name: str, blob_path: str | Path) -> None:
        self.resource_name = resource_name
        self.blob_path = Path(blob_path)

    def _client(self):
        try:
            from google.cloud import kms
        except ImportError as exc:  # pragma: no cover
            raise KeyProviderError(
                "GCP KMS support needs the SDK: pip install 'deidkit[gcp]'"
            ) from exc
        return kms.KeyManagementServiceClient()

    def get_key(self) -> bytes:
        if not self.blob_path.exists():
            raise KeyProviderError(
                f"wrapped key blob not found: {self.blob_path}"
            )
        resp = self._client().decrypt(
            request={
                "name": self.resource_name,
                "ciphertext": self.blob_path.read_bytes(),
            }
        )
        return resp.plaintext

    def describe(self) -> str:
        return f"gcpkms:{self.resource_name} (blob at {self.blob_path})"

    def provision(self) -> str:
        if self.blob_path.exists():
            raise KeyProviderError(
                f"{self.blob_path} already exists. Refusing to overwrite: that "
                "would orphan every surrogate already issued."
            )
        plaintext = Fernet.generate_key()
        resp = self._client().encrypt(
            request={"name": self.resource_name, "plaintext": plaintext}
        )
        self.blob_path.parent.mkdir(parents=True, exist_ok=True)
        self.blob_path.write_bytes(resp.ciphertext)
        self.blob_path.chmod(0o600)
        return (
            f"generated a data key, wrapped it under {self.resource_name}, and "
            f"wrote the blob to {self.blob_path}"
        )


# ----------------------------------------------------------------------
# resolution
# ----------------------------------------------------------------------
def from_uri(uri: str) -> KeyProvider:
    """Build a provider from a key URI. See the module docstring for forms."""
    if "://" not in uri and ":" not in uri:
        raise KeyProviderError(f"not a key URI: {uri!r}")

    scheme, _, rest = uri.partition(":")
    scheme = scheme.lower()

    if scheme == "env":
        return EnvKeyProvider(rest or "DEIDKIT_VAULT_KEY")

    if scheme == "file":
        return FileKeyProvider(rest)

    if scheme == "awskms":
        parsed = urlparse(uri)
        query = parse_qs(parsed.query)
        key_id = rest.split("?", 1)[0]
        blob = query.get("blob", [""])[0]
        if not blob:
            raise KeyProviderError(
                "awskms URIs need ?blob=<path> for the wrapped key, e.g. "
                "awskms:arn:aws:kms:us-east-1:123:key/abc?blob=/vault/key.blob"
            )
        return AwsKmsKeyProvider(
            key_id, blob, region=query.get("region", [None])[0]
        )

    if scheme == "azurekv":
        # azurekv:https://myvault.vault.azure.net/secret-name
        body = rest.lstrip("/")
        if "://" in body:
            head, _, name = body.rpartition("/")
            return AzureKeyVaultProvider(head, name)
        raise KeyProviderError(
            "azurekv URIs look like azurekv:https://<vault>.vault.azure.net/<secret>"
        )

    if scheme == "gcpkms":
        query = parse_qs(urlparse(uri).query)
        resource = rest.split("?", 1)[0]
        blob = query.get("blob", [""])[0]
        if not blob:
            raise KeyProviderError("gcpkms URIs need ?blob=<path>")
        return GcpKmsKeyProvider(resource, blob)

    raise KeyProviderError(
        f"unknown key URI scheme {scheme!r}; expected one of "
        "env, file, awskms, azurekv, gcpkms"
    )


def resolve(
    key_uri: str | None = None, *, require_managed: bool = False
) -> KeyProvider:
    """Resolve a provider from a URI, then ``DEIDKIT_KEY_URI``, then the env var.

    ``require_managed`` refuses an unmanaged source. Set it in production
    deployments -- via ``DEIDKIT_REQUIRE_MANAGED_KEY=1`` -- so that a
    misconfiguration fails loudly instead of quietly falling back to an
    environment variable.
    """
    uri = key_uri or os.environ.get("DEIDKIT_KEY_URI")
    provider = from_uri(uri) if uri else EnvKeyProvider()

    if not require_managed:
        require_managed = os.environ.get(
            "DEIDKIT_REQUIRE_MANAGED_KEY", ""
        ).lower() in {"1", "true", "yes"}

    if require_managed and not provider.managed:
        raise KeyProviderError(
            f"a managed key source is required but the key comes from "
            f"{provider.describe()}. Point DEIDKIT_KEY_URI at a KMS or Key "
            "Vault, or unset DEIDKIT_REQUIRE_MANAGED_KEY if this really is a "
            "development environment."
        )
    return provider


def is_valid_fernet_key(key: bytes) -> bool:
    """Cheap shape check, so a truncated secret fails with a clear message."""
    try:
        return len(base64.urlsafe_b64decode(key)) == 32
    except Exception:
        return False
