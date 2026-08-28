"""The crosswalk vault.

Holds the only mapping between real identifiers and the surrogates that appear
in the data tiers, plus each subject's date-shift offset. It is the single
highest-value target in the system and is designed accordingly:

* **Surrogates are random, never derived.** HIPAA 164.514(c) requires that a
  re-identification code not be derived from, or related to, information about
  the individual. ``HMAC(subject_id)`` is derived from it; a random token is
  not. Randomness also removes the brute-force exposure that hashing a small
  keyspace (an MRN, a national ID) always carries.
* **The lookup index does not leak the original.** Forward lookups need to find
  "the surrogate already issued for this subject" without storing the subject
  identifier in the clear, so the index key is an HMAC under a pepper derived
  from the vault key, and the original is held separately under authenticated
  encryption.
* **Reverse lookups are break-glass and logged.** ``reverse()`` demands a
  justification string and writes an access-log row before returning. There is
  no unlogged path to a real identifier.
* **Destroying the vault is a deliberate act with consequences.** It is what
  converts pseudonymised data into anonymous data. ``destroy()`` exists so that
  it can be scheduled rather than forgotten.

Deployment: the vault file, its key, and the accounts that reach them must be
separate from every data tier. No principal that can read a data tier may read
the vault.
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

_ENV_KEY = "DEIDKIT_VAULT_KEY"

#: Surrogate alphabet: unambiguous under transcription and reading aloud.
#: Crockford-style -- no I, L, O, U.
_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

#: Label suffixes for LABEL_MAP: A..Z then AA, AB, ...
_LABEL_LETTERS = [chr(c) for c in range(65, 91)] + [
    chr(a) + chr(b) for a in range(65, 91) for b in range(65, 91)
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS surrogate (
    entity      TEXT NOT NULL,
    lookup      BLOB NOT NULL,
    surrogate   TEXT NOT NULL,
    original_ct BLOB NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (entity, lookup)
);
CREATE UNIQUE INDEX IF NOT EXISTS surrogate_unique
    ON surrogate (entity, surrogate);

CREATE TABLE IF NOT EXISTS date_offset (
    entity      TEXT NOT NULL,
    lookup      BLOB NOT NULL,
    offset_days INTEGER NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (entity, lookup)
);

CREATE TABLE IF NOT EXISTS access_log (
    at            TEXT NOT NULL,
    action        TEXT NOT NULL,
    entity        TEXT,
    surrogate     TEXT,
    operator      TEXT NOT NULL,
    justification TEXT NOT NULL,
    n_records     INTEGER
);

CREATE TABLE IF NOT EXISTS vault_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class VaultError(RuntimeError):
    pass


class Vault:
    """Encrypted crosswalk store.

    Parameters
    ----------
    path:
        SQLite file. Keep it off any volume that also holds a data tier, and
        out of any backup set shared with one -- a restore that lands both in
        the same place silently undoes the whole design.
    key:
        Fernet key. Omitted, it is read from ``DEIDKIT_VAULT_KEY``. In
        production this comes from a KMS/HSM, not an environment variable.
    operator:
        Recorded on every access-log row.
    """

    def __init__(
        self,
        path: str | Path,
        key: bytes | str | None = None,
        *,
        operator: str = "unknown",
        key_uri: str | None = None,
        require_managed: bool = False,
    ) -> None:
        raw: bytes | str | None = key
        self.key_source = "explicit"

        if raw is None:
            # One resolution path for every deployment: a URI argument, then
            # DEIDKIT_KEY_URI, then the bare environment variable. Production
            # sets DEIDKIT_REQUIRE_MANAGED_KEY so the last fallback is refused
            # rather than silently accepted.
            from .keyprovider import KeyProviderError, resolve

            try:
                provider = resolve(key_uri, require_managed=require_managed)
                raw = provider.get_key()
                self.key_source = provider.describe()
            except KeyProviderError as exc:
                raise VaultError(str(exc)) from exc

        if not raw:
            raise VaultError(
                f"no vault key supplied and {_ENV_KEY} is unset. "
                "Generate one with Vault.generate_key(), or point "
                "DEIDKIT_KEY_URI at a key service."
            )
        if isinstance(raw, str):
            raw = raw.encode()
        try:
            self._fernet = Fernet(raw)
        except (ValueError, TypeError) as exc:
            raise VaultError(f"invalid vault key: {exc}") from exc

        # One secret in, two derived: the AEAD key (Fernet, above) and a pepper
        # for the lookup index. Deriving rather than reusing keeps the index
        # keys unusable for decryption.
        self._pepper = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"deidkit/vault/lookup-pepper/v1",
        ).derive(raw)

        self.path = Path(path)
        self.operator = operator
        new = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.executescript(_SCHEMA)
        if new:
            self._db.execute(
                "INSERT OR IGNORE INTO vault_meta VALUES (?, ?)",
                ("created_at", _now()),
            )
        # Verify the key matches this vault before any use, so a wrong key
        # fails loudly instead of quietly issuing a second surrogate space.
        self._check_key()
        self._db.commit()

    # ------------------------------------------------------------------
    # key management
    # ------------------------------------------------------------------
    @staticmethod
    def generate_key() -> str:
        """A fresh Fernet key. Store it in a KMS, never beside the vault file."""
        return Fernet.generate_key().decode()

    def _check_key(self) -> None:
        row = self._db.execute(
            "SELECT value FROM vault_meta WHERE key = 'key_check'"
        ).fetchone()
        if row is None:
            token = self._fernet.encrypt(b"deidkit/vault/key-check/v1")
            self._db.execute(
                "INSERT INTO vault_meta VALUES (?, ?)", ("key_check", token.decode())
            )
            return
        try:
            if self._fernet.decrypt(row[0].encode()) != b"deidkit/vault/key-check/v1":
                raise VaultError("vault key check failed")
        except InvalidToken as exc:
            raise VaultError(
                f"the key supplied does not open {self.path}. Refusing to "
                "continue: proceeding would issue a second, disjoint surrogate "
                "space for the same subjects."
            ) from exc

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _lookup(self, entity: str, original: str) -> bytes:
        h = hmac.HMAC(self._pepper, hashes.SHA256())
        h.update(entity.encode("utf-8"))
        h.update(b"\x00")
        h.update(original.encode("utf-8"))
        return h.finalize()

    def _new_surrogate(self, entity: str, prefix: str, length: int) -> str:
        """A random, non-derived surrogate unique within the entity space."""
        for _ in range(64):
            body = "".join(secrets.choice(_ALPHABET) for _ in range(length))
            cand = f"{prefix}-{body}" if prefix else body
            hit = self._db.execute(
                "SELECT 1 FROM surrogate WHERE entity = ? AND surrogate = ?",
                (entity, cand),
            ).fetchone()
            if hit is None:
                return cand
        raise VaultError(
            f"could not find a free surrogate for entity {entity!r} at length "
            f"{length}; raise 'length'"
        )

    # ------------------------------------------------------------------
    # forward mapping -- the hot path
    # ------------------------------------------------------------------
    def surrogate_for(
        self,
        entity: str,
        original: str,
        *,
        prefix: str = "",
        length: int = 8,
    ) -> str:
        """Return the surrogate for ``original``, issuing one if needed.

        Idempotent: the same original always maps to the same surrogate, which
        is what keeps joins intact across domains and across drops.
        """
        lk = self._lookup(entity, original)
        row = self._db.execute(
            "SELECT surrogate FROM surrogate WHERE entity = ? AND lookup = ?",
            (entity, lk),
        ).fetchone()
        if row is not None:
            return row[0]

        surrogate = self._new_surrogate(entity, prefix, length)
        self._db.execute(
            "INSERT INTO surrogate (entity, lookup, surrogate, original_ct, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                entity,
                lk,
                surrogate,
                self._fernet.encrypt(original.encode("utf-8")),
                _now(),
            ),
        )
        return surrogate

    def surrogate_map(
        self,
        entity: str,
        originals: Iterable[str],
        *,
        prefix: str = "",
        length: int = 8,
    ) -> dict[str, str]:
        """Batch form of :meth:`surrogate_for`. Commits once."""
        out: dict[str, str] = {}
        for o in originals:
            if o in out:
                continue
            out[o] = self.surrogate_for(entity, o, prefix=prefix, length=length)
        self._db.commit()
        return out

    # ------------------------------------------------------------------
    # neutral labels (blinding, not privacy)
    # ------------------------------------------------------------------
    def label_map(
        self, entity: str, originals: Iterable[str], *, prefix: str = "TRT"
    ) -> dict[str, str]:
        """Map values to ``TRT A``, ``TRT B``, ... -- stable and reversible.

        Assignment order is random rather than alphabetical or first-seen:
        ordered labels would leak the arms' order in the protocol, which for a
        dose-escalation study is most of what blinding was protecting.

        Labels live in the same table as the surrogates, so reversal goes
        through the same logged break-glass path.
        """
        values = list(dict.fromkeys(str(o) for o in originals))
        out: dict[str, str] = {}

        pending: list[str] = []
        for v in values:
            row = self._db.execute(
                "SELECT surrogate FROM surrogate WHERE entity = ? AND lookup = ?",
                (entity, self._lookup(entity, v)),
            ).fetchone()
            if row is not None:
                out[v] = row[0]
            else:
                pending.append(v)

        if pending:
            used = {
                r[0]
                for r in self._db.execute(
                    "SELECT surrogate FROM surrogate WHERE entity = ?", (entity,)
                )
            }
            free = [
                f"{prefix} {a}"
                for a in _LABEL_LETTERS
                if f"{prefix} {a}" not in used
            ]
            if len(free) < len(pending):
                raise VaultError(
                    f"only {len(free)} free labels for {len(pending)} new values "
                    f"in entity {entity!r}"
                )
            secrets.SystemRandom().shuffle(pending)
            for v, label in zip(pending, free):
                self._db.execute(
                    "INSERT INTO surrogate (entity, lookup, surrogate, original_ct,"
                    " created_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        entity,
                        self._lookup(entity, v),
                        label,
                        self._fernet.encrypt(v.encode("utf-8")),
                        _now(),
                    ),
                )
                out[v] = label
            self._db.commit()
        return out

    # ------------------------------------------------------------------
    # date offsets
    # ------------------------------------------------------------------
    def offset_for(
        self,
        original: str,
        *,
        entity: str = "subject",
        low: int = 180,
        high: int = 540,
    ) -> int:
        """Per-subject date-shift offset in days, stable once issued.

        Uniform across all of that subject's dates, so intervals are preserved
        while absolute linkage is broken. Sign is randomised.
        """
        lk = self._lookup(entity, original)
        row = self._db.execute(
            "SELECT offset_days FROM date_offset WHERE entity = ? AND lookup = ?",
            (entity, lk),
        ).fetchone()
        if row is not None:
            return row[0]
        magnitude = secrets.randbelow(high - low + 1) + low
        offset = magnitude if secrets.randbits(1) else -magnitude
        self._db.execute(
            "INSERT INTO date_offset (entity, lookup, offset_days, created_at)"
            " VALUES (?, ?, ?, ?)",
            (entity, lk, offset, _now()),
        )
        return offset

    def offset_map(
        self, originals: Iterable[str], *, entity: str = "subject", **kw
    ) -> dict[str, int]:
        out: dict[str, int] = {}
        for o in originals:
            if o not in out:
                out[o] = self.offset_for(o, entity=entity, **kw)
        self._db.commit()
        return out

    # ------------------------------------------------------------------
    # reverse mapping -- break-glass only
    # ------------------------------------------------------------------
    def reverse(
        self, entity: str, surrogate: str, *, justification: str
    ) -> str | None:
        """Recover the original identifier. Always logged.

        ``justification`` is mandatory and free-text: the safety report number,
        the data query reference. There is no unlogged path to a real
        identifier, and no default that lets a caller omit the reason.
        """
        if not justification or not justification.strip():
            raise VaultError("reverse() requires a non-empty justification")

        row = self._db.execute(
            "SELECT original_ct FROM surrogate WHERE entity = ? AND surrogate = ?",
            (entity, surrogate),
        ).fetchone()
        self._db.execute(
            "INSERT INTO access_log (at, action, entity, surrogate, operator,"
            " justification, n_records) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                _now(),
                "reverse" if row else "reverse_miss",
                entity,
                surrogate,
                self.operator,
                justification.strip(),
                1 if row else 0,
            ),
        )
        self._db.commit()
        if row is None:
            return None
        return self._fernet.decrypt(row[0]).decode("utf-8")

    def access_log(self) -> list[dict[str, object]]:
        cur = self._db.execute(
            "SELECT at, action, entity, surrogate, operator, justification,"
            " n_records FROM access_log ORDER BY at"
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, object]:
        by_entity = dict(
            self._db.execute(
                "SELECT entity, COUNT(*) FROM surrogate GROUP BY entity"
            ).fetchall()
        )
        return {
            "path": str(self.path),
            "key_source": self.key_source,
            "surrogates_by_entity": by_entity,
            "date_offsets": self._db.execute(
                "SELECT COUNT(*) FROM date_offset"
            ).fetchone()[0],
            "reverse_lookups": self._db.execute(
                "SELECT COUNT(*) FROM access_log WHERE action = 'reverse'"
            ).fetchone()[0],
        }

    def destroy(self, *, confirm: str, operator: str, justification: str) -> None:
        """Irreversibly destroy the crosswalk.

        This is the act that converts pseudonymised data into anonymous data.
        It should appear on a retention schedule with a date against it, not
        happen by accident -- hence the explicit confirmation phrase.
        """
        if confirm != "DESTROY CROSSWALK":
            raise VaultError(
                "destroy() requires confirm='DESTROY CROSSWALK'. After this, no "
                "subject in any published tier can ever be re-identified -- "
                "including for safety reporting."
            )
        receipt = {
            "destroyed_at": _now(),
            "operator": operator,
            "justification": justification,
            "stats_at_destruction": self.stats(),
        }
        self._db.close()
        self.path.unlink(missing_ok=True)
        Path(str(self.path) + ".destroyed.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )

    def commit(self) -> None:
        self._db.commit()

    def close(self) -> None:
        self._db.commit()
        self._db.close()

    def __enter__(self) -> Vault:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
