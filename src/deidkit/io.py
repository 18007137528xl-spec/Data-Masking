"""Reading and writing study data.

EDC and SDTM exports rarely arrive as CSV. SAS transport (``.xpt``) is the
regulatory submission format and ``.sas7bdat`` is what most CRO pipelines pass
around internally, so both are first-class here via ``pyreadstat``. Parquet is
the sensible internal format for the published tiers.

Every read records a SHA-256 of the file bytes. That checksum goes into the
manifest, so a published tier can always be traced to the exact input it came
from -- which is the first question anyone asks when a number looks wrong.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

READABLE = {".csv", ".tsv", ".txt", ".parquet", ".pq", ".xpt", ".sas7bdat"}

#: Working files that live beside published data but are not domains.
#:
#: The review queue matters most here. It holds the ORIGINAL text of every
#: flagged row -- that is its whole purpose, a human has to read the untouched
#: value to adjudicate it -- so it carries raw PHI and, in a blinded study, the
#: compound name. Discovering it as a domain would feed it back through the
#: pipeline; leaving it in the published tier would put unredacted text in the
#: directory analysts read. It belongs with the vault, under the same controls.
NOT_DOMAINS = frozenset(
    {
        "REVIEW_QUEUE", "REVIEW-QUEUE", "RQ", "QUEUE",
        "STEWARD_REVIEW", "STEWARD-REVIEW",
        "MANIFEST", "CONTRACT", "README",
    }
)

#: Remote URI schemes handled through fsspec. Object storage is how the three
#: zones stay genuinely separate in a cloud deployment: quarantine, the LDS
#: tier, and the de-identified tier each get their own bucket with its own
#: policy, rather than three directories one IAM role can all reach.
_REMOTE = re.compile(r"^(s3|s3a|az|abfs|abfss|gs|gcs|https?|file)://", re.IGNORECASE)


def is_remote(path: str | Path) -> bool:
    return bool(_REMOTE.match(str(path))) and not str(path).lower().startswith(
        "file://"
    )


def _fs(path: str):
    """Resolve an fsspec filesystem for a remote URI."""
    try:
        import fsspec
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"reading {path} needs fsspec plus the backend for its scheme: "
            "pip install 'deidkit[aws]' (s3fs), '[azure]' (adlfs) or "
            "'[gcp]' (gcsfs)"
        ) from exc
    return fsspec.core.url_to_fs(path)


def _open(path: str | Path, mode: str = "rb"):
    if is_remote(path):
        fs, inner = _fs(str(path))
        return fs.open(inner, mode)
    return open(path, mode)


def checksum(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of the file bytes.

    Recorded in the manifest so a published tier can always be traced to the
    exact input it came from -- the first question anyone asks when a number
    looks wrong.
    """
    h = hashlib.sha256()
    with _open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def read_table(path: str | Path) -> pd.DataFrame:
    """Read one domain table, preserving values as strings where ambiguous.

    Dates in SDTM are character ``--DTC`` fields and must not be coerced to
    datetimes on read: partial values like ``2015-03`` would either fail or be
    silently completed to a day that was never recorded.
    """
    text = str(path)
    suffix = Path(text.split("?", 1)[0]).suffix.lower()

    # pandas resolves s3:// / az:// / gs:// itself when the fsspec backend is
    # installed, so remote and local take the same path for these formats.
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(text, dtype=str, keep_default_na=True)
    if suffix == ".tsv":
        return pd.read_csv(text, sep="\t", dtype=str, keep_default_na=True)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(text)

    if suffix in {".xpt", ".sas7bdat"}:
        try:
            import pyreadstat
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                f"reading {suffix} requires pyreadstat: pip install 'deidkit[sas]'"
            ) from exc
        reader = (
            pyreadstat.read_xport if suffix == ".xpt" else pyreadstat.read_sas7bdat
        )
        if is_remote(text):
            # pyreadstat needs a real path, so stream the object to a temp file.
            # Nothing durable is written: the file is unlinked on close, which
            # matters because this content is quarantine-zone data.
            import tempfile

            with _open(text, "rb") as src, tempfile.NamedTemporaryFile(
                suffix=suffix
            ) as tmp:
                while block := src.read(1 << 22):
                    tmp.write(block)
                tmp.flush()
                frame, _meta = reader(tmp.name)
                return frame
        frame, _meta = reader(text)
        return frame

    raise ValueError(
        f"unsupported input {Path(text).name!r}; expected one of {sorted(READABLE)}"
    )


def discover(directory: str | Path) -> dict[str, str]:
    """Map ``DOMAIN -> path`` for readable tables in a directory or prefix.

    Domain name is the upper-cased stem, matching SDTM convention (``ae.xpt``
    -> ``AE``). Works on a local directory or an object-storage prefix.
    """
    text = str(directory)

    if is_remote(text):
        fs, inner = _fs(text)
        out: dict[str, str] = {}
        scheme = text.split("://", 1)[0]
        for entry in sorted(fs.ls(inner, detail=False)):
            name = entry.rsplit("/", 1)[-1]
            stem = Path(name).stem.upper()
            if Path(name).suffix.lower() in READABLE and stem not in NOT_DOMAINS:
                out[stem] = f"{scheme}://{entry.lstrip('/')}"
        if not out:
            raise FileNotFoundError(f"no readable tables under {text}")
        return out

    d = Path(text)
    if not d.is_dir():
        raise NotADirectoryError(d)
    local: dict[str, str] = {}
    for p in sorted(d.iterdir()):
        if (
            p.is_file()
            and p.suffix.lower() in READABLE
            and p.stem.upper() not in NOT_DOMAINS
        ):
            local[p.stem.upper()] = str(p)
    return local


def load_study(directory: str | Path) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Read every table in a directory. Returns ``(frames, checksums)``."""
    paths = discover(directory)
    frames = {name: read_table(p) for name, p in paths.items()}
    sums = {name: checksum(p) for name, p in paths.items()}
    return frames, sums


def write_table(frame: pd.DataFrame, path: str | Path) -> None:
    text = str(path)
    if is_remote(text):
        if Path(text).suffix.lower() in {".parquet", ".pq"}:
            frame.to_parquet(text, index=False)
        else:
            frame.to_csv(text, index=False)
        return
    p = Path(text)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(p, index=False)
    else:
        frame.to_csv(p, index=False)


def write_study(
    frames: dict[str, pd.DataFrame], directory: str | Path, fmt: str = "parquet"
) -> dict[str, str]:
    text = str(directory).rstrip("/")
    if not is_remote(text):
        Path(text).mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, frame in frames.items():
        target = f"{text}/{name.lower()}.{fmt}"
        write_table(frame, target)
        written[name] = target
    return written


def write_text(content: str, path: str | Path) -> None:
    """Write a manifest or review queue, local or remote."""
    text = str(path)
    if is_remote(text):
        with _open(text, "wb") as fh:
            fh.write(content.encode("utf-8"))
        return
    p = Path(text)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
