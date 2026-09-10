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

READABLE = {
    ".csv", ".tsv", ".txt", ".parquet", ".pq", ".xpt", ".sas7bdat",
    # Excel is how clinical data actually moves between people, whatever the
    # transfer specification says. A tool for inbound EDC data that cannot
    # read a workbook is refusing the most likely input it will ever be given.
    ".xlsx", ".xlsm",
}

#: Excel workbooks, which need sheet-aware discovery: one workbook commonly
#: holds every domain as a separate sheet.
EXCEL = {".xlsx", ".xlsm"}

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


def sheet_split(path: str | Path) -> tuple[str, str | None]:
    """Split ``workbook.xlsx::SheetName`` into the file and the sheet."""
    text = str(path)
    if "::" in text:
        base, sheet = text.rsplit("::", 1)
        return base, sheet
    return text, None


def checksum(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of the file bytes.

    Recorded in the manifest so a published tier can always be traced to the
    exact input it came from -- the first question anyone asks when a number
    looks wrong.
    """
    path, _sheet = sheet_split(path)
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
    text, sheet = sheet_split(text)
    suffix = Path(text.split("?", 1)[0]).suffix.lower()

    if suffix in EXCEL:
        # dtype=str for the same reason as CSV, and with more force: Excel
        # will have already turned 2015-03 into a datetime and eaten the
        # leading zero off site 002 before this tool ever sees the file. What
        # is read here cannot undo that -- it can only avoid adding to it.
        return pd.read_excel(
            text, sheet_name=sheet or 0, dtype=str, keep_default_na=True,
            engine="openpyxl",
        )

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
        if not p.is_file() or p.suffix.lower() not in READABLE:
            continue
        if p.stem.upper() in NOT_DOMAINS:
            continue
        if p.suffix.lower() in EXCEL:
            # One workbook, many domains. "study.xlsx" holding sheets DM, AE
            # and LB is three tables, not one -- and addressing them needs the
            # sheet in the path, hence the "::" suffix that read_table splits.
            for sheet in _excel_sheets(p):
                if sheet.upper() in NOT_DOMAINS:
                    continue
                name = sheet.upper() if len(_excel_sheets(p)) > 1 else p.stem.upper()
                local[name] = f"{p}::{sheet}"
            continue
        local[p.stem.upper()] = str(p)
    return local


def _excel_sheets(path: str | Path) -> list[str]:
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "reading .xlsx requires openpyxl: pip install 'deidkit[excel]'"
        ) from exc
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        return [ws.title for ws in book.worksheets if ws.max_row and ws.max_row > 1]
    finally:
        book.close()


def explain_empty(directory: str | Path) -> str:
    """Say why a directory yielded no tables, naming what was actually there.

    "no readable tables in D:\\study" is true and useless: it does not say
    what the tool looked at, what it can read, or which of the two is wrong.
    The usual answer is a file extension away -- an .xlsx before this tool
    read Excel, or a Windows Explorer that hides extensions so nobody can see
    the file is .xls rather than .xlsx.
    """
    d = Path(str(directory))
    if not d.exists():
        return f"{d} does not exist"
    if not d.is_dir():
        return (
            f"{d} is a file, not a folder. Point this at the FOLDER that "
            "holds your tables -- one domain per file, or one workbook with "
            "a sheet per domain."
        )
    entries = sorted(p for p in d.iterdir() if p.is_file())
    if not entries:
        return f"{d} is empty"
    lines = [f"{d} holds {len(entries)} file(s), none of them readable:"]
    for p in entries[:15]:
        why = (
            "skipped: reserved name (working file, not a domain)"
            if p.suffix.lower() in READABLE
            else f"unsupported extension {p.suffix or '(none)'}"
        )
        lines.append(f"  {p.name:<44} {why}")
    if len(entries) > 15:
        lines.append(f"  ... and {len(entries) - 15} more")
    lines.append(f"readable: {', '.join(sorted(READABLE))}")
    if any(p.suffix.lower() == ".xls" for p in entries):
        lines.append(
            "\n.xls is the pre-2007 binary format and is not supported. "
            "Open it in Excel and Save As .xlsx."
        )
    if not any(p.suffix for p in entries):
        lines.append(
            "\nSome files have no extension at all. Windows Explorer hides "
            "known extensions by default (View > Show > File name "
            "extensions) -- what looks like a bare name may be an .xls."
        )
    return "\n".join(lines)


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

    if fmt in {"xlsx", "excel"}:
        return _write_workbook(frames, text)

    written: dict[str, str] = {}
    for name, frame in frames.items():
        target = f"{text}/{name.lower()}.{fmt}"
        write_table(frame, target)
        written[name] = target
    return written


def _write_workbook(
    frames: dict[str, pd.DataFrame], directory: str
) -> dict[str, str]:
    """One workbook, one sheet per domain, every cell stored as text.

    Excel output exists for a reason stronger than convenience. A published
    tier written as CSV and then opened in Excel -- which is what happens to
    it -- is a tier Excel gets to reinterpret on the way in: 2015-03 becomes a
    date, site 002 loses its zero, a long subject number turns into
    scientific notation. The de-identification survives that; the data does
    not.

    Cells written as text are read back as text, so the values a steward
    approved are the values the analyst sees.
    """
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "writing .xlsx requires openpyxl: pip install 'deidkit[excel]'"
        ) from exc
    if is_remote(directory):
        raise ValueError(
            "xlsx output to object storage is not supported; write locally "
            "and upload, or use --format parquet"
        )
    target = f"{directory}/{Path(directory).name}.xlsx"
    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        for name, frame in frames.items():
            # Sheet names: 31 characters, and none of : \ / ? * [ ]
            sheet = re.sub(r"[:\\/?*\[\]]", "_", str(name))[:31] or "SHEET"
            frame.astype("string").to_excel(writer, sheet_name=sheet, index=False)
        for sheet in writer.book.worksheets:
            for row in sheet.iter_rows():
                for cell in row:
                    cell.number_format = "@"  # text, so Excel stops guessing
    return {name: f"{target}::{name}" for name in frames}


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
