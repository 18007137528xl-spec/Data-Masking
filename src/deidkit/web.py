"""A local review app for the five gated steps.

The pipeline was usable only from a terminal, by someone who could hold the
order of five commands and the difference between two CSVs in their head. The
person who actually owns the decisions -- who knows whether MHTERM is safe to
publish and whether this study's sites are small enough to worry about -- is
often not that person. This serves the same five steps as one page.

What it deliberately does NOT change:

* **The gate.** Approval still goes through ``decisions.apply_sheet``, still
  refuses a sheet with a blank row, and still produces a contract signed over
  the rules. The web form is another way to fill the sheet, not another way to
  get past it.
* **Where the data is.** Nothing is uploaded. The server reads the drop from
  the filesystem it is running on, which is the PHI server, and writes the
  tiers back to it. The browser receives counts, column names, and rules.
* **Where the key is.** The vault key is read from the process environment,
  exactly as the CLI reads it. It is never accepted from the browser: a key
  typed into a form ends up in the request log, the browser's autofill, and
  the operator's screen-share.

Bound to 127.0.0.1. A PHI server with a de-identification console on its
external interface is a worse problem than the one this solves, so binding
anywhere else takes an explicit flag and prints a warning that says why.

Implemented on ``http.server`` rather than a framework on purpose: the machine
this runs on frequently has no route to a package index, and a review console
that cannot be installed is not a review console.
"""

from __future__ import annotations

import json
import threading
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from . import decisions as dec
from . import io as dio
from . import profile as prof
from .contract import Contract
from .pipeline import DeidPipeline
from .vault import Vault, VaultError

# ----------------------------------------------------------------------
# session state
# ----------------------------------------------------------------------


@dataclass
class Session:
    """One study being worked on. Single user, single drop, in memory.

    Held in memory rather than on disk because it is scaffolding: the durable
    artefacts are the approved contract, the manifest and the tier, and those
    are written through the same code the CLI uses. Restarting the server
    loses the half-filled sheet and nothing else.
    """

    directory: str | None = None
    raw: bool = False
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    checksums: dict[str, str] = field(default_factory=dict)
    draft: Contract | None = None
    sheet: pd.DataFrame | None = None
    approved: Contract | None = None
    result: Any = None
    out_dir: str | None = None
    # What the page needs to rebuild itself after a refresh. The browser holds
    # no state of its own: close the tab, reopen the address, and the review
    # carries on from where it was. Only restarting the SERVER loses it.
    notes: list[dict[str, str]] = field(default_factory=list)
    last_approval: dict[str, Any] | None = None
    last_run: dict[str, Any] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def state(self) -> dict[str, Any]:
        sheet = self.sheet
        filled = 0 if sheet is None else int((sheet["decision"] != "").sum())
        return {
            "directory": self.directory,
            "raw": self.raw,
            "domains": {k: len(v) for k, v in self.frames.items()},
            "rules": 0 if sheet is None else len(sheet),
            "decided": filled,
            "approved": self.approved is not None,
            "approved_by": (
                self.approved.approval.approved_by
                if self.approved and self.approved.approval
                else None
            ),
            "fingerprint": (
                self.approved.approval.rules_fingerprint
                if self.approved and self.approved.approval
                else None
            ),
            "ran": self.result is not None,
            "out_dir": self.out_dir,
        }


class ApiError(Exception):
    """A message meant for the person, not a stack trace.

    ``extra`` travels to the page alongside the message, for the one case
    where the page has to offer a choice rather than just report: discarding
    decisions already recorded.
    """

    def __init__(self, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.extra = extra


# ----------------------------------------------------------------------
# the five steps
# ----------------------------------------------------------------------
def do_profile(s: Session, body: dict[str, Any]) -> dict[str, Any]:
    directory = (body.get("directory") or "").strip()
    if not directory:
        raise ApiError("No folder given.")

    # Reading a drop starts the review over. With decisions already recorded
    # that silently threw away a steward's afternoon -- the CLI refuses the
    # same thing without --force-decisions, and so does this.
    decided = s.state()["decided"]
    if decided and not body.get("force"):
        raise ApiError(
            f"{decided} decision(s) are already recorded for {s.directory}. "
            "Reading a folder starts the review over and discards them.",
            needs_force=True,
            decided=decided,
        )

    if not Path(directory).is_dir():
        raise ApiError(f"{directory} is not a folder on this server.")

    try:
        frames, checksums = dio.load_study(directory)
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the operator
        raise ApiError(str(exc)) from exc
    if not frames:
        raise ApiError(dio.explain_empty(directory))

    raw = bool(body.get("raw"))
    draft, suggestions = prof.draft_contract(
        frames,
        source=body.get("source") or Path(directory).name,
        raw_edc=raw,
        k_target=int(body.get("k_target") or 5),
        join_key_template=body.get("offset_key") or None,
        subject_id_template=body.get("id_template") or None,
    )
    sheet = dec.build_sheet(draft, suggestions)

    s.directory, s.raw = directory, raw
    s.frames, s.checksums = frames, checksums
    s.draft, s.sheet = draft, sheet
    s.approved = s.result = None
    s.last_approval = s.last_run = None
    s.notes = _profile_notes(draft, frames, raw)

    return {
        "state": s.state(),
        "notes": s.notes,
        "rows": _sheet_rows(sheet),
    }


def _profile_notes(
    contract: Contract, frames: dict[str, pd.DataFrame], raw: bool
) -> list[dict[str, str]]:
    """The warnings the CLI prints, as structured items rather than a wall.

    A warning nobody reads is not a warning, and in a browser a paragraph of
    grey text is exactly that. Each one carries its own severity so the page
    can put the two that stop a run above the ones that are informational.
    """
    from .contract import NEEDS_ANCHOR

    notes: list[dict[str, str]] = []
    anchor_used = any(
        f.treatment in NEEDS_ANCHOR for d in contract.domains for f in d.fields
    )
    anchor_cols = set(map(str, frames.get(contract.anchor.domain, pd.DataFrame()).columns))
    if anchor_used and contract.anchor.date_column not in anchor_cols:
        notes.append({
            "level": "halt",
            "text": f"No reference start date found: {contract.anchor.date_column} "
            f"is not in {contract.anchor.domain}. Any study-day conversion "
            "will refuse to run. Audit timestamps are deliberately not used "
            "as an anchor -- they date the paperwork, not the patient.",
        })
    if anchor_used and contract.anchor.subject_column not in anchor_cols:
        notes.append({
            "level": "halt",
            "text": f"The anchor's subject column {contract.anchor.subject_column} "
            f"is not in {contract.anchor.domain}.",
        })
    if not raw and not any(
        str(c).upper().endswith("DTC") for f in frames.values() for c in f.columns
    ):
        notes.append({
            "level": "warn",
            "text": "No column in this drop ends in DTC, so it is very likely a "
            "raw EDC extract rather than SDTM. Without the raw setting, dates "
            "become study days -- which needs an anchor this drop has no "
            "column for, and discards the written date form that a raw -> "
            "SDTM model has to learn.",
        })
    if raw:
        notes.append({
            "level": "info",
            "text": "Raw mode: dates are shifted per subject with their written "
            "form intact.",
        })
    return notes


def _sheet_rows(sheet: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {k: ("" if pd.isna(v) else v) for k, v in row.items()} | {"index": i}
        for i, row in enumerate(sheet.to_dict("records"))
    ]


def do_decide(s: Session, body: dict[str, Any]) -> dict[str, Any]:
    """Record decisions. Accepts a partial set; approval is what demands all."""
    if s.sheet is None:
        raise ApiError("Nothing profiled yet.")
    changed = False
    for item in body.get("rows") or []:
        i = int(item["index"])
        for col in ("decision", "decision_treatment", "decision_params", "steward_note"):
            if col in item:
                new = str(item[col] or "")
                if s.sheet.at[i, col] != new:
                    s.sheet.at[i, col] = new
                    changed = True

    # A signature covers the rules as they were when it was given. Change one
    # afterwards and the approval in hand no longer describes the sheet, but
    # Run would still have used it -- signed rules A, reviewed rules B,
    # published under A. The approval is withdrawn instead, and the page has
    # to be signed again before anything runs.
    withdrawn = changed and s.approved is not None
    if withdrawn:
        s.approved = None
        s.last_approval = s.last_run = None
        s.result = None
    return {"state": s.state(), "approval_withdrawn": withdrawn}


def do_approve(s: Session, body: dict[str, Any]) -> dict[str, Any]:
    if s.sheet is None or s.draft is None:
        raise ApiError("Nothing profiled yet.")
    who = (body.get("approved_by") or "").strip()
    if not who:
        raise ApiError(
            "Who is approving? This is recorded in the contract and the "
            "manifest, and it should be a person rather than a service account."
        )
    try:
        approved, stats = dec.apply_sheet(
            s.sheet,
            s.draft,
            approved_by=who,
            frames=s.frames,
            plan_file="(reviewed in the deidkit console)",
            note=body.get("note") or None,
        )
    except dec.DecisionError as exc:
        raise ApiError(str(exc)) from exc

    s.approved = approved
    out = (body.get("contract_out") or "").strip()
    written = None
    if out:
        approved.to_yaml(out)
        written = out
    s.last_approval = {
        "stats": stats,
        "contract_written_to": written,
        "fingerprint": approved.approval.rules_fingerprint,
    }
    s.last_run = None
    return {"state": s.state(), **s.last_approval}


def do_run(s: Session, body: dict[str, Any]) -> dict[str, Any]:
    if s.approved is None:
        raise ApiError("This contract has not been approved.")
    out_dir = (body.get("out") or "").strip()
    vault_path = (body.get("vault") or "").strip()
    if not out_dir or not vault_path:
        raise ApiError("Both an output folder and a vault file are needed.")

    try:
        vault = Vault(vault_path, operator=body.get("operator") or "console")
    except VaultError as exc:
        raise ApiError(str(exc)) from exc

    pipeline = DeidPipeline(
        s.approved, vault, operator=body.get("operator") or "console"
    )
    try:
        result = pipeline.run(s.frames, checksums=s.checksums)
    except Exception as exc:  # noqa: BLE001
        raise ApiError(f"{type(exc).__name__}: {exc}") from exc

    written = dio.write_study(
        result.frames, out_dir, fmt=body.get("format") or "csv"
    )

    # The same two blocks the CLI adds, for the same reason. A manifest that is
    # silent about review lets a tier produced here be mistaken later for one
    # nobody signed -- and a console is exactly where that mistake would be
    # made, because approving took two clicks rather than a command.
    result.manifest["vault"] = {"key_source": vault.key_source}
    result.manifest["review"] = (
        {"reviewed": True, **s.approved.approval.model_dump(mode="json")}
        if s.approved.approval is not None
        else {"reviewed": False, "reason": "no steward approved this contract"}
    )
    manifest_path = str(Path(out_dir) / "manifest.json")
    dio.write_text(json.dumps(result.manifest, indent=2, default=str), manifest_path)

    review_dir = f"{out_dir}_review"
    queue_path = None
    if not result.review_queue.empty:
        queue_path = str(Path(review_dir) / "review_queue.csv")
        dio.write_text(result.review_queue.to_csv(index=False), queue_path)
    if result.risk_report is not None:
        dio.write_text(
            json.dumps(
                result.risk_report.to_dict(include_class_values=True),
                indent=2,
                default=str,
            ),
            str(Path(review_dir) / "risk_detail.json"),
        )

    s.result, s.out_dir = result, out_dir
    s.last_run = {
        "summary": result.summary(),
        "written": written,
        "manifest": manifest_path,
        "queue": queue_path,
        "review_dir": review_dir,
        "risk": (
            result.risk_report.to_dict() if result.risk_report is not None else None
        ),
        "queue_rows": len(result.review_queue),
    }
    return {"state": s.state(), **s.last_run}


def do_queue(s: Session) -> dict[str, Any]:
    """The free-text queue. This is the one response that carries PHI.

    It has to: the queue exists so a person can read the original text and
    judge it, and a redacted queue would be unreviewable. It travels over
    loopback to a browser on the same machine, and the page says plainly what
    is on the screen.
    """
    if s.result is None:
        raise ApiError("Nothing has been run yet.")
    q = s.result.review_queue
    return {
        "columns": list(q.columns),
        "rows": q.head(500).to_dict("records"),
        "total": len(q),
        "phi": True,
    }


def do_resume(s: Session) -> dict[str, Any]:
    """Everything the page needs to put itself back where it was."""
    return {
        "state": s.state(),
        "rows": _sheet_rows(s.sheet) if s.sheet is not None else [],
        "notes": s.notes,
        "approval": s.last_approval,
        "run": s.last_run,
    }


ROUTES = {
    "/api/state": lambda s, b: {"state": s.state()},
    "/api/profile": do_profile,
    "/api/decide": do_decide,
    "/api/approve": do_approve,
    "/api/run": do_run,
    "/api/treatments": lambda s, b: {"text": dec.treatment_reference()},
}


# ----------------------------------------------------------------------
# http
# ----------------------------------------------------------------------
def _page() -> str:
    return (Path(__file__).parent / "console.html").read_text(encoding="utf-8")


def _guide() -> str | None:
    """The illustrated guide, served from the same process.

    Shipped inside the package rather than linked from a wiki, because the
    machine this runs on often has no route to a wiki -- and a steward stuck on
    step 2 should not need a second machine to find out what step 2 means.
    """
    path = Path(__file__).parent / "guide.html"
    return path.read_text(encoding="utf-8") if path.exists() else None


def make_handler(session: Session):
    class Handler(BaseHTTPRequestHandler):
        server_version = "deidkit"

        def log_message(self, fmt, *args):  # noqa: A003 - quiet by default
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            # Nothing here should ever be embedded, cached by a proxy, or
            # reached from a page the operator happens to have open.
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: dict) -> None:
            self._send(
                code,
                json.dumps(payload, default=str).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send(200, _page().encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/guide":
                text = _guide()
                if text is None:
                    self._json(404, {"error": "the guide is not installed"})
                else:
                    self._send(200, text.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/queue":
                self._guarded(lambda: do_queue(session))
                return
            if path == "/api/resume":
                self._guarded(lambda: do_resume(session))
                return
            if path == "/api/state":
                self._guarded(lambda: {"state": session.state()})
                return
            self._json(404, {"error": "no such path"})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            fn = ROUTES.get(path)
            if fn is None:
                self._json(404, {"error": "no such path"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json(400, {"error": "malformed request"})
                return
            self._guarded(lambda: fn(session, body))

        def _guarded(self, fn) -> None:
            """One place where a failure becomes a sentence rather than a 500.

            An operator who is not the author gets told what went wrong and
            what to do; the traceback goes to the console the server was
            started from, where an engineer can find it.
            """
            try:
                with session.lock:
                    self._json(200, fn())
            except ApiError as exc:
                self._json(400, {"error": str(exc), **exc.extra})
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self._json(
                    500,
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "detail": "The full traceback is on the terminal that "
                        "started the server.",
                    },
                )

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    session = Session()
    httpd = ThreadingHTTPServer((host, port), make_handler(session))
    print(f"deidkit console on http://{host}:{port}")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(
            "  WARNING: bound to a non-loopback address. This console reads the\n"
            "  quarantine drop and displays unredacted free text from the review\n"
            "  queue. On the server that holds PHI, that is a bigger exposure\n"
            "  than the manual workflow it replaces. Put it behind loopback and\n"
            "  an SSH tunnel unless something else is already authenticating it."
        )
    print("  the vault key is read from this process's environment, not the browser")
    print("  Ctrl-C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
