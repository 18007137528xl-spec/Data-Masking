"""The review console.

A second front door onto the pipeline is a second chance to lose the property
that makes it trustworthy. What is asserted here is that the console is only a
different way to FILL the decision sheet, never a way around it:

* a sheet with a blank row is refused, with the same message the CLI gives
* the contract it produces carries a real signature over the real rules
* the key never travels over HTTP
* the console binds to loopback

The happy path is tested through the API functions rather than over a socket:
the HTTP layer here is a thin shell over these calls, and a test that binds a
port tests the operating system more than the code.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from deidkit import web
from deidkit.vault import Vault


@pytest.fixture
def drop(tmp_path) -> Path:
    d = tmp_path / "quarantine"
    d.mkdir()
    rows = []
    for s in range(1, 25):
        site = f"{(s % 3) + 1:03d}"
        for v in (1, 2, 3):
            rows.append(
                {
                    "SUBJECT": f"{site}-{s:04d}",
                    "SITEID": site,
                    "VISIT": f"V{v}",
                    "LBDAT": f"{(s % 28) + 1:02d}/03/2025",
                    "LBORRES": 130 + s,
                }
            )
    pd.DataFrame(rows).to_csv(d / "LB.csv", index=False)
    return d


@pytest.fixture
def session(monkeypatch) -> web.Session:
    monkeypatch.setenv("DEIDKIT_VAULT_KEY", Vault.generate_key())
    return web.Session()


def test_the_console_walks_the_five_steps(session, drop, tmp_path):
    out = web.do_profile(session, {"directory": str(drop), "raw": True})
    assert out["state"]["rules"] > 0
    assert out["state"]["decided"] == 0

    # every row starts blank, and approval refuses while any of them is
    with pytest.raises(web.ApiError, match="no decision"):
        web.do_approve(session, {"approved_by": "steward"})

    web.do_decide(
        session,
        {"rows": [{"index": r["index"], "decision": "OK"} for r in out["rows"]]},
    )
    assert session.state()["decided"] == session.state()["rules"]

    signed = web.do_approve(
        session,
        {
            "approved_by": "steward@example.com",
            "contract_out": str(tmp_path / "study.yaml"),
        },
    )
    assert signed["fingerprint"].startswith("sha256:")
    assert Path(signed["contract_written_to"]).exists()

    ran = web.do_run(
        session,
        {
            "out": str(tmp_path / "tier"),
            "vault": str(tmp_path / "vault.db"),
            "operator": "steward",
            "format": "csv",
        },
    )
    assert ran["state"]["ran"] is True
    published = Path(ran["written"]["LB"])
    assert published.exists()

    frame = pd.read_csv(published, dtype=str)
    # the shaped surrogate and its site segment survive the console path too
    for subj, site in zip(frame["SUBJECT"], frame["SITEID"]):
        assert subj.startswith(site)
    assert set(frame["SUBJECT"]).isdisjoint(
        set(pd.read_csv(drop / "LB.csv", dtype=str)["SUBJECT"])
    )

    manifest = json.loads(Path(ran["manifest"]).read_text())
    assert manifest["review"]["reviewed"] is True
    assert manifest["review"]["approved_by"] == "steward@example.com"
    assert "key_source" in manifest["vault"]


def test_approval_needs_a_named_person(session, drop):
    out = web.do_profile(session, {"directory": str(drop), "raw": True})
    web.do_decide(
        session,
        {"rows": [{"index": r["index"], "decision": "OK"} for r in out["rows"]]},
    )
    with pytest.raises(web.ApiError, match="Who is approving"):
        web.do_approve(session, {"approved_by": "   "})


def test_running_without_an_approval_is_refused(session, drop, tmp_path):
    web.do_profile(session, {"directory": str(drop), "raw": True})
    with pytest.raises(web.ApiError, match="not been approved"):
        web.do_run(
            session, {"out": str(tmp_path / "t"), "vault": str(tmp_path / "v.db")}
        )


def test_a_missing_folder_is_a_sentence_not_a_traceback(session):
    with pytest.raises(web.ApiError, match="is not a folder"):
        web.do_profile(session, {"directory": "/no/such/place"})
    with pytest.raises(web.ApiError, match="No folder given"):
        web.do_profile(session, {})


def test_the_key_is_never_taken_from_the_request(session, drop, tmp_path):
    """A key typed into a browser is a key in the request log and the autofill.

    The run path reads the environment, as the CLI does; a key supplied in the
    request body must have no effect at all.
    """
    out = web.do_profile(session, {"directory": str(drop), "raw": True})
    web.do_decide(
        session,
        {"rows": [{"index": r["index"], "decision": "OK"} for r in out["rows"]]},
    )
    web.do_approve(session, {"approved_by": "steward"})

    real = os.environ.pop("DEIDKIT_VAULT_KEY")
    try:
        with pytest.raises(web.ApiError):
            web.do_run(
                session,
                {
                    "out": str(tmp_path / "t"),
                    "vault": str(tmp_path / "v.db"),
                    "key": real,
                    "vault_key": real,
                    "DEIDKIT_VAULT_KEY": real,
                },
            )
    finally:
        os.environ["DEIDKIT_VAULT_KEY"] = real


def test_the_page_is_self_contained():
    """No CDN. The server that holds PHI usually has no route to one."""
    page = (Path(web.__file__).parent / "console.html").read_text(encoding="utf-8")
    assert "<title>" in page
    for host in ("http://", "https://", "//cdn", "cdnjs", "unpkg", "googleapis"):
        assert host not in page, f"the console reaches out to {host}"


def test_the_default_bind_is_loopback():
    import inspect

    sig = inspect.signature(web.serve)
    assert sig.parameters["host"].default == "127.0.0.1"


def test_a_raw_looking_drop_profiled_without_raw_is_flagged(session, drop):
    out = web.do_profile(session, {"directory": str(drop), "raw": False})
    levels = {n["level"] for n in out["notes"]}
    assert "warn" in levels
    assert any("raw EDC extract" in n["text"] for n in out["notes"])
