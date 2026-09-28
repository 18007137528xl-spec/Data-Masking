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


def test_reading_again_does_not_silently_discard_a_review(session, drop):
    """A steward who re-clicks Read after an hour of review loses nothing.

    The CLI refuses to overwrite a filled sheet without --force-decisions. The
    console used to do it without asking, and a browser makes the click easy.
    """
    out = web.do_profile(session, {"directory": str(drop), "raw": True})
    web.do_decide(session, {"rows": [{"index": 0, "decision": "OK"}]})

    with pytest.raises(web.ApiError) as err:
        web.do_profile(session, {"directory": str(drop), "raw": True})
    assert err.value.extra["needs_force"] is True
    assert err.value.extra["decided"] == 1
    assert session.state()["decided"] == 1  # nothing was touched

    again = web.do_profile(session, {"directory": str(drop), "raw": True, "force": True})
    assert again["state"]["decided"] == 0
    assert len(again["rows"]) == len(out["rows"])


def test_the_page_can_rebuild_itself_after_a_refresh(session, drop, tmp_path):
    """Everything the page shows comes back from the server, at every stage."""
    assert web.do_resume(session)["rows"] == []

    out = web.do_profile(session, {"directory": str(drop), "raw": True})
    web.do_decide(session, {"rows": [{"index": 0, "decision": "CHANGE",
                                      "decision_treatment": "drop"}]})
    back = web.do_resume(session)
    assert back["rows"][0]["decision"] == "CHANGE"
    assert back["rows"][0]["decision_treatment"] == "drop"
    assert back["notes"] == out["notes"]
    assert back["approval"] is None and back["run"] is None

    web.do_decide(session, {"rows": [
        {"index": r["index"], "decision": "OK"} for r in out["rows"][1:]
    ]})
    web.do_approve(session, {"approved_by": "steward"})
    assert web.do_resume(session)["approval"]["fingerprint"].startswith("sha256:")

    web.do_run(session, {"out": str(tmp_path / "t"), "vault": str(tmp_path / "v.db")})
    back = web.do_resume(session)
    assert back["run"]["queue_rows"] == 0
    assert "summary" in back["run"]


def test_every_risk_field_the_page_reads_is_one_the_server_sends():
    """The k verdict box never rendered: the page read risk.k, the server sent k_min.

    Nothing failed -- the box was simply absent, and a steward who has never
    seen it cannot miss it. This ties the page to the payload so a rename on
    either side breaks a test instead of quietly removing the verdict.
    """
    import re

    from deidkit.risk import RiskReport

    page = (Path(web.__file__).parent / "console.html").read_text(encoding="utf-8")
    read = set(re.findall(r"\brisk\.(\w+)", page))
    assert read, "the page reads no risk fields at all"

    import inspect

    # the authoritative list: the keys to_dict actually emits
    sent = set(re.findall(r'"(\w+)":', inspect.getsource(RiskReport.to_dict)))
    missing = sorted(read - sent)
    assert not missing, f"the page reads risk fields the server never sends: {missing}"


def test_changing_a_decision_after_signing_withdraws_the_signature(session, drop, tmp_path):
    """Signed rules A, reviewed rules B, published under A -- not possible."""
    out = web.do_profile(session, {"directory": str(drop), "raw": True})
    web.do_decide(session, {"rows": [
        {"index": r["index"], "decision": "OK"} for r in out["rows"]
    ]})
    web.do_approve(session, {"approved_by": "steward"})
    assert session.state()["approved"]

    # re-sending the same decision is not a change
    same = web.do_decide(session, {"rows": [{"index": 0, "decision": "OK"}]})
    assert same["approval_withdrawn"] is False
    assert session.state()["approved"]

    moved = web.do_decide(session, {"rows": [
        {"index": 0, "decision": "CHANGE", "decision_treatment": "drop"}
    ]})
    assert moved["approval_withdrawn"] is True
    assert not session.state()["approved"]
    with pytest.raises(web.ApiError, match="not been approved"):
        web.do_run(session, {"out": str(tmp_path / "t"), "vault": str(tmp_path / "v.db")})


def test_the_guide_is_self_contained():
    page = Path(web.__file__).parent / "guide.html"
    if not page.exists():
        pytest.skip("guide not built")
    text = page.read_text(encoding="utf-8")
    for host in ("http://", "https://", "//cdn", "googleapis"):
        # the one http:// allowed is the loopback address the guide tells you to open
        assert text.replace("http://127.0.0.1", "").count(host) == 0, host


def test_a_port_already_in_use_is_a_sentence_not_a_traceback(capsys):
    """The usual cause is the console already running in another window."""
    import socket

    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = busy.getsockname()[1]
        assert web.serve(port=port) == 1
    err = capsys.readouterr().err
    assert "may already be running" in err
    assert f"--port {port + 1}" in err


def test_the_launchers_find_the_key_the_right_way_round():
    """A key already configured on the machine wins; the dev key is a fallback.

    Reading the dev key first would silently override a production KMS setup
    with the key that sits beside the vault -- the exact arrangement the
    production configuration exists to avoid.
    """
    root = Path(web.__file__).resolve().parents[2]
    bat = (root / "serve.bat").read_text()
    sh = (root / "serve.sh").read_text()
    assert bat.index("if defined DEIDKIT_KEY_URI") < bat.index("dev-vault.key\" (")
    assert bat.index("if defined DEIDKIT_VAULT_KEY") < bat.index("set /p DEIDKIT_VAULT_KEY")
    assert sh.index("DEIDKIT_KEY_URI:-") < sh.index("tr -d")
    for text in (bat, sh):
        assert "DEVELOPMENT" in text
        assert "serve --open" in text
