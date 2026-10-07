"""Protocol masking: propose, decide, write, and prove nothing survived."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

docx = pytest.importorskip("docx")
pytest.importorskip("pypdf")
pytest.importorskip("lxml")

from deidkit import Vault, protocol as pr  # noqa: E402
from deidkit import web  # noqa: E402


@pytest.fixture()
def vault(tmp_path) -> Vault:
    v = Vault(tmp_path / "vault.db", key=Vault.generate_key(), operator="pytest")
    yield v
    v.close()


def make_protocol(path: Path) -> Path:
    d = docx.Document()
    sec = d.sections[0]
    sec.header.paragraphs[0].text = "Protocol No.: BDM-AI-2025-001  Confidential"
    sec.footer.paragraphs[0].text = "Contact: jane.roe@acme-onc.com, Tel: +1 617 555 0142"
    d.add_heading("A Phase 3 Study of Zentolimab (ACM-4417) in NSCLC", 0)
    d.add_paragraph("Sponsor: Acme Oncology Inc.")
    p = d.add_paragraph()
    r = p.add_run("Subjects receive Zento")
    r.bold = True
    p.add_run("limab 200 mg Q3W or placebo. Registered as NCT01234567.")
    d.add_paragraph("Coordinating investigator: Dr. Maria Gonzalez, Massachusetts General Hospital.")
    d.add_paragraph("本研究由北京协和医院牵头，申办方：上海恒星医药有限公司，研究者：王建国。")
    t = d.add_table(rows=1, cols=2)
    t.cell(0, 0).text = "EudraCT"
    t.cell(0, 1).text = "2024-512345-12-00"
    d.core_properties.author = "Jane Roe"
    d.core_properties.title = "Zentolimab protocol"
    d.save(path)
    return path


def tiny_pdf(path: Path, lines: list[str]) -> Path:
    body = "BT /F1 12 Tf 72 720 Td " + " ".join(f"({x}) Tj 0 -16 Td" for x in lines) + " ET"
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(body)} >>\nstream\n{body}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    path.write_bytes(out)
    return path


def _by_term(terms):
    return {t.term: t for t in terms}


def test_the_scan_finds_every_kind_of_name(tmp_path, vault):
    terms = _by_term(pr.scan([make_protocol(tmp_path / "p.docx")], vault))
    expect = {
        "Zentolimab": "drug", "ACM-4417": "drug", "Acme Oncology Inc": "sponsor",
        "BDM-AI-2025-001": "study_id", "NCT01234567": "study_id",
        "2024-512345-12-00": "study_id", "Maria Gonzalez": "person",
        "王建国": "person", "jane.roe@acme-onc.com": "contact",
        "+1 617 555 0142": "contact", "Massachusetts General Hospital": "site",
        "北京协和医院": "site", "上海恒星医药有限公司": "sponsor",
    }
    for term, cat in expect.items():
        assert term in terms, term
        assert terms[term].category == cat, term
    # the split-run mention ("Zento" bold + "limab") is counted too
    assert terms["Zentolimab"].count == 3


def test_a_drug_the_data_labelled_keeps_its_label(tmp_path, vault):
    vault.label_map("treatment_product", ["ZENTOLIMAB"], prefix="DRUG")
    data_label = vault.labelled("treatment_product")["ZENTOLIMAB"]
    terms = _by_term(pr.scan([make_protocol(tmp_path / "p.docx")], vault))
    assert terms["Zentolimab"].proposed == data_label
    assert terms["Zentolimab"].source == "data"


def test_the_masked_copy_keeps_its_structure_and_loses_every_term(tmp_path, vault):
    (tmp_path / "in").mkdir()
    src = make_protocol(tmp_path / "in" / "p.docx")
    terms = pr.scan([src], vault)
    for t in terms:
        t.decision = "OK"
    res = pr.apply([src], terms, vault, tmp_path / "out", approved_by="tester")
    assert res["manifest"]["residual_occurrences"] == 0
    out = tmp_path / "out" / "p.docx"
    text = "\n".join(pr.read_text(out))
    for t in terms:
        assert t.term.lower() not in text.lower(), t.term
    # still a Word document with its table, header and the bold run
    d = docx.Document(out)
    assert d.tables[0].cell(0, 0).text == "EudraCT"
    assert any(r.bold for p in d.paragraphs for r in p.runs)
    assert "Protocol No.:" in d.sections[0].header.paragraphs[0].text
    assert d.core_properties.author == ""
    # the originals are beside the output, not in it
    assert (tmp_path / "out_review" / "protocol_terms.csv").exists()
    assert not any("terms" in n for n in zipfile.ZipFile(out).namelist())


def test_keep_and_change_are_honoured(tmp_path, vault):
    src = make_protocol(tmp_path / "p.docx")
    terms = pr.scan([src], vault)
    for t in terms:
        t.decision = "OK"
    by = _by_term(terms)
    by["NCT01234567"].decision = "KEEP"
    by["Acme Oncology Inc"].decision = "CHANGE"
    by["Acme Oncology Inc"].replacement = "the Sponsor"
    pr.apply([src], terms, vault, tmp_path / "out", approved_by="tester")
    text = "\n".join(pr.read_text(tmp_path / "out" / "p.docx"))
    assert "NCT01234567" in text
    assert "Sponsor: the Sponsor" in text


def test_nothing_is_written_until_every_term_is_decided(tmp_path, vault):
    src = make_protocol(tmp_path / "p.docx")
    terms = pr.scan([src], vault)
    with pytest.raises(ValueError, match="no decision"):
        pr.apply([src], terms, vault, tmp_path / "out", approved_by="tester")
    for t in terms:
        t.decision = "OK"
    with pytest.raises(ValueError, match="named person"):
        pr.apply([src], terms, vault, tmp_path / "out", approved_by=" ")
    assert not (tmp_path / "out").exists()


def test_a_pdf_is_masked_as_text(tmp_path, vault):
    src = tiny_pdf(tmp_path / "p.pdf", ["Sponsor: Acme Oncology Inc.",
                                        "Study drug Zentolimab, NCT01234567."])
    terms = pr.scan([src], vault)
    for t in terms:
        t.decision = "OK"
    res = pr.apply([src], terms, vault, tmp_path / "out", approved_by="tester")
    out = (tmp_path / "out" / "p.txt").read_text(encoding="utf-8")
    assert "Zentolimab" not in out and "NCT01234567" not in out and "Acme" not in out
    assert res["manifest"]["residual_occurrences"] == 0


def test_the_console_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("DEIDKIT_VAULT_KEY", Vault.generate_key())
    (tmp_path / "in").mkdir()
    src = make_protocol(tmp_path / "in" / "p.docx")
    s = web.Session()
    j = web.do_p_scan(s, {"path": str(src), "vault": str(tmp_path / "v.db")})
    assert j["rows"]
    j = web.do_p_add(s, {"term": "Q3W", "category": "drug"})
    assert j["rows"][0]["term"] == "Q3W" and j["rows"][0]["decision"] == "OK"
    with pytest.raises(web.ApiError):
        web.do_p_apply(s, {"approved_by": "t", "out": str(tmp_path / "o")})
    web.do_p_decide(s, {"rows": [{"index": i, "decision": "OK"} for i in range(len(j["rows"]))]})
    with pytest.raises(web.ApiError, match="other than the folder"):
        web.do_p_apply(s, {"approved_by": "t", "out": str(tmp_path / "in")})
    j = web.do_p_apply(s, {"approved_by": "t", "out": str(tmp_path / "o")})
    assert j["result"]["manifest"]["residual_occurrences"] == 0
    with pytest.raises(web.ApiError) as e:
        web.do_p_scan(s, {"path": str(src), "vault": str(tmp_path / "v.db")})
    assert e.value.extra.get("needs_force")
