"""Protocol masking: propose, decide, write, and prove nothing survived."""

from __future__ import annotations

import re
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


def test_a_pdf_is_redacted_in_place(tmp_path, vault):
    pm = pytest.importorskip("pymupdf")
    src = tmp_path / "p.pdf"
    doc = pm.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Sponsor: Acme Oncology Inc.", fontsize=11)
    page.insert_text((72, 96), "Study drug Zentolimab, NCT01234567.", fontsize=11)
    page.insert_text((72, 120), "研究者：王建国，北京协和医院", fontname="china-s", fontsize=11)
    page.draw_rect(pm.Rect(60, 60, 400, 130))  # a table border must survive
    doc.new_page().insert_text((72, 72), "Page two mentions Zentolimab again.")
    doc.set_metadata({"title": "Zentolimab protocol", "author": "Jane Roe"})
    doc.save(str(src))

    terms = pr.scan([src], vault)
    found = {t.term for t in terms}
    assert {"Zentolimab", "NCT01234567", "王建国", "北京协和医院"} <= found
    for t in terms:
        t.decision = "OK"
    res = pr.apply([src], terms, vault, tmp_path / "out", approved_by="tester")
    out = tmp_path / "out" / "p.pdf"
    assert out.exists() and not (tmp_path / "out" / "p.txt").exists()
    assert res["manifest"]["residual_occurrences"] == 0
    with pm.open(str(out)) as masked:
        assert masked.page_count == 2
        text = "".join(pg.get_text() for pg in masked)
        assert "Zentolimab" not in text and "NCT01234567" not in text
        assert "王建国" not in text and "Acme" not in text
        assert "DRUG" in text and "Sponsor:" in text  # replacement drawn, rest kept
        assert masked[0].get_drawings()  # the border is still there
        assert not (masked.metadata or {}).get("author")


def test_without_pymupdf_a_pdf_falls_back_to_text(tmp_path, vault, monkeypatch):
    monkeypatch.setattr(pr, "_pymupdf", lambda: None)
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


def test_the_names_a_review_found_left_behind(tmp_path, vault):
    """From a real review of a masked protocol: a surname left after its
    first name, the file name, an abbreviation, a code with a space, a brand
    in a reference, and two companies named without Inc. or Ltd."""
    d = docx.Document()
    d.add_paragraph("Prepared by @Xiaofeng Li")
    d.add_paragraph("Short title: EMPA add-on to metformin, 24 weeks")
    d.add_paragraph("Investigational product: empagliflozin (BI 10773)")
    d.add_paragraph("Developed by Boehringer Ingelheim and Eli Lilly.")
    d.add_paragraph("1. JARDIANCE (empagliflozin) tablets, prescribing information.")
    src = tmp_path / "EX-EMPA-301_Protocol.docx"
    d.save(src)

    terms = pr.scan([src], vault)
    by = _by_term(terms)
    assert "Xiaofeng Li" in by and by["Xiaofeng Li"].category == "person"
    for t in ("EMPA", "JARDIANCE", "BI 10773", "Boehringer Ingelheim", "Eli Lilly"):
        assert t in by, t
    # the abbreviation and the brand become the same label as the drug
    assert by["EMPA"].proposed == by["empagliflozin"].proposed
    assert by["JARDIANCE"].proposed == by["empagliflozin"].proposed
    assert by["BI 10773"].proposed == by["empagliflozin"].proposed
    for t in terms:
        t.decision = "OK"
    res = pr.apply([src], terms, vault, tmp_path / "out", approved_by="tester")
    (out,) = (tmp_path / "out").glob("*.docx")
    assert "EMPA" not in out.name
    text = "\n".join(pr.read_text(out))
    for gone in ("Li", "EMPA", "empagliflozin", "JARDIANCE", "10773",
                 "Boehringer", "Lilly", "Xiaofeng"):
        assert gone not in text.split() and gone.lower() not in text.lower().split(), gone
    assert res["manifest"]["residual_occurrences"] == 0
    manifest = (tmp_path / "out" / "protocol_manifest.json").read_text(encoding="utf-8")
    assert "EMPA" not in manifest


def test_a_name_wrapped_onto_the_next_pdf_line_is_still_redacted(tmp_path, vault):
    pm = pytest.importorskip("pymupdf")
    src = tmp_path / "p.pdf"
    doc = pm.open()
    page = doc.new_page()
    # one paragraph, wrapped by width so the company name breaks over two lines
    page.insert_textbox(pm.Rect(72, 72, 172, 160),
                        "Funded by Example Biologics, Inc and run by Dr. Xiaofeng Li",
                        fontsize=11)
    page.insert_text((72, 170), "Title: lead", fontsize=11)
    page.insert_text((72, 200), "Schedule of Activities", fontsize=11)
    page.insert_text((72, 214), "Clinic visits every 4 weeks.", fontsize=11)
    doc.save(str(src))

    terms = pr.scan([src], vault)
    by = _by_term(terms)
    assert "Example Biologics, Inc" in by
    assert "Xiaofeng Li" in by and "Xiaofeng Li Title" not in by
    assert not any("Activities Clinic" in t for t in by)
    for t in terms:
        t.decision = "OK"
    res = pr.apply([src], terms, vault, tmp_path / "out", approved_by="tester")
    assert res["manifest"]["residual_occurrences"] == 0, res["residual_terms"]
    text = "".join(pg.get_text() for pg in pm.open(str(tmp_path / "out" / "p.pdf")))
    assert "Example" not in text and "Biologics" not in text and "Xiaofeng" not in text


def test_compound_names_and_codes_are_high_and_standards_are_not_compounds(tmp_path, vault):
    d = docx.Document()
    d.add_paragraph("Subjects receive Zentolimab (ACM-4417) or SHR-1210 and BI 10773.")
    d.add_paragraph("Prohibited: ketoconazole. 本品为恒星替尼片。")
    d.add_paragraph("Conducted per ISO 14155 and IEC 62304; see Table 1001.")
    src = tmp_path / "p.docx"
    d.save(src)
    terms = _by_term(pr.scan([src], vault))
    for t in ("Zentolimab", "ACM-4417", "SHR-1210", "BI 10773", "ketoconazole"):
        assert terms[t].category == "drug" and terms[t].confidence == "high", t
    assert any("替尼" in k and v.confidence == "high" for k, v in terms.items())
    for t in ("ISO 14155", "IEC 62304"):
        assert t not in terms, t


def _scan_lines(tmp_path, vault, lines):
    d = docx.Document()
    for line in lines:
        d.add_paragraph(line)
    src = tmp_path / "names.docx"
    d.save(src)
    return {t.term: t for t in pr.scan([src], vault)}


def test_names_are_found_however_a_protocol_writes_them(tmp_path, vault):
    terms = _scan_lines(tmp_path, vault, [
        "Principal Investigator: John Smith",
        "Prepared by: Wei Zhang",
        "Approved by: Zhang Wei",
        "Sponsor Signatory", "Robert Brown", "Vice President, Clinical Development",
        "The team includes WANG Jianguo and Li Xiaofeng.",
        "Please contact Xiaofeng Li or Jian-Guo Chen.",
        "本研究由北京协和医院张伟主任医师牵头，李娜教授参与。",
        "联系人：刘洋；签字：陈静",
        "Study Statistician: Anna Müller",
        "Printed Name: Sarah Johnson",
        "Investigator: John A. Doe",
        "Kevin Lee", "Clinical Project Manager",
        "Name: Michael O'Neil",
        "Data Manager: HUANG Lei",
    ])
    for name in ("John Smith", "Wei Zhang", "Zhang Wei", "Robert Brown",
                 "WANG Jianguo", "Li Xiaofeng", "Xiaofeng Li", "Jian-Guo Chen",
                 "张伟", "李娜", "刘洋", "陈静", "Anna Müller", "Sarah Johnson",
                 "John A. Doe", "Kevin Lee", "Michael O'Neil", "HUANG Lei"):
        assert name in terms and terms[name].category == "person", name
        assert terms[name].confidence == "high", name
    for role in ("Sponsor Signatory", "Clinical Project Manager"):
        assert role not in terms, role


def test_headings_and_places_are_not_people(tmp_path, vault):
    terms = _scan_lines(tmp_path, vault, [
        "Adverse Events",
        "The investigator will record every adverse event.",
        "Informed Consent", "Investigator responsibilities",
        "Sites in Hong Kong and Long Term Follow-up visits.",
        "研究者应当在签署知情同意书后开始治疗。",
    ])
    people = {k for k, t in terms.items() if t.category == "person"}
    assert not people, people


def test_phone_numbers_are_found_without_a_label(tmp_path, vault):
    terms = _scan_lines(tmp_path, vault, [
        "Call +86 21 6123 4567 or 138-1234-5678 or 021-61234567.",
        "US (617) 555-0142, hotline 400-820-1234.",
        "Phone Number: 010 8765 4321",
        "Visit 2024-03-12, version 1.0.2, EudraCT 2024-512345-12-00, Day 120.",
    ])
    contacts = {k for k, t in terms.items() if t.category == "contact"}
    assert contacts == {"+86 21 6123 4567", "138-1234-5678", "021-61234567",
                        "(617) 555-0142", "400-820-1234", "010 8765 4321"}, contacts


def test_the_name_model_queues_full_names_only(tmp_path, vault):
    class Eager:  # stands in for Presidio, which calls any capital a person
        def detect(self, text):
            from deidkit.freetext import Finding
            out = []
            for m in re.finditer(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", text):
                out.append(Finding("PERSON", m.start(), m.end(), m.group(), 0.85, "fake"))
            return out

    d = docx.Document()
    for line in ("When the Site reports Concomitant Medications, see Week 4.",
                 "Questions go to Harriet Vance."):
        d.add_paragraph(line)
    src = tmp_path / "n.docx"
    d.save(src)
    terms = {t.term: t for t in pr.scan([src], vault, detector=Eager())}
    people = {k for k, t in terms.items() if t.category == "person"}
    assert people == {"Harriet Vance"}, people
    assert terms["Harriet Vance"].confidence == "medium"


def test_more_ways_a_protocol_writes_a_name(tmp_path, vault):
    terms = _scan_lines(tmp_path, vault, [
        "Investigator: Smith, John",
        "Sponsor Representative: SMITH, Jane",
        "Principal Investigator: JOHN DOE",
        "Reviewed and signed by Peter Parker MD.",
        "Cardiology input from Alan Grant, MD, FACC.",
        "Laura Palmer", "Department of Oncology", "Twin Peaks General",
        "Data reviewed by J. Watson and J.A. Holmes. Approved by U.S. Food and Drug Administration.",
        "Hong Kong team: Chan Tai Man; Taiwan team: Hsu Chih-Wei; Korea: Kim Min-jun.",
        "Contact Ellen Ripley (ellen.ripley@weyland.com) for safety questions.",
        "主要研究者", "王小明", "北京协和医院 肿瘤科",
        "赵丽", "联系电话：010-12345678",
        "Tom Hardy", "tom.hardy@example.org",
        "Questions: Lisa Wong.",
        "Adverse Events", "The investigator will report all events on Park Avenue.",
    ])
    people = {k for k, t in terms.items() if t.category == "person"}
    assert people == {
        "Smith, John", "SMITH, Jane", "JOHN DOE", "Peter Parker", "Alan Grant",
        "Laura Palmer", "J. Watson", "J.A. Holmes", "Chan Tai Man", "Hsu Chih-Wei",
        "Kim Min-jun", "Ellen Ripley", "王小明", "赵丽", "Tom Hardy", "Lisa Wong",
    }, people


def test_every_name_in_a_list_is_found(tmp_path, vault):
    terms = _scan_lines(tmp_path, vault, [
        "Investigators: John Smith, Mary Jones, Wei Zhang and Tom Brown.",
        "Steering committee: Dr. Ann Lee, Dr. Raj Patel (Chair), Omar Haddad, MD, Karen Liu, PhD",
        "研究者：张伟、李娜、王芳、欧阳明",
        "Writing group: Hannah Fischer; Pierre Dubois; Sofia Rossi",
        "Endpoints: Overall Survival, Progression Free Survival and Adverse Events.",
    ])
    people = {k for k, t in terms.items() if t.category == "person"}
    for name in ("John Smith", "Mary Jones", "Wei Zhang", "Tom Brown", "Ann Lee",
                 "Raj Patel", "Omar Haddad", "Karen Liu", "张伟", "李娜", "王芳",
                 "欧阳明", "Hannah Fischer", "Pierre Dubois", "Sofia Rossi"):
        assert name in people, (name, people)
    assert not people & {"Overall Survival", "Progression Free Survival",
                         "Adverse Events"}


def test_a_pdf_replacement_keeps_the_size_style_and_colour_of_the_line(tmp_path):
    pm = pytest.importorskip("pymupdf")
    doc = pm.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Sponsor: Acme Oncology Incorporated today.",
                     fontname="tiro", fontsize=11)
    page.insert_text((72, 100), "Signed: Wei Zhang", fontname="hebo", fontsize=14,
                     color=(0, 0, 0.6))
    page.insert_text((72, 130), "Ask Li now.", fontname="helv", fontsize=12)
    src = tmp_path / "in.pdf"
    doc.save(src)
    before = {sp["text"]: sp for b in doc[0].get_text("dict")["blocks"]
              for ln in b["lines"] for sp in ln["spans"]}
    pr._write_pdf(src, tmp_path / "out.pdf", {
        "Acme Oncology Incorporated": "SPONSOR A", "Wei Zhang": "PERSON B",
        "Li": "PERSON C"})
    out = pm.open(tmp_path / "out.pdf")
    spans = {sp["text"]: sp for b in out[0].get_text("dict")["blocks"]
             for ln in b["lines"] for sp in ln["spans"]}
    a, b, c = spans["SPONSOR A"], spans["PERSON B"], spans["PERSON C"]
    assert a["font"].startswith("Times") and a["size"] == pytest.approx(11, abs=0.1)
    assert "Bold" in b["font"] and b["color"] == 0x000099
    assert b["size"] >= 14 * 0.9                    # nearly full size
    # far longer than "Li": narrowed, and it does not run into "now."
    now = out[0].search_for("now.")[0]
    assert c["bbox"][2] <= now.x0 + 1
