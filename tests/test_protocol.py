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
    # a short term before an underlined bold heading: "XYZ Medical Monitor"
    page.insert_text((72, 160), "XYZ Medical Monitor", fontname="hebo", fontsize=12)
    under = page.search_for("Medical Monitor")[0]
    page.draw_line((under.x0, 162), (under.x1, 162), color=(0, 0, 0), width=0.8)
    src = tmp_path / "in.pdf"
    doc.save(src)
    pr._write_pdf(src, tmp_path / "out.pdf", {
        "Acme Oncology Incorporated": "SPONSOR A", "Wei Zhang": "PERSON B",
        "Li": "PERSON C", "XYZ": "SPONSOR L"})
    out = pm.open(tmp_path / "out.pdf")[0]
    spans = [sp for b in out.get_text("dict")["blocks"]
             for ln in b["lines"] for sp in ln["spans"]]

    def span_of(word):
        sp = next(sp for sp in spans if word in sp["text"])
        return {**sp, "bbox": tuple(out.search_for(word)[0])}

    a, b, c, d = (span_of(k) for k in ("SPONSOR A", "PERSON B", "PERSON C", "SPONSOR L"))
    assert a["font"].startswith("Times") and a["size"] == pytest.approx(11, abs=0.1)
    assert "Bold" in b["font"] and b["color"] == 0x000099
    # longer than the word it replaced: full size, and the line moved over
    for sp, size in ((b, 14), (c, 12), (d, 12)):
        assert sp["size"] == pytest.approx(size, abs=0.2), sp["text"]
    now = out.search_for("now.")[0]
    assert c["bbox"][2] <= now.x0 + 1
    mm = out.search_for("Medical Monitor")[0]
    assert d["bbox"][2] <= mm.x0 + 1
    # the underline moved with the heading
    lines = [dr["rect"] for dr in out.get_drawings() if dr["rect"].y0 > 155]
    assert lines and abs(lines[0].x0 - mm.x0) < 2 and abs(lines[0].x1 - mm.x1) < 4


def test_first_name_and_surname_in_capitals(tmp_path, vault):
    terms = _scan_lines(tmp_path, vault, [
        "Principal Investigator: John SMITH",
        "Medical Monitor: Dr. Mary-Jane O'NEIL, MD",
        "The protocol was reviewed by Wei ZHANG and Anna-Lena SCHMIDT-WEBER.",
        "Signed by Peter J. HOLLOWAY on behalf of the sponsor.",
        "Steering committee: Luis GARCIA, Aiko TANAKA, Raj PATEL",
        "Coordinating centre lead: Ming LI",
        # none of these is a person
        "This Phase III study uses RECIST and CTCAE grading; Visit EOT follows Cycle IV.",
        "Refer to the SAP. Grade AE terms per MedDRA. Serum ALT and Plasma PK samples.",
    ])
    people = {k for k, t in terms.items() if t.category == "person"}
    for name in ("John SMITH", "Mary-Jane O'NEIL", "Wei ZHANG",
                 "Anna-Lena SCHMIDT-WEBER", "Peter J. HOLLOWAY", "Luis GARCIA",
                 "Aiko TANAKA", "Raj PATEL", "Ming LI"):
        assert name in people, (name, people)
    assert all(not re.search(r"\b(III|RECIST|CTCAE|EOT|IV|SAP|AE|ALT|PK)\b", p)
               for p in people), people


def test_names_with_capitals_inside_and_particles(tmp_path, vault):
    terms = _scan_lines(tmp_path, vault, [
        "Principal Investigator: DeAnn MacArthur",
        "Medical Monitor: Sarah McDonald, MD",
        "Approved by: LaToya DeVito",
        "Steering committee: Ludwig van Beethoven, Maria de la Cruz, DeAnn SMITH",
        "Coded with MedDRA and analysed in SAS by the CRO.",
    ])
    people = {k for k, t in terms.items() if t.category == "person"}
    for name in ("DeAnn MacArthur", "Sarah McDonald", "LaToya DeVito",
                 "Ludwig van Beethoven", "Maria de la Cruz", "DeAnn SMITH"):
        assert name in people, (name, people)
    assert not any("MedDRA" in p or "SAS" in p for p in people)


# ----------------------------------------------------------------------
# pictures, embedded objects and hidden text
# ----------------------------------------------------------------------
def _png(w: int = 4, h: int = 3) -> bytes:
    return pr._placeholder_png(w, h)


def _protocol_with_pictures(path: Path) -> Path:
    """A Word protocol with a picture in the body and the header, alt text
    that names the sponsor, a tracked deletion, a mailto field and link, an
    embedded workbook and a first-page thumbnail."""
    import io

    d = docx.Document()
    d.sections[0].header.paragraphs[0].add_run().add_picture(io.BytesIO(_png()))
    d.add_paragraph("Sponsor: Acme Oncology Inc.")
    d.add_paragraph().add_run().add_picture(io.BytesIO(_png(6, 2)))
    d.save(path)

    W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    with zipfile.ZipFile(path) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    doc = parts["word/document.xml"].decode("utf-8")
    doc = re.sub(r'(<wp:docPr [^>]*?)(/?>)', r'\1 descr="Acme Oncology logo" title="Acme"\2',
                 doc, count=1)
    extra = (
        f'<w:p><w:del w:id="91" w:author="x" w:date="2025-01-01T00:00:00Z">'
        f'<w:r><w:delText>Zentolimab was deleted here</w:delText></w:r></w:del></w:p>'
        f'<w:p><w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        f'<w:r><w:instrText xml:space="preserve"> HYPERLINK "mailto:jane.roe@acme-onc.com" </w:instrText></w:r>'
        f'<w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>jane.roe@acme-onc.com</w:t></w:r>'
        f'<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>'
        f'<w:p><w:r><w:object>'
        f'<v:shape xmlns:v="urn:schemas-microsoft-com:vml" id="ole1" alt="Acme budget" style="width:50pt;height:30pt">'
        f'<v:imagedata xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        f'xmlns:o="urn:schemas-microsoft-com:office:office" r:id="rIdOlePic" o:title="budget"/></v:shape>'
        f'<o:OLEObject xmlns:o="urn:schemas-microsoft-com:office:office" '
        f'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        f'Type="Embed" ProgID="Excel.Sheet.12" ShapeID="ole1" r:id="rIdOle"/>'
        f'</w:object></w:r></w:p>'
    )
    doc = doc.replace("<w:sectPr", extra + "<w:sectPr", 1)
    parts["word/document.xml"] = doc.encode("utf-8")
    rels = parts["word/_rels/document.xml.rels"].decode("utf-8")
    R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    rels = rels.replace("</Relationships>",
        f'<Relationship Id="rIdOle" Type="{R}/oleObject" Target="embeddings/oleObject1.bin"/>'
        f'<Relationship Id="rIdOlePic" Type="{R}/image" Target="media/olepreview.png"/>'
        f'<Relationship Id="rIdLink" Type="{R}/hyperlink" Target="mailto:jane.roe@acme-onc.com" '
        f'TargetMode="External"/></Relationships>')
    parts["word/_rels/document.xml.rels"] = rels.encode("utf-8")
    parts["word/embeddings/oleObject1.bin"] = b"Acme Oncology budget workbook"
    parts["word/media/olepreview.png"] = _png(5, 5)
    parts["docProps/thumbnail.png"] = _png(8, 8)
    root_rels = parts["_rels/.rels"].decode("utf-8").replace("</Relationships>",
        '<Relationship Id="rIdThumb" Type="http://schemas.openxmlformats.org/package/2006/'
        'relationships/metadata/thumbnail" Target="docProps/thumbnail.png"/></Relationships>')
    parts["_rels/.rels"] = root_rels.encode("utf-8")
    ct = parts["[Content_Types].xml"].decode("utf-8")
    if 'Extension="bin"' not in ct:
        ct = ct.replace("</Types>", '<Default Extension="bin" '
                        'ContentType="application/vnd.openxmlformats-officedocument.oleObject"/></Types>')
    parts["[Content_Types].xml"] = ct.encode("utf-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in parts.items():
            z.writestr(n, b)
    assert W  # namespace used above
    return path


def test_word_pictures_objects_and_hidden_text_are_removed(tmp_path, vault):
    src = _protocol_with_pictures(tmp_path / "p.docx")
    terms = pr.scan([src], vault)
    terms = [t for t in terms if t.term != "Zentolimab"]
    terms.append(pr.Term(term="Zentolimab", category="drug", count=1,
                         proposed="DRUG A", decision="OK"))
    for t in terms:
        t.decision = t.decision or "OK"
    res = pr.apply([src], terms, vault, tmp_path / "out", approved_by="tester")
    out = tmp_path / "out" / "p.docx"
    f = res["files"][0]
    with zipfile.ZipFile(src) as z:     # python-docx's template brings a thumbnail too
        thumbs = sum(1 for n in z.namelist() if n.startswith("docProps/thumbnail"))
    assert f["images_removed"] == 3 + thumbs  # body, header, OLE preview, thumbnails
    assert f["objects_removed"] == 1
    assert f["images_left"] == 0 and res["manifest"]["images_left"] == 0

    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert not [n for n in names if "media/" in n and n != pr._PLACEHOLDER]
        assert not [n for n in names if "embeddings/" in n or "thumbnail" in n]
        body = z.read("word/document.xml").decode("utf-8")
        rels = z.read("word/_rels/document.xml.rels").decode("utf-8")
        assert "Acme" not in body and "budget" not in body
        assert "Zentolimab" not in body          # the tracked deletion too
        assert "jane.roe" not in body and "jane.roe" not in rels
        assert pr.REMOVED_LABEL in body
        assert "OLEObject" not in body and "oleObject" not in rels
    docx.Document(str(out))                      # still opens
    assert res["manifest"]["residual_occurrences"] == 0


def test_a_deleted_revision_is_read_by_the_scan(tmp_path):
    src = _protocol_with_pictures(tmp_path / "p.docx")
    assert any("Zentolimab was deleted here" in b for b in pr.read_text(src))


def test_pdf_pictures_and_attachments_are_removed(tmp_path, vault):
    pm = pytest.importorskip("pymupdf")
    src = tmp_path / "p.pdf"
    doc = pm.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Sponsor: Acme Oncology Inc.", fontsize=11)
    page.insert_image(pm.Rect(72, 100, 272, 200), stream=_png(20, 10))
    # A letterhead picture behind text: the picture goes, the text stays.
    page.insert_image(pm.Rect(72, 400, 400, 460), stream=_png(30, 6))
    page.insert_text((80, 430), "Text printed over a background", fontsize=11)
    doc.embfile_add("budget.xlsx", b"Acme budget")
    doc.save(str(src))

    terms = pr.scan([src], vault)
    for t in terms:
        t.decision = "OK"
    res = pr.apply([src], terms, vault, tmp_path / "out", approved_by="tester")
    f = res["files"][0]
    assert f["images_removed"] == 2 and f["objects_removed"] == 1
    assert f["images_left"] == 0
    with pm.open(str(tmp_path / "out" / "p.pdf")) as masked:
        pg = masked[0]
        assert not pg.get_images(full=True) and not masked.embfile_names()
        text = pg.get_text()
        assert "Text printed over a background" in text
        assert pr.REMOVED_LABEL in text          # the label in the first frame
        assert pg.get_drawings()                 # the grey box


def test_every_drug_in_an_inhibitor_list_is_masked(tmp_path, vault):
    d = docx.Document()
    d.add_paragraph(
        "Prohibited: strong CYP3A4 inhibitors (e.g., ketoconazole, itraconazole, "
        "clarithromycin, ritonavir, cobicistat, nefazodone, telithromycin, "
        "verapamil, diltiazem or St. John's wort) and inducers such as rifampin, "
        "carbamazepine, phenytoin, phenobarbital and enzalutamide.")
    d.add_paragraph("Moderate inhibitors:")
    for x in ("fluconazole", "erythromycin", "aprepitant", "grapefruit juice", "imatinib"):
        d.add_paragraph(x)
    d.add_paragraph("Patients should avoid other herbal supplements.")
    src = tmp_path / "p.docx"
    d.save(src)
    found = {t.term.lower(): t for t in pr.scan([src], vault)}
    for drug in ("ketoconazole", "itraconazole", "clarithromycin", "ritonavir",
                 "cobicistat", "nefazodone", "telithromycin", "verapamil", "diltiazem",
                 "rifampin", "carbamazepine", "phenytoin", "phenobarbital",
                 "enzalutamide", "fluconazole", "erythromycin", "aprepitant", "imatinib"):
        assert drug in found and found[drug].category == "drug", drug
        assert found[drug].confidence == "high", drug
    assert found["st. john's wort"].confidence == "medium"
    assert found["grapefruit juice"].confidence == "medium"
    assert not {"inducers", "herbal supplements", "other herbal supplements",
                "moderate inhibitors", "prohibited"} & set(found)
