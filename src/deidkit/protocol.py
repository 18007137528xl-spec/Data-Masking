"""Masking a study protocol: sponsor, drug, study numbers, people, sites.

A protocol is not PHI, but it names everything the data tier was careful to
hide: the sponsor, the compound and its code, the protocol and registry
numbers, the investigators and the hospitals. A corpus that ships a
de-identified dataset beside the protocol it came from has published the
study's identity in the second file.

The flow is the data flow in miniature. The tool proposes; a person decides
every term; nothing is written until someone has put their name to the list:

1. ``scan`` reads the documents and proposes terms, each with a category, a
   count, where it was seen, and a proposed replacement
2. a person marks every term OK (replace as proposed), CHANGE (replace with
   their own text) or KEEP (not sensitive), and adds what the scan missed
3. ``apply`` writes masked copies and checks them: no approved term may
   survive anywhere in the output

Replacements come from the same vault as the data, so the protocol and the
tier say the same thing: a drug the data relabelled ``DRUG A`` is ``DRUG A``
in the protocol too, and a study number gets the same surrogate whichever
file meets it first.

Word documents are rewritten in place, keeping every table, heading, style
and number: body, tables, text boxes, headers, footers, footnotes, endnotes
and comments, plus the document properties, tracked deletions, field codes
and link targets. A PDF can only be read as text, so with PyMuPDF installed
it is redacted in place -- the text removed from the file, the replacement
drawn where it stood, pages and layout kept -- and without it the masked copy
is a text file.

Pictures are not read, so none is kept: a logo, a scanned signature, a
chemical structure or a study schema can carry exactly what the text masking
took out. Every picture is removed from the file and a grey box with a cross
drawn in its frame, so the layout and the figure numbering stay. Embedded
objects (a workbook or drawing inside a Word file, an attachment inside a
PDF) and Word's first-page thumbnail go the same way. The read-back counts
whatever picture is left; the answer has to be none.
"""

from __future__ import annotations

import csv
import hashlib
import json
import posixpath
import re
import struct
import zipfile
import zlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import blinding
from .dummytext import dummy_value
from .vault import Vault

READABLE = (".docx", ".pdf", ".txt", ".md")

CATEGORIES = {
    "drug": "drug or compound",
    "sponsor": "sponsor or company",
    "study_id": "protocol / registry number",
    "person": "person",
    "contact": "e-mail, phone, web address",
    "place": "address or place name",
    "site": "hospital, university, CRO",
}

DECISIONS = ("OK", "CHANGE", "KEEP")


@dataclass
class Term:
    term: str
    category: str
    count: int = 0
    confidence: str = "medium"
    source: str = "pattern"
    example: str = ""
    proposed: str = ""
    decision: str = ""
    replacement: str = ""
    #: Another name for a term already on the list -- the abbreviation EMPA,
    #: the brand JARDIANCE for empagliflozin -- which takes the same
    #: replacement, so one drug does not become DRUG A and DRUG C.
    alias_of: str = ""

    def final(self) -> str | None:
        """What the term becomes, or None when it stays."""
        if self.decision == "KEEP":
            return None
        if self.decision == "CHANGE":
            return self.replacement
        return self.proposed


# ----------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_TEXT_PARTS = re.compile(
    r"^word/(document|header\d*|footer\d*|footnotes|endnotes|comments)\.xml$"
)


def documents(path: str | Path) -> list[Path]:
    """The protocol files at ``path``: one file, or every readable file in a
    folder. Word lock files (~$...) are skipped."""
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() not in READABLE:
            raise ValueError(
                f"{p.name}: not a protocol format this reads "
                f"({', '.join(READABLE)})"
            )
        return [p]
    if p.is_dir():
        found = sorted(
            f for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in READABLE
            and not f.name.startswith("~$")
        )
        if not found:
            raise ValueError(f"no {', '.join(READABLE)} file in {p}")
        return found
    raise ValueError(f"{p} does not exist on this server")


#: Where a paragraph keeps text: what is shown, what a tracked change
#: deleted (still in the file, and back on screen with one click on Reject),
#: and field codes -- HYPERLINK "mailto:..." is one.
_TEXT_TAGS = ("t", "delText", "instrText")


def _docx_paragraphs(root) -> Iterable[list[Any]]:
    """Each paragraph's text nodes, in order -- and only its own: a text box
    inside a paragraph is a paragraph of its own and is yielded separately.
    Shown text, deleted text and field codes come as three separate runs of
    nodes, so a term is never matched across two of them."""
    for p in root.iter(_W + "p"):
        for tag in _TEXT_TAGS:
            nodes = [
                t for t in p.iter(_W + tag)
                if next(t.iterancestors(_W + "p"), None) is p
            ]
            if nodes:
                yield nodes


def _pdf_blocks(page) -> list[str]:
    """A page's text, one string per block, where a line that ran to the
    right edge and wrapped is joined to the next with a newline, and a line
    that stopped short (a heading, the end of a paragraph) is followed by a
    blank line. Only a wrapped line may lend its last words to the next."""
    out = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        lines = [ln for ln in block.get("lines", []) if ln.get("spans")]
        if not lines:
            continue
        right = max(ln["bbox"][2] for ln in lines)
        width = right - min(ln["bbox"][0] for ln in lines) or 1
        parts = []
        for i, ln in enumerate(lines):
            text = "".join(sp["text"] for sp in ln["spans"])
            if i:
                wrapped = lines[i - 1]["bbox"][2] >= right - 0.12 * width
                parts.append("\n" if wrapped else "\n\n")
            parts.append(text)
        out.append("".join(parts))
    return out


def read_text(path: Path) -> list[str]:
    """The document as a list of text blocks (paragraphs, or PDF pages)."""
    suffix = path.suffix.lower()
    if suffix == ".docx":
        from lxml import etree

        blocks: list[str] = []
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if _TEXT_PARTS.match(name):
                    root = etree.fromstring(z.read(name))
                    blocks += ["".join(t.text or "" for t in ns)
                               for ns in _docx_paragraphs(root)]
                elif name.endswith(".rels"):
                    blocks += [
                        rel.get("Target") or ""
                        for rel in etree.fromstring(z.read(name))
                        if rel.get("TargetMode") == "External"
                        and (rel.get("Type") or "").endswith("/hyperlink")
                    ]
            if "docProps/core.xml" in z.namelist():
                core = etree.fromstring(z.read("docProps/core.xml"))
                blocks += [e.text for e in core.iter() if e.text and e.text.strip()]
        return blocks
    if suffix == ".pdf":
        pm = _pymupdf()
        if pm is not None:
            with pm.open(str(path)) as doc:
                return [t for page in doc for t in _pdf_blocks(page)] + [
                    v for v in (doc.metadata or {}).values() if isinstance(v, str) and v
                ]
        from pypdf import PdfReader

        return [page.extract_text() or "" for page in PdfReader(str(path)).pages]
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


# ----------------------------------------------------------------------
# finding candidates
# ----------------------------------------------------------------------
_CO_SUFFIX = (
    r"(?:Inc\.?|Incorporated|Ltd\.?|Limited|LLC|L\.L\.C\.|GmbH|AG|S\.A\.|"
    r"S\.p\.A\.|N\.V\.|B\.V\.|plc|PLC|Corp\.?|Corporation|Co\.,? ?Ltd\.?|"
    r"Pharmaceuticals?|Pharma|Therapeutics|Biosciences|Biotech(?:nology)?|"
    r"Biologics|Biopharma(?:ceuticals)?|Oncology|Laboratories)"
)
_CAP = r"(?!(?:Sponsor|Prescribing|Information|Study|Protocol|The|By|From|For|And|Of|US|Page|Title|Version)\b)[A-Z][\w&'-]*"
# Where a Chinese name can start: after punctuation, a space, or one of the
# little words that sit in front of a name ("由北京协和医院牵头"). Without it the
# match runs back to the start of the sentence.
_ZH_START = r"(?:(?<=[由在于和与及是的为从经将对向被把：:，,。；;、\s（(“\"])|^)"
_ZH = r"[\u4e00-\u9fa5]"
# A Chinese match still starts at the beginning of its sentence when the
# sentence begins the paragraph ("本研究由北京协和医院..."). Cut it after the
# last of these words -- not 和 or 与, which sit inside names (北京协和医院).
_ZH_CUT = re.compile(r"^.*[由在于是为从经将对向被把的]")
# International nonproprietary name stems: the ending says "drug" whatever
# the rest of the word is (zentolimab, osimertinib, olaparib).
_INN = (
    r"mab|cept|tinib|ciclib|parib|lisib|rafenib|degib|zomib|nib|"
    r"previr|asvir|buvir|vir|glutide|tide|gliflozin|gliptin|platin|taxel|"
    r"rubicin|mustine|sartan|pril|olol|conazole|prazole|mycin|cillin|"
    r"relin|relix|lukast|dronate|parin|xaban|gatran|sentan|fungin|"
    r"floxacin|oxacin|cycline|vastatin|statin|setron|tegravir|kinra|leukin|"
    # the ones a concomitant-medication list is made of: verapamil, diltiazem,
    # amlodipine, carbamazepine, rifampin, midazolam, dexamethasone ... --
    # each checked against an English word list: not histidine, turpentine
    # or triazole
    r"pamil|tiazem|dipine|azepine|apine|azodone|azolam|azepam|triptan|oxetine|"
    r"pramine|triptyline|profen|fenac|coxib|[aeio]tidine|barbital|ampin|ampicin|"
    r"butin|apentine|cistat|nidazole|bendazole|olone|sone|semide|"
    r"thiazide|afil|oquine|otecan|trexate|fosfamide|phosphamide|citabine|"
    r"uracil|poetin|grastim|limus|sporin|sporine|vudine|darone|cainide|ytoin|"
    r"bopag|quidar|cridar"
)
_PHARMA = (
    r"Boehringer[ -]Ingelheim|Eli Lilly(?: and Company)?|Lilly|Pfizer|Merck(?: Sharp & Dohme)?|"
    r"MSD|Novartis|Hoffmann-La Roche|Roche|Genentech|AstraZeneca|GlaxoSmithKline|GSK|"
    r"Sanofi(?:-Aventis)?|Bayer|AbbVie|Amgen|Bristol[- ]Myers Squibb|BMS|Johnson & Johnson|"
    r"Janssen|Takeda|Astellas|Daiichi[- ]Sankyo|Eisai|Otsuka|Novo Nordisk|Gilead|Regeneron|"
    r"Vertex|Biogen|Moderna|BioNTech|Servier|Ipsen|Teva|Viatris|Chugai|Shionogi|"
    r"Jiangsu Hengrui|Hengrui|BeiGene|Innovent|Junshi|Hansoh|CSPC|Sino Biopharm|Zai Lab|"
    r"Akeso|Hutchmed|Fosun"
)
_PHARMA_ZH = r"恒瑞医药|恒瑞|百济神州|信达生物|君实生物|豪森药业|石药集团|正大天晴|再鼎医药|康方生物|和黄医药|复星医药|勃林格殷格翰|礼来|辉瑞|默沙东|诺华|罗氏|阿斯利康|葛兰素史克|赛诺菲|拜耳|艾伯维|安进|百时美施贵宝|强生|杨森|武田|安斯泰来|第一三共|卫材|诺和诺德|吉利德"
_ZH_DRUG = r"单抗|替尼|西尼|帕利|司他|他汀|沙坦|普利|洛尔|霉素|西林|铂|紫杉醇|比星|那肽|鲁肽|格列净|格列汀|韦"
#: Roles a protocol writes a person's name after: "Medical Monitor: Jane
#: Roe", "Approved by: Wei Zhang", "Printed Name: Sarah Johnson".
_ROLE_EN = (
    r"(?:Principal Investigator|Coordinating Investigator|Lead Investigator|"
    r"Sub-?Investigator|Investigator|PI|Medical Monitor|Medical Expert|"
    r"Medical Officer|Study Physician|Responsible Physician|Safety Physician|"
    r"Sponsor(?:'s)?\s+(?:Representative|Signatory|Medical Officer|Contact)|"
    r"Study Director|Clinical Project Manager|Project Manager|Project Lead|"
    r"Study Manager|Clinical Research Associate|CRA|Biostatistician|"
    r"(?:Study\s+)?Statistician|Data Manager|Pharmacovigilance Contact|"
    r"Contact Person|Contact|Author|Prepared by|Written by|Reviewed by|"
    r"Approved by|Signed by|Printed Name|Full Name|Name)"
)
#: A person's name as protocols write it: Sarah Johnson, John A. Doe,
#: Michael O'Neil, WANG Jianguo, Jianguo WANG, Anna Müller.
# Ends at the end of a word: without that, "John SMITH" read as "John S".
# Capitals inside a name are allowed where a lower-case letter follows each:
# DeAnn, McDonald, MacArthur, LaToya, DeVito -- but not MedDRA or SAS.
_NAME_WORD = (
    r"[A-Z][a-zà-ÿ]*(?:[A-Z][a-zà-ÿ]+)*"
    r"(?:['’-][A-Za-zà-ÿ][a-zà-ÿ]*(?:[A-Z][a-zà-ÿ]+)*)*(?![A-Za-zà-ÿ])"
)
#: Lower-case particles inside a name: Ludwig van Beethoven, Maria de la Cruz.
_PARTICLE = r"(?:van|von|der|den|de|del|della|di|da|du|la|le|bin|binti|al|el|ten|ter|dos|das)"
_NSEP = rf"\s+(?:{_PARTICLE}\s+){{0,2}}"
#: A surname in capitals, as many protocols and most CVs write it: SMITH,
#: O'NEIL, SMITH-JONES, LI.
_CAPS_SURNAME = r"(?:[A-Z]['’])?[A-Z]{2,}(?:['’-][A-Z]{2,})*(?![A-Za-z])"
_PNAME = (
    # John SMITH, John A. SMITH, Mary-Jane O'NEIL, Ludwig van BEETHOVEN
    rf"{_NAME_WORD}(?:\s+(?:{_NAME_WORD}|[A-Z]\.))?{_NSEP}{_CAPS_SURNAME}"
    rf"|{_NAME_WORD}(?:\s+[A-Z]\.)?(?:{_NSEP}{_NAME_WORD}){{1,2}}"
    r"|[A-Z]{2,}(?:\s+[A-Z][a-z]+(?:-[A-Za-z]+)?){1,2}"
    # JOHN SMITH, SMITH JOHN A.
    r"|[A-Z]{2,}(?:\s+[A-Z]\.)?(?:\s+[A-Z]{2,}){1,2}"
    # Smith, John  /  SMITH, John A.
    rf"|(?:{_NAME_WORD}|[A-Z]{{2,}}),\s*{_NAME_WORD}(?:\s+[A-Z]\.)?"
)

_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    # registry and protocol numbers
    ("study_id", "high", re.compile(r"\bNCT\d{8}\b")),
    ("study_id", "high", re.compile(r"\b(?:19|20)\d{2}-5\d{5}-\d{2}(?:-\d{2})?\b")),  # CTIS
    ("study_id", "high", re.compile(r"\b(?:19|20)\d{2}-\d{6}-\d{2}\b")),  # EudraCT
    ("study_id", "high", re.compile(
        r"(?i:\bIND\s*(?:No\.?|Number|#)?\s*[:：#]?\s*)(\d{2,3},\d{3}|\d{5,6})(?![\d,])")),
    ("study_id", "high", re.compile(
        r"(?i:protocol\s*(?:no\.?|number|id|code|编号)?\s*[:：#]?\s*)"
        r"([A-Z0-9][A-Z0-9_./-]*\d[A-Z0-9_./-]*)")),
    ("study_id", "high", re.compile(r"(?:方案编号|试验编号|研究编号)\s*[:：]?\s*([A-Za-z0-9_./-]{4,})")),
    # BDM-AI-2025-001, TIG-2026-001: letters then two or more hyphen groups
    ("study_id", "medium", re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+){2,5}\b")),
    # Compound names and codes are high: in a protocol they are masked by
    # default, and a person un-masks the few that should stay (a rescue
    # medication, a probe substrate) rather than confirming each one.
    # compound codes: MK-3475, BMS-936558, ABBV951, and with a space,
    # BI 10773 -- but not a year (ICH 2016) or a short count (Day 120)
    ("drug", "high", re.compile(r"\b[A-Z]{1,5}-?\d{3,6}[A-Z]?\b")),
    ("drug", "high", re.compile(r"\b[A-Z]{2,5} (?!(?:19|20)\d\d\b)\d{4,6}[A-Z]?\b")),
    # trade marks
    ("drug", "high", re.compile(r"\b([A-Z][A-Za-z0-9-]{2,})\s?[®™]")),
    # INN stems: zentolimab, osimertinib -- and their Chinese counterparts
    ("drug", "high", re.compile(rf"\b([A-Za-z]{{3,}}(?:{_INN}))\b", re.I)),
    ("drug", "high", re.compile(rf"{_ZH_START}({_ZH}{{1,8}}?(?:{_ZH_DRUG}))")),
    # companies
    ("sponsor", "medium", re.compile(
        rf"\b((?:(?:{_CAP}|&)\s+){{0,4}}{_CAP},?\s+{_CO_SUFFIX})(?![\w])")),
    ("sponsor", "high", re.compile(r"(?im)^\s*sponsor(?:\s+name)?\s*[:：]\s*(.{3,80}?)\s*$")),
    ("sponsor", "medium", re.compile(
        rf"{_ZH_START}((?:{_ZH}|[（）()]){{2,24}}?(?:股份有限公司|有限公司|集团|制药|药业|医药|生物科技|生物技术))")),
    ("sponsor", "medium", re.compile(r"(?:申办方|申办者)(?:名称)?\s*[:：]\s*([^\s，,。；;]{2,40})")),
    ("site", "medium", re.compile(r"\b(IQVIA|Parexel|PPD|ICON plc|Syneos Health|Labcorp|Covance|"
                                  r"Medpace|PRA Health Sciences|Fortrea|Tigermed|WuXi \w+)\b")),
    ("site", "medium", re.compile(r"(泰格医药|药明康德|药明\w{0,4})")),
    # institutions
    ("site", "medium", re.compile(
        rf"\b((?:{_CAP}\s+){{0,6}}(?:Hospital|University|Medical Cent(?:er|re)|Cancer Cent(?:er|re)|"
        rf"Clinic|Institute|Health System|School of Medicine|College)"
        rf"(?:\s+(?:of|for)\s+(?:{_CAP}\s?){{1,5}})?)")),
    ("site", "medium", re.compile(
        rf"{_ZH_START}({_ZH}{{2,25}}?(?:医院|大学|研究所|研究院|医学院|卫生院))")),
    # big pharma by name, with no Inc. or Ltd. to give it away
    ("sponsor", "high", re.compile(rf"\b({_PHARMA})\b")),
    ("sponsor", "high", re.compile(rf"({_PHARMA_ZH})")),
    # people
    ("person", "high", re.compile(r"(?<![\w.])@([A-Z][a-z'-]+(?:\s+[A-Z][a-z'-]+){0,2})")),
    ("person", "high", re.compile(
        rf"\b(?:Dr|Prof|Professor|Mr|Ms|Mrs)\.?\s+({_PNAME}|{_NAME_WORD}|{_CAPS_SURNAME})")),
    ("person", "high", re.compile(
        rf"\b({_PNAME}),\s*"
        r"(?:MD|M\.D\.|PhD|Ph\.D\.|MBBS|MBChB|PharmD|DO|D\.O\.|RN|BSN|NP|PA-C|MSc|MS|"
        r"MPH|DrPH|FRCP|FRCPC|FACP|FACC|FRACP|MBA)\b")),
    ("person", "high", re.compile(  # John Smith MD, Jane Roe PhD
        rf"\b({_PNAME})\s+"
        r"(?:MD|M\.D\.|PhD|Ph\.D\.|MBBS|PharmD)\b")),
    ("person", "high", re.compile(
        r"(?:研究者|主要研究者|协调研究者|分中心研究者|联系人|医学监查员|监查员|姓名|"
        r"负责人|签名|签字|申办方代表|项目经理|统计师|统计负责人|医学经理|撰写人|"
        r"审核人|批准人|报告人|起草人|作者|数据管理员|药物警戒联系人)"
        r"\s*[:：]\s*([\u4e00-\u9fa5·]{2,5})")),
    # contact details
    ("contact", "high", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("contact", "high", re.compile(r"\bhttps?://[^\s<>\"')]+|\bwww\.[^\s<>\"')]+")),
    ("contact", "high", re.compile(
        r"(?i:(?:tel|phone|telephone|fax|mobile|cell|mob|contact\s+number|电话|传真|手机)"
        r"(?:\s*(?:no\.?|number|nr\.?|#|号码|号))?\.?\s*[:：]?\s*)"
        r"(\+?\(?\d[\d\s().-]{6,}\d)")),
    # Unlabelled, by shape alone. Each is specific enough that a date, a
    # version or a protocol number does not fit it.
    ("contact", "high", re.compile(  # +86 21 6123 4567, +1 (617) 555-0142
        r"(?<![\w+])(\+\s?\d{1,3}[\s.-]?(?:\(?\d{1,4}\)?[\s.-]?){2,5}\d{2,4})(?!\d)")),
    ("contact", "high", re.compile(  # 01 58 10 08 81, 01.58.10.08.81 (France)
        r"(?<![\d.+-])(0[1-9](?:[\s.]\d{2}){4})(?![\d.])")),
    ("contact", "high", re.compile(  # 138 1234 5678, 138-1234-5678
        r"(?<![\d-])(1[3-9]\d[\s-]?\d{4}[\s-]?\d{4})(?![\d-])")),
    ("contact", "high", re.compile(  # 021-61234567, (021) 6123-4567, 010 12345678
        r"(?<![\d-])(\(?0\d{2,3}\)?[\s-]\d{3,4}[\s-]?\d{4})(?![\d-])")),
    ("contact", "high", re.compile(  # 400-820-1234, 800 123 4567
        r"(?<![\d-])([48]00[\s-]?\d{3}[\s-]?\d{4})(?![\d-])")),
    ("contact", "high", re.compile(  # (617) 555-0142, 617.555.0142
        r"(?<![\d-])(\(\d{3}\)\s?\d{3}[\s.-]\d{4}|\d{3}[.-]\d{3}[.-]\d{4})(?![\d-])")),
    # people, by the email written beside them: John Smith (j.smith@x.org)
    ("person", "high", re.compile(
        rf"({_PNAME})\s*[(<\[,:–-]?\s*(?:mailto:)?[\w.+-]+@[\w-]+\.")),
    # people, by the role written in front of them
    ("person", "high", re.compile(
        rf"(?i:\b{_ROLE_EN})\s*[:：]\s*(?:(?:Dr|Prof|Mr|Ms|Mrs)\.?\s+)?({_PNAME})")),
]

#: Standards, regulations and scales written letters-then-number, which the
#: compound-code patterns would otherwise take for MK-3475.
_STANDARD = re.compile(
    r"^(?:ISO|IEC|EN|ASTM|ANSI|CLSI|DIN|BS|JIS|USP|EP|JP|CFR|ICH|NMPA|FDA|EMA|"
    r"WHO|ICD|SOP|ECOG|NYHA|VAS|NRS|RECIST|CTCAE|MEDDRA|SAS|GB|YY|WS|TABLE|"
    r"FIGURE|SECTION|APPENDIX|PAGE|VERSION|ROOM|SUITE)[\s-]?\d",
    re.IGNORECASE,
)

#: Words that end a name rather than continue it.
_NAME_STOP = {
    "and", "or", "of", "the", "at", "in", "on", "for", "to", "with", "md", "phd",
    "hospital", "university", "medical", "center", "centre", "clinic", "institute",
    "sponsor", "study", "protocol", "investigator", "monitor", "department",
    "page", "version", "date", "tel", "phone", "email", "fax",
    "title", "role", "signature", "name", "position", "affiliation",
    "principal", "director", "manager", "statistician", "physician", "vice",
    "president", "chief", "officer", "head", "lead", "coordinator",
    "associate", "professor", "printed", "approved", "reviewed", "signed",
}
#: Words in front of "(drug)" that are not a brand.
_NOT_A_BRAND = {
    "tablet", "tablets", "capsule", "capsules", "dose", "doses", "study", "drug",
    "placebo", "treatment", "arm", "group", "oral", "injection", "solution",
    "administered", "receive", "receiving", "with", "plus", "versus", "and",
}

#: Words a pattern can produce that name nothing.
_NOT_A_NAME = {
    "the", "this", "study", "sponsor", "protocol", "clinical", "university",
    "hospital", "institute", "college", "clinic", "the university",
}


#: Words protocol headings are made of. A line of two capitalised words
#: beside a role is a name -- unless it is "Adverse Events".
_HEADING_WORDS = frozenset(
    """adverse events event consent informed schedule assessments procedures
    objectives objective endpoints endpoint safety efficacy statistical analysis
    analyses inclusion exclusion criteria treatment treatments dosing dose visit
    visits screening randomization randomisation background rationale
    introduction synopsis appendix references contacts contact signatures
    approval amendment summary design population data management quality ethics
    records monitoring publication termination withdrawal discontinuation
    pharmacokinetics laboratory vital signs physical examination history
    concomitant medications medication prohibited contraception pregnancy
    reporting serious unexpected investigational product accountability storage
    labeling labelling blinding unblinding emergency overview table figure list
    abbreviations sample size estimand hypothesis follow up end period phase
    general administrative responsibilities committee board review approved
    confidential confidentiality sponsor information clinical trial"""
    # laboratory and clinical measures: "Absolute Neutrophil Count" beside
    # a Medical Monitor line, or "Platelet Count" in the same list, is a
    # lab test, not a person
    """count counts absolute neutrophil neutrophils platelet platelets
    hemoglobin haemoglobin hematocrit haematocrit lymphocyte lymphocytes
    leukocyte leukocytes creatinine clearance bilirubin albumin glucose sodium
    potassium calcium magnesium protein alkaline phosphatase alanine aspartate
    aminotransferase transaminase serum plasma urine urinalysis ratio pressure
    blood cell cells body mass index temperature test compound score grade
    function level levels normal limit upper chemistry hematology haematology
    coagulation prothrombin thromboplastin partial normalized antigen
    testosterone prostate specific heart electrocardiogram""".split()
)

_ROLE_NEAR = re.compile(
    r"(?i)\b(?:investigator|monitor|physician|director|manager|statistician|"
    r"signature|signed|approved|reviewed|prepared|author|sponsor|M\.?D\.?|"
    r"Ph\.?D\.?|MBBS|title|position|vice president|head of|lead|"
    # an affiliation or a contact line under a name says the same
    r"department|dept\.?|hospital|university|institute|clinic|cent(?:er|re)|"
    r"school|college|faculty|division|affiliation|address|e-?mail|tel|phone|fax)\b"
)
_ROLE_NEAR_ZH = re.compile(
    r"(研究者|医院|科室|主任|医师|医生|教授|签名|签字|日期|联系人|电话|职务|职称|单位|"
    r"负责人|监查员|经理|统计师|邮箱)"
)

_PINYIN_SURNAMES = frozenset(
    """wang li zhang liu chen yang huang zhao wu zhou xu sun ma zhu hu guo he
    gao lin luo zheng liang xie song tang han feng deng cao peng zeng xiao tian
    dong pan yuan cai jiang yu du ye cheng wei su lu ding ren shen yao cui
    zhong tan fan jin shi liao jia xia fu fang bai zou meng xiong qin qiu yin
    xue yan duan lei hou long tao gu mao hao gong shao wan qian kong xiang wen
    kang niu ge xing hong lai qi mo lv lyu ou yi""".split()
)
_PINYIN_SYLLABLE = (
    r"(?:zh|ch|sh|[bpmfdtnlgkhjqxrzcsyw])?"
    r"(?:iang|iong|uang|ang|eng|ing|ong|ian|iao|uai|uan|ai|ei|ao|ou|an|en|in|"
    r"un|ia|ie|iu|ua|uo|ui|ue|er|a|o|e|i|u|v)"
)
_PINYIN_GIVEN = re.compile(rf"(?i)(?:{_PINYIN_SYLLABLE}){{1,3}}")
#: Pairs of pinyin-looking words that are places, not people.
_ADDRESS_WORDS = {"avenue", "road", "street", "building", "floor", "district",
                  "province", "city", "county", "village", "town", "lane", "drive"}
_PINYIN_PLACES = {"hong kong", "tai wan", "ma cau", "xi an", "he nan", "he bei"}


def _ner_person(frag: str) -> bool:
    """Keep a name the model found only when it reads as a full name.

    The model calls a capitalised word at the start of a sentence a person
    ("When", "Site", "Concomitant"), and a protocol is wall-to-wall
    capitalised headings. A single word, a heading phrase ("Concomitant
    Medications") or a drug name is dropped rather than queued: every one
    of them was a row someone had to dismiss by hand.
    """
    from .freetext import CLINICAL_ALLOWLIST, _looks_like_drug_name

    words = frag.strip().split()
    if len(words) < 2 or not re.fullmatch(_PNAME, frag.strip()):
        return False
    low = [w.lower().strip(".,'’") for w in words]
    return not any(
        w in _HEADING_WORDS or w in _NAME_STOP or w in _NOT_A_NAME
        or w in CLINICAL_ALLOWLIST or _looks_like_drug_name(w)
        for w in low
    )


#: Romanised surnames that are not Mandarin pinyin: Cantonese (Chan, Wong),
#: Wade-Giles (Hsu, Chang, Tsai) and Korean (Kim, Choi). Beside one of these
#: any capitalised given name counts, hyphenated or not: Chan Tai Man,
#: Hsu Chih-Wei, Kim Min-jun.
_OTHER_SURNAMES = frozenset(
    """chan wong cheung leung lam chow tsang kwok ng lee yeung ho tse lau
    fong yip mak poon siu tam tong tsui yau yuen hui kwan law choy chu
    hsu chang tsai chiang kuo hsieh tseng chou chao yeh hung kao chien
    chuang hsiao hsueh kim choi jung kang cho yoon jang lim shin kwon hwang
    seo""".split()
)
_LEAD_WORDS = {"contact", "questions", "please", "call", "email", "ask", "by",
               "from", "dear", "attn", "attention", "cc", "to"}
_INITIAL_STOP = {"food", "drug", "department", "government", "national", "army",
                 "navy", "patent", "pharmacopeia", "pharmacopoeia", "postal",
                 "state", "states", "district", "court", "census", "code"}
#: Words an outline item starts with: "C. All subjects", "M. Individual
#: stopping rules". A letter, a full stop and one of these is a heading.
_OUTLINE_WORDS = frozenset("""all any each every individual individuals other others
    subject subjects patient patients participant participants total overall general
    specific additional prior previous current new primary secondary exploratory key
    unbound bound free mean median maximum minimum number the this these that those
    in on of for with without if when where after before during within use used using
    women men female male children adult adults healthy normal abnormal laboratory
    samples sample results evaluation evaluations criteria""".split())


#: Capitalised words a protocol puts in front of an acronym: "Phase III",
#: "Visit EOT", "per RECIST", "Grade AE". None of them is a first name.
_NOT_GIVEN = frozenset(
    """phase visit day week cycle grade version section table figure appendix
    per see using total serum plasma urine dose arm cohort part stage level
    study protocol amendment sponsor site form module schedule annex chapter
    and or the a an of in on at by for to with from via all each every any
    no not if when where while after before during within between under over
    patients patient subjects subject participants investigators investigator
    adverse serious laboratory central local refer according based including
    other new final draft revised original clinical medical global regional
    page date time type class model method criteria report data rate risk""".split()
)
#: Capitals that are acronyms or numerals, never a surname.
_NOT_SURNAME = frozenset(
    """AE AES SAE SAES TEAE ECG EKG CT MRI PET PK PD ADA DLT MTD RP2D ORR OS
    PFS DFS DOR CR PR SD PD BOR ITT PP FAS SAP CSR CRF ECRF EDC IRB IEC ICF
    DSMB IDMC GCP ICH FDA EMA NMPA CDE WHO NCI CTCAE RECIST IRECIST MEDDRA
    WHODRUG SOC PT LLT HLT HLGT ULN LLN ALT AST ALP GGT LDH BUN INR APTT HBV
    HCV HIV COVID QTC QTCF BMI BP HR RR EOT EOS FU SOA IP IMP NIMP PI CRA CRO
    USA UK EU PRC II III IV VI VII VIII IX XI XII ID NO NA ND UNK TBD TBC
    MD PHD RN NP DO MS MBA MPH IV IM SC PO QD BID TID QW Q2W Q3W""".split()
)


def _caps_surname_names(text: str) -> Iterable[tuple[str, int, int]]:
    """John SMITH, Mary-Jane O'NEIL, Wei ZHANG: a first name followed by a
    surname in capitals, with no role or title needed -- the capitals are
    the cue. A word that only ever stands before an acronym (Phase III,
    Visit EOT) is not a first name, and an acronym is not a surname."""
    # One line at a time, and a rejected match is tried again from its next
    # word: "Contact" / "Marie-Claire DUBOIS" in a table read as one match
    # starting at the heading, and the name inside it was never looked at.
    pat = re.compile(
        rf"(?<![A-Za-z])({_NAME_WORD}(?:[ \t]+(?:{_NAME_WORD}|[A-Z]\.))?[ \t]+{_CAPS_SURNAME})"
    )
    pos = 0
    while True:
        m = pat.search(text, pos)
        if not m:
            return
        words = m.group(1).split()
        given, surname = words[0], words[-1]
        parts = re.split(r"['’-]", surname)
        if (len(given) < 2 or given.lower() in _NOT_GIVEN | _HEADING_WORDS | _NAME_STOP
                or any(w.lower() in _NOT_GIVEN | _HEADING_WORDS for w in words[1:-1])
                or any(p in _NOT_SURNAME for p in parts) or surname in _NOT_SURNAME
                # FDCB, CRP: an acronym has no vowel more often than not
                or not re.search(r"[AEIOUY]", surname)):
            pos = m.start(1) + len(given)
            continue
        yield m.group(1), m.start(1), m.end(1)
        pos = m.end(1)


def _initial_names(text: str) -> Iterable[tuple[str, int, int]]:
    """J. Smith, J.A. Smith, J. A. Smith: initials then a surname. Medium --
    "U.S. Food" has the same shape, and the obvious ones are filtered."""
    for m in re.finditer(
        rf"(?<![\w.])((?:[A-Z]\.\s?){{1,2}}(?=[A-Z][a-z]{{2}}){_NAME_WORD})", text
    ):
        last = m.group(1).split()[-1].split(".")[-1].lower()
        if last in _INITIAL_STOP | _HEADING_WORDS | _NAME_STOP | _OUTLINE_WORDS:
            continue
        yield m.group(1), m.start(1), m.end(1)


_LIST_SEP = re.compile(r"\s*(?:,|;|，|、|；|/|\band\b|&|和|及|与|以及)\s*")
_CREDENTIAL = re.compile(
    r"(?:MD|M\.D\.|PhD|Ph\.D\.|MBBS|MBChB|PharmD|DO|RN|BSN|NP|PA-C|MSc|MS|MPH|"
    r"DrPH|FRCP|FRCPC|FACP|FACC|FRACP|MBA|博士|教授|主任医师|副主任医师|医师)\.?$"
)
_TITLE_PREFIX = re.compile(r"^(?:Dr|Prof|Professor|Mr|Ms|Mrs)\.?\s+")


def _list_item_name(item: str) -> str | None:
    """The name in one list item, if the item is nothing but a name (with a
    title in front or a role in brackets after allowed)."""
    from .profile import _CN_COMPOUND_SURNAMES, _CN_SURNAMES

    item = re.sub(r"\s*[(（][^)）]{1,30}[)）]\s*$", "", item.strip()).strip(" .。")
    item = _TITLE_PREFIX.sub("", item)
    if re.fullmatch(r"[\u4e00-\u9fa5·]{2,4}", item):
        if item[0] in _CN_SURNAMES or item.startswith(_CN_COMPOUND_SURNAMES):
            return item
        return None
    if not re.fullmatch(_PNAME, item) or "," in item:
        return None
    if any(w.lower().strip(".'’") in _HEADING_WORDS | _NAME_STOP | _ADDRESS_WORDS
           for w in item.split()):
        return None
    return item


def _list_names(line: str, found: dict) -> Iterable[tuple[str, int, int, str]]:
    """Every name in a list that already holds a found name -- or, with no
    name found yet, a list of three or more items that are all names."""
    parts, pos = [], 0
    for m in _LIST_SEP.finditer(line):
        parts.append((line[pos:m.start()], pos))
        pos = m.end()
    parts.append((line[pos:], pos))
    if len(parts) < 2:
        return
    items = []
    for raw, at in parts:
        if not raw.strip() or _CREDENTIAL.fullmatch(raw.strip()):
            continue  # "John Smith, MD, Mary Jones": the MD is not an item
        # the first item carries the lead-in: "Investigators: John Smith"
        lead = re.match(r"^.*[:：]\s*", raw)
        if lead:
            raw, at = raw[lead.end():], at + lead.end()
        items.append((raw, at, _list_item_name(raw)))
    names = [x for x in items if x[2]]
    seeded = any(_key(n) in found and found[_key(n)].category == "person"
                 for _, _, n in names)
    if not names or (not seeded and (len(names) < 3 or len(names) < len(items))):
        return
    conf = "high" if seeded or all(
        re.search(r"[\u4e00-\u9fa5]", n) for _, _, n in names) else "medium"
    for raw, at, n in names:
        off = raw.find(n)
        yield n, at + off, at + off + len(n), conf


_DRUG_SEP = re.compile(r"\s*(?:,|;|，|、|；|/|[()\[\]（）]|\band\b|\bor\b|&|和|及|与|或)\s*")
_DRUG_LEAD = re.compile(
    r"^.*?(?:[(:：]|\b(?:such as|e\.\s?g\.?,?|eg,|i\.\s?e\.?,?|including|includes?|"
    r"for example|for instance|like)\s+)", re.I)
#: Words that make a list item a description, not a drug: "other strong
#: inhibitors", "herbal supplements", "calcium channel blockers".
_NOT_A_DRUG_WORD = frozenset({
    "other", "others", "drug", "drugs", "medication", "medications", "medicine",
    "medicines", "inhibitor", "inhibitors", "inducer", "inducers", "substrate",
    "substrates", "strong", "moderate", "potent", "weak", "agent", "agents",
    "product", "products", "herbal", "supplement", "supplements", "the", "a", "an",
    "etc", "all", "any", "these", "those", "certain", "some", "combination",
    "treatment", "therapy", "therapies", "patient", "patients", "subject", "subjects",
    "blocker", "blockers", "antifungal", "antifungals", "antibiotic", "antibiotics",
    "antiretroviral", "antiretrovirals", "anticonvulsant", "anticonvulsants",
    "antidepressant", "antidepressants", "antiepileptic", "antiepileptics", "class",
    "classes", "use", "used", "not", "limited", "to", "with", "without", "is", "are",
    "be", "may", "should", "must", "within", "days", "weeks", "prior", "before",
    "after", "during", "study", "dose", "doses", "known", "listed", "below", "above",
    "table", "appendix", "section", "see", "refer", "list", "following", "include",
    # the study's own product is not one of the drugs in a list of others
    "test", "compound", "investigational", "placebo", "matching", "vehicle",
    # how and when, not what: "pembrolizumab, lenvatinib, orally"
    "orally", "oral", "daily", "weekly", "intravenously", "intravenous",
    "subcutaneously", "alone", "monotherapy", "both", "either", "each",
    "respectively", "together", "concurrently", "once", "twice", "food", "water",
})


def _drug_item(item: str) -> str | None:
    """The drug in one list item, if the item is nothing but a name: one to
    three words of letters (St. John's wort, grapefruit juice)."""
    item = _DRUG_LEAD.sub("", item.strip())
    item = item.strip(" \t.。;:()（）[]\"'“”")
    words = item.split()
    if not 1 <= len(words) <= 3 or len(item) < 4:
        return None
    if not all(re.fullmatch(r"[A-Za-z][A-Za-z'’.-]*", w) for w in words):
        return None
    if any(w.lower().strip(".'’") in _NOT_A_DRUG_WORD for w in words):
        return None
    return item


def _list_drugs(line: str, found: dict) -> Iterable[tuple[str, int, int, str]]:
    """Every drug in a list that already holds two found drugs: "strong
    CYP3A4 inhibitors (e.g., ketoconazole, itraconazole, clarithromycin,
    nefazodone, verapamil or St. John's wort)". The stems find some of them;
    a list masked halfway is worse than either choice made fully.

    A list is a run of items with nothing but names in it: brackets
    separate items too ("elacridar [GF120918]", "eltrombopag)"), and an
    item that is a sentence ("within 7 days or 5 half-lives") ends the run.
    So a whole paragraph can be read at once -- a list running over several
    printed lines is one list -- without the paragraph's ordinary words
    being taken for drugs."""
    parts, pos = [], 0
    for m in _DRUG_SEP.finditer(line):
        parts.append((line[pos:m.start()], pos))
        pos = m.end()
    parts.append((line[pos:], pos))

    def is_known(raw: str) -> bool:
        k = _key(raw.strip(" \t.。;:"))
        return k in found and found[k].category in ("drug", "study_id")

    run: list[tuple[str, int, str | None]] = []
    for raw, at in parts + [("<end of list>", len(line))]:
        if not raw.strip():
            continue
        name = _drug_item(raw)
        if name or is_known(raw):
            run.append((raw, at, name))
            continue
        drugs = sum(1 for r, _, n in run
                    if n and _key(n) in found and found[_key(n)].category == "drug")
        if len(run) >= 3 and drugs >= 2:
            for r, a, n in run:
                if n and _key(n) not in found:
                    off = r.find(n)
                    yield n, a + off, a + off + len(n), "high" if " " not in n else "medium"
        run = []


#: A label a study number is written after, alone on its line or in its
#: table cell, with the number on the next line or in the next cell:
#: "IND #" | "123456", "EudraCT Number" | "2018-001234-56".
_ID_LABEL = re.compile(
    r"(?i)^\s*(?:(?:US\s+|FDA\s+)?IND|EudraCT|CTIS|EU\s*CT|NCT|ClinicalTrials\.gov|"
    r"Registry|Registration|(?:(?:Sponsor|Company)\s+)?(?:Protocol|Study|Trial))"
    r"(?:\s*(?:No\.?|Number|#|ID|Identifier|Code|Registration\s+(?:No\.?|Number)))?"
    r"\s*[:：#]?\s*$"
)
_ID_QUALIFIED = re.compile(r"(?i)(?:No\.?|Number|#|ID|Identifier|Code)\s*[:：#]?\s*$")
_IND_VALUE = re.compile(r"\d{2,3},\d{3}|\d{5,6}")
_ID_VALUE = re.compile(r"(?=[A-Za-z0-9_./-]*[A-Za-z-])(?=[A-Za-z0-9_./-]*\d)[A-Za-z0-9][A-Za-z0-9_./-]{4,39}")


def _labelled_ids(lines: list[str]) -> Iterable[tuple[str, str]]:
    """Study numbers written in the cell or on the line after their label.
    Yields (value, context). IND takes a bare 5-6 digit number (123,456
    too); the others need letters or a hyphen with the digits, so a
    "Protocol" heading followed by "2.0" is not a number."""
    lines = [ln.strip() for ln in lines]
    for i, label in enumerate(lines):
        m = _ID_LABEL.match(label)
        if not m:
            continue
        head = label.split()[0].upper() if label.split() else ""
        if head in ("PROTOCOL", "STUDY", "TRIAL", "SPONSOR", "COMPANY") and not _ID_QUALIFIED.search(label):
            continue
        value = next((ln for ln in lines[i + 1:i + 3] if ln), "")
        ind = re.search(r"(?i)\bIND\b", label)
        if ind and _IND_VALUE.fullmatch(value):
            yield value, f"{label} {value}"
        elif not ind and _ID_VALUE.fullmatch(value) and not re.fullmatch(r"[\d./-]+", value.replace("-", "")):
            yield value, f"{label} {value}"


# ----------------------------------------------------------------------
# addresses and places
# ----------------------------------------------------------------------
_US_STATES = (
    r"A[LKZR]|C[AOT]|D[EC]|FL|GA|HI|I[DLNA]|K[SY]|LA|M[EDAINSOT]|N[EVHJMYCD]|"
    r"O[HKR]|PA|RI|S[CD]|T[NX]|UT|V[TA]|W[AVIY]|PR|"
    # Canadian provinces
    r"AB|BC|MB|NB|NL|NS|NT|NU|ON|PE|QC|SK|YT"
)
_POSTCODE = r"\d{5}(?:-\d{4})?|[A-Z]\d[A-Z][ \t]?\d[A-Z]\d"
_STREET_SUFFIX = (
    r"(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|Drive|Dr\.?|"
    r"Lane|Ln\.?|Way|Parkway|Pkwy\.?|Place|Pl\.?|Court|Ct\.?|Square|Sq\.?|"
    r"Terrace|Highway|Hwy\.?|Circle|Plaza|Row|Crescent|Close)(?![A-Za-z])"
)
_CITY = r"[A-Z][a-z]+(?:[ -](?:[A-Z][a-z]+|am|an|der|de|la|le|sur|upon)){0,3}"
#: 525 Market Street, Suite 2800, San Francisco, CA 94105, USA
_ADDRESS = re.compile(
    rf"\b(?P<street>\d{{1,6}}[A-Za-z]?(?:-\d{{1,6}})?[ \t]+(?:[A-Z][A-Za-z.'’-]*[ \t]+){{1,4}}{_STREET_SUFFIX}"
    rf"|(?:\d{{1,6}}[ \t]*[–-][ \t]*)?\d{{1,4}}(?:st|nd|rd|th)[ \t]+{_STREET_SUFFIX})"
    r"(?:[ \t]+(?:NW|NE|SW|SE|N|S|E|W)(?![A-Za-z]))?"
    r"(?:,?[ \t]+(?:Suite|Ste\.?|Floor|Fl\.?|Unit|Building|Bldg\.?|Room|Rm\.?)[ \t]*[\w-]+)?"
    rf"(?:,[ \t]*(?P<city>{_CITY}))?"
    rf"(?:,?[ \t]*(?:{_US_STATES})(?![A-Za-z]))?"
    rf"(?:,?[ \t]*(?:{_POSTCODE}))?"
)
#: San Francisco, CA 94105 -- without the street
_CITY_STATE_ZIP = re.compile(
    rf"\b(?P<city>{_CITY}),[ \t]*(?:{_US_STATES})[ \t]+(?:{_POSTCODE})\b"
)
#: 11 rue Watt 75013 Paris, 12 avenue de la République, 69002 Lyon
_FR_ADDRESS = re.compile(
    r"\b\d{1,4}(?:[ \t]?(?:bis|ter))?,?[ \t]+(?:rue|avenue|av\.|boulevard|bd|place|quai|allée|"
    r"chemin|impasse|cours|route)[ \t]+(?:(?:de|du|des|la|le|l['’]|d['’])[ \t]*)*"
    r"[A-ZÉÈ][\w'’-]*(?:[ \t]+(?:de|du|des|la|le|[A-ZÉÈ][\w'’-]*)){0,3},?[ \t]+(?:F-)?\d{5}"
    rf"[ \t]+(?P<city>{_CITY})"
)
#: Luis A. de Herrera 1248/1806, 11300 Montevideo -- the number after the
#: street, then a postcode and the city
_NUMBER_AFTER = re.compile(
    r"\b(?:[A-Z][\w'’.-]*|de|del|la|los|las|da|do|di|van|von)"
    r"(?:[ \t]+(?:[A-Z][\w'’.-]*|de|del|la|los|las|da|do|di|van|von)){0,4}"
    r"[ \t]+\d{1,5}(?:/\d{1,5})?,[ \t]*(?:[A-Z]{1,2}-)?\d{4,5}"
    rf"[ \t]+(?P<city>{_CITY})"
)
#: Binger Straße 173, 55216 Ingelheim am Rhein
_STRASSE = re.compile(
    r"\b[A-ZÄÖÜ][a-zäöüß]+(?:-[A-ZÄÖÜ][a-zäöüß]+)*[ -]?(?:[Ss]traße|[Ss]trasse|weg|platz|allee|gasse|ring|damm)"
    rf"[ \t]+\d{{1,4}}[a-z]?(?:,[ \t]*(?:D-)?\d{{4,5}}[ \t]+(?P<city>{_CITY}))?"
)
#: 北京市朝阳区建国路88号, 上海市浦东新区张江路100号
_ZH_ADDRESS = re.compile(
    r"[\u4e00-\u9fa5]{2,8}?(?:省|市|自治区)[\u4e00-\u9fa5]{0,12}?(?:市|区|县)?"
    r"[\u4e00-\u9fa5\d]{1,16}?(?:路|街|大道|道|巷)[\d\-]+号(?:[\u4e00-\u9fa5\d]{0,10}(?:楼|室|层|座))?"
)
#: Places the name model finds that are not worth masking: a country or a
#: region says nothing about who ran the study.
_NOT_A_PLACE = frozenset(x.strip() for x in """
    united states, usa, us, u.s., u.s.a., america, north america, south america,
    latin america, europe, european union, eu, asia, asia-pacific, africa,
    middle east, oceania, china, japan, korea, south korea, canada, mexico,
    brazil, argentina, chile, australia, new zealand, india, uk, united kingdom,
    great britain, britain, england, scotland, wales, germany, france, italy,
    spain, portugal, poland, russia, ukraine, netherlands, belgium, switzerland,
    austria, sweden, norway, denmark, finland, ireland, israel, turkey, greece,
    hungary, czech republic, czechia, romania, bulgaria, taiwan, singapore,
    thailand, vietnam, malaysia, philippines, indonesia, south africa, egypt,
    saudi arabia, west, east, north, south""".split(","))


def _places(text: str) -> Iterable[tuple[str, int, int]]:
    """Addresses, and the city in each: the address is replaced as a whole,
    the city on its own wherever else it is written."""
    for pat in (_ADDRESS, _CITY_STATE_ZIP, _STRASSE, _FR_ADDRESS, _NUMBER_AFTER, _ZH_ADDRESS):
        for m in pat.finditer(text):
            whole = m.group(0).rstrip(" ,")
            yield whole, m.start(), m.start() + len(whole)
            city = m.groupdict().get("city")
            if city and city.lower() not in _NOT_A_PLACE and city.lower() not in _HEADING_WORDS:
                yield city, m.start("city"), m.end("city")


def _ner_place(frag: str) -> bool:
    """A place the name model found: a city, a street, a state -- not a
    country, a region, or a capitalised protocol word."""
    frag = frag.strip()
    low = frag.lower()
    if len(frag) < 3 or not frag[0].isupper() or low in _NOT_A_PLACE:
        return False
    return not any(w in _HEADING_WORDS or w in _NAME_STOP or w in _NOT_A_NAME
                   for w in low.split())


def _pinyin_names(text: str) -> Iterable[tuple[str, int, int]]:
    """Chinese names romanised: Zhang Wei, Wei Zhang, WANG Jianguo,
    Jian-Guo Wang -- a pinyin surname beside a given name made of pinyin
    syllables, in either order. Without a title or a role in front of them,
    which is how most protocols list their staff."""
    word = r"[A-Z][a-z]+(?:-[A-Za-z][a-z]+)?|[A-Z]{2,}"
    for m in re.finditer(rf"\b({word})\s+({word})\b", text):
        a, b = m.group(1), m.group(2)
        if f"{a} {b}".lower() in _PINYIN_PLACES:
            continue
        if {a.lower(), b.lower()} & (_HEADING_WORDS | _NAME_STOP | _ADDRESS_WORDS):
            continue
        for sur, given in ((a, b), (b, a)):
            g = given.replace("-", "")
            if sur.isupper() and given.isupper():
                continue
            if (sur.lower() in _PINYIN_SURNAMES and len(g) >= 2
                    and (_PINYIN_GIVEN.fullmatch(g) or "-" in given)):
                yield m.group(0), m.start(), m.end()
                break
            if sur.lower() in _OTHER_SURNAMES and 2 <= len(g) <= 12:
                yield m.group(0), m.start(), m.end()
                break


def _zh_titled_names(text: str) -> Iterable[tuple[str, int, int]]:
    """王建国教授, 张伟 主任医师: a name followed by a title. Read backwards
    from the title, three characters then two then four, and kept only when
    it starts with a surname -- so 研究者应当 and 由北京 never become names."""
    from .profile import _CN_COMPOUND_SURNAMES, _CN_SURNAMES

    titles = re.compile(
        r"(?:副主任医师|主任医师|主治医师|住院医师|副教授|教授|副研究员|研究员|"
        r"博士|医生|医师|主任|经理|总监|老师)"
    )
    for m in titles.finditer(text):
        end = m.start()
        while end and text[end - 1] in " \t":
            end -= 1
        for n in (3, 2, 4):
            name = text[max(0, end - n):end]
            if (len(name) == n and re.fullmatch(r"[\u4e00-\u9fa5]+", name)
                    and (name[0] in _CN_SURNAMES
                         or name.startswith(_CN_COMPOUND_SURNAMES))):
                yield name, end - n, end
                break


def _key(text: str) -> str:
    """A term as a lookup key: case and the kind of whitespace do not count,
    so "Example\nBiologics" found across a line break is the same term."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _boundary(term: str) -> str:
    """A whole-term regex: word edges where the term starts or ends with a
    word character, none for CJK, where words are not separated. Spaces
    inside the term match any run of whitespace, line breaks included: a
    name wrapped onto the next line of a PDF is still the name."""
    term = re.sub(r"\s+", " ", term.strip())
    esc = r"\s+".join(re.escape(w) for w in term.split(" "))
    lead = r"(?<![A-Za-z0-9_])" if re.match(r"[A-Za-z0-9_]", term) else ""
    trail = r"(?![A-Za-z0-9_])" if re.search(r"[A-Za-z0-9_]$", term) else ""
    return lead + esc + trail


def _context(text: str, start: int, end: int, width: int = 40) -> str:
    a, b = max(0, start - width), min(len(text), end + width)
    return ("…" if a else "") + text[a:b].replace("\n", " ") + ("…" if b < len(text) else "")


def _known_from_data(vault: Vault) -> tuple[dict[str, str], set[str]]:
    """What the data side already relabelled.

    Returns the exact values with their labels (an arm string keeps the
    arm's label, a product name the product's), and the drug words inside
    them, which are only evidence that a word is a drug: 'pembrolizumab'
    inside two arms is neither arm, so it gets a DRUG label of its own.
    """
    labels: dict[str, str] = {}
    words: set[str] = set()
    for entity in ("treatment_product", "treatment"):
        for original, label in vault.labelled(entity).items():
            labels.setdefault(original.lower(), label)
            words |= {t.lower() for t in blinding.derive_terms([original])}
    return labels, words


def scan(
    paths: Iterable[Path], vault: Vault, detector: Any = None,
    *, propose_replacements: bool = True,
) -> list[Term]:
    """Propose the terms to mask, with a count and an example for each."""
    paths = list(paths)
    # The file name is the first place a study's identity leaks:
    # EX-EMPA-301_Protocol.pdf. It is scanned like a line of text.
    texts = [re.sub(r"[_]+", " ", p.stem) for p in paths]
    texts += [t for p in paths for t in read_text(p)]
    found: dict[str, Term] = {}

    def add(term: str, category: str, confidence: str, source: str,
            text: str, start: int, end: int) -> None:
        term = re.sub(r"\s+", " ", term).strip(
            " \t,.;:：，。" if category == "contact" else " \t,.;:：，。()（）"
        )
        if category in ("sponsor", "site"):
            # "Sponsor: Medivation, Inc., 525 Market Street, San Francisco":
            # the company, then its address -- a place of its own
            for pat in (_ADDRESS, _CITY_STATE_ZIP, _STRASSE, _FR_ADDRESS, _NUMBER_AFTER,
                        _ZH_ADDRESS):
                cut = pat.search(term)
                if cut and cut.start() > 0:
                    term = term[:cut.start()].rstrip(" \t,;:：，")
        if re.search(r"[\u4e00-\u9fa5]", term) and category in ("site", "sponsor", "drug"):
            cut = _ZH_CUT.match(term)
            if cut and 0 < cut.end() and len(term) - cut.end() >= 3:
                start += cut.end()
                term = term[cut.end():]
        if category == "person" and not re.search(r"[\u4e00-\u9fa5]", term):
            # "Xiaofeng" found, "Xiaofeng Li" written: a surname after the
            # match belongs to it, or the masked line reads "PERSON B Li".
            # On the same line only: in a table the next cell follows the
            # name after a line break ("Jean-Pierre MARTIN" / "Senior
            # Clinical Project Coordinator").
            tail = re.match(
                rf"(?:[ \t]+(?:{_PARTICLE}[ \t]+){{0,2}}(?:{_NAME_WORD}|(?!(?:MD|PHD|RN|NP|DO|MS|MBA|MPH|"
                r"PI|CRA|CEO|CMO|VP|USA|UK|PRC)\b)[A-Z]{2,}(?:['’-][A-Z]{2,})*)){1,2}\b",
                text[end:],
            )
            if tail and not any(w.lower() in _NAME_STOP for w in tail.group(0).split()):
                term += tail.group(0)
                end += len(tail.group(0))
        if category == "person" and not re.search(r"[\u4e00-\u9fa5]", term):
            # "Contact Ellen Ripley", "Questions: Lisa Wong": the word in
            # front is the sentence, not the name.
            words = term.split()
            while len(words) > 2 and words[0].lower().strip(":,") in (
                _NAME_STOP | _HEADING_WORDS | _LEAD_WORDS
            ):
                start += len(words[0]) + 1
                words = words[1:]
            term = " ".join(words)
        if len(term) < 2 or term.lower() in _NOT_A_NAME:
            return
        if category == "person" and not re.search(r"[\u4e00-\u9fa5]", term) and any(
                w.lower().strip(".,'’") in _NAME_STOP | _HEADING_WORDS for w in term.split()):
            return  # "Coordinator, Project": a job title in "Surname, Given" shape
        key = _key(term)
        if key in found:
            if _rank(confidence) > _rank(found[key].confidence):
                found[key].confidence = confidence
            return
        found[key] = Term(term, category, confidence=confidence, source=source,
                          example=_context(text, start, end))

    labels, words = _known_from_data(vault)
    known = set(labels) | words
    for text in texts:
        low = text.lower()
        for term in known:
            if len(term) < 4:
                continue
            m = re.search(_boundary(term), low)
            if m:
                add(text[m.start():m.end()], "drug", "high", "data", text,
                    m.start(), m.end())
        # One line at a time: a name does not continue onto the next line
        # of the source, and letting a pattern run across one is how
        # "Schedule of Activities" + "Clinic visit" became "Activities Clinic".
        lines = text.splitlines()
        for n, line in enumerate(lines):
            for category, conf, pat in _PATTERNS:
                for m in pat.finditer(line):
                    g = 1 if m.groups() and m.group(1) else 0
                    term = m.group(g)
                    # ...except where a PDF wrapped the name itself: a match
                    # at the very start of a line takes the capitalised
                    # words ending the line before ("Example" / "Biologics,
                    # Inc"), or a masked copy reads "Example SPONSOR A".
                    if (category in ("sponsor", "site") and n
                            and not line[:m.start(g)].strip()):
                        prev = re.search(rf"((?:{_CAP}[ \t]+){{0,2}}{_CAP})[ \t]*$", lines[n - 1])
                        if prev and not any(w.lower() in _NAME_STOP for w in prev.group(1).split()):
                            term = prev.group(1) + " " + term
                    # and a person's surname wrapped to the next line
                    # ("Dr. Xiaofeng" / "Li") is looked for there.
                    ctx = line
                    if category == "person" and n + 1 < len(lines) and lines[n + 1]:
                        ctx = line + " " + lines[n + 1]
                    if category == "drug" and _STANDARD.match(term):
                        continue  # ISO 14155, IEC 62304: a standard, not a compound
                    add(term, category, conf, "pattern", ctx, m.start(g), m.end(g))
        for term, start, end in _pinyin_names(text):
            add(term, "person", "high", "pattern", text, start, end)
        for term, start, end in _zh_titled_names(text):
            add(term, "person", "high", "pattern", text, start, end)
        for term, start, end in _initial_names(text):
            add(term, "person", "medium", "pattern", text, start, end)
        for term, start, end in _caps_surname_names(text):
            add(term, "person", "high", "pattern", text, start, end)
        for term, start, end in _places(text):
            add(term, "place", "high", "pattern", text, start, end)
        if detector is not None and text.strip():
            for f in detector.detect(text):
                cat = {"PERSON": "person", "EMAIL_ADDRESS": "contact",
                       "PHONE_NUMBER": "contact", "URL": "contact",
                       "LOCATION": "place", "GPE": "place"}.get(f.entity_type)
                if cat:
                    frag = text[f.start:f.end]
                    if cat == "person" and not _ner_person(frag):
                        continue
                    if cat == "place" and not _ner_place(frag):
                        continue
                    add(frag, cat, "medium", "ner", text, f.start, f.end)

    # A Chinese name alone in its cell or line, beside a short role cell.
    from .profile import _CN_COMPOUND_SURNAMES, _CN_SURNAMES

    for i, text in enumerate(texts):
        line = re.sub(r"\s+", "", text)
        if not (re.fullmatch(r"[\u4e00-\u9fa5·]{2,4}", line)
                and (line[0] in _CN_SURNAMES or line.startswith(_CN_COMPOUND_SURNAMES))):
            continue
        near = " ".join(t for t in texts[max(0, i - 2):i] + texts[i + 1:i + 3]
                        if len(t.strip()) <= 40)
        if _ROLE_NEAR_ZH.search(near) and not _ROLE_NEAR_ZH.search(line):
            add(line, "person", "high", "signature block", near + " " + text,
                len(near) + 1, len(near) + 1 + len(text))

    # A name alone on its line, beside a line that says what the person is:
    # a signature block, or a table row "Kevin Lee | Clinical Project Manager".
    for i, text in enumerate(texts):
        line = text.strip()
        if (not re.fullmatch(_PNAME, line) or line.lower() in _NOT_A_NAME
                or _ROLE_NEAR.search(line)
                or any(w.lower().strip("'’") in _HEADING_WORDS | _NAME_STOP
                       for w in line.split())):
            continue  # "Clinical Project Manager", "Adverse Events": not people
        # Only short neighbours count: a signature block or a table row, not
        # a heading followed by a paragraph that mentions the investigator.
        near_blocks = [t for t in texts[max(0, i - 2):i] + texts[i + 1:i + 3]
                       if len(t.strip()) <= 80]
        near = " ".join(near_blocks)
        if _ROLE_NEAR.search(near) or re.search(r"[\w.+-]+@[\w-]+\.", near):
            add(line, "person", "high", "signature block", near + " " + line,
                len(near) + 1, len(near) + 1 + len(line))

    # The same, one line at a time: a PDF table often comes out as one block
    # per row or per column ("Valerie Rossi" / "Director, Quality Assurance").
    flat_lines = [ln.strip() for text in texts for ln in text.splitlines() if ln.strip()]
    for i, line in enumerate(flat_lines):
        if (not re.fullmatch(_PNAME, line) or line.lower() in _NOT_A_NAME
                or _ROLE_NEAR.search(line)
                or any(w.lower().strip("'’") in _HEADING_WORDS | _NAME_STOP | _OUTLINE_WORDS
                       for w in line.split())):
            continue
        near = " ".join(t for t in flat_lines[max(0, i - 2):i] + flat_lines[i + 1:i + 3]
                        if len(t) <= 80)
        if _ROLE_NEAR.search(near):
            add(line, "person", "high", "table row", near + " " + line,
                len(near) + 1, len(near) + 1 + len(line))

    # A list of people: "Investigators: John Smith, Mary Jones and Wei Zhang",
    # "研究者：张伟、李娜、王芳". The patterns above find the name the cue sits
    # next to and stop; every other item in the same list is a name too.
    for text in texts:
        for line in text.splitlines():
            for term, start, end, conf in _list_names(line, found):
                add(term, "person", conf, "list", line, start, end)

    # A study number in the cell or on the line after its label.
    flat = [ln for text in texts for ln in text.splitlines()]
    for value, ctx in _labelled_ids(flat):
        key = _key(value)
        if key in found and found[key].category == "drug" and found[key].source == "pattern":
            found[key].category = "study_id"   # ABC-123 after "Protocol No.": not a compound
        add(value, "study_id", "high", "label", ctx, len(ctx) - len(value), len(ctx))

    # A list of drugs, in a sentence or one per bullet or table cell.
    for text in texts:
        para = re.sub(r"\s*\n\s*", " ", text)
        for term, start, end, conf in _list_drugs(para, found):
            add(term, "drug", conf, "list", para, start, end)
    run: list[str] = []
    for text in texts + [""]:
        item = _drug_item(text) if len(text) <= 60 else None
        if item and "\n" not in text:
            run.append(item)
            continue
        if len(run) >= 3 and sum(
                1 for x in run if _key(x) in found and found[_key(x)].category == "drug") >= 2:
            joined_run = ", ".join(run)
            for x in run:
                if _key(x) not in found:
                    i = joined_run.find(x)
                    add(x, "drug", "high" if " " not in x else "medium", "list",
                        joined_run, i, i + len(x))
        run = []

    # The company behind "Merck Sharp & Dohme LLC" is also written "Merck".
    # Its first word is offered as a term of its own, at low confidence:
    # often it is the name, sometimes it is a place ("Jiangsu").
    joined = "\n".join(texts)
    for t in list(found.values()):
        if t.category == "sponsor" and " " in t.term:
            head = t.term.split()[0].strip(",")
            if len(head) >= 4 and head.lower() not in found and head[0].isupper():
                m = re.search(_boundary(head), joined)
                if m:
                    add(head, "sponsor", "low", "pattern", joined, m.start(), m.end())

    # Other names for a drug already found: its abbreviation (EMPA for
    # empagliflozin, in a short title or a file name), its brand
    # ("JARDIANCE (empagliflozin)", "empagliflozin (Jardiance®)") and its
    # development code ("empagliflozin (BI 10773)"). Each is tied to the
    # drug, so it is replaced with the same label.
    aliases: dict[str, str] = {}
    drugs = [t for t in list(found.values())
             if t.category == "drug" and t.term.isalpha() and len(t.term) >= 6]
    for d in drugs:
        for x in re.finditer(r"\b[A-Z]{4,6}\b", joined):
            tok = x.group(0)
            if d.term.upper().startswith(tok) and tok != d.term.upper():
                add(tok, "drug", "high", "abbreviation", joined, x.start(), x.end())
                aliases[tok.lower()] = d.term.lower()
        # The drug name in any case, the brand with its capital: under one
        # case-blind flag "inhibitors (boceprevir" made "inhibitors" a brand
        # of boceprevir, and every "inhibitors (" in the protocol DRUG A.
        name = f"(?i:{_boundary(d.term)})"
        for pat in (
            rf"\b([A-Z][A-Za-z]{{3,}})\s*[®™]?\s*\(\s*{name}",
            rf"{name}\s*\(\s*([A-Z][A-Za-z]{{3,}}|[A-Z]{{1,5}}[- ]?\d{{3,6}}[A-Z]?)\s*[®™]?\s*\)",
        ):
            for x in re.finditer(pat, joined):
                brand = x.group(1)
                low = brand.lower()
                if (low in _NOT_A_BRAND or low == d.term.lower()
                        or low in _NOT_A_DRUG_WORD or low in _HEADING_WORDS):
                    continue
                add(brand, "drug", "high", "brand", joined, x.start(1), x.end(1))
                aliases[brand.lower()] = d.term.lower()

    # Count what each term would actually replace, longest first: a mention
    # of "Pembrolizumab" inside "Pembrolizumab 200 mg Q3W" belongs to the
    # longer term, and a term nothing is left for is dropped.
    counts: dict[str, int] = {}
    m = _matcher({t.term: "" for t in found.values()})
    if m is not None:
        for x in m[0].finditer(joined):
            counts[_key(x.group(0))] = counts.get(_key(x.group(0)), 0) + 1
    kept = []
    for key, t in found.items():
        t.count = counts.get(key, 0)
        if t.count:
            if key in aliases and aliases[key] in found:
                t.alias_of = found[aliases[key]].term
            if propose_replacements and not t.alias_of:
                t.proposed = propose(t, vault, labels)
            kept.append(t)
    if propose_replacements:
        for t in kept:
            if t.alias_of:
                t.proposed = found[t.alias_of.lower()].proposed
    order = {"low": 0, "medium": 1, "high": 2}
    kept.sort(key=lambda t: (order[t.confidence], -t.count, t.term.lower()))
    return kept


def _rank(confidence: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(confidence, 0)


# ----------------------------------------------------------------------
# replacements
# ----------------------------------------------------------------------
def propose(t: Term, vault: Vault, labels: dict[str, str] | None = None) -> str:
    """The replacement for one term, from the vault so it is stable and the
    same as the data's wherever the data met the same thing."""
    key = t.term.strip()
    if t.category == "drug":
        labels = labels if labels is not None else _known_from_data(vault)[0]
        if key.lower() in labels:
            return labels[key.lower()]
        return vault.label_map("treatment_product", [key.upper()], prefix="DRUG")[key.upper()]
    if t.category == "sponsor":
        return vault.label_map("sponsor", [key.upper()], prefix="SPONSOR")[key.upper()]
    if t.category == "site":
        return vault.label_map("institution", [key.upper()], prefix="INSTITUTION")[key.upper()]
    if t.category == "person":
        return vault.label_map("person", [key.upper()], prefix="PERSON")[key.upper()]
    if t.category == "study_id":
        return vault.surrogate_for("studyid", key, preserve_format=True)
    if t.category == "place":
        return vault.label_map("place", [key.upper()], prefix="PLACE")[key.upper()]
    return dummy_value(key, vault)  # contact details: same shape, no content


def rules_digest(terms: list[Term]) -> str:
    payload = sorted((t.term, t.decision, t.final() or "") for t in terms)
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# writing
# ----------------------------------------------------------------------
def _matcher(mapping: dict[str, str]) -> tuple[re.Pattern[str], dict[str, str]] | None:
    """One regex for every term, longest first, so 'MK-3475-189' is replaced
    as a whole before 'MK-3475' can take a bite out of it."""
    if not mapping:
        return None
    terms = sorted(mapping, key=len, reverse=True)
    pat = re.compile("|".join(_boundary(t) for t in terms), re.I)
    return pat, {_key(t): r for t, r in mapping.items()}


def replace_text(text: str, mapping: dict[str, str]) -> tuple[str, int]:
    m = _matcher(mapping)
    if m is None or not text:
        return text, 0
    pat, lookup = m
    n = 0

    def sub(x: re.Match[str]) -> str:
        nonlocal n
        n += 1
        return lookup[_key(x.group(0))]

    return pat.sub(sub, text), n


def _rewrite_paragraph(nodes: list[Any], pat, lookup) -> int:
    """Replace across the text nodes of one paragraph. A term split over
    runs ("Pembro" in bold, "lizumab" plain) is still one term; its
    replacement takes the formatting of the run where it began."""
    texts = [t.text or "" for t in nodes]
    owner = [i for i, s in enumerate(texts) for _ in s]
    full = "".join(texts)
    matches = list(pat.finditer(full))
    if not matches:
        return 0
    out = [""] * len(nodes)
    pos = 0
    for m in matches:
        for c in range(pos, m.start()):
            out[owner[c]] += full[c]
        out[owner[m.start()]] += lookup[_key(m.group(0))]
        pos = m.end()
    for c in range(pos, len(full)):
        out[owner[c]] += full[c]
    for t, s in zip(nodes, out):
        if t.text != s:
            t.text = s
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return len(matches)


# ----------------------------------------------------------------------
# pictures and embedded objects
# ----------------------------------------------------------------------
#: What a removed picture says, as its alt text and, in a PDF, in its frame.
REMOVED_LABEL = "Figure removed"

_PLACEHOLDER = "word/media/deidkit_figure_removed.png"
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_V = "{urn:schemas-microsoft-com:vml}"
_O = "{urn:schemas-microsoft-com:office:office}"
_R_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
_CT = "{http://schemas.openxmlformats.org/package/2006/content-types}"


def _placeholder_png(w: int = 160, h: int = 120) -> bytes:
    """A light grey box with a border and a cross -- the drawing convention
    for "no picture here". Word stretches it to the picture's frame."""
    fill, line = b"\xe8\xe8\xe8", b"\x96\x96\x96"
    rows = []
    for y in range(h):
        row = bytearray(b"\x00")
        for x in range(w):
            edge = x < 2 or y < 2 or x >= w - 2 or y >= h - 2
            cross = abs(x * h - y * w) <= w or abs((w - 1 - x) * h - y * w) <= w
            row += line if edge or cross else fill
        rows.append(bytes(row))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows), 9))
            + chunk(b"IEND", b""))


def _scrub_pictures(root, ole_ids: set[str], mapping: dict[str, str]) -> tuple[int, int]:
    """Alt text and names of pictures say what the picture was ("Acme logo",
    "signature_jsmith.png"): a picture's become REMOVED_LABEL; any other
    shape's are masked like text. An embedded object whose part is going is
    taken out of the document, leaving its preview picture -- which is
    itself replaced. Returns (terms replaced, objects removed)."""
    replaced = 0
    for e in root.iter():
        if not isinstance(e.tag, str):
            continue
        local = e.tag.rsplit("}", 1)[-1]
        if local in ("docPr", "cNvPr"):
            holder = e.getparent()
            if local == "docPr":
                picture = holder is not None and next(holder.iter(_A + "blip"), None) is not None
            else:
                picture = holder is not None and holder.tag.endswith("}nvPicPr")
            if picture:
                e.set("name", f"Figure {e.get('id', '')}".strip())
                e.set("descr", REMOVED_LABEL)
                e.attrib.pop("title", None)
                continue
            for a in ("name", "descr", "title"):
                if e.get(a):
                    new, k = replace_text(e.get(a), mapping)
                    if k:
                        e.set(a, new)
                        replaced += k
        elif e.tag == _V + "imagedata":
            e.attrib.pop(_O + "title", None)
            shape = e.getparent()
            if shape is not None and shape.get("alt") is not None:
                shape.set("alt", REMOVED_LABEL)
    gone = 0
    for e in list(root.iter(_O + "OLEObject")):
        if e.get(_R_ID) in ole_ids:
            e.getparent().remove(e)
            gone += 1
    return replaced, gone


def _write_docx(src: Path, dst: Path, mapping: dict[str, str]) -> dict[str, int]:
    from lxml import etree

    def xml(root) -> bytes:
        return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)

    m = _matcher(mapping)
    with zipfile.ZipFile(src) as zin:
        items = [(i, zin.read(i.filename)) for i in zin.infolist()]
    names = {i.filename for i, _ in items}
    out: dict[str, bytes] = {}
    replaced = 0

    # 1. Relationships. Every picture is pointed at the placeholder, so the
    # original is left unreferenced and dropped; an embedded object and the
    # first-page thumbnail lose their relationship and their part.
    pictures: set[str] = set()
    linked = objects = 0
    ole_ids: dict[str, set[str]] = {}
    referenced: set[str] = set()
    for item, data in items:
        name = item.filename
        if not name.endswith(".rels"):
            continue
        root = etree.fromstring(data)
        base = posixpath.dirname(posixpath.dirname(name))
        source = posixpath.join(base, posixpath.basename(name)[: -len(".rels")])
        for rel in list(root):
            kind = (rel.get("Type") or "").rsplit("/", 1)[-1]
            target = rel.get("Target") or ""
            external = rel.get("TargetMode") == "External"
            resolved = None if external else (
                target.lstrip("/") if target.startswith("/")
                else posixpath.normpath(posixpath.join(base, target)))
            if kind in ("image", "hdphoto"):
                if external:
                    linked += 1
                    rel.attrib.pop("TargetMode")
                else:
                    pictures.add(resolved)
                rel.set("Target", posixpath.relpath(_PLACEHOLDER, base or "."))
            elif kind in ("oleObject", "package", "thumbnail"):
                if kind == "thumbnail":
                    pictures.add(resolved or target)
                else:
                    objects += 1
                ole_ids.setdefault(source, set()).add(rel.get("Id"))
                root.remove(rel)
            else:
                if kind == "hyperlink" and external and m is not None:
                    new, k = replace_text(target, mapping)
                    if k:
                        rel.set("Target", new)
                        replaced += k
                if resolved:
                    referenced.add(resolved)
        out[name] = xml(root)
    # Anything under media/ or embeddings/ nothing points at any more goes:
    # the originals just unhooked, and orphans an editor left behind.
    drop = {n for n in names
            if re.search(r"(^|/)(media|embeddings)/", n) and n not in referenced
            and n != _PLACEHOLDER}
    drop |= {p for p in pictures if p in names}

    # 2. The parts with text: mask it, and scrub the pictures' alt text.
    for item, data in items:
        name = item.filename
        if _TEXT_PARTS.match(name):
            root = etree.fromstring(data)
            if m is not None:
                for nodes in _docx_paragraphs(root):
                    replaced += _rewrite_paragraph(nodes, *m)
            k, _ = _scrub_pictures(root, ole_ids.get(name, set()), mapping)
            replaced += k
            out[name] = xml(root)
        elif name in ole_ids:
            root = etree.fromstring(data)
            _scrub_pictures(root, ole_ids[name], mapping)
            out[name] = xml(root)
        elif name == "docProps/core.xml":
            root = etree.fromstring(data)
            for e in root.iter():
                tag = etree.QName(e).localname
                if tag in ("creator", "lastModifiedBy"):
                    e.text = ""
                elif e.text and m is not None:
                    e.text, k = replace_text(e.text, mapping)
                    replaced += k
            out[name] = xml(root)
        elif name == "docProps/app.xml":
            root = etree.fromstring(data)
            for e in root.iter():
                if etree.QName(e).localname in ("Company", "Manager"):
                    e.text = ""
            out[name] = xml(root)
        elif name == "[Content_Types].xml":
            root = etree.fromstring(data)
            for e in list(root):
                if e.tag == _CT + "Override" and (e.get("PartName") or "").lstrip("/") in drop:
                    root.remove(e)
            if not any(e.tag == _CT + "Default" and (e.get("Extension") or "").lower() == "png"
                       for e in root):
                d = etree.SubElement(root, _CT + "Default")
                d.set("Extension", "png")
                d.set("ContentType", "image/png")
            out[name] = xml(root)

    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for item, data in items:
            if item.filename in drop or item.filename == _PLACEHOLDER:
                continue
            zout.writestr(item, out.get(item.filename, data))
        zout.writestr(_PLACEHOLDER, _placeholder_png())
    return {"replaced": replaced,
            "images_removed": len(pictures) + linked,
            "objects_removed": objects}


def images_left(path: Path) -> int:
    """Pictures and embedded objects still in a written file. The answer has
    to be 0; the read-back reports anything else."""
    suffix = path.suffix.lower()
    if suffix == ".docx":
        with zipfile.ZipFile(path) as z:
            return sum(
                1 for n in z.namelist()
                if n != _PLACEHOLDER and (
                    re.search(r"(^|/)(media|embeddings)/", n)
                    or n.startswith("docProps/thumbnail"))
            )
    if suffix == ".pdf":
        pm = _pymupdf()
        if pm is None:
            return 0
        with pm.open(str(path)) as doc:
            return (sum(len(page.get_images(full=True)) for page in doc)
                    + len(doc.embfile_names()))
    return 0


def _draw_placeholder(pm, page, box) -> None:
    """The grey box with a cross where a picture was. Not where text sits
    on top of the picture (a background, a letterhead): the text stays
    readable and the box is left out."""
    box = pm.Rect(box) & page.rect
    if box.is_empty or box.width < 2 or box.height < 2:
        return
    if any(b[6] == 0 and pm.Rect(b[:4]).intersects(box) for b in page.get_text("blocks")):
        return
    grey, fill = (0.59, 0.59, 0.59), (0.91, 0.91, 0.91)
    shape = page.new_shape()
    shape.draw_rect(box)
    shape.finish(color=grey, fill=fill, width=0.5)
    shape.draw_line(box.tl, box.br)
    shape.draw_line(box.tr, box.bl)
    shape.finish(color=grey, width=0.5)
    shape.commit()
    if box.width >= 80 and box.height >= 20:
        mid = (box.y0 + box.y1) / 2
        label = pm.Rect(box.x0, mid - 7, box.x1, mid + 7)
        page.draw_rect(pm.Rect(box.x0 + box.width / 2 - 40, mid - 7,
                               box.x0 + box.width / 2 + 40, mid + 7),
                       color=None, fill=fill, width=0)
        page.insert_textbox(label, REMOVED_LABEL, fontname="helv", fontsize=8,
                            color=(0.3, 0.3, 0.3), align=1)


def _remove_pdf_pictures(pm, page) -> int:
    """Take every picture off the page -- out of the file, not covered --
    and draw the placeholder in its frame. Text over a picture stays."""
    infos = [i for i in page.get_image_info(xrefs=True)
             if not pm.Rect(i["bbox"]).is_empty]
    if not infos:
        return 0
    boxes = [pm.Rect(i["bbox"]) for i in infos]
    if hasattr(pm, "PDF_REDACT_TEXT_NONE"):
        for b in boxes:
            page.add_redact_annot(b, fill=False, cross_out=False)
        page.apply_redactions(
            images=pm.PDF_REDACT_IMAGE_REMOVE,
            graphics=getattr(pm, "PDF_REDACT_LINE_ART_NONE", 0),
            text=pm.PDF_REDACT_TEXT_NONE,
        )
    else:
        # An older PyMuPDF would take the text over a picture with it.
        for xref in {i["xref"] for i in infos if i.get("xref")}:
            page.delete_image(xref)
    for b in boxes:
        _draw_placeholder(pm, page, b)
    return len(infos)


def _pymupdf():
    try:
        import pymupdf  # noqa: F401

        return pymupdf
    except ImportError:
        return None


def output_name(src: Path) -> str:
    if src.suffix.lower() == ".pdf" and _pymupdf() is None:
        return src.stem + ".txt"
    return src.stem + src.suffix.lower()


def _base14(span: dict[str, Any], text: str) -> str:
    """The standard PDF font closest to the one the replaced text was in.

    The original font is usually embedded as a subset holding only the
    glyphs the document used, so it cannot write new text; the nearest of
    the fonts every PDF reader has is the honest substitute. Serif stays
    serif, bold stays bold."""
    name = (span.get("font") or "").lower()
    flags = int(span.get("flags") or 0)
    bold = bool(flags & 16) or any(w in name for w in ("bold", "black", "heavy", "semibold"))
    italic = bool(flags & 2) or "italic" in name or "oblique" in name
    if not text.isascii():
        serif_cjk = any(w in name for w in ("song", "ming", "simsun", "serif", "kai", "fang"))
        return "china-s" if serif_cjk else "china-ss"
    if flags & 8 or any(w in name for w in ("courier", "mono", "consolas")):
        family = ("cour", "cobo", "coit", "cobi")
    elif flags & 4 or any(w in name for w in (
            "times", "serif", "roman", "georgia", "garamond", "cambria", "book")):
        family = ("tiro", "tibo", "tiit", "tibi")
    else:
        family = ("helv", "hebo", "heit", "hebi")
    return family[(1 if bold else 0) + (2 if italic else 0)]


def _rgb(color: int) -> tuple[float, float, float]:
    return ((color >> 16 & 255) / 255, (color >> 8 & 255) / 255, (color & 255) / 255)


def _write_pdf(src: Path, dst: Path, mapping: dict[str, str]) -> dict[str, int]:
    """Redact each term where it sits on the page and write the replacement
    in its place, so the masked PDF keeps its pages, tables and layout.

    Redaction removes the text from the file, not just from view -- a black
    box over a selectable word is the classic leak. Matching is done per
    line on the characters' own positions, with the same whole-term rule as
    everywhere else, over each block with its lines joined, so a term
    wrapped onto the next line is found and redacted on both. A word split
    by a hyphen at the line end is not; the read-back counts it.

    A replacement is written in the size, colour and nearest standard font
    of the text it replaces. When it is longer than the original ("ACME" ->
    SPONSOR A) the rest of the line moves right to make room -- rewritten in
    the nearest standard font, each run kept to its original width, and an
    underline beneath it moved with it -- instead of the replacement being
    squeezed into the old word's space at a fraction of the line's size.
    Only when the line would then run past the page edge is the replacement
    narrowed, and only after that does its size give way.
    """
    pm = _pymupdf()
    m = _matcher(mapping)
    doc = pm.open(str(src))
    replaced = images = empty = objects = 0
    for page in doc:
        if not page.get_text("text").strip():
            empty += 1
        images += _remove_pdf_pictures(pm, page)
        for a in list(page.annots(types=[pm.PDF_ANNOT_FILE_ATTACHMENT]) or []):
            page.delete_annot(a)
            objects += 1
        if m is None:
            continue
        pat, lookup = m
        line_hits: dict[tuple[int, int], list[tuple[int, int, str]]] = {}
        lines_of: dict[tuple[int, int], list[dict[str, Any]]] = {}
        wrapped: list[tuple[list[list[Any]], str]] = []
        raw = page.get_text("rawdict")
        for bi, block in enumerate(raw.get("blocks", [])):
            # The block's lines joined with a newline that has no box, so a
            # term wrapped onto the next line ("Example" / "Biologics, Inc")
            # is still found -- and redacted on both lines.
            chars: list[dict[str, Any]] = []
            for li, line in enumerate(block.get("lines", [])):
                if chars:
                    chars.append({"c": "\n", "bbox": None})
                own = [{**ch, "span": span, "line": (bi, li)}
                       for span in line["spans"] for ch in span["chars"]]
                for k, ch in enumerate(own):
                    ch["k"] = k
                lines_of[(bi, li)] = own
                chars += own
            text = "".join(c["c"] for c in chars)
            for x in pat.finditer(text):
                seg = [c for c in chars[x.start():x.end()] if c["bbox"] is not None]
                new_text = lookup[_key(x.group(0))]
                if not seg:
                    continue
                if len({c["line"] for c in seg}) == 1:
                    line_hits.setdefault(seg[0]["line"], []).append(
                        (seg[0]["k"], seg[-1]["k"] + 1, new_text))
                    continue
                pieces: list[list[Any]] = []
                for c in seg:
                    if pieces and pieces[-1][3] == c["line"]:
                        pieces[-1][0] |= pm.Rect(c["bbox"])
                    else:
                        pieces.append([pm.Rect(c["bbox"]), c["origin"], c["span"], c["line"]])
                wrapped.append((pieces, new_text))
        if not line_hits and not wrapped:
            continue

        drawings = page.get_drawings()
        plans, underlines = [], []
        for key, hits in line_hits.items():
            plan = _plan_line(pm, page, lines_of[key], sorted(hits), drawings)
            plans.append(plan)
            underlines += plan["underlines"]
        for plan in plans:
            for box in plan["redact"]:
                page.add_redact_annot(box, fill=False, cross_out=False)
        for pieces, _ in wrapped:
            for box, *_ in pieces:
                page.add_redact_annot(box, fill=False, cross_out=False)
        page.apply_redactions(
            images=pm.PDF_REDACT_IMAGE_NONE,
            graphics=getattr(pm, "PDF_REDACT_LINE_ART_NONE", 0),
        )
        # An underline under text that moved is removed and drawn again
        # under it. Needs a PyMuPDF that can redact line art alone; on an
        # older one the underline stays where it was.
        if underlines:
            for u in underlines:
                pad = max(1.0, float(u["width"]))
                page.add_redact_annot(u["rect"] + (-pad, -pad, pad, pad),
                                      fill=False, cross_out=False)
            try:
                page.apply_redactions(
                    images=pm.PDF_REDACT_IMAGE_NONE,
                    graphics=pm.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
                    text=pm.PDF_REDACT_TEXT_NONE,
                )
                moved = True
            except (TypeError, AttributeError):
                moved = False
        for plan in plans:
            for item in plan["draw"]:
                _draw_run(pm, page, *item)
        if underlines and moved:
            for u in underlines:
                r = u["rect"] + (u["shift"], 0, u["shift"] + u["grow"], 0)
                if u["stroke"]:
                    page.draw_line(r.bl, r.br, color=u["color"] or (0, 0, 0),
                                   width=u["width"])
                else:
                    page.draw_rect(r, color=None, fill=u["color"] or (0, 0, 0), width=0)
        # A wrapped term is written once, in the widest of its pieces -- a
        # fragment at the end of a line has no room for it.
        for pieces, new_text in wrapped:
            box, origin, span, _ = max(pieces, key=lambda p: p[0].width)
            _draw_run(pm, page, origin, new_text, span, box.width)
        replaced += sum(len(h) for h in line_hits.values()) + len(wrapped)
    for name in doc.embfile_names():
        doc.embfile_del(name)
        objects += 1
    # Title, author, subject, keywords: the protocol's name is usually there.
    doc.set_metadata({})
    try:
        doc.del_xml_metadata()
    except Exception:  # noqa: BLE001 - older PyMuPDF
        pass
    doc.save(str(dst), garbage=4, deflate=True, clean=True)
    doc.close()
    out = {"replaced": replaced, "images_removed": images, "objects_removed": objects}
    if empty:
        out["pages_without_text"] = empty
    return out


def _draw_run(pm, page, origin, text: str, span: dict[str, Any], width: float | None,
              squeeze: float | None = None) -> None:
    """Write ``text`` on a baseline in the span's size, colour and nearest
    standard font. With ``width``: fitted to it -- stretched or narrowed to
    within 15% for a run of the original line (so its words land where they
    were), or narrowed to 60% and then shrunk for a replacement given no
    more room than that."""
    size = span["size"]
    font = _base14(span, text)
    need = pm.get_text_length(text, fontname=font, fontsize=size) or 1
    if squeeze is None:
        squeeze = 1.0 if width is None else max(0.6, min(1.0, width / need))
        if width is not None and need * squeeze > width:
            size = max(4.0, size * width / (need * squeeze))
    page.insert_text(
        origin, text, fontname=font, fontsize=size,
        color=_rgb(span.get("color", 0)),
        morph=(pm.Point(origin), pm.Matrix(squeeze, 1)),
    )


def _plan_line(pm, page, chars: list[dict[str, Any]], hits: list[tuple[int, int, str]],
               drawings: list[dict[str, Any]]) -> dict[str, Any]:
    """What to redact and what to write for one line with replacements."""
    def box(a: int, b: int):
        r = pm.Rect(chars[a]["bbox"])
        for c in chars[a + 1:b]:
            r |= pm.Rect(c["bbox"])
        return r

    def need(text: str, span: dict[str, Any]) -> float:
        return pm.get_text_length(text, fontname=_base14(span, text), fontsize=span["size"])

    grow = [max(0.0, need(t, chars[a]["span"]) - box(a, b).width) for a, b, t in hits]
    total = sum(grow)
    if total <= 0.5:
        # Every replacement fits where its term was: write each in place.
        return {
            "redact": [box(a, b) for a, b, _ in hits],
            "draw": [(chars[a]["origin"], t, chars[a]["span"], box(a, b).width)
                     for a, b, t in hits],
            "underlines": [],
        }

    # The rest of the line moves right by what the replacements need --
    # as far as the page allows. Past that, every replacement is narrowed
    # by the same factor (and _draw_run shrinks it only beyond 60%).
    line_end = box(0, len(chars)).x1
    room = max(0.0, page.rect.x1 - 18 - line_end)
    factor = 1.0 if total <= room else room / total
    first = hits[0][0]
    tail = box(first, len(chars))
    tail = pm.Rect(tail.x0, tail.y0 + tail.height * 0.15, tail.x1, tail.y1 - tail.height * 0.15)
    draw: list[tuple[Any, ...]] = []
    shift, i, hi = 0.0, first, 0
    shifts: list[tuple[float, float]] = []  # (original x, shift from there on)
    while i < len(chars):
        if hi < len(hits) and i == hits[hi][0]:
            a, b, t = hits[hi]
            old = box(a, b)
            span = chars[a]["span"]
            want = old.width + grow[hi] * factor
            o = chars[a]["origin"]
            draw.append(((o[0] + shift, o[1]), t, span, want))
            shift += want - old.width
            shifts.append((old.x1, shift))
            i, hi = b, hi + 1
            continue
        # a run of original characters in one span, up to the next hit
        j = i
        stop = hits[hi][0] if hi < len(hits) else len(chars)
        while j < stop and chars[j]["span"] is chars[i]["span"]:
            j += 1
        run = "".join(c["c"] for c in chars[i:j])
        if run.strip():
            # skip leading spaces: the run starts where its first glyph was
            lead = len(run) - len(run.lstrip())
            k = i + lead
            r = box(k, j)
            span = chars[k]["span"]
            text = run.strip()
            fit = r.width / (need(text, span) or 1)
            o = chars[k]["origin"]
            draw.append(((o[0] + shift, o[1]), text, span, None,
                         max(0.85, min(1.15, fit))))
        i = j

    # Underlines beneath the moved text move with it.
    base = chars[first]["origin"][1]
    size = chars[first]["span"]["size"]
    underlines = []
    for d in drawings:
        r = d.get("rect")
        if r is None or r.height > 2.5 or r.width < 3:
            continue
        if not (base - 1 <= r.y0 <= base + size * 0.4):
            continue
        if r.x1 <= tail.x0 or r.x0 >= line_end + 1:
            continue
        s = 0.0
        for x, sh in shifts:
            if r.x0 >= x - 0.5:
                s = sh
        grow_by = shift - s if r.x0 < tail.x0 else 0.0
        underlines.append({"rect": pm.Rect(r), "shift": s, "grow": grow_by,
                           "color": d.get("color") or d.get("fill"),
                           "width": d.get("width") or 1.0,
                           "stroke": d.get("type") in ("s", "fs") and r.height < 0.5})
    return {"redact": [tail], "draw": draw, "underlines": underlines}


def masked_name(src: Path, mapping: dict[str, str]) -> str:
    """The output file name, with the same terms replaced as in the text --
    a masked EX-EMPA-301_Protocol.pdf must not keep its name."""
    name = output_name(src)
    stem, dot, ext = name.rpartition(".")
    readable = re.sub(r"_+", " ", stem)
    masked, n = replace_text(readable, mapping)
    if not n:
        return name
    masked = re.sub(r'[\\/:*?"<>|\s]+', "_", masked).strip("_")
    return f"{masked}.{ext}"


def write(src: Path, out_dir: Path, mapping: dict[str, str]) -> dict[str, Any]:
    """Write the masked copy of one document and say what was done."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / masked_name(src, mapping)
    suffix = src.suffix.lower()
    info: dict[str, Any] = {"source": src.name, "output": dst.name,
                            "sha256_in": hashlib.sha256(src.read_bytes()).hexdigest()}
    if suffix == ".docx":
        info |= _write_docx(src, dst, mapping)
    elif suffix == ".pdf" and _pymupdf() is not None:
        info |= _write_pdf(src, dst, mapping)
    else:
        pages = read_text(src)
        total = 0
        chunks = []
        for n, page in enumerate(pages, 1):
            text, k = replace_text(page, mapping)
            total += k
            chunks.append(f"=== page {n} ===\n{text}" if suffix == ".pdf" else text)
        dst.write_text("\n".join(chunks) if suffix != ".pdf" else "\n\n".join(chunks),
                       encoding="utf-8")
        info["replaced"] = total
        if suffix == ".pdf":
            empty = sum(1 for p in pages if not p.strip())
            info["pages"] = len(pages)
            if empty:
                info["pages_without_text"] = empty  # scanned: needs OCR first
    return info


def residual_terms(out_paths: Iterable[Path], terms: Iterable[str]) -> dict[str, int]:
    """Which approved terms still appear in the output, and how often. The
    answer has to be empty; anything else is a leak the report names."""
    terms = [t for t in terms if t]
    if not terms:
        return {}
    pat = re.compile("|".join(_boundary(t) for t in sorted(terms, key=len, reverse=True)), re.I)
    by_key = {_key(t): t for t in terms}
    left: dict[str, int] = {}
    for p in out_paths:
        for text in [re.sub(r"_+", " ", p.stem)] + read_text(p):
            for x in pat.finditer(text):
                name = by_key.get(_key(x.group(0)), x.group(0))
                left[name] = left.get(name, 0) + 1
    return left


def residual(out_paths: Iterable[Path], terms: Iterable[str]) -> int:
    return sum(residual_terms(out_paths, terms).values())


def _made_by_masking(term: str, replacements: set[str]) -> bool:
    """A flag the replacements themselves raised: a label ("DRUG A"), part
    of one, or text with a label in it -- "DRUG AI DRUG AG" reads like a
    company ending in AG, "P-gp DRUG BD" and "Unbound PLACE K" like a name
    with its surname in capitals. With the labels taken out, at most one
    word may be left -- one word alone is never proposed as a name -- so
    "SPONSOR A Acme Oncology Inc." is still flagged."""
    low = term.lower()
    if low in replacements or any(low in r for r in replacements):
        return True
    rest, hit = low, False
    for r in sorted((r for r in replacements if r), key=len, reverse=True):
        rest, n = re.subn(_boundary(r), " ", rest)
        hit = hit or n > 0
    words = re.findall(r"[a-z\u4e00-\u9fa5][a-z\u4e00-\u9fa5'’.-]{2,}", rest)
    return hit and len(words) <= 1


def apply(
    paths: list[Path], terms: list[Term], vault: Vault, out_dir: str | Path,
    *, approved_by: str, detector: Any = None,
) -> dict[str, Any]:
    """Write every masked document, check it, and write the record."""
    blank = [t.term for t in terms if t.decision not in DECISIONS]
    if blank:
        raise ValueError(f"{len(blank)} term(s) have no decision yet")
    bad = [t.term for t in terms if t.decision == "CHANGE" and not t.replacement.strip()]
    if bad:
        raise ValueError(f"{len(bad)} CHANGE term(s) have no replacement text")
    if not approved_by.strip():
        raise ValueError("the masking must be approved by a named person")

    mapping = {t.term: t.final() for t in terms if t.final() is not None}
    out = Path(out_dir)
    files = [write(p, out, mapping) for p in paths]
    outs = [out / f["output"] for f in files]
    for f, o in zip(files, outs):
        f["images_left"] = images_left(o)
    left_terms = residual_terms(outs, mapping)
    left = sum(left_terms.values())
    # Anything the scan would still flag in the masked copy, ignoring what was
    # deliberately kept and what the replacements themselves look like.
    kept = {t.term.lower() for t in terms if t.decision == "KEEP"}
    replacements = {r.lower() for r in mapping.values()}
    again = [
        asdict(t) for t in scan(outs, vault, detector, propose_replacements=False)
        if t.term.lower() not in kept and not _made_by_masking(t.term, replacements)
    ]

    by_cat: dict[str, int] = {}
    for t in terms:
        if t.final() is not None:
            by_cat[t.category] = by_cat.get(t.category, 0) + 1
    manifest = {
        "tool": "deidkit protocol",
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "approved_by": approved_by,
        "rules_fingerprint": rules_digest(terms),
        # Output names only: the original file name is part of what is
        # being masked, so it goes to the _review folder with the terms.
        "files": [{k: v for k, v in f.items() if k != "source"} for f in files],
        "terms_replaced_by_category": by_cat,
        "terms_kept": len(kept),
        "residual_occurrences": left,
        "images_left": sum(f["images_left"] for f in files),
        "still_flagged_after_masking": len(again),
        "notes": [
            "Pictures are not read, so every one is removed: images_removed "
            "counts them per file, objects_removed the embedded objects, and "
            "images_left the ones the read-back still found (it must be 0).",
            "Text drawn as vector graphics (a logo made of shapes, text "
            "turned into outlines) is not text and not a picture: it survives.",
            "A PDF is redacted in place: the text is removed from the file and "
            "the replacement drawn where it stood. A term broken across two "
            "lines is not matched there and shows up as residual.",
            "The term list with the originals is in the _review folder beside "
            "this one, not here.",
        ],
    }
    (out / "protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    review = out.parent / f"{out.name}_review"
    review.mkdir(parents=True, exist_ok=True)
    save_terms(terms, review / "protocol_terms.csv")
    (review / "files.json").write_text(
        json.dumps([{"source": f["source"], "output": f["output"]} for f in files],
                   indent=2, ensure_ascii=False), encoding="utf-8")
    (review / "still_flagged.json").write_text(
        json.dumps(again, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"manifest": manifest, "out_dir": str(out), "review_dir": str(review),
            "still_flagged": again, "files": files,
            # On the page only: the manifest keeps the count, not the names.
            "residual_terms": left_terms}


# ----------------------------------------------------------------------
# the term sheet, for the command line
# ----------------------------------------------------------------------
_SHEET = ["term", "category", "count", "confidence", "source", "example",
          "proposed", "decision", "replacement", "alias_of"]


def save_terms(terms: list[Term], path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=_SHEET)
        w.writeheader()
        for t in terms:
            w.writerow({k: getattr(t, k) for k in _SHEET})


def load_terms(path: str | Path) -> list[Term]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        t = Term(**{k: (int(r[k] or 0) if k == "count" else (r.get(k) or ""))
                    for k in _SHEET})
        t.decision = t.decision.strip().upper()
        if t.category not in CATEGORIES:
            raise ValueError(f"{t.term!r}: unknown category {t.category!r} "
                             f"(one of {', '.join(CATEGORIES)})")
        out.append(t)
    return out
