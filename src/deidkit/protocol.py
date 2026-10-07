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
and comments, plus the document properties. A PDF can only be read as text,
so its masked copy is a text file. Images are not read at all -- a sponsor
logo survives -- and the report says how many there were.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
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


def _docx_paragraphs(root) -> Iterable[list[Any]]:
    """Each paragraph's text nodes, in order -- and only its own: a text box
    inside a paragraph is a paragraph of its own and is yielded separately."""
    for p in root.iter(_W + "p"):
        nodes = [
            t for t in p.iter(_W + "t")
            if next(t.iterancestors(_W + "p"), None) is p
        ]
        if nodes:
            yield nodes


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
            if "docProps/core.xml" in z.namelist():
                core = etree.fromstring(z.read("docProps/core.xml"))
                blocks += [e.text for e in core.iter() if e.text and e.text.strip()]
        return blocks
    if suffix == ".pdf":
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
_CAP = r"[A-Z][\w&'.-]*"
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
    r"floxacin|oxacin|cycline|vastatin|statin|setron|tegravir|kinra|leukin"
)
_ZH_DRUG = r"单抗|替尼|西尼|帕利|司他|他汀|沙坦|普利|洛尔|霉素|西林|铂|紫杉醇|比星|那肽|鲁肽|格列净|格列汀|韦"
_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    # registry and protocol numbers
    ("study_id", "high", re.compile(r"\bNCT\d{8}\b")),
    ("study_id", "high", re.compile(r"\b(?:19|20)\d{2}-5\d{5}-\d{2}(?:-\d{2})?\b")),  # CTIS
    ("study_id", "high", re.compile(r"\b(?:19|20)\d{2}-\d{6}-\d{2}\b")),  # EudraCT
    ("study_id", "high", re.compile(
        r"(?i:\bIND\s*(?:No\.?|Number|#)?\s*[:：]?\s*)(\d{5,6})\b")),
    ("study_id", "high", re.compile(
        r"(?i:protocol\s*(?:no\.?|number|id|code|编号)?\s*[:：#]?\s*)"
        r"([A-Z0-9][A-Z0-9_./-]*\d[A-Z0-9_./-]*)")),
    ("study_id", "high", re.compile(r"(?:方案编号|试验编号|研究编号)\s*[:：]?\s*([A-Za-z0-9_./-]{4,})")),
    # BDM-AI-2025-001, TIG-2026-001: letters then two or more hyphen groups
    ("study_id", "medium", re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+){2,5}\b")),
    # compound codes: MK-3475, BMS-936558, ABBV951
    ("drug", "medium", re.compile(r"\b[A-Z]{1,5}-?\d{3,6}[A-Z]?\b")),
    # trade marks
    ("drug", "high", re.compile(r"\b([A-Z][A-Za-z0-9-]{2,})\s?[®™]")),
    # INN stems: zentolimab, osimertinib -- and their Chinese counterparts
    ("drug", "medium", re.compile(rf"\b([A-Za-z]{{3,}}(?:{_INN}))\b", re.I)),
    ("drug", "medium", re.compile(rf"{_ZH_START}({_ZH}{{1,8}}?(?:{_ZH_DRUG}))")),
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
    # people
    ("person", "high", re.compile(
        r"\b(?:Dr|Prof|Professor|Mr|Ms|Mrs)\.?\s+([A-Z][a-z'-]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z'-]+){0,2})")),
    ("person", "high", re.compile(
        r"\b([A-Z][a-z'-]+(?:\s+[A-Z]\.)?\s+[A-Z][a-z'-]+),\s*"
        r"(?:MD|M\.D\.|PhD|Ph\.D\.|MBBS|PharmD|DO|RN|MSc|MPH|FRCP)\b")),
    ("person", "high", re.compile(
        r"(?:研究者|主要研究者|联系人|医学监查员|姓名|负责人)\s*[:：]\s*([\u4e00-\u9fa5·]{2,5})")),
    # contact details
    ("contact", "high", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("contact", "high", re.compile(r"\bhttps?://[^\s<>\"')]+|\bwww\.[^\s<>\"')]+")),
    ("contact", "high", re.compile(
        r"(?i:(?:tel|phone|telephone|fax|mobile|cell|电话|传真|手机)\.?\s*[:：]?\s*)"
        r"(\+?\d[\d\s().-]{6,}\d)")),
]

#: Words a pattern can produce that name nothing.
_NOT_A_NAME = {
    "the", "this", "study", "sponsor", "protocol", "clinical", "university",
    "hospital", "institute", "college", "clinic", "the university",
}


def _boundary(term: str) -> str:
    """A whole-term regex: word edges where the term starts or ends with a
    word character, none for CJK, where words are not separated."""
    esc = re.escape(term)
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
    texts = [t for p in paths for t in read_text(p)]
    found: dict[str, Term] = {}

    def add(term: str, category: str, confidence: str, source: str,
            text: str, start: int, end: int) -> None:
        term = term.strip(" \t,.;:：，。()（）")
        if re.search(r"[\u4e00-\u9fa5]", term) and category in ("site", "sponsor", "drug"):
            cut = _ZH_CUT.match(term)
            if cut and 0 < cut.end() and len(term) - cut.end() >= 3:
                start += cut.end()
                term = term[cut.end():]
        if len(term) < 2 or term.lower() in _NOT_A_NAME:
            return
        key = term.lower()
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
        for category, conf, pat in _PATTERNS:
            for m in pat.finditer(text):
                g = 1 if m.groups() and m.group(1) else 0
                add(m.group(g), category, conf, "pattern", text, m.start(g), m.end(g))
        if detector is not None and text.strip():
            for f in detector.detect(text):
                cat = {"PERSON": "person", "EMAIL_ADDRESS": "contact",
                       "PHONE_NUMBER": "contact", "URL": "contact"}.get(f.entity_type)
                if cat:
                    add(text[f.start:f.end], cat, "low", "ner", text, f.start, f.end)

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

    # Count what each term would actually replace, longest first: a mention
    # of "Pembrolizumab" inside "Pembrolizumab 200 mg Q3W" belongs to the
    # longer term, and a term nothing is left for is dropped.
    counts: dict[str, int] = {}
    m = _matcher({t.term: "" for t in found.values()})
    if m is not None:
        for x in m[0].finditer(joined):
            counts[x.group(0).lower()] = counts.get(x.group(0).lower(), 0) + 1
    kept = []
    for key, t in found.items():
        t.count = counts.get(key, 0)
        if t.count:
            if propose_replacements:
                t.proposed = propose(t, vault, labels)
            kept.append(t)
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
    return pat, {t.lower(): r for t, r in mapping.items()}


def replace_text(text: str, mapping: dict[str, str]) -> tuple[str, int]:
    m = _matcher(mapping)
    if m is None or not text:
        return text, 0
    pat, lookup = m
    n = 0

    def sub(x: re.Match[str]) -> str:
        nonlocal n
        n += 1
        return lookup[x.group(0).lower()]

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
        out[owner[m.start()]] += lookup[m.group(0).lower()]
        pos = m.end()
    for c in range(pos, len(full)):
        out[owner[c]] += full[c]
    for t, s in zip(nodes, out):
        if t.text != s:
            t.text = s
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return len(matches)


def _write_docx(src: Path, dst: Path, mapping: dict[str, str]) -> dict[str, int]:
    from lxml import etree

    m = _matcher(mapping)
    replaced = images = 0
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.startswith("word/media/"):
                images += 1
            if m is not None and _TEXT_PARTS.match(item.filename):
                root = etree.fromstring(data)
                for nodes in _docx_paragraphs(root):
                    replaced += _rewrite_paragraph(nodes, *m)
                data = etree.tostring(root, xml_declaration=True,
                                      encoding="UTF-8", standalone=True)
            elif item.filename == "docProps/core.xml":
                root = etree.fromstring(data)
                for e in root.iter():
                    tag = etree.QName(e).localname
                    if tag in ("creator", "lastModifiedBy"):
                        e.text = ""
                    elif e.text and m is not None:
                        e.text, k = replace_text(e.text, mapping)
                        replaced += k
                data = etree.tostring(root, xml_declaration=True,
                                      encoding="UTF-8", standalone=True)
            elif item.filename == "docProps/app.xml":
                root = etree.fromstring(data)
                for e in root.iter():
                    if etree.QName(e).localname in ("Company", "Manager"):
                        e.text = ""
                data = etree.tostring(root, xml_declaration=True,
                                      encoding="UTF-8", standalone=True)
            zout.writestr(item, data)
    return {"replaced": replaced, "images_not_read": images}


def output_name(src: Path) -> str:
    return src.stem + (".txt" if src.suffix.lower() == ".pdf" else src.suffix.lower())


def write(src: Path, out_dir: Path, mapping: dict[str, str]) -> dict[str, Any]:
    """Write the masked copy of one document and say what was done."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / output_name(src)
    suffix = src.suffix.lower()
    info: dict[str, Any] = {"source": src.name, "output": dst.name,
                            "sha256_in": hashlib.sha256(src.read_bytes()).hexdigest()}
    if suffix == ".docx":
        info |= _write_docx(src, dst, mapping)
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


def residual(out_paths: Iterable[Path], terms: Iterable[str]) -> int:
    """How many times an approved term still appears in the output. The
    answer has to be zero; anything else is a leak the report names."""
    terms = [t for t in terms if t]
    if not terms:
        return 0
    pat = re.compile("|".join(_boundary(t) for t in sorted(terms, key=len, reverse=True)), re.I)
    return sum(len(pat.findall(t)) for p in out_paths for t in read_text(p))


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
    left = residual(outs, mapping)
    # Anything the scan would still flag in the masked copy, ignoring what was
    # deliberately kept and what the replacements themselves look like.
    kept = {t.term.lower() for t in terms if t.decision == "KEEP"}
    replacements = {r.lower() for r in mapping.values()}
    again = [
        asdict(t) for t in scan(outs, vault, detector, propose_replacements=False)
        if t.term.lower() not in kept and t.term.lower() not in replacements
        and not any(t.term.lower() in r for r in replacements)
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
        "files": files,
        "terms_replaced_by_category": by_cat,
        "terms_kept": len(kept),
        "residual_occurrences": left,
        "still_flagged_after_masking": len(again),
        "notes": [
            "Images are not read: a logo or a scanned signature in the document "
            "survives masking. images_not_read counts them per file.",
            "A PDF is masked as text; its layout is not kept.",
            "The term list with the originals is in the _review folder beside "
            "this one, not here.",
        ],
    }
    (out / "protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    review = out.parent / f"{out.name}_review"
    review.mkdir(parents=True, exist_ok=True)
    save_terms(terms, review / "protocol_terms.csv")
    (review / "still_flagged.json").write_text(
        json.dumps(again, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"manifest": manifest, "out_dir": str(out), "review_dir": str(review),
            "still_flagged": again}


# ----------------------------------------------------------------------
# the term sheet, for the command line
# ----------------------------------------------------------------------
_SHEET = ["term", "category", "count", "confidence", "source", "example",
          "proposed", "decision", "replacement"]


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
