"""
Hierarchical legal-aware chunking.

  Level 1: document boundary (handled by the caller)
  Level 2: legal section boundary (Article / Section / § / Chapter / N.M)
  Level 3: paragraph boundary
  Level 4: fixed-size fallback (~512 tokens) with overlap

We split on section boundaries first so "Article 5(1)" never gets cleaved
mid-sentence. Oversized sections fall through to LangChain's recursive
splitter with paragraph/sentence separators.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.pdf_parser import ParsedPage, page_detected_section


# Headings live on their own line — require the section-id token to be
# at line-start with little or no trailing prose.
SECTION_PATTERNS = [
    re.compile(
        r"(?:^|\n)[ \t]*(Article\s+\d+(?:\.\d+)*[A-Za-z]?(?:\([^)]+\))?)\s*[\r\n]"
    ),
    re.compile(r"(?:^|\n)[ \t]*(Section\s+\d+(?:\.\d+)*[A-Za-z]?)\s*[\r\n]"),
    re.compile(r"(?:^|\n)[ \t]*(§\s*\d+(?:\.\d+)*[A-Za-z]?)\s*[\r\n]"),
    re.compile(r"(?:^|\n)[ \t]*(Chapter\s+[IVXLCDM\d]+)\s*[\r\n]"),
    re.compile(r"(?:^|\n)[ \t]*(\d+\.\d+(?:\.\d+)*\s+[A-Z][A-Za-z][^\n]{0,80})\s*[\r\n]"),
]


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: Dict


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _split_by_sections(text: str) -> List[Dict]:
    """
    Split a page's text on legal-section boundaries.
    Returns list of {text, heading|None}, in order, with any leading
    preamble before the first detected section preserved.
    """
    boundaries = []
    for rx in SECTION_PATTERNS:
        for m in rx.finditer(text):
            heading = m.group(1).strip()
            # Position is start of captured group, not the leading whitespace.
            idx = m.start() + m.group(0).find(heading)
            boundaries.append((idx, heading))

    if not boundaries:
        return [{"text": text.strip(), "heading": None}]

    boundaries.sort(key=lambda x: x[0])
    # Dedup boundaries at identical offsets.
    dedup = []
    for b in boundaries:
        if not dedup or dedup[-1][0] != b[0]:
            dedup.append(b)

    slices: List[Dict] = []
    if dedup[0][0] > 0:
        pre = text[: dedup[0][0]].strip()
        if pre:
            slices.append({"text": pre, "heading": None})

    for i, (start, heading) in enumerate(dedup):
        end = dedup[i + 1][0] if i + 1 < len(dedup) else len(text)
        slc = text[start:end].strip()
        if slc:
            slices.append({"text": slc, "heading": heading})

    return slices


def _extract_sub_section(heading: str | None) -> str:
    if not heading:
        return ""
    m = re.search(r"(\d+(?:\.\d+)*(?:\([^)]+\))*)", heading)
    return m.group(1) if m else ""


def _slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\.[a-z0-9]+$", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")[:40]


def infer_doc_type(filename: str) -> str:
    f = filename.lower()
    if re.search(r"(contract|nda|agreement|sla)", f):
        return "contract"
    if re.search(r"(case|judgment|opinion|ruling)", f):
        return "case_law"
    if re.search(r"(gdpr|regulation|directive|act|statute|law)", f):
        return "regulation"
    if re.search(r"(policy|guideline|guidance)", f):
        return "policy"
    return "document"


def chunk_pages(
    pages: List[ParsedPage],
    *,
    doc_id: str,
    doc_name: str,
    doc_type: str,
    jurisdiction: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> List[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    # Per-doc nonce so re-uploading the same filename doesn't collide on
    # the chunks PRIMARY KEY.
    doc_prefix = doc_id.replace("-", "")[:8]
    from datetime import datetime

    date_ingested = datetime.utcnow().strftime("%Y-%m-%d")
    chunks: List[Chunk] = []
    running_heading: str | None = None
    counter = 0

    for page in pages:
        if not page.raw_text or not page.raw_text.strip():
            continue

        page_heading = page.detected_section or page_detected_section(page.raw_text)
        if page_heading:
            running_heading = page_heading

        slices = _split_by_sections(page.raw_text)
        for slc in slices:
            section_title = slc["heading"] or running_heading or "Unknown"
            if slc["heading"]:
                running_heading = slc["heading"]
            sub_section = _extract_sub_section(slc["heading"])
            base_id = f"{doc_prefix}_{_slugify(doc_name)}_p{page.page_number}_c{counter}"

            text = slc["text"]
            if len(text) <= chunk_size:
                meta = {
                    "source": doc_name,
                    "doc_type": doc_type,
                    "jurisdiction": jurisdiction,
                    "date_ingested": date_ingested,
                    "page": page.page_number,
                    "section": section_title,
                    "sub_section": sub_section,
                    "chunk_id": base_id,
                    "chunk_type": "section",
                    "char_count": len(text),
                    "token_count": estimate_tokens(text),
                }
                chunks.append(Chunk(chunk_id=base_id, text=text, metadata=meta))
                counter += 1
                continue

            sub_texts = splitter.split_text(text)
            for j, sub in enumerate(sub_texts):
                sub_id = f"{base_id}_s{j}"
                meta = {
                    "source": doc_name,
                    "doc_type": doc_type,
                    "jurisdiction": jurisdiction,
                    "date_ingested": date_ingested,
                    "page": page.page_number,
                    "section": section_title,
                    "sub_section": sub_section,
                    "chunk_id": sub_id,
                    "chunk_type": "sub_chunk",
                    "char_count": len(sub),
                    "token_count": estimate_tokens(sub),
                    "sub_index": j,
                }
                chunks.append(Chunk(chunk_id=sub_id, text=sub, metadata=meta))
                counter += 1

    return chunks
