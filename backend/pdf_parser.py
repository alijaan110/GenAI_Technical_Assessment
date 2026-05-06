"""
Production PDF parser using pdfplumber, exactly as recommended by the
assessment doc. Extracts per-page text plus structural cues (font sizes
& heading detection) we use to attribute citations to a section name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import pdfplumber

HEADING_PATTERNS = [
    re.compile(r"^(article\s+\d+(?:\.\d+)*[a-z]?)\b.*$", re.IGNORECASE),
    re.compile(r"^(section\s+\d+(?:\.\d+)*[a-z]?)\b.*$", re.IGNORECASE),
    re.compile(r"^(§\s*\d+(?:\.\d+)*[a-z]?)\b.*$"),
    re.compile(r"^(chapter\s+[ivxlcdm\d]+)\b.*$", re.IGNORECASE),
    re.compile(r"^(part\s+[ivxlcdm\d]+)\b.*$", re.IGNORECASE),
    re.compile(r"^(\d+\.\d+(?:\.\d+)*\s+[A-Z].{0,80})$"),
    re.compile(r"^(\d+\.\s+[A-Z].{0,80})$"),
    re.compile(r"^([A-Z][A-Z0-9 \-,&]{6,80})$"),
]


def detect_heading(line: str) -> Optional[str]:
    s = line.strip()
    if not s or len(s) > 120:
        return None
    for rx in HEADING_PATTERNS:
        m = rx.match(s)
        if m:
            return m.group(1).strip() if m.lastindex else s
    return None


def page_detected_section(text: str) -> Optional[str]:
    for line in text.splitlines():
        h = detect_heading(line)
        if h:
            return h
    return None


@dataclass
class ParsedPage:
    page_number: int
    raw_text: str
    detected_section: Optional[str] = None
    font_sizes: List[float] = field(default_factory=list)
    tables: List[List[List[str]]] = field(default_factory=list)


def parse_pdf(file_path: str) -> List[ParsedPage]:
    """
    Walks the PDF page-by-page, preserving page numbers, harvesting font
    sizes (so callers can do larger-font-as-heading inference), and pulling
    tables as structured data.
    """
    pages: List[ParsedPage] = []
    current_section: Optional[str] = None

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            raw_text = page.extract_text() or ""

            # Heuristic 1: regex-based heading detection on the raw text
            sec = page_detected_section(raw_text) or current_section

            # Heuristic 2: font-size-based heading detection
            font_sizes: List[float] = []
            try:
                words = page.extract_words(extra_attrs=["size", "fontname"])
                font_sizes = [w["size"] for w in words if w.get("size")]
                if font_sizes:
                    avg = sum(font_sizes) / len(font_sizes)
                    big = [w["text"] for w in words if w.get("size", 0) > avg * 1.25]
                    if big:
                        candidate = " ".join(big[:8]).strip()
                        if 4 <= len(candidate) <= 120:
                            sec = candidate
            except Exception:
                # pdfplumber occasionally raises on malformed pages — keep going.
                pass

            if sec:
                current_section = sec

            tables: List[List[List[str]]] = []
            try:
                tables = page.extract_tables() or []
            except Exception:
                pass

            pages.append(
                ParsedPage(
                    page_number=i + 1,
                    raw_text=raw_text,
                    detected_section=current_section,
                    font_sizes=font_sizes,
                    tables=tables,
                )
            )

    return pages
