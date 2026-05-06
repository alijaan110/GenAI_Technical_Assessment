"""
Document ingestion pipeline.

  parse PDF (pdfplumber) → hierarchical legal chunking → OpenAI embeddings →
  persist to SQLite → mirror into both:
    • dense backend (Qdrant if reachable, else in-memory fallback)
    • BM25 sparse store (rebuilt from SQLite)

On a cold start, both indexes hydrate from SQLite so re-ingest isn't required.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

from backend.bm25_store import get_bm25, reset_bm25
from backend.chunker import chunk_pages, infer_doc_type
from backend.db import get_db
from backend.embedder import embed_documents
from backend.pdf_parser import parse_pdf
from backend.settings import get_settings
from backend.vector_store import active_backend_name, get_backend, reset_backend


@dataclass
class IngestResult:
    document_id: str
    total_pages: int
    total_chunks: int
    doc_type: str
    backend: str


def process_document(
    file_path: str,
    original_filename: str,
    file_size: int,
    *,
    doc_type_override: Optional[str] = None,
    jurisdiction_override: Optional[str] = None,
) -> IngestResult:
    db = get_db()
    settings = get_settings()
    doc_id = str(uuid.uuid4())
    doc_type = doc_type_override or infer_doc_type(original_filename)
    jurisdiction = jurisdiction_override or "Unknown"

    db.execute(
        """INSERT INTO documents
        (id, filename, file_path, status, file_size_bytes, doc_type, jurisdiction)
        VALUES (?, ?, ?, 'processing', ?, ?, ?)""",
        (doc_id, original_filename, file_path, file_size, doc_type, jurisdiction),
    )
    db.commit()

    try:
        # 1. Parse pages with structure
        pages = parse_pdf(file_path)
        if not pages or all(not p.raw_text.strip() for p in pages):
            raise ValueError(
                "This PDF appears to be image-only or scanned (no extractable text). "
                "We don't currently OCR images — please upload a text-based PDF, or run an OCR tool first."
            )

        # 2. Hierarchical legal chunking
        try:
            chunk_size = int(settings.chunk_size or "900")
            chunk_overlap = int(settings.chunk_overlap or "180")
        except ValueError:
            chunk_size, chunk_overlap = 900, 180

        chunks = chunk_pages(
            pages,
            doc_id=doc_id,
            doc_name=original_filename,
            doc_type=doc_type,
            jurisdiction=jurisdiction,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not chunks:
            raise ValueError("No chunks were produced from the document.")

        # 3. Embed
        vectors = embed_documents([c.text for c in chunks])

        # 4. Persist to SQLite (canonical) + dense + BM25
        backend = get_backend()
        items: List[Dict] = []
        with db:
            for idx, c in enumerate(chunks):
                db.execute(
                    """INSERT INTO chunks
                    (id, document_id, chunk_index, page_number, section_title, sub_section,
                     chunk_text, chunk_type, char_count, token_count, embedding_vector, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        c.chunk_id,
                        doc_id,
                        idx,
                        c.metadata["page"],
                        c.metadata["section"],
                        c.metadata["sub_section"],
                        c.text,
                        c.metadata["chunk_type"],
                        c.metadata["char_count"],
                        c.metadata["token_count"],
                        json.dumps(vectors[idx]),
                        json.dumps(c.metadata),
                    ),
                )
                meta = {**c.metadata, "document_id": doc_id, "chunk_id": c.chunk_id}
                items.append(
                    {
                        "chunk_id": c.chunk_id,
                        "text": c.text,
                        "metadata": meta,
                        "vector": vectors[idx],
                    }
                )
        backend.upsert(items)
        get_bm25().rebuild()

        db.execute(
            """UPDATE documents
            SET status = 'completed', total_pages = ?, total_chunks = ?,
                updated_at = CURRENT_TIMESTAMP WHERE id = ?""",
            (len(pages), len(chunks), doc_id),
        )
        db.commit()

        return IngestResult(
            document_id=doc_id,
            total_pages=len(pages),
            total_chunks=len(chunks),
            doc_type=doc_type,
            backend=active_backend_name(),
        )
    except Exception as e:
        db.execute("UPDATE documents SET status = 'failed' WHERE id = ?", (doc_id,))
        db.commit()
        # Clean up the upload artifact on failure.
        try:
            os.unlink(file_path)
        except OSError:
            pass
        raise


def purge_document(document_id: str) -> None:
    """Surgical removal from dense backend + BM25 + disk + SQLite."""
    db = get_db()
    row = db.execute(
        "SELECT file_path FROM documents WHERE id = ?", (document_id,)
    ).fetchone()
    if row and row["file_path"] and os.path.exists(row["file_path"]):
        try:
            os.unlink(row["file_path"])
        except OSError:
            pass
    with db:
        db.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        db.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    try:
        get_backend().delete_by_document(document_id)
    except Exception:
        # Backend might not be hydrated yet — re-probe will reflect the new state.
        reset_backend()
    reset_bm25()
