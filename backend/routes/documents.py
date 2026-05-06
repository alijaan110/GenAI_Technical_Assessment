"""Document upload, list, delete."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.db import get_db
from backend.document_service import process_document, purge_document

router = APIRouter(tags=["documents"])

UPLOAD_DIR = Path(os.getcwd()) / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/api/rag/upload")
async def upload_document(
    file: UploadFile = File(...),
    doc_type: Optional[str] = Form(None),
    jurisdiction: Optional[str] = Form(None),
):
    if not file.filename:
        raise HTTPException(400, "Missing filename")

    # Persist the upload to disk first; processing happens synchronously
    # against the file path (pdfplumber wants a path or file-like object).
    dest = UPLOAD_DIR / uuid.uuid4().hex
    contents = await file.read()
    if not contents:
        raise HTTPException(400, "Empty file")
    dest.write_bytes(contents)

    try:
        result = process_document(
            str(dest),
            file.filename,
            len(contents),
            doc_type_override=doc_type,
            jurisdiction_override=jurisdiction,
        )
    except ValueError as e:
        # Friendly client-facing 400 for known input issues (e.g. image-only PDF).
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

    return {
        "success": True,
        "document_id": result.document_id,
        "total_pages": result.total_pages,
        "total_chunks": result.total_chunks,
        "doc_type": result.doc_type,
        "backend": result.backend,
    }


@router.get("/api/rag/documents")
def list_documents():
    db = get_db()
    rows = db.execute(
        """SELECT id, filename, file_path, upload_date, status, total_pages,
                  total_chunks, file_size_bytes, doc_type, jurisdiction
           FROM documents ORDER BY created_at DESC"""
    ).fetchall()
    return {"documents": [dict(r) for r in rows]}


@router.delete("/api/rag/documents/{document_id}")
def delete_document(document_id: str):
    purge_document(document_id)
    return {"success": True}
