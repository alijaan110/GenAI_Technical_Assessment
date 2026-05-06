"""Evaluation endpoints (test sets, auto-gen, run, results, report)."""

from __future__ import annotations

import json
import uuid
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from backend.dataset_generator import auto_generate_test_set
from backend.db import get_db
from backend.eval_service import execute_evaluation, run_evaluation

router = APIRouter(tags=["evaluation"])


class TestQuestion(BaseModel):
    question: str
    expected_answer: str
    question_type: Optional[str] = "factual"
    source_doc: Optional[str] = None
    source_page: Optional[int] = None


class TestQuestionsCreate(BaseModel):
    test_set_id: Optional[str] = None
    questions: List[TestQuestion]


class AutoGenRequest(BaseModel):
    n_per_chunk: Optional[int] = 1
    max_chunks: Optional[int] = 14
    document_id: Optional[str] = None


class RunEvalRequest(BaseModel):
    test_set_id: str


@router.get("/api/evaluation/test-questions")
def list_test_sets():
    db = get_db()
    rows = db.execute(
        """SELECT test_set_id, COUNT(*) AS count, MIN(created_at) AS created_at
           FROM test_questions GROUP BY test_set_id
           ORDER BY MIN(created_at) DESC"""
    ).fetchall()
    return {"test_sets": [dict(r) for r in rows]}


@router.get("/api/evaluation/test-questions/{set_id}")
def list_questions(set_id: str):
    db = get_db()
    rows = db.execute(
        "SELECT * FROM test_questions WHERE test_set_id = ? ORDER BY created_at",
        (set_id,),
    ).fetchall()
    return {"questions": [dict(r) for r in rows]}


@router.post("/api/evaluation/test-questions")
def add_test_questions(payload: TestQuestionsCreate):
    if not payload.questions:
        raise HTTPException(400, "questions must be a non-empty array")
    db = get_db()
    set_id = payload.test_set_id or str(uuid.uuid4())
    with db:
        for q in payload.questions:
            db.execute(
                """INSERT INTO test_questions
                   (id, test_set_id, question, expected_answer, question_type, source_doc, source_page)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    set_id,
                    q.question,
                    q.expected_answer,
                    q.question_type or "factual",
                    q.source_doc,
                    q.source_page,
                ),
            )
    return {"success": True, "test_set_id": set_id, "count": len(payload.questions)}


@router.post("/api/evaluation/test-questions/auto-generate")
async def auto_generate(payload: AutoGenRequest):
    try:
        result = await auto_generate_test_set(
            n_per_chunk=payload.n_per_chunk or 2,
            max_chunks=payload.max_chunks or 10,
            document_id=payload.document_id,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"success": True, **result}


@router.post("/api/evaluation/run")
async def post_run(payload: RunEvalRequest, background_tasks: BackgroundTasks):
    eval_id = await run_evaluation(payload.test_set_id)
    background_tasks.add_task(execute_evaluation, eval_id, payload.test_set_id)
    return {"success": True, "evaluation_id": eval_id}


@router.get("/api/evaluation")
def list_evaluations():
    db = get_db()
    rows = db.execute("SELECT * FROM evaluations ORDER BY started_at DESC").fetchall()
    return {"evaluations": [dict(r) for r in rows]}


@router.get("/api/evaluation/results/{eval_id}")
def get_results(eval_id: str):
    db = get_db()
    eval_row = db.execute("SELECT * FROM evaluations WHERE id = ?", (eval_id,)).fetchone()
    if not eval_row:
        raise HTTPException(404, "Evaluation not found")
    rows = db.execute(
        """SELECT er.*, tq.question, tq.expected_answer
           FROM evaluation_results er
           JOIN test_questions tq ON er.question_id = tq.id
           WHERE er.evaluation_id = ?""",
        (eval_id,),
    ).fetchall()
    detailed = []
    for r in rows:
        try:
            contexts = json.loads(r["retrieved_contexts"]) if r["retrieved_contexts"] else []
        except Exception:
            contexts = []
        try:
            issues = json.loads(r["issues"]) if r["issues"] else []
        except Exception:
            issues = []
        detailed.append(
            {
                "question": r["question"],
                "expected_answer": r["expected_answer"],
                "answer": r["generated_answer"],
                "contexts": contexts,
                "scores": {
                    "faithfulness": r["faithfulness_score"],
                    "answer_relevancy": r["relevancy_score"],
                    "context_precision": r["precision_score"],
                    "context_recall": r["recall_score"],
                    "answer_correctness": r["correctness_score"],
                },
                "has_hallucination": bool(r["has_hallucination"]),
                "issues": issues,
            }
        )
    out = dict(eval_row)
    out["detailed_results"] = detailed
    return out


@router.get("/api/evaluation/results/{eval_id}/report", response_class=PlainTextResponse)
def get_report(eval_id: str):
    db = get_db()
    row = db.execute(
        "SELECT report_markdown FROM evaluations WHERE id = ?", (eval_id,)
    ).fetchone()
    if not row or not row["report_markdown"]:
        raise HTTPException(404, "No report")
    return PlainTextResponse(row["report_markdown"], media_type="text/markdown")


@router.delete("/api/evaluation/test-questions/{set_id}")
def delete_test_set(set_id: str):
    db = get_db()
    with db:
        db.execute("DELETE FROM test_questions WHERE test_set_id = ?", (set_id,))
    return {"success": True, "deleted_set": set_id}


@router.delete("/api/evaluation/{eval_id}")
def delete_evaluation(eval_id: str):
    db = get_db()
    with db:
        db.execute("DELETE FROM evaluation_results WHERE evaluation_id = ?", (eval_id,))
        db.execute("DELETE FROM evaluations WHERE id = ?", (eval_id,))
    return {"success": True, "deleted_evaluation": eval_id}

