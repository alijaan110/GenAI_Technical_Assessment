"""
RAGAS-style RAG evaluation — five canonical metrics + derived hallucination rate.
LLM-as-judge implementation (the same methodology RAGAS uses internally),
written inline so we don't pull a heavy extra dependency for the assessment.

Runs are launched as background tasks via FastAPI's BackgroundTasks.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage

from backend.db import get_db
from backend.llm import get_llm
from backend.rag_service import handle_rag_query


# ── Score parsing ────────────────────────────────────────────────
def _parse_score(raw: str) -> float:
    if not raw:
        return 0.0
    m = re.search(r"(?:^|[^\d.])(0?\.\d+|1(?:\.0+)?|0|1)", raw)
    if not m:
        return 0.0
    try:
        return max(0.0, min(1.0, float(m.group(1))))
    except ValueError:
        return 0.0


def _parse_json_score(raw: str, key: str) -> float:
    try:
        cleaned = raw.replace("```json", "").replace("```", "")
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end < 0:
            raise ValueError("no json")
        obj = json.loads(cleaned[start : end + 1])
        v = obj.get(key)
        if isinstance(v, (int, float)):
            return max(0.0, min(1.0, float(v)))
    except Exception:
        pass
    return _parse_score(raw)


# ── Prompts (one per metric) ─────────────────────────────────────
FAITHFULNESS = """You are a strict RAG evaluator. Decide whether the ANSWER is fully supported by the CONTEXT.
Score from 0.0 (entirely fabricated / contradicted) to 1.0 (every claim grounded in context).

Step 1: list the atomic factual claims in the answer.
Step 2: for each claim, check whether the context supports it.
Step 3: report final score.

CONTEXT:
{context}

ANSWER:
{answer}

Return ONLY JSON: {{"score": <0..1>, "unsupported_claims": <int>}}"""

RELEVANCY = """Determine whether the ANSWER directly addresses the QUESTION.
Score from 0.0 (off-topic) to 1.0 (directly and fully answers).

QUESTION: {question}
ANSWER: {answer}

Return ONLY JSON: {{"score": <0..1>}}"""

PRECISION = """Decide what fraction of the retrieved CONTEXT chunks were actually relevant to the QUESTION.
Score from 0.0 (none relevant) to 1.0 (every chunk relevant).

QUESTION: {question}
CONTEXT (chunks separated by ---):
{context}

Return ONLY JSON: {{"score": <0..1>, "relevant_chunks": <int>, "total_chunks": <int>}}"""

RECALL = """Decide what fraction of the GROUND TRUTH answer is covered by the retrieved CONTEXT.
Score from 0.0 (context misses everything) to 1.0 (context fully covers the ground truth).

GROUND TRUTH: {ground_truth}
CONTEXT:
{context}

Return ONLY JSON: {{"score": <0..1>}}"""

CORRECTNESS = """Compare the GENERATED ANSWER to the GROUND TRUTH for semantic correctness.
Score from 0.0 (contradicts ground truth) to 1.0 (semantically equivalent).
Minor wording differences are fine; the meaning must match.

QUESTION: {question}
GROUND TRUTH: {ground_truth}
GENERATED ANSWER: {answer}

Return ONLY JSON: {{"score": <0..1>, "explanation": "<short reason>"}}"""


@dataclass
class _Row:
    question: str
    expected: str
    answer: str
    contexts: List[str]
    faithfulness: float
    relevancy: float
    precision: float
    recall: float
    correctness: float


# ── Public entrypoint ────────────────────────────────────────────
async def run_evaluation(test_set_id: str) -> str:
    """
    Kick off a full RAGAS-style eval. Returns the new evaluation_id.
    The caller schedules the actual work as a BackgroundTask.
    """
    db = get_db()
    eval_id = str(uuid.uuid4())
    with db:
        db.execute(
            """INSERT INTO evaluations (id, test_set_id, status)
               VALUES (?, ?, 'running')""",
            (eval_id, test_set_id),
        )
    return eval_id


async def execute_evaluation(eval_id: str, test_set_id: str) -> None:
    db = get_db()
    questions = db.execute(
        "SELECT * FROM test_questions WHERE test_set_id = ?", (test_set_id,)
    ).fetchall()
    if not questions:
        with db:
            db.execute(
                """UPDATE evaluations SET status = 'failed',
                   analysis_text = 'No test questions found in this set' WHERE id = ?""",
                (eval_id,),
            )
        return

    llm = get_llm()
    rows: List[_Row] = []
    sum_f = sum_r = sum_p = sum_rec = sum_c = 0.0
    hallucinations = passed = 0

    for q in questions:
        try:
            rag = await handle_rag_query(q["question"], top_k=5, use_hybrid=True)
            answer = rag.answer
            contexts = [s.excerpt for s in rag.sources]
            ctx_joined = "\n---\n".join(contexts) or "(no context retrieved)"
            gt = q["expected_answer"] or ""

            tasks = [
                _judge(llm, FAITHFULNESS.format(context=ctx_joined, answer=answer)),
                _judge(llm, RELEVANCY.format(question=q["question"], answer=answer)),
                _judge(llm, PRECISION.format(question=q["question"], context=ctx_joined)),
                _judge(llm, RECALL.format(ground_truth=gt, context=ctx_joined)) if gt else _const("0"),
                _judge(
                    llm,
                    CORRECTNESS.format(
                        question=q["question"], ground_truth=gt, answer=answer
                    ),
                )
                if gt
                else _const("0"),
            ]
            f_raw, r_raw, p_raw, rec_raw, c_raw = await asyncio.gather(*tasks)

            f = _parse_json_score(f_raw, "score")
            r = _parse_json_score(r_raw, "score")
            p = _parse_json_score(p_raw, "score")
            rec = _parse_json_score(rec_raw, "score") if gt else 0.0
            c = _parse_json_score(c_raw, "score") if gt else 0.0
            hallucinated = f < 0.7

            sum_f += f
            sum_r += r
            sum_p += p
            sum_rec += rec
            sum_c += c
            if hallucinated:
                hallucinations += 1
            if f >= 0.7 and r >= 0.7 and (not gt or c >= 0.7):
                passed += 1

            issues: List[str] = []
            if f < 0.7:
                issues.append("Low faithfulness — answer may contain hallucinated content")
            if r < 0.7:
                issues.append("Low relevancy — answer doesn't directly address the question")
            if p < 0.6:
                issues.append("Low context precision — retriever returned irrelevant chunks")
            if gt and rec < 0.6:
                issues.append("Low context recall — retriever missed relevant passages")
            if gt and c < 0.7:
                issues.append("Low correctness — generated answer differs from ground truth")

            with db:
                db.execute(
                    """INSERT INTO evaluation_results
                    (evaluation_id, question_id, generated_answer, retrieved_contexts,
                     faithfulness_score, relevancy_score, precision_score, recall_score,
                     correctness_score, has_hallucination, issues)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        eval_id,
                        q["id"],
                        answer,
                        json.dumps(contexts),
                        f,
                        r,
                        p,
                        rec,
                        c,
                        1 if hallucinated else 0,
                        json.dumps(issues),
                    ),
                )

            rows.append(
                _Row(
                    question=q["question"],
                    expected=gt,
                    answer=answer,
                    contexts=contexts,
                    faithfulness=f,
                    relevancy=r,
                    precision=p,
                    recall=rec,
                    correctness=c,
                )
            )
        except Exception as e:
            with db:
                db.execute(
                    """INSERT INTO evaluation_results
                    (evaluation_id, question_id, generated_answer, retrieved_contexts,
                     faithfulness_score, relevancy_score, precision_score, recall_score,
                     correctness_score, has_hallucination, issues)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        eval_id,
                        q["id"],
                        f"Error: {e}",
                        json.dumps([]),
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        json.dumps([f"Evaluation error: {e}"]),
                    ),
                )

    n = len(questions)
    avg = lambda s: s / max(1, n)
    avg_f = avg(sum_f)
    report = _render_report(
        avg_f=avg_f,
        avg_r=avg(sum_r),
        avg_p=avg(sum_p),
        avg_rec=avg(sum_rec),
        avg_c=avg(sum_c),
        hallucination_rate=hallucinations / n,
        rows=rows,
    )

    with db:
        db.execute(
            """UPDATE evaluations
               SET status='completed',
                   overall_faithfulness=?,
                   overall_relevancy=?,
                   overall_precision=?,
                   overall_recall=?,
                   overall_correctness=?,
                   hallucination_rate=?,
                   total_questions=?,
                   passed_questions=?,
                   failed_questions=?,
                   report_markdown=?,
                   completed_at=CURRENT_TIMESTAMP
             WHERE id=?""",
            (
                avg_f,
                avg(sum_r),
                avg(sum_p),
                avg(sum_rec),
                avg(sum_c),
                hallucinations / n,
                n,
                passed,
                n - passed,
                report,
                eval_id,
            ),
        )


async def _judge(llm, prompt: str) -> str:
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content if isinstance(response.content, str) else str(response.content)


async def _const(v: str) -> str:
    return v


def _render_report(
    *,
    avg_f: float,
    avg_r: float,
    avg_p: float,
    avg_rec: float,
    avg_c: float,
    hallucination_rate: float,
    rows: List[_Row],
) -> str:
    fmt = lambda n: f"{n:.3f}"
    pct = lambda n: f"{n*100:.1f}%"

    head = (
        "| Question | Faithfulness | Ans.Relevancy | Context Precision "
        "| Context Recall | Correctness |"
    )
    sep = "|----------|-------------:|--------------:|------------------:|---------------:|------------:|"
    body_lines = []
    for r in rows:
        q = r.question
        if len(q) > 50:
            q = q[:47] + "…"
        q = q.replace("|", "\\|")
        body_lines.append(
            f"| {q} | {fmt(r.faithfulness)} | {fmt(r.relevancy)} | "
            f"{fmt(r.precision)} | {fmt(r.recall)} | {fmt(r.correctness)} |"
        )
    body = "\n".join(body_lines) if body_lines else "| _no rows_ |"
    avg_line = (
        f"| **AVERAGE** | **{fmt(avg_f)}** | **{fmt(avg_r)}** | **{fmt(avg_p)}** "
        f"| **{fmt(avg_rec)}** | **{fmt(avg_c)}** |"
    )

    interpretation = (
        "✅ All metrics meet production targets. Ship it."
        if avg_f >= 0.85
        else "⚠️ High hallucination — strengthen system prompt and lower temperature."
        if avg_f < 0.7
        else "🟡 Acceptable but room for improvement — tune retrieval thresholds and chunk metadata."
    )

    return "\n".join(
        [
            "# RAG Evaluation Report",
            "",
            "## Overall Scores",
            f"- **Faithfulness:** {fmt(avg_f)}",
            f"- **Answer Relevancy:** {fmt(avg_r)}",
            f"- **Context Precision:** {fmt(avg_p)}",
            f"- **Context Recall:** {fmt(avg_rec)}",
            f"- **Answer Correctness:** {fmt(avg_c)}",
            f"- **Hallucination Rate:** {pct(hallucination_rate)}",
            "",
            "## Per-Question Breakdown",
            head,
            sep,
            body,
            avg_line,
            "",
            "## Interpretation",
            interpretation,
        ]
    )
