"""
SQLite layer (stdlib `sqlite3` for zero deps). One connection per process —
SQLite handles multi-thread reads fine; we set check_same_thread=False because
FastAPI dispatches across the asyncio worker thread pool.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional

_db: Optional[sqlite3.Connection] = None


def _safe_alter(conn: sqlite3.Connection, sql: str) -> None:
    """ALTER TABLE that swallows 'duplicate column' errors so re-runs are idempotent."""
    try:
        conn.execute(sql)
    except sqlite3.OperationalError:
        pass


def init_db() -> sqlite3.Connection:
    global _db
    if _db is not None:
        return _db

    data_dir = Path(os.getcwd()) / "data"
    data_dir.mkdir(exist_ok=True)
    db_path = data_dir / "genai_assessment.db"

    _db = sqlite3.connect(
        db_path,
        check_same_thread=False,
        detect_types=sqlite3.PARSE_DECLTYPES,
    )
    _db.row_factory = sqlite3.Row
    _db.execute("PRAGMA journal_mode = WAL")
    _db.execute("PRAGMA foreign_keys = ON")

    _db.executescript(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key       VARCHAR(100) PRIMARY KEY,
            value     TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS documents (
            id              VARCHAR(36) PRIMARY KEY,
            filename        VARCHAR(255) NOT NULL,
            file_path       VARCHAR(500) NOT NULL,
            upload_date     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status          VARCHAR(20) NOT NULL,
            total_pages     INTEGER,
            total_chunks    INTEGER,
            file_size_bytes INTEGER,
            doc_type        VARCHAR(50),
            jurisdiction    VARCHAR(50),
            metadata        TEXT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id               VARCHAR(64) PRIMARY KEY,
            document_id      VARCHAR(36) NOT NULL,
            chunk_index      INTEGER NOT NULL,
            page_number      INTEGER,
            section_title    VARCHAR(255),
            sub_section      VARCHAR(255),
            chunk_text       TEXT NOT NULL,
            chunk_type       VARCHAR(20),
            char_count       INTEGER,
            token_count      INTEGER,
            embedding_vector TEXT,
            metadata         TEXT,
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);

        CREATE TABLE IF NOT EXISTS chat_sessions (
            id            VARCHAR(36) PRIMARY KEY,
            title         VARCHAR(255) NOT NULL,
            message_count INTEGER DEFAULT 0,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS chat_messages (
            id              VARCHAR(36) PRIMARY KEY,
            session_id      VARCHAR(36) NOT NULL,
            role            VARCHAR(16) NOT NULL,
            content         TEXT NOT NULL,
            sources         TEXT,
            is_grounded     INTEGER,
            retrieval_score REAL,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id);

        CREATE TABLE IF NOT EXISTS queries (
            id                    VARCHAR(36) PRIMARY KEY,
            session_id            VARCHAR(36),
            question              TEXT NOT NULL,
            answer                TEXT,
            top_k                 INTEGER DEFAULT 5,
            use_hybrid            INTEGER DEFAULT 1,
            is_grounded           INTEGER,
            retrieval_score       REAL,
            model_used            VARCHAR(50),
            response_time_seconds REAL,
            created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS query_sources (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id        VARCHAR(36) NOT NULL,
            chunk_id        VARCHAR(64) NOT NULL,
            relevance_score REAL,
            rank_position   INTEGER,
            FOREIGN KEY (query_id) REFERENCES queries(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS test_questions (
            id              VARCHAR(36) PRIMARY KEY,
            test_set_id     VARCHAR(36),
            question        TEXT NOT NULL,
            expected_answer TEXT,
            question_type   VARCHAR(50),
            source_doc      VARCHAR(255),
            source_page     INTEGER,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS evaluations (
            id                   VARCHAR(36) PRIMARY KEY,
            test_set_id          VARCHAR(36),
            status               VARCHAR(20),
            overall_faithfulness REAL,
            overall_relevancy    REAL,
            overall_precision    REAL,
            overall_recall       REAL,
            overall_correctness  REAL,
            hallucination_rate   REAL,
            total_questions      INTEGER,
            passed_questions     INTEGER,
            failed_questions     INTEGER,
            analysis_text        TEXT,
            report_markdown      TEXT,
            started_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at         TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS evaluation_results (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluation_id        VARCHAR(36) NOT NULL,
            question_id          VARCHAR(36) NOT NULL,
            generated_answer     TEXT,
            retrieved_contexts   TEXT,
            faithfulness_score   REAL,
            relevancy_score      REAL,
            precision_score      REAL,
            recall_score         REAL,
            correctness_score    REAL,
            has_hallucination    INTEGER,
            issues               TEXT,
            FOREIGN KEY (evaluation_id) REFERENCES evaluations(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS agent_runs (
            id                       VARCHAR(36) PRIMARY KEY,
            query                    TEXT NOT NULL,
            status                   VARCHAR(30),
            current_step             VARCHAR(50),
            output_format            VARCHAR(20),
            search_strategy          VARCHAR(20),
            needs_clarification      INTEGER DEFAULT 0,
            clarification_question   TEXT,
            user_clarification       TEXT,
            final_output             TEXT,
            summary                  TEXT,
            reasoning_log            TEXT,
            error_log                TEXT,
            execution_time_seconds   REAL,
            thread_id                VARCHAR(36),
            started_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at             TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS agent_steps (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          VARCHAR(36) NOT NULL,
            step_name       VARCHAR(50) NOT NULL,
            step_order      INTEGER,
            status          VARCHAR(20),
            tool_used       VARCHAR(50),
            input_data      TEXT,
            output_data     TEXT,
            error_message   TEXT,
            started_at      TIMESTAMP,
            completed_at    TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES agent_runs(id) ON DELETE CASCADE
        );
        """
    )

    # Idempotent migrations for older DBs that may pre-date some columns.
    for stmt in [
        "ALTER TABLE queries ADD COLUMN session_id VARCHAR(36)",
        "ALTER TABLE queries ADD COLUMN is_grounded INTEGER",
        "ALTER TABLE queries ADD COLUMN retrieval_score REAL",
        "ALTER TABLE evaluations ADD COLUMN overall_recall REAL",
        "ALTER TABLE evaluations ADD COLUMN overall_correctness REAL",
        "ALTER TABLE evaluations ADD COLUMN report_markdown TEXT",
        "ALTER TABLE evaluation_results ADD COLUMN retrieved_contexts TEXT",
        "ALTER TABLE evaluation_results ADD COLUMN recall_score REAL",
        "ALTER TABLE evaluation_results ADD COLUMN correctness_score REAL",
        "ALTER TABLE test_questions ADD COLUMN source_doc VARCHAR(255)",
        "ALTER TABLE test_questions ADD COLUMN source_page INTEGER",
        "ALTER TABLE agent_runs ADD COLUMN search_strategy VARCHAR(20)",
        "ALTER TABLE agent_runs ADD COLUMN needs_clarification INTEGER DEFAULT 0",
        "ALTER TABLE agent_runs ADD COLUMN clarification_question TEXT",
        "ALTER TABLE agent_runs ADD COLUMN user_clarification TEXT",
        "ALTER TABLE agent_runs ADD COLUMN summary TEXT",
        "ALTER TABLE agent_runs ADD COLUMN thread_id VARCHAR(36)",
    ]:
        _safe_alter(_db, stmt)

    _db.commit()
    return _db


def get_db() -> sqlite3.Connection:
    if _db is None:
        return init_db()
    return _db
