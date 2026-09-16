"""SQLite-backed transcription job queue."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class QueueJob:
    id: str
    status: str
    audio_path: str
    output_dir: str
    language: Optional[str]
    labels: str
    created_at: str
    updated_at: str
    diarize: bool = False
    engine: str = "whisper-bn"
    model: Optional[str] = None
    result_json_path: Optional[str] = None
    error: Optional[str] = None


class TranscriptionQueue:
    """Small durable SQLite queue for one-host ASR services."""

    def __init__(self, db_path: str = "./transcripts/transcription_queue.sqlite3"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    audio_path TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    language TEXT,
                    labels TEXT NOT NULL,
                    diarize INTEGER NOT NULL DEFAULT 0,
                    engine TEXT NOT NULL DEFAULT 'whisper-bn',
                    model TEXT,
                    result_json_path TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            for ddl in (
                "ALTER TABLE jobs ADD COLUMN diarize INTEGER NOT NULL DEFAULT 0",
                "ALTER TABLE jobs ADD COLUMN engine TEXT NOT NULL DEFAULT 'whisper-bn'",
                "ALTER TABLE jobs ADD COLUMN model TEXT",
            ):
                try:
                    conn.execute(ddl)
                except sqlite3.OperationalError:
                    pass
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at)")

    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def create_job(
        self,
        audio_path: str,
        output_dir: str,
        language: Optional[str] = "bn",
        labels: str = "Agent,Customer",
        diarize: bool = False,
        engine: str = "whisper-bn",
        model: Optional[str] = None,
    ) -> QueueJob:
        job = QueueJob(
            id=str(uuid.uuid4()),
            status="queued",
            audio_path=audio_path,
            output_dir=output_dir,
            language=language,
            labels=labels,
            diarize=diarize,
            engine=engine,
            model=model,
            created_at=self._now(),
            updated_at=self._now(),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (id, status, audio_path, output_dir, language, labels, diarize, engine, model, created_at, updated_at)
                VALUES (:id, :status, :audio_path, :output_dir, :language, :labels, :diarize, :engine, :model, :created_at, :updated_at)
                """,
                asdict(job),
            )
        return job

    def get_job(self, job_id: str) -> Optional[QueueJob]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def list_jobs(self, limit: int = 100) -> list[QueueJob]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def claim_next(self) -> Optional[QueueJob]:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if not row:
                conn.commit()
                return None
            conn.execute(
                "UPDATE jobs SET status = 'processing', updated_at = ? WHERE id = ?",
                (self._now(), row["id"]),
            )
            conn.commit()
        return self.get_job(row["id"])

    def complete_job(self, job_id: str, result_json_path: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'completed', result_json_path = ?, error = NULL, updated_at = ?
                WHERE id = ?
                """,
                (result_json_path, self._now(), job_id),
            )

    def fail_job(self, job_id: str, error: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'failed', error = ?, updated_at = ? WHERE id = ?",
                (error, self._now(), job_id),
            )

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> QueueJob:
        data = {key: row[key] for key in row.keys()}
        data["diarize"] = bool(data.get("diarize", False))
        data["engine"] = data.get("engine") or "whisper-bn"
        return QueueJob(**data)

    @staticmethod
    def job_to_dict(job: QueueJob) -> dict:
        data = asdict(job)
        data["engine"] = job.engine or "whisper-bn"
        data["result_path"] = job.result_json_path
        data["transcript"] = None
        if job.result_json_path and Path(job.result_json_path).exists():
            try:
                transcript = json.loads(Path(job.result_json_path).read_text(encoding="utf-8"))
                data["transcript"] = transcript
                data["result"] = transcript  # Backwards-compatible alias.
            except Exception:
                data["result"] = None
        return data
