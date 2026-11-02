"""SQLite: відповіді, автоматичні оцінки та ручна розмітка."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import config


def _connect() -> sqlite3.Connection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                external_id TEXT,
                question TEXT NOT NULL UNIQUE,
                reference TEXT,
                topic TEXT
            );

            CREATE TABLE IF NOT EXISTS answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER NOT NULL,
                mode TEXT NOT NULL,
                answer TEXT,
                contexts_json TEXT,
                latency REAL,
                tokens INTEGER,
                cost REAL,
                created_at TEXT NOT NULL,
                UNIQUE(question_id, mode),
                FOREIGN KEY(question_id) REFERENCES questions(id)
            );

            CREATE TABLE IF NOT EXISTS scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                answer_id INTEGER NOT NULL,
                method TEXT NOT NULL,
                criterion TEXT NOT NULL,
                value REAL,
                cost REAL,
                latency REAL,
                run_index INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                UNIQUE(answer_id, method, criterion, run_index),
                FOREIGN KEY(answer_id) REFERENCES answers(id)
            );

            CREATE TABLE IF NOT EXISTS annotations (
                answer_id INTEGER NOT NULL,
                annotator TEXT NOT NULL,
                faithfulness REAL,
                completeness REAL,
                usefulness REAL,
                trust REAL,
                hallucination REAL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (answer_id, annotator),
                FOREIGN KEY(answer_id) REFERENCES answers(id)
            );

            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT,
                mode TEXT,
                method TEXT,
                message TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS method_runs (
                answer_id INTEGER NOT NULL,
                method TEXT NOT NULL,
                run_index INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                PRIMARY KEY (answer_id, method, run_index)
            );
            """
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def upsert_question(question: str, reference: str | None = None, external_id: str | None = None, topic: str | None = None) -> int:
    init_db()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO questions (external_id, question, reference, topic)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(question) DO UPDATE SET
                reference = COALESCE(excluded.reference, questions.reference),
                external_id = COALESCE(excluded.external_id, questions.external_id),
                topic = COALESCE(excluded.topic, questions.topic)
            """,
            (external_id, question, reference, topic),
        )
        row = connection.execute("SELECT id FROM questions WHERE question = ?", (question,)).fetchone()
        return int(row["id"])


def find_answer(question: str, mode: str) -> dict | None:
    init_db()
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT a.*, q.question, q.reference, q.external_id, q.topic
            FROM answers a
            JOIN questions q ON q.id = a.question_id
            WHERE q.question = ? AND a.mode = ?
            """,
            (question, mode),
        ).fetchone()
    return _answer_dict(row) if row else None


def save_answer(res: dict, reference: str | None = None, external_id: str | None = None, topic: str | None = None, replace: bool = False) -> int:
    question_id = upsert_question(res["question"], reference, external_id, topic)
    existing = find_answer(res["question"], res["mode"])
    if existing and not replace:
        return int(existing["id"])
    payload = (
        res.get("answer"),
        json.dumps(res.get("contexts") or [], ensure_ascii=False),
        res.get("latency"),
        res.get("tokens"),
        res.get("cost"),
        _now(),
        question_id,
        res["mode"],
    )
    init_db()
    with _connect() as connection:
        if existing and replace:
            connection.execute("DELETE FROM scores WHERE answer_id = ?", (existing["id"],))
            connection.execute("DELETE FROM method_runs WHERE answer_id = ?", (existing["id"],))
            connection.execute(
                """
                UPDATE answers
                SET answer = ?, contexts_json = ?, latency = ?, tokens = ?, cost = ?, created_at = ?
                WHERE id = ?
                """,
                (*payload[:6], existing["id"]),
            )
            return int(existing["id"])
        cursor = connection.execute(
            """
            INSERT INTO answers (answer, contexts_json, latency, tokens, cost, created_at, question_id, mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        return int(cursor.lastrowid)


def save_score(answer_id: int, method: str, criterion: str, value: float, cost: float = 0.0, latency: float = 0.0, run_index: int = 0) -> None:
    init_db()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO scores (answer_id, method, criterion, value, cost, latency, run_index, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(answer_id, method, criterion, run_index) DO UPDATE SET
                value = excluded.value,
                cost = excluded.cost,
                latency = excluded.latency,
                created_at = excluded.created_at
            """,
            (answer_id, method, criterion, value, cost, latency, run_index, _now()),
        )


def scored_criteria(answer_id: int, method: str, run_index: int = 0) -> set[str]:
    init_db()
    with _connect() as connection:
        rows = connection.execute(
            "SELECT criterion FROM scores WHERE answer_id = ? AND method = ? AND run_index = ?",
            (answer_id, method, run_index),
        ).fetchall()
    return {row["criterion"] for row in rows}


def mark_method_done(answer_id: int, method: str, run_index: int = 0) -> None:
    init_db()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO method_runs (answer_id, method, run_index, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(answer_id, method, run_index) DO NOTHING
            """,
            (answer_id, method, run_index, _now()),
        )


def method_done(answer_id: int, method: str, run_index: int = 0) -> bool:
    init_db()
    with _connect() as connection:
        row = connection.execute(
            "SELECT 1 FROM method_runs WHERE answer_id = ? AND method = ? AND run_index = ?",
            (answer_id, method, run_index),
        ).fetchone()
    return row is not None


def log_error(question: str, mode: str, method: str, message: str) -> None:
    init_db()
    with _connect() as connection:
        connection.execute(
            "INSERT INTO errors (question, mode, method, message, created_at) VALUES (?, ?, ?, ?, ?)",
            (question, mode, method, message[:2000], _now()),
        )


def save_annotation(answer_id: int, annotator: str, values: dict) -> None:
    init_db()
    fields = ("faithfulness", "completeness", "usefulness", "trust", "hallucination")
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO annotations (
                answer_id, annotator, faithfulness, completeness, usefulness, trust, hallucination, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(answer_id, annotator) DO UPDATE SET
                faithfulness = excluded.faithfulness,
                completeness = excluded.completeness,
                usefulness = excluded.usefulness,
                trust = excluded.trust,
                hallucination = excluded.hallucination,
                created_at = excluded.created_at
            """,
            (answer_id, annotator, *[values.get(field) for field in fields], _now()),
        )


def _answer_dict(row: sqlite3.Row) -> dict:
    data = dict(row)
    data["contexts"] = json.loads(data.pop("contexts_json") or "[]")
    return data


def load_answers() -> list[dict]:
    init_db()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT a.*, q.question, q.reference, q.external_id, q.topic
            FROM answers a
            JOIN questions q ON q.id = a.question_id
            ORDER BY a.id
            """
        ).fetchall()
        score_rows = connection.execute("SELECT * FROM scores ORDER BY id").fetchall()
        note_rows = connection.execute("SELECT * FROM annotations").fetchall()
    by_answer: dict[int, dict] = {}
    for row in rows:
        item = _answer_dict(row)
        item["scores"] = []
        item["annotations"] = {}
        by_answer[item["id"]] = item
    for row in score_rows:
        item = by_answer.get(row["answer_id"])
        if item:
            item["scores"].append(dict(row))
    for row in note_rows:
        item = by_answer.get(row["answer_id"])
        if item:
            item["annotations"][row["annotator"]] = dict(row)
    return list(by_answer.values())


def to_dataframe(aggregate: str = "mean"):
    """Рядок = відповідь. Оцінки методів усереднені за повторними прогонами."""
    import pandas as pd

    records = []
    for item in load_answers():
        row = {
            "answer_id": item["id"],
            "question_id": item["question_id"],
            "question": item["question"],
            "reference": item.get("reference"),
            "topic": item.get("topic"),
            "mode": item["mode"],
            "answer": item["answer"],
            "latency": item.get("latency"),
            "tokens": item.get("tokens"),
            "generation_cost": item.get("cost"),
        }
        grouped: dict[tuple[str, str], list[float]] = {}
        run_cost: dict[tuple[str, int], float] = {}
        run_time: dict[tuple[str, int], float] = {}
        for score in item["scores"]:
            if score["value"] is None:
                continue
            grouped.setdefault((score["method"], score["criterion"]), []).append(float(score["value"]))
            key = (score["method"], int(score["run_index"]))
            run_cost[key] = run_cost.get(key, 0.0) + float(score["cost"] or 0)
            run_time[key] = run_time.get(key, 0.0) + float(score["latency"] or 0)
        for (method, criterion), values in grouped.items():
            ordered = sorted(values)
            center = ordered[len(ordered) // 2] if aggregate == "median" else sum(ordered) / len(ordered)
            row[f"{method}__{criterion}"] = center
            row[f"{method}__{criterion}__std"] = _std(ordered)
            row[f"{method}__{criterion}__n"] = len(ordered)
        methods = {method for method, _run in run_cost}
        for method in methods:
            costs = [value for (name, _run), value in run_cost.items() if name == method]
            times = [value for (name, _run), value in run_time.items() if name == method]
            row[f"{method}__cost"] = sum(costs) / len(costs)
            row[f"{method}__latency"] = sum(times) / len(times)
        for annotator, note in item["annotations"].items():
            prefix = "expert" if annotator == "expert1" else annotator
            for field in ("faithfulness", "completeness", "usefulness", "trust", "hallucination"):
                row[f"{prefix}_{field}"] = note.get(field)
        records.append(row)
    return pd.DataFrame(records)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def list_answer_ids() -> list[int]:
    init_db()
    with _connect() as connection:
        rows = connection.execute("SELECT id FROM answers ORDER BY id").fetchall()
    return [int(row["id"]) for row in rows]
