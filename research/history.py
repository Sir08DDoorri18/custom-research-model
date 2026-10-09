"""Saved conversations: Chainlit's thread history, kept in a SQLite file on this PC.

Chainlit stores threads, messages and elements through its SQLAlchemy data layer. Two local
pieces make that work without a database server or cloud storage:
- SQLite can't hold lists, so list values (tags) are stored as JSON text and read back.
- Elements (the answer window, the parallel panel, added files) go to a folder instead of
  S3/GCS; Chainlit only ships cloud stores.
Conversations older than `storage.history_days` (models.yaml) are deleted at startup.
Nothing here leaves the PC.
"""
from __future__ import annotations

import contextvars
import datetime as dt
import json
import secrets
import sqlite3
from pathlib import Path

from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.data.storage_clients.base import BaseStorageClient

from .config import ROOT

DB = ROOT / ".cache" / "history.db"
FILES = ROOT / ".cache" / "history_files"
SECRET = ROOT / ".cache" / "auth_secret"
FILES_URL = "/research/files/"

# Columns are the keys Chainlit 2.x writes (StepDict, ElementDict); it inserts whatever keys it has.
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    "id" TEXT PRIMARY KEY, "identifier" TEXT NOT NULL UNIQUE, "metadata" TEXT NOT NULL, "createdAt" TEXT);
CREATE TABLE IF NOT EXISTS threads (
    "id" TEXT PRIMARY KEY, "createdAt" TEXT, "name" TEXT, "userId" TEXT, "userIdentifier" TEXT,
    "tags" TEXT, "metadata" TEXT);
CREATE TABLE IF NOT EXISTS steps (
    "id" TEXT PRIMARY KEY, "name" TEXT NOT NULL, "type" TEXT NOT NULL, "threadId" TEXT NOT NULL,
    "parentId" TEXT, "command" TEXT, "modes" TEXT, "streaming" BOOLEAN NOT NULL, "waitForAnswer" BOOLEAN,
    "isError" BOOLEAN, "metadata" TEXT, "tags" TEXT, "input" TEXT, "output" TEXT, "createdAt" TEXT,
    "start" TEXT, "end" TEXT, "generation" TEXT, "showInput" TEXT, "defaultOpen" BOOLEAN,
    "autoCollapse" BOOLEAN, "language" TEXT, "icon" TEXT, "indent" INT);
CREATE TABLE IF NOT EXISTS elements (
    "id" TEXT PRIMARY KEY, "threadId" TEXT, "type" TEXT, "url" TEXT, "chainlitKey" TEXT, "path" TEXT,
    "name" TEXT NOT NULL, "display" TEXT, "objectKey" TEXT, "size" TEXT, "page" INT, "language" TEXT,
    "forId" TEXT, "mime" TEXT, "props" TEXT, "autoPlay" BOOLEAN, "playerConfig" TEXT);
CREATE TABLE IF NOT EXISTS feedbacks (
    "id" TEXT PRIMARY KEY, "forId" TEXT NOT NULL, "threadId" TEXT NOT NULL, "value" INT NOT NULL, "comment" TEXT);
CREATE INDEX IF NOT EXISTS steps_thread ON steps ("threadId");
CREATE INDEX IF NOT EXISTS elements_thread ON elements ("threadId");
CREATE INDEX IF NOT EXISTS threads_user ON threads ("userId");
"""
# Columns (and the aliases Chainlit's queries give them) that hold JSON.
JSON_COLUMNS = {"props", "metadata", "generation", "tags", "modes", "thread_metadata", "thread_tags",
                "step_metadata", "step_generation", "step_tags"}


def auth_secret() -> str:
    """Signs the browser's login token. Kept in a file so a restart doesn't log the tab out."""
    if not SECRET.exists():
        SECRET.parent.mkdir(parents=True, exist_ok=True)
        SECRET.write_text(secrets.token_urlsafe(48), encoding="utf-8")
    return SECRET.read_text(encoding="utf-8").strip()


def ensure_schema() -> None:
    DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB) as db:
        db.executescript(SCHEMA)


class LocalStorage(BaseStorageClient):
    """Element files in .cache/history_files, served back by the app at FILES_URL."""

    async def upload_file(self, object_key: str, data, mime: str = "application/octet-stream",
                          overwrite: bool = True, content_disposition: str | None = None) -> dict:
        path = file_path(object_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
        return {"object_key": object_key, "url": FILES_URL + object_key}

    async def delete_file(self, object_key: str) -> bool:
        path = file_path(object_key)
        if path.exists():
            path.unlink()
        return True

    async def get_read_url(self, object_key: str) -> str:
        return FILES_URL + object_key

    async def close(self) -> None:
        pass


def file_path(object_key: str) -> Path:
    """Where an element file lives; refuses keys that would point outside the folder."""
    path = (FILES / object_key).resolve()
    if not path.is_relative_to(FILES.resolve()):
        raise ValueError(f"bad object key: {object_key}")
    return path


_raw = contextvars.ContextVar("history_raw_rows", default=False)


class DataLayer(SQLAlchemyDataLayer):
    """Chainlit's SQL layer is written for PostgreSQL, which hands JSON columns back as objects;
    SQLite hands back text. Lists going in are stored as JSON, and JSON columns coming out are
    decoded, so the rest of Chainlit sees what it expects."""

    def __init__(self):
        ensure_schema()
        super().__init__(f"sqlite+aiosqlite:///{DB.as_posix()}", storage_provider=LocalStorage())

    async def execute_sql(self, query: str, parameters: dict):
        params = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, tuple)) else v
                  for k, v in parameters.items()}
        result = await super().execute_sql(query, params)
        if isinstance(result, list) and not _raw.get():
            for row in result:
                for col in JSON_COLUMNS.intersection(row):
                    row[col] = _decoded(row[col])
        return result

    async def get_element(self, thread_id: str, element_id: str):
        token = _raw.set(True)  # this one runs json.loads on "props" itself
        try:
            return await super().get_element(thread_id, element_id)
        finally:
            _raw.reset(token)


def _decoded(value):
    if isinstance(value, str) and value[:1] in ("{", "["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def prune(days: float) -> int:
    """Delete conversations older than `days` with their messages and files. Returns how many."""
    if not DB.exists():
        return 0
    cutoff = (dt.datetime.now() - dt.timedelta(days=days)).isoformat()
    with sqlite3.connect(DB) as db:
        old = [r[0] for r in db.execute('SELECT "id" FROM threads WHERE "createdAt" < ?', (cutoff,))]
        if old:
            marks = ",".join("?" * len(old))
            for table in ("steps", "elements", "feedbacks"):
                db.execute(f'DELETE FROM {table} WHERE "threadId" IN ({marks})', old)
            db.execute(f'DELETE FROM threads WHERE "id" IN ({marks})', old)
        kept = {r[0] for r in db.execute('SELECT "objectKey" FROM elements') if r[0]}
        threads = {r[0] for r in db.execute('SELECT "id" FROM threads')}
    _sweep(kept)
    from . import checkpoint  # progress of questions in conversations that are gone
    checkpoint.prune(threads)
    return len(old)


def _sweep(kept: set[str], grace: float = 86400) -> None:
    """Remove element files no element points to (deleted conversations, and the first copy of
    an element saved before its conversation had an owner), then empty folders. Files younger
    than `grace` seconds are left alone: they may belong to a message being saved right now."""
    if not FILES.exists():
        return
    now = dt.datetime.now().timestamp()
    for path in FILES.rglob("*"):
        if path.is_file() and path.relative_to(FILES).as_posix() not in kept and now - path.stat().st_mtime > grace:
            path.unlink(missing_ok=True)
    for folder in sorted((p for p in FILES.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
        try:
            folder.rmdir()  # only succeeds when empty
        except OSError:
            pass


def previous_turns(thread: dict, limit: int = 6000) -> str:
    """The last questions and answers of a saved conversation, as context for the next question
    after it is reopened (the chat model starts fresh and has no memory of it)."""
    lines = []
    for step in thread.get("steps") or []:
        if step.get("type") == "user_message" and step.get("output"):
            lines.append(f"질문: {step['output'].strip()}")
        elif step.get("type") == "assistant_message" and step.get("output"):
            lines.append(f"답변: {step['output'].strip()}")
    text = "\n\n".join(lines)
    return text[-limit:]
