import sqlite3
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "chat_history.db")

def _init():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON conversations(session_id)")

_init()

def save_message(session_id: str, role: str, content: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO conversations (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.now(timezone.utc).isoformat())
        )
