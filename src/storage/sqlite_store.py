import sqlite3
import json
from pathlib import Path
from src.config import SQLITE_PATH

def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(SQLITE_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            entity_type TEXT NOT NULL,
            wikipedia_title TEXT NOT NULL,
            chunk_count INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            entity_name TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn

def store_chunks(chunks: list[dict]) -> None:
    conn = init_db()
    for chunk in chunks:
        conn.execute(
            "INSERT OR REPLACE INTO chunks (chunk_id, entity_name, entity_type, chunk_index, text) VALUES (?,?,?,?,?)",
            (chunk["chunk_id"], chunk["entity_name"], chunk["entity_type"], chunk["chunk_index"], chunk["text"])
        )
    entity_counts: dict[str, dict] = {}
    for chunk in chunks:
        name = chunk["entity_name"]
        if name not in entity_counts:
            entity_counts[name] = {"entity_type": chunk["entity_type"], "count": 0}
        entity_counts[name]["count"] += 1
    for name, info in entity_counts.items():
        conn.execute(
            "INSERT OR REPLACE INTO entities (name, entity_type, wikipedia_title, chunk_count) VALUES (?,?,?,?)",
            (name, info["entity_type"], name, info["count"])
        )
    conn.commit()
    conn.close()
    print(f"Stored {len(chunks)} chunks in SQLite at {SQLITE_PATH}")

def get_entity_stats() -> list[dict]:
    conn = init_db()
    rows = conn.execute("SELECT name, entity_type, chunk_count FROM entities ORDER BY entity_type, name").fetchall()
    conn.close()
    return [{"name": r[0], "entity_type": r[1], "chunk_count": r[2]} for r in rows]
