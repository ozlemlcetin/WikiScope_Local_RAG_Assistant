import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

DB_DIR = BASE_DIR / "db"
CHROMA_DIR = DB_DIR / "chroma"
SQLITE_PATH = DB_DIR / "metadata.sqlite"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"

TOP_K = 5
CHROMA_COLLECTION = "wikipedia_entities"

for d in [RAW_DIR / "people", RAW_DIR / "places", PROCESSED_DIR, METADATA_DIR, CHROMA_DIR]:
    d.mkdir(parents=True, exist_ok=True)
