import json
from pathlib import Path
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, PROCESSED_DIR
from src.processing.cleaner import clean_text
from src.entities import ALL_ENTITIES
from src.ingestion.ingest import get_raw_path

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks

def build_chunks() -> list[dict]:
    all_chunks = []
    for entity in ALL_ENTITIES:
        path = get_raw_path(entity)
        if not path.exists():
            print(f"  [missing] {entity.name} - run ingest first")
            continue
        raw = path.read_text(encoding="utf-8")
        text = clean_text(raw)
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{entity.name.replace(' ', '_')}_{idx}",
                "entity_name": entity.name,
                "entity_type": entity.entity_type,
                "text": chunk,
                "chunk_index": idx,
            })
        print(f"  [chunked] {entity.name}: {len(chunks)} chunks")
    return all_chunks

def save_chunks(chunks: list[dict]) -> Path:
    out = PROCESSED_DIR / "chunks.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(chunks)} chunks to {out}")
    return out
