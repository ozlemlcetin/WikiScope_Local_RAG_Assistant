#!/usr/bin/env python3
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processing.chunker import build_chunks, save_chunks
from src.storage.sqlite_store import store_chunks
from src.storage.vector_store import add_chunks
from src.generation.ollama_client import embed_text, check_ollama_health
from src.config import PROCESSED_DIR

if __name__ == "__main__":
    print("=== WikiScope: Building Vector Database ===\n")
    if not check_ollama_health():
        print("ERROR: Ollama is not running. Start it with: ollama serve")
        sys.exit(1)

    print("Step 1: Chunking documents...")
    chunks = build_chunks()
    save_chunks(chunks)

    print("\nStep 2: Storing chunks in SQLite...")
    store_chunks(chunks)

    print("\nStep 3: Generating embeddings and storing in Chroma...")
    print(f"  Embedding {len(chunks)} chunks (this may take a few minutes)...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        emb = embed_text(chunk["text"])
        embeddings.append(emb)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(chunks)} embedded...")
    add_chunks(chunks, embeddings)
    print("\n=== Done! Vector database is ready. Run: streamlit run app.py ===")
