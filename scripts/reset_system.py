#!/usr/bin/env python3
import sys
import shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROCESSED_DIR, METADATA_DIR, DB_DIR, CHROMA_DIR, SQLITE_PATH

def reset(keep_raw: bool = True):
    print("=== WikiScope: Resetting system ===")
    paths_to_clear = [PROCESSED_DIR, METADATA_DIR, CHROMA_DIR]
    if not keep_raw:
        from src.config import RAW_DIR
        paths_to_clear.append(RAW_DIR)
    for p in paths_to_clear:
        if p.exists():
            shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)
            print(f"  Cleared: {p}")
    if SQLITE_PATH.exists():
        SQLITE_PATH.unlink()
        print(f"  Deleted: {SQLITE_PATH}")
    print("\nReset complete. Run ingest_data.py and build_vector_db.py to start fresh.")

if __name__ == "__main__":
    keep_raw = "--full" not in sys.argv
    if not keep_raw:
        confirm = input("This will delete ALL data including raw Wikipedia files. Are you sure? [y/N] ")
        if confirm.lower() != "y":
            print("Aborted.")
            sys.exit(0)
    reset(keep_raw=keep_raw)
