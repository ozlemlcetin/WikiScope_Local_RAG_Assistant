#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.ingest import ingest_all

if __name__ == "__main__":
    force = "--force" in sys.argv
    print("=== WikiScope: Ingesting Wikipedia data ===")
    if force:
        print("Force mode: re-downloading all articles\n")
    else:
        print("Skipping already downloaded articles (use --force to re-download)\n")
    ingest_all(force=force)
    print("\nDone.")
