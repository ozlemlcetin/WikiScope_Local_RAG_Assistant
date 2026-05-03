#!/usr/bin/env python3
"""Quick smoke-test for WikiScope: runs five demo queries and prints results."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.answer_generator import answer

QUERIES = [
    "Who was Albert Einstein and what is he known for?",
    "What did Marie Curie discover?",
    "Which famous place is located in Turkey?",
    "Compare Hagia Sophia and Pyramids of Giza.",
    "Who is the president of Mars?",
]


def main() -> None:
    print("=" * 70)
    print("WikiScope — Demo Test Run")
    print("=" * 70)

    for i, query in enumerate(QUERIES, 1):
        print(f"\n[{i}/{len(QUERIES)}] {query}")
        print("-" * 60)

        t0 = time.perf_counter()
        result = answer(query, top_k=5, debug=False)
        elapsed = time.perf_counter() - t0

        entities = list(dict.fromkeys(c["entity_name"] for c in result["chunks"]))
        print(f"Route    : {result['route']}")
        print(f"Entities : {entities}")
        print(f"Time     : {elapsed:.2f}s")
        print(f"Answer   :\n{result['answer']}")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
