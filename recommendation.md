# WikiScope — Recommendations and Future Improvements

## Current Limitations

1. **Fixed entity list** — only 40 entities are searchable; questions about any other topic return the fallback response.
2. **Static data** — Wikipedia snapshots are captured at ingest time and not refreshed automatically.
3. **Keyword-based routing** — query router may misroute ambiguous queries that lack clear person/place signals.
4. **Character-based chunking** — chunks split on character count, not sentence or paragraph boundaries, which can cut sentences mid-way.
5. **Single language** — only English Wikipedia is supported.

## Recommended Improvements

### Short-term (v1.1)

- **Semantic chunking** — split on sentence or paragraph boundaries using `nltk` or `spaCy` instead of fixed character counts. This would preserve coherent context windows and improve retrieval quality.
- **Confidence-threshold fallback** — automatically trigger the "I don't know" fallback when all retrieved chunk distances exceed a tunable threshold, rather than relying solely on LLM judgment.
- **Entity expansion UI** — allow users to add new entities at runtime (enter a Wikipedia title, trigger fetch + embed + store) without re-running the full pipeline.
- **Answer citations** — highlight which specific Wikipedia paragraph each answer sentence came from, improving transparency.

### Medium-term (v1.2)

- **Chat history context** — include the last N turns in the prompt to support follow-up questions ("What else did she discover?").
- **Structured metadata filters** — let users narrow the search to a specific entity or entity type directly from the sidebar.
- **Cross-encoder re-ranking** — add a local cross-encoder (e.g., `ms-marco-MiniLM` via `sentence-transformers`) after vector retrieval to improve chunk selection precision beyond distance-based ordering.
- **Docker packaging** — containerise Ollama, ChromaDB, SQLite, and the Streamlit app so the system can be reproduced on any machine with a single `docker compose up`.

### Long-term (v2.0)

- **Automatic Wikipedia refresh** — periodic re-ingestion (e.g., nightly cron) to keep the local snapshot current.
- **Evaluation harness** — a set of ground-truth Q&A pairs to measure RAG quality (precision, recall, faithfulness, hallucination rate) across all 40 entities.
- **Graph RAG** — build entity relationship graphs to answer relational questions ("Which scientists knew each other?").
- **Multi-collection ChromaDB** — separate collections per entity type for cleaner isolation and faster filtered searches.
- **Larger dataset** — extend beyond 40 entities by supporting batched Wikipedia category imports (e.g., all Nobel laureates).

## Model Recommendations

| Use Case | Recommended Model |
|---|---|
| Better answers, moderate speed | `llama3.1:8b` or `mistral:7b` |
| Faster responses, smaller device | `phi3:mini` |
| Better embeddings | `mxbai-embed-large` |

## Chunking Strategy Notes

The current character-based chunking (800 chars, 120 overlap) is a pragmatic baseline that works reliably but occasionally splits sentences mid-way. A semantic chunking approach — splitting on paragraph or sentence boundaries — would preserve more coherent context windows and likely improve both retrieval precision and answer quality. `nltk.sent_tokenize` or `spaCy`'s sentence segmenter are lightweight options that require no additional model downloads.
