# WikiScope — Recommendations and Future Improvements

## Current Limitations

1. **Fixed entity list** — only 40 entities are searchable.
2. **Static data** — Wikipedia snapshots are captured at ingest time and not refreshed.
3. **Simple keyword routing** — query router may misroute ambiguous queries.
4. **No re-ranking** — chunks are returned in embedding similarity order only.

## Recommended Improvements

### Short-term (v1.1)
- **Hybrid search** — combine BM25 keyword search with vector similarity for better recall.
- **Re-ranking** — add a cross-encoder re-ranker (e.g., ms-marco-MiniLM) to improve chunk selection precision.
- **Confidence scoring** — threshold on embedding distance to auto-trigger the fallback response.
- **Entity expansion** — allow users to add new entities without re-running the full pipeline.

### Medium-term (v1.2)
- **Streaming responses** — stream tokens from Ollama to reduce perceived latency.
- **Chat history context** — include the last N turns in the prompt for follow-up questions.
- **Structured metadata filters** — let users filter by entity type in the UI sidebar.
- **Answer citations** — show which specific Wikipedia paragraph the answer came from.

### Long-term (v2.0)
- **Automatic Wikipedia refresh** — periodic re-ingestion to keep data current.
- **Multi-collection Chroma** — separate collections per entity type for cleaner architecture.
- **Graph RAG** — build entity relationship graphs to answer relational questions.
- **Evaluation harness** — a set of ground-truth Q&A pairs to measure RAG quality (precision, recall, faithfulness).

## Model Recommendations

| Use Case | Recommended Model |
|---|---|
| Better answers, slower | llama3.1:8b or mistral:7b |
| Faster, smaller device | phi3:mini |
| Better embeddings | mxbai-embed-large |

## Chunking Strategy Notes

The current character-based chunking (800 chars, 120 overlap) is a pragmatic baseline. Semantic chunking — splitting on paragraph or sentence boundaries — would preserve more coherent context windows and likely improve retrieval quality. Consider `nltk` sentence tokenization or spaCy for a simple upgrade.
