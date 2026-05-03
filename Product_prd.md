# WikiScope Local RAG Assistant — Product Requirements Document

## Overview

WikiScope is a fully local Retrieval-Augmented Generation (RAG) system for university coursework. It answers questions about famous people and places using Wikipedia as its knowledge base. No external LLM APIs are used. Wikipedia is accessed only during ingestion; after that, local data, embeddings, vector database, retrieval, and generation run on localhost.

## Goals

- Demonstrate RAG architecture without third-party LLM APIs.
- Implement chunking, embedding, retrieval, and generation manually.
- Provide a usable chat interface for homework demonstration.

## Non-Goals

- Real-time Wikipedia updates.
- Multi-language support.
- Production-grade scaling.

## Entities

### Required People (10)
Albert Einstein, Marie Curie, Leonardo da Vinci, William Shakespeare, Ada Lovelace, Nikola Tesla, Lionel Messi, Cristiano Ronaldo, Taylor Swift, Frida Kahlo.

### Additional People (10)
Isaac Newton, Charles Darwin, Cleopatra, Napoleon Bonaparte, Mahatma Gandhi, Nelson Mandela, Wolfgang Amadeus Mozart, Vincent van Gogh, Stephen Hawking, Aristotle.

### Required Places (10)
Eiffel Tower, Great Wall of China, Taj Mahal, Grand Canyon, Machu Picchu, Colosseum, Hagia Sophia, Statue of Liberty, Pyramids of Giza, Mount Everest.

### Additional Places (10)
Stonehenge, Angkor Wat, Petra, Chichen Itza, Acropolis of Athens, Vatican City, Niagara Falls, Amazon rainforest, Sahara Desert, Great Barrier Reef.

## Functional Requirements

| ID | Requirement |
|----|-------------|
| F1 | Fetch and store Wikipedia articles locally via MediaWiki HTTP API |
| F2 | Chunk articles with configurable size (800 chars) and overlap (120 chars) |
| F3 | Generate embeddings with `nomic-embed-text` via Ollama |
| F4 | Store vectors in ChromaDB with `entity_type` metadata (person / place) |
| F5 | Route queries to `people`, `places`, or `both` via keyword + entity-name matching |
| F6 | Hybrid retrieval: entity-name boost → lexical location injection → vector search → rerank |
| F7 | Generate grounded answers with `llama3.2:3b` via Ollama |
| F8 | Return "I don't know based on the local Wikipedia data." when context is insufficient |
| F9 | Streamlit chat interface with persistent session history |
| F10 | Show/hide retrieved source chunks panel in UI |
| F11 | Comparison mode: side-by-side entity summaries with deterministic same-dimension diff bullets |
| F12 | Evidence signal badge: `entity match` / `lexical match` / `vector search` / `insufficient local evidence` |
| F13 | Latency display: total response time shown below each answer |
| F14 | Streaming response display: typewriter-effect word-by-word rendering (display-level, post-generation) |
| F15 | Two-model comparison: optional side-by-side answers from a second local Ollama model |
| F16 | Example question buttons in sidebar for quick demo access |
| F17 | Local dataset coverage panel: lists all 40 ingested entities in the sidebar |

## Non-Functional Requirements

- Runs fully offline after initial Wikipedia fetch.
- Ingest pipeline completes in under 10 minutes on a modern Mac.
- Query response time under 30 seconds on Apple Silicon.

## Architecture

```
User Query
    │
    ▼
Query Router
(keyword matching + entity-name scan)
    │
    ├─ person → filter ChromaDB by entity_type=person
    ├─ place  → filter ChromaDB by entity_type=place
    └─ both   → no filter
         │
         ▼
    Hybrid Retrieval
         │
         ├─ 1. Entity-name boost
         │       Named entities found in query → inject matching chunks
         │       at distance=0.0 (highest priority sentinel)
         │
         ├─ 2. Lexical location injection
         │       Location keywords in query → inject location-field chunks
         │       at distance=0.1 (second-priority sentinel)
         │
         ├─ 3. Vector search (nomic-embed-text via Ollama)
         │       Embed query → ChromaDB top-K L2 search
         │
         └─ 4. Reranking
                 Merge + deduplicate; boosted chunks always rank first
                 │
                 ▼
         Context Builder
         (formats chunk text + entity metadata for prompt)
                 │
                 ▼
         Prompt Construction
         (RAG prompt or comparison prompt)
                 │
                 ▼
         llama3.2:3b (Ollama)
                 │
                 ▼
         Post-processing
         (comparison: strip LLM diff → rebuild deterministically from chunk text)
                 │
                 ▼
         Answer → Streamlit UI
```

## Comparison Mode Detail

For queries that mention two entities (e.g., "Compare X and Y"), the system:

1. Routes to hybrid retrieval for both entities.
2. Sends a structured comparison prompt requesting per-entity summaries and a diff section.
3. Strips the LLM-generated diff section from the response.
4. Rebuilds the diff section deterministically: sentences from each entity's chunks are matched using 11 named dimension patterns (birth/origin, death, nationality, field/discipline, achievement, construction date, designer/builder, purpose/use, location, height/size, material/composition). Only sentences that match the same dimension group produce a valid bullet: `Entity A fact — vs — Entity B fact`.
5. Falls back to a conservative Python-only answer if the LLM response fails validation.
