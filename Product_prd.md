# WikiScope Local RAG Assistant — Product Requirements Document

## Overview

WikiScope is a fully local Retrieval-Augmented Generation (RAG) system for university coursework. It answers questions about famous people and places using Wikipedia as its knowledge base. All computation runs on the user's machine with no external API calls.

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
| F1 | Fetch and store Wikipedia articles locally |
| F2 | Chunk articles with configurable size and overlap |
| F3 | Generate embeddings with nomic-embed-text via Ollama |
| F4 | Store vectors in ChromaDB with entity_type metadata |
| F5 | Route queries to people, places, or both |
| F6 | Retrieve top-K relevant chunks |
| F7 | Generate grounded answers with llama3.2:3b |
| F8 | Return fallback when context is insufficient |
| F9 | Streamlit chat interface |
| F10 | Show/hide retrieved chunks in UI |

## Non-Functional Requirements

- Runs fully offline after initial Wikipedia fetch.
- Ingest pipeline completes in under 10 minutes on a modern Mac.
- Query response time under 30 seconds on Apple Silicon.

## Architecture

```
User Query
    │
    ▼
Query Router (keyword + entity name matching)
    │
    ├─ person → filter Chroma by entity_type=person
    ├─ place  → filter Chroma by entity_type=place
    └─ both   → no filter
         │
         ▼
    Embed Query (nomic-embed-text via Ollama)
         │
         ▼
    Chroma Vector Search (top-K)
         │
         ▼
    Context Builder
         │
         ▼
    Prompt Construction
         │
         ▼
    llama3.2:3b (Ollama)
         │
         ▼
    Answer → Streamlit UI
```
