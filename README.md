# WikiScope Local RAG Assistant

A fully local Retrieval-Augmented Generation (RAG) system that answers questions about famous people and places using locally stored Wikipedia data. No external LLM APIs are used. Wikipedia is accessed only during the one-time ingestion step; after that, inference, embeddings, vector storage, retrieval, and generation all run on localhost.

## Features

**Required features:**
- Local-only RAG: answers are grounded exclusively in retrieved Wikipedia excerpts
- 20 famous people + 20 famous places (40 entities total)
- Data ingestion and chunking pipeline (Wikipedia MediaWiki API → character chunks)
- Local embeddings via `nomic-embed-text` through Ollama
- Vector storage in ChromaDB with entity-type metadata
- Query routing: `people`, `places`, or `both` based on query keywords and named entities
- Comparison mode with deterministic same-dimension difference detection
- "I don't know based on the local Wikipedia data." fallback when evidence is insufficient
- Streamlit chat interface at `localhost:8501`
- Retrieved sources panel (toggle in sidebar)

**Optional demo features:**
- Evidence signal badge: `entity match` / `lexical match` / `vector search` / `insufficient local evidence`
- Latency display: total response time shown below each answer
- Example question buttons in the sidebar
- Local dataset coverage panel (shows all configured local dataset entities)
- Streaming response display (typewriter effect — display-level only, after full answer generation)
- Optional side-by-side comparison of answers from two local Ollama generation models

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| LLM runtime | Ollama |
| Answer generation | llama3.2:3b (local) |
| Embeddings | nomic-embed-text (local, via Ollama) |
| Vector store | ChromaDB (persistent, local) |
| Metadata store | SQLite (local) |
| Chat UI | Streamlit |
| Wikipedia data | MediaWiki HTTP API (fetched once at ingest time) |

## Repository Structure

```
wikiscope-local-rag/
├── app.py                        # Streamlit chat application
├── requirements.txt              # Python dependencies
├── README.md
├── Product_prd.md                # Product requirements document
├── recommendation.md             # Future improvement notes
├── scripts/
│   ├── ingest_data.py            # Download Wikipedia articles
│   ├── build_vector_db.py        # Chunk, embed, store in ChromaDB + SQLite
│   ├── run_demo_tests.py         # CLI smoke test for all demo queries
│   └── reset_system.py           # Clear DB and/or raw data
├── src/
│   ├── config.py                 # Paths, model names, chunk settings
│   ├── entities.py               # Entity list (20 people + 20 places)
│   ├── generation/
│   │   ├── answer_generator.py   # RAG answer pipeline, comparison logic
│   │   └── ollama_client.py      # Ollama HTTP client (generate + embed)
│   ├── ingestion/
│   │   ├── ingest.py             # Orchestrates Wikipedia fetch for all entities
│   │   └── wikipedia_fetcher.py  # MediaWiki API fetcher with retry/backoff
│   ├── processing/
│   │   ├── chunker.py            # Character-based chunking with overlap
│   │   └── cleaner.py            # Text normalisation
│   ├── retrieval/
│   │   ├── context_builder.py    # Formats retrieved chunks into LLM context
│   │   ├── query_router.py       # Routes queries to people / places / both
│   │   └── retriever.py          # Hybrid retrieval: entity boost + lexical + vector
│   └── storage/
│       ├── sqlite_store.py       # SQLite chunk metadata store
│       └── vector_store.py       # ChromaDB wrapper
└── data/                         # Created at ingest time (git-ignored)
    ├── raw/                      # Downloaded Wikipedia text files
    ├── processed/                # chunks.jsonl
    └── metadata/                 # entities.json, failed_entities.json
```

## Setup and Installation (macOS)

### 1. Create and activate a virtual environment

```bash
cd wikiscope-local-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Install Ollama

Download from https://ollama.com/download or use Homebrew:

```bash
brew install ollama
```

### 3. Pull the required models

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### 4. Start Ollama (keep running in a separate terminal)

```bash
ollama serve
```

### 5. Ingest Wikipedia data

```bash
python scripts/ingest_data.py
```

Downloads Wikipedia articles for all 40 entities to `data/raw/`. Uses the MediaWiki API with automatic retry and rate-limit backoff. Already-downloaded articles are skipped unless `--force` is passed.

### 6. Build the vector database

```bash
python scripts/build_vector_db.py
```

Chunks all articles (800-character chunks, 120-character overlap), generates embeddings with `nomic-embed-text`, and stores everything in ChromaDB and SQLite.

### 7. Launch the Streamlit app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.



## Demo Queries

The following queries demonstrate the full range of system capabilities:

| Query | Expected behaviour |
|---|---|
| `Who was Albert Einstein and what is he known for?` | Returns theory of relativity, quantum theory contributions, and E = mc² |
| `What did Marie Curie discover?` | Returns radium and polonium |
| `Which famous place is located in Turkey?` | Returns Hagia Sophia, Istanbul |
| `Compare Hagia Sophia and Pyramids of Giza.` | Returns per-entity facts + deterministic same-dimension difference bullet |
| `Who is the president of Mars?` | Returns "I don't know based on the local Wikipedia data." |

## Running the Demo Test Script

```bash
python scripts/run_demo_tests.py
```

Runs all five demo queries non-interactively and prints route, retrieved entities, elapsed time, and the full answer for each.

## Resetting the System

```bash
# Clear DB only (keep raw Wikipedia files — fast re-index):
python scripts/reset_system.py

# Full reset (deletes everything including raw files):
python scripts/reset_system.py --full
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `Ollama is not running` warning in UI | Run `ollama serve` in a separate terminal |
| Generation fails / model error from Ollama | Pull the required models: `ollama pull llama3.2:3b` and `ollama pull nomic-embed-text` |
| `ModuleNotFoundError: No module named 'chromadb'` | Activate the virtual environment: `source .venv/bin/activate` |
| `Vectors in DB: ⚠️ Not built yet` in sidebar | Run `python scripts/build_vector_db.py` |
| Answer is always "I don't know" | Run ingestion and build scripts; check that `data/raw/` is populated |
| `nomic-embed-text` appears in model comparison selector | This is an embedding model, not a generation model — the app filters it out automatically |
| Second model comparison unavailable | Install another generation model: `ollama pull phi3:mini` or `ollama pull mistral` |
| Streamlit won't open | Make sure port 8501 is free; try `streamlit run app.py --server.port 8502` |

## Limitations

- Answers are limited to the 40 locally ingested entities; questions about other topics return the fallback response.
- Data is not connected to live Wikipedia — it reflects the state of Wikipedia at ingest time.
- Comparison quality depends on the content of retrieved chunks; not all entity pairs have clearly matched same-dimension facts.
- Streaming display is typewriter-effect only — the full answer is computed first, then rendered word by word.
- Two-model comparison requires a second Ollama generation model to be installed locally.
- Very long sentences containing abbreviations like `c.` (circa) are handled robustly, but highly unusual formatting in Wikipedia articles may occasionally produce incomplete excerpts.
