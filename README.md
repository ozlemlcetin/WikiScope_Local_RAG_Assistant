# WikiScope Local RAG Assistant

A fully local Wikipedia RAG (Retrieval-Augmented Generation) system that answers questions about famous people and places. No external LLM APIs — everything runs on your Mac.

## Tech Stack

- **Python 3.11+**
- **Ollama** — local LLM runtime
- **llama3.2:3b** — answer generation
- **nomic-embed-text** — local embeddings
- **ChromaDB** — vector database
- **SQLite** — metadata storage
- **Streamlit** — chat UI

## Installation (macOS)

### 1. Clone and set up Python environment

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

### 3. Pull required models

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

Downloads 40 Wikipedia articles (20 people + 20 places) to `data/raw/`.

### 6. Build the vector database

```bash
python scripts/build_vector_db.py
```

Chunks, embeds, and stores all documents in ChromaDB and SQLite.

### 7. Launch the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Example Questions

**People:**
- Who was Albert Einstein and what did he discover?
- What are the main contributions of Marie Curie?
- Tell me about Ada Lovelace's work.
- How many Ballon d'Or awards has Lionel Messi won?

**Places:**
- Where is the Eiffel Tower located and when was it built?
- How tall is Mount Everest?
- What is the history of the Colosseum?

**Comparisons:**
- Compare Einstein and Newton.
- What are the similarities between the Taj Mahal and the Pyramids of Giza?

## Failure Cases

- Questions outside the 40 ingested entities → "I don't know based on the local Wikipedia data."
- Very recent events (post Wikipedia snapshot) → may not have data.
- Ambiguous queries → routed to "both" category, may return less relevant results.

## Reset

```bash
# Reset DB only (keeps raw Wikipedia files):
python scripts/reset_system.py

# Full reset (re-downloads everything):
python scripts/reset_system.py --full
```
