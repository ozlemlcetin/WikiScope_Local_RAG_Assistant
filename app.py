import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.generation.answer_generator import answer
from src.generation.ollama_client import check_ollama_health, list_local_models
from src.storage.vector_store import collection_count
from src.config import LLM_MODEL, EMBED_MODEL, TOP_K

st.set_page_config(page_title="WikiScope", page_icon="🔍", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("WikiScope")
    st.caption("Local Wikipedia RAG Assistant")
    st.divider()
    st.subheader("System Status")
    ollama_ok = check_ollama_health()
    st.write("Ollama:", "✅ Running" if ollama_ok else "❌ Not running")
    try:
        count = collection_count()
        st.write(f"Vectors in DB: {count:,}")
    except Exception:
        st.write("Vectors in DB: ⚠️ Not built yet")
    st.divider()
    st.subheader("Configuration")
    st.write(f"LLM: `{LLM_MODEL}`")
    st.write(f"Embeddings: `{EMBED_MODEL}`")
    top_k = st.slider("Top-K chunks", min_value=1, max_value=10, value=TOP_K)
    show_chunks = st.checkbox("Show retrieved chunks", value=False)
    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# --- Main ---
st.title("WikiScope Local RAG Assistant")
st.caption("Ask questions about famous people and places — answers from local Wikipedia data only.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chunks") and show_chunks:
            with st.expander("Retrieved chunks"):
                for i, chunk in enumerate(msg["chunks"]):
                    st.markdown(f"**Chunk {i+1}** — {chunk['entity_name']} ({chunk['entity_type']}) | dist: {chunk['distance']:.4f}")
                    st.text(chunk["text"][:400] + "..." if len(chunk["text"]) > 400 else chunk["text"])

if not ollama_ok:
    st.warning("Ollama is not running. Start it with: `ollama serve`")

if prompt := st.chat_input("Ask about a person or place..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = answer(prompt, top_k=top_k)
        st.markdown(result["answer"])
        st.caption(f"Route: `{result['route']}` | Chunks retrieved: {len(result['chunks'])}")
        if show_chunks and result["chunks"]:
            with st.expander("Retrieved chunks"):
                for i, chunk in enumerate(result["chunks"]):
                    st.markdown(f"**Chunk {i+1}** — {chunk['entity_name']} ({chunk['entity_type']}) | dist: {chunk['distance']:.4f}")
                    st.text(chunk["text"][:400] + "..." if len(chunk["text"]) > 400 else chunk["text"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "chunks": result["chunks"],
    })
