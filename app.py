import re
import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# ── Comparison answer sanitiser ───────────────────────────────────────────────

_DIFF_SECTION_RE = re.compile(
    r"(?i)(directly\s+supported\s+differences|supported\s+differences)\s*:",
)

_FALLBACK_SENTENCE = (
    "No directly comparable same-dimension differences were found in the retrieved excerpts."
)


def _sanitize_comparison_answer(text: str, entity_names: list[str] | None = None) -> str:
    """Clean up a comparison answer in-place; pass non-comparison answers through unchanged.

    Steps:
    1. Locate the diff section (either heading variant).
    2. Normalize heading to "Directly supported differences:".
    3. Remove bullets that lack "— vs —", have an empty side, or (when exactly 2
       entity names are known) do not contain both entity names.
    4. If all bullets were removed, insert the safe fallback sentence.
    5. Collapse triple+ newlines.
    """
    m = _DIFF_SECTION_RE.search(text)
    if not m:
        return text  # not a comparison answer

    # Normalize heading
    text = text[:m.start()] + "Directly supported differences:" + text[m.end():]

    # Re-locate section after normalization
    section_start = text.index("Directly supported differences:") + len("Directly supported differences:")
    # Find end of section (next heading-like line or end of string)
    section_end_m = re.search(r"\n\s*\n\s*[A-Z]", text[section_start:])
    section_end = section_start + section_end_m.start() if section_end_m else len(text)

    section_body = text[section_start:section_end]
    lines = section_body.split("\n")

    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Pass through blank lines and non-bullet lines unchanged
        if not stripped or not re.match(r"^[-*•]", stripped):
            kept.append(line)
            continue
        bullet_content = re.sub(r"^[-*•]\s*", "", stripped)
        # Must contain — vs —
        if "—" not in bullet_content or " vs " not in bullet_content.lower():
            if re.search(r"—\s*vs\s*—", bullet_content, re.IGNORECASE):
                pass
            else:
                continue
        parts = re.split(r"—\s*vs\s*—", bullet_content, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) != 2:
            continue
        left, right = parts[0].strip(), parts[1].strip()
        # Both sides must be non-empty
        if not left or not right:
            continue
        # Both entity names must appear (when exactly 2 known)
        if entity_names and len(entity_names) == 2:
            a_low, b_low = entity_names[0].lower(), entity_names[1].lower()
            bullet_low = bullet_content.lower()
            if a_low not in bullet_low or b_low not in bullet_low:
                continue
        kept.append(line)

    # Check if any real bullet survived
    has_real_bullet = any(
        re.match(r"^[-*•]", l.strip()) and "—" in l for l in kept
    )
    if not has_real_bullet:
        # Remove any existing fallback lines and insert canonical one
        kept = [l for l in kept if not re.search(
            r"no (?:safe difference|directly comparable)", l, re.IGNORECASE
        )]
        kept.append(f"- {_FALLBACK_SENTENCE}")

    new_section = "\n".join(kept)
    text = text[:section_start] + new_section + text[section_end:]
    return re.sub(r"\n{3,}", "\n\n", text).strip()

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
    debug_mode = st.checkbox("Debug mode (terminal output)", value=False)
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
            result = answer(prompt, top_k=top_k, debug=debug_mode)
        chunk_entities = list(dict.fromkeys(c["entity_name"] for c in result.get("chunks", [])))
        entity_names = chunk_entities if len(chunk_entities) == 2 else None
        display_answer = _sanitize_comparison_answer(result["answer"], entity_names)
        st.markdown(display_answer)
        st.caption(f"Route: `{result['route']}` | Chunks retrieved: {len(result['chunks'])}")
        if show_chunks and result["chunks"]:
            with st.expander("Retrieved chunks"):
                for i, chunk in enumerate(result["chunks"]):
                    st.markdown(f"**Chunk {i+1}** — {chunk['entity_name']} ({chunk['entity_type']}) | dist: {chunk['distance']:.4f}")
                    st.text(chunk["text"][:400] + "..." if len(chunk["text"]) > 400 else chunk["text"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": display_answer,
        "chunks": result["chunks"],
    })
