import re
import time
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
from src.entities import PEOPLE, PLACES

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
    stream_display = st.checkbox("Stream response display", value=False)
    debug_mode = st.checkbox("Debug mode (terminal output)", value=False)
    st.divider()

    # ── Two-model comparison ───────────────────────────────────────────────
    _EMBED_SUBSTRINGS = ("embed", "nomic-embed", "mxbai-embed", "bge")

    def _is_generation_model_candidate(model_name: str) -> bool:
        """Return True if the model is likely a text-generation model, not embedding-only."""
        lower = model_name.lower()
        return not any(sub in lower for sub in _EMBED_SUBSTRINGS)

    st.subheader("Model comparison")
    compare_models = st.checkbox("Compare two local models", value=False)
    second_model: str | None = None
    if compare_models:
        all_models = list_local_models()
        generation_models = [
            m for m in all_models
            if m != LLM_MODEL and m != EMBED_MODEL and _is_generation_model_candidate(m)
        ]
        if generation_models:
            second_model = st.selectbox("Second generation model", generation_models)
        else:
            st.warning(
                "No second local generation model found. "
                "Install another Ollama chat model to enable comparison. "
                "Example: `ollama pull phi3:mini` or `ollama pull mistral`"
            )
    st.divider()

    # ── Example questions ──────────────────────────────────────────────────
    st.subheader("Try an example")
    _EXAMPLES = [
        "Who was Albert Einstein and what is he known for?",
        "What did Marie Curie discover?",
        "Which famous place is located in Turkey?",
        "Compare Hagia Sophia and Pyramids of Giza.",
        "Who is the president of Mars?",
    ]
    for ex in _EXAMPLES:
        if st.button(ex, key=f"ex_{ex}"):
            st.session_state["pending_prompt"] = ex
    st.divider()

    # ── Dataset coverage ───────────────────────────────────────────────────
    with st.expander("Local dataset coverage"):
        st.markdown(f"**People ({len(PEOPLE)})**")
        st.markdown(", ".join(p.name for p in PEOPLE))
        st.markdown(f"**Places ({len(PLACES)})**")
        st.markdown(", ".join(p.name for p in PLACES))
    st.divider()

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# --- Main ---
st.title("WikiScope Local RAG Assistant")
st.caption("Ask questions about famous people and places — answers from local Wikipedia data only.")

if "messages" not in st.session_state:
    st.session_state.messages = []

def _stream_markdown(text: str):
    """Yield words one at a time for typewriter-style display via st.write_stream."""
    for word in text.split(" "):
        yield word + " "


def _render_sources(chunks: list[dict]) -> None:
    """Render a clean sources panel inside an expander."""
    with st.expander("Sources used"):
        for i, chunk in enumerate(chunks):
            dist = chunk["distance"]
            dist_str = f"{dist:.4f}" if dist not in (0.0, 0.1) else ("boosted" if dist == 0.0 else "lexical")
            st.markdown(
                f"**Source {i + 1}** — {chunk['entity_name']} "
                f"({chunk['entity_type']}) | distance: {dist_str}"
            )
            snippet = chunk["text"][:500]
            if len(chunk["text"]) > 500:
                snippet += "…"
            st.caption(snippet)
            if i < len(chunks) - 1:
                st.divider()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            meta = msg["meta"]
            st.caption(
                f"Route: `{meta['route']}` | "
                f"Chunks: {meta['chunks']} | "
                f"Time: {meta['elapsed']:.2f}s | "
                f"Evidence: {meta.get('evidence', '—')}"
            )
        if msg.get("chunks") and show_chunks:
            _render_sources(msg["chunks"])

def _evidence_signal(answer_text: str, chunks: list[dict]) -> str:
    if answer_text.startswith("I don't know based on the local Wikipedia data"):
        return "insufficient local evidence"
    if any(c.get("distance") == 0.0 for c in chunks):
        return "entity match"
    if any(c.get("distance") == 0.1 for c in chunks):
        return "lexical match"
    return "vector search"


if not ollama_ok:
    st.warning("Ollama is not running. Start it with: `ollama serve`")

# Consume a pending_prompt set by an example button, or fall back to chat input.
_pending = st.session_state.pop("pending_prompt", None)
prompt = _pending or st.chat_input("Ask about a person or place...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            t0 = time.perf_counter()
            result = answer(prompt, top_k=top_k, debug=debug_mode)
            elapsed = time.perf_counter() - t0

        chunk_entities = list(dict.fromkeys(c["entity_name"] for c in result.get("chunks", [])))
        entity_names = chunk_entities if len(chunk_entities) == 2 else None
        display_answer = _sanitize_comparison_answer(result["answer"], entity_names)
        evidence = _evidence_signal(display_answer, result["chunks"])

        # Display: streaming typewriter or plain markdown
        if stream_display:
            st.write_stream(_stream_markdown(display_answer))
        else:
            st.markdown(display_answer)

        st.caption(
            f"Route: `{result['route']}` | "
            f"Chunks: {len(result['chunks'])} | "
            f"Time: {elapsed:.2f}s | "
            f"Evidence: {evidence}"
        )
        if show_chunks and result["chunks"]:
            _render_sources(result["chunks"])

        # Optional second-model comparison
        if compare_models and second_model:
            with st.expander(f"Second model answer — `{second_model}`"):
                try:
                    with st.spinner(f"Querying {second_model}…"):
                        t1 = time.perf_counter()
                        result2 = answer(
                            prompt, top_k=top_k, debug=False, model_name=second_model
                        )
                        elapsed2 = time.perf_counter() - t1
                    display2 = _sanitize_comparison_answer(result2["answer"], entity_names)
                    evidence2 = _evidence_signal(display2, result2["chunks"])
                    st.markdown(display2)
                    st.caption(
                        f"Model: `{second_model}` | "
                        f"Route: `{result2['route']}` | "
                        f"Chunks: {len(result2['chunks'])} | "
                        f"Time: {elapsed2:.2f}s | "
                        f"Evidence: {evidence2}"
                    )
                except Exception as exc:
                    st.error(f"Second model failed: {exc}")

    meta = {
        "route": result["route"],
        "chunks": len(result["chunks"]),
        "elapsed": elapsed,
        "evidence": evidence,
    }
    st.session_state.messages.append({
        "role": "assistant",
        "content": display_answer,
        "chunks": result["chunks"],
        "meta": meta,
    })
