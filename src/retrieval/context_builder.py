def build_context(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    parts = []
    seen = set()
    for chunk in chunks:
        key = f"{chunk['entity_name']}_{chunk['chunk_index']}"
        if key in seen:
            continue
        seen.add(key)
        header = f"[{chunk['entity_name']} ({chunk['entity_type']})]"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)
