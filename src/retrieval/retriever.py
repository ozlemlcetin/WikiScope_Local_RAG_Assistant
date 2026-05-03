import json
from src.config import TOP_K, PROCESSED_DIR
from src.generation.ollama_client import embed_text
from src.storage.vector_store import query_collection, get_chunks_by_entity_name
from src.retrieval.query_router import route_query, get_matched_entities, get_location_keywords

# Module-level cache: loaded once per process on first lexical search call
_chunks_cache: list[dict] | None = None


def _load_chunks() -> list[dict]:
    global _chunks_cache
    if _chunks_cache is None:
        path = PROCESSED_DIR / "chunks.jsonl"
        if not path.exists():
            _chunks_cache = []
        else:
            with path.open(encoding="utf-8") as f:
                _chunks_cache = [json.loads(line) for line in f if line.strip()]
    return _chunks_cache


def _lexical_location_search(
    location_keywords: set[str],
    entity_type: str = "place",
    top_n: int = 3,
    debug: bool = False,
) -> list[dict]:
    """
    Scan chunks.jsonl for chunks whose text or entity_name contains a location keyword.
    Score by hit count and early appearance, return top_n as retriever-compatible dicts.
    """
    all_chunks = _load_chunks()
    candidates: list[tuple] = []  # (score_key, chunk_dict)

    for raw in all_chunks:
        if raw.get("entity_type") != entity_type:
            continue
        text = raw.get("text", "")
        text_lower = text.lower()
        entity_lower = raw.get("entity_name", "").lower()

        hits = [kw for kw in location_keywords if kw in text_lower or kw in entity_lower]
        if not hits:
            continue

        hit_count = len(hits)
        # Bonus: keyword in first 500 chars means it's in the intro (most informative)
        early_hit = any(kw in text_lower[:500] for kw in hits)

        sort_key = (-hit_count, 0 if early_hit else 1, raw.get("chunk_index", 0))
        candidates.append((sort_key, {
            "text": text,
            "entity_name": raw.get("entity_name", ""),
            "entity_type": raw.get("entity_type", ""),
            "chunk_index": raw.get("chunk_index", 0),
            "distance": 0.1,  # sentinel: lexical match, not vector distance
        }))

    candidates.sort(key=lambda x: x[0])
    results = [item[1] for item in candidates[:top_n]]

    if debug:
        print(f"[DEBUG] lexical_location_candidates: {len(candidates)} found, "
              f"injecting top {len(results)}")
        for c in results:
            print(f"[DEBUG]   lexical: entity={c['entity_name']!r} "
                  f"chunk_index={c['chunk_index']} dist={c['distance']}")

    return results


def _rerank(
    chunks: list[dict],
    route: str,
    location_keywords: set[str],
    debug: bool = False,
) -> list[dict]:
    """Sort chunks: entity-boosted first, lexical-injected second, type-match, kw-hits, distance."""
    scored = []
    for c in chunks:
        text_lower = c["text"].lower()
        kw_hits = [kw for kw in location_keywords if kw in text_lower]
        is_entity_boosted = c["distance"] == 0.0
        is_lexical = c["distance"] == 0.1
        type_match = (route == "places" and c["entity_type"] == "place") or \
                     (route == "people" and c["entity_type"] == "person") or \
                     route == "both"
        sort_key = (
            0 if is_entity_boosted else (1 if is_lexical else 2),  # boost tiers
            0 if type_match else 1,
            -len(kw_hits),
            c["distance"],
        )
        scored.append((sort_key, kw_hits, c))
        if debug and kw_hits:
            print(f"[DEBUG]   rerank boost: entity={c['entity_name']!r} "
                  f"kw_hits={kw_hits} dist={c['distance']:.4f}")
    scored.sort(key=lambda x: x[0])
    return [item[2] for item in scored]


def retrieve(query: str, top_k: int = TOP_K, debug: bool = False) -> list[dict]:
    route = route_query(query)
    matched_entities = get_matched_entities(query)
    location_keywords = get_location_keywords(query)
    is_comparison = any(kw in query.lower() for kw in
                        {"compare", "vs", "versus", "difference", "similarities", "similar", "contrast"})

    if debug:
        print(f"\n[DEBUG] query           : {query!r}")
        print(f"[DEBUG] route           : {route}")
        print(f"[DEBUG] comparison      : {is_comparison}")
        print(f"[DEBUG] matched_entities: {matched_entities}")
        print(f"[DEBUG] location_kw     : {location_keywords}")

    seen: set[str] = set()
    chunks: list[dict] = []

    # 1. Entity-name boosting (distance=0.0): guarantee chunks for named entities.
    #    For comparison queries, fetch more chunks per entity so each side is well-represented.
    boost_per_entity = 4 if is_comparison else 3
    for entity_name in matched_entities:
        boosted = get_chunks_by_entity_name(entity_name, top_k=boost_per_entity)
        for c in boosted:
            key = f"{c['entity_name']}_{c['chunk_index']}"
            if key not in seen:
                seen.add(key)
                chunks.append(c)

    # 2. Lexical location injection (distance=0.1): find place chunks containing
    #    country/city keywords that vector search would miss
    if route == "places" and location_keywords:
        lexical = _lexical_location_search(location_keywords, entity_type="place",
                                           top_n=3, debug=debug)
        for c in lexical:
            key = f"{c['entity_name']}_{c['chunk_index']}"
            if key not in seen:
                seen.add(key)
                chunks.append(c)

    # 3. Vector search
    embedding = embed_text(query)
    if route == "people":
        results = query_collection(embedding, top_k=top_k, entity_type_filter="person")
    elif route == "places":
        results = query_collection(embedding, top_k=top_k, entity_type_filter="place")
    else:
        results = query_collection(embedding, top_k=top_k, entity_type_filter=None)

    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            key = f"{meta.get('entity_name', '')}_{meta.get('chunk_index', 0)}"
            if key not in seen:
                seen.add(key)
                chunks.append({
                    "text": doc,
                    "entity_name": meta.get("entity_name", ""),
                    "entity_type": meta.get("entity_type", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "distance": dist,
                })

    if debug:
        print(f"[DEBUG] pre-rerank chunk count: {len(chunks)}")

    # 4. Rerank then truncate.
    #    For comparison queries, protect all entity-boosted chunks from truncation so
    #    each entity keeps its dedicated context even when top_k is small.
    chunks = _rerank(chunks, route, location_keywords, debug=debug)
    if is_comparison and matched_entities:
        n_boosted = sum(1 for c in chunks if c["distance"] == 0.0)
        effective_k = max(top_k, n_boosted)
    else:
        effective_k = top_k
    chunks = chunks[:effective_k]

    if debug:
        print(f"[DEBUG] final chunk order ({len(chunks)} chunks):")
        for i, c in enumerate(chunks):
            preview = c["text"][:150].replace("\n", " ")
            print(f"[DEBUG]   [{i}] entity={c['entity_name']!r} "
                  f"type={c['entity_type']!r} dist={c['distance']:.4f} | {preview!r}")

    return chunks
