from src.config import TOP_K
from src.generation.ollama_client import embed_text
from src.storage.vector_store import query_collection
from src.retrieval.query_router import route_query

def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    route = route_query(query)
    embedding = embed_text(query)
    if route == "people":
        results = query_collection(embedding, top_k=top_k, entity_type_filter="person")
    elif route == "places":
        results = query_collection(embedding, top_k=top_k, entity_type_filter="place")
    else:
        results = query_collection(embedding, top_k=top_k, entity_type_filter=None)

    chunks = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text": doc,
                "entity_name": meta.get("entity_name", ""),
                "entity_type": meta.get("entity_type", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "distance": dist,
            })
    return chunks
