import chromadb
from chromadb.config import Settings
from src.config import CHROMA_DIR, CHROMA_COLLECTION

def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))

def get_or_create_collection(client: chromadb.PersistentClient):
    return client.get_or_create_collection(name=CHROMA_COLLECTION)

def add_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    ids = [c["chunk_id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    metas = [{"entity_name": c["entity_name"], "entity_type": c["entity_type"],
               "chunk_index": c["chunk_index"]} for c in chunks]
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            documents=docs[i:i+batch_size],
            metadatas=metas[i:i+batch_size],
        )
    print(f"Added {len(ids)} vectors to Chroma collection '{CHROMA_COLLECTION}'")

def query_collection(query_embedding: list[float], top_k: int = 5, entity_type_filter: str | None = None) -> dict:
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    where = {"entity_type": entity_type_filter} if entity_type_filter else None
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return results

def get_chunks_by_entity_name(entity_name: str, top_k: int = 3) -> list[dict]:
    """Directly fetch stored chunks for a specific entity by metadata filter."""
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    results = collection.get(
        where={"entity_name": entity_name},
        include=["documents", "metadatas"],
        limit=top_k,
    )
    chunks = []
    if results["documents"]:
        for doc, meta in zip(results["documents"], results["metadatas"]):
            chunks.append({
                "text": doc,
                "entity_name": meta.get("entity_name", ""),
                "entity_type": meta.get("entity_type", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "distance": 0.0,
            })
    return chunks

def collection_count() -> int:
    client = get_chroma_client()
    col = get_or_create_collection(client)
    return col.count()
