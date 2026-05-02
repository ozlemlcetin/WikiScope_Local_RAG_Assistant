from src.retrieval.retriever import retrieve
from src.retrieval.context_builder import build_context
from src.generation.ollama_client import generate_text
from src.retrieval.query_router import route_query

FALLBACK = "I don't know based on the local Wikipedia data."

def build_prompt(query: str, context: str) -> str:
    return f"""You are WikiScope, a helpful assistant that answers questions strictly using the provided Wikipedia context.

RULES:
- Answer ONLY based on the context below.
- Do NOT use any external knowledge or make things up.
- If the context does not contain enough information to answer, respond exactly with:
  "I don't know based on the local Wikipedia data."
- Be concise and factual.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

def answer(query: str, top_k: int = 5) -> dict:
    route = route_query(query)
    chunks = retrieve(query, top_k=top_k)
    context = build_context(chunks)

    if not context.strip():
        return {"answer": FALLBACK, "chunks": [], "route": route, "context": ""}

    prompt = build_prompt(query, context)
    response = generate_text(prompt)

    if not response or len(response.strip()) < 5:
        response = FALLBACK

    return {
        "answer": response,
        "chunks": chunks,
        "route": route,
        "context": context,
    }
