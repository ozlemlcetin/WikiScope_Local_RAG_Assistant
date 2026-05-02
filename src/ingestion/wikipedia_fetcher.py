import requests
import time

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

def fetch_wikipedia_text(title: str) -> str:
    params = {
        "action": "query",
        "prop": "extracts",
        "exlimit": 1,
        "titles": title,
        "explaintext": True,
        "exsectionformat": "plain",
        "format": "json",
    }
    headers = {"User-Agent": "WikiScope-RAG/1.0 (homework project)"}
    resp = requests.get(WIKIPEDIA_API, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    if "missing" in page:
        raise ValueError(f"Wikipedia page not found: {title}")
    text = page.get("extract", "")
    if not text:
        raise ValueError(f"Empty extract for: {title}")
    time.sleep(0.5)
    return text
