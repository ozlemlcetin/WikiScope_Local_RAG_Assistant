import requests
import time

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "WikiScope-RAG/1.0 (university homework; ozlemcetin)"}
REQUEST_DELAY = 2.0
MAX_RETRIES = 5

def _fetch_one_title(title: str) -> str:
    """Fetch extract for a single title with retry/backoff. Raises on failure or missing page."""
    params = {
        "action": "query",
        "prop": "extracts",
        "exlimit": 1,
        "titles": title,
        "explaintext": True,
        "exsectionformat": "plain",
        "format": "json",
    }
    backoff = 2.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(WIKIPEDIA_API, params=params, headers=HEADERS, timeout=30)
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = float(resp.headers.get("Retry-After", backoff))
                wait = max(retry_after, backoff)
                print(f"\n    [{resp.status_code}] waiting {wait:.0f}s (attempt {attempt}/{MAX_RETRIES})...", end=" ", flush=True)
                time.sleep(wait)
                backoff = min(backoff * 2, 60)
                continue
            resp.raise_for_status()
            data = resp.json()
            pages = data["query"]["pages"]
            page = next(iter(pages.values()))
            if "missing" in page:
                raise ValueError(f"Wikipedia page not found: {title}")
            text = page.get("extract", "")
            if not text:
                raise ValueError(f"Empty extract for: {title}")
            time.sleep(REQUEST_DELAY)
            return text
        except (requests.ConnectionError, requests.Timeout) as e:
            print(f"\n    [network error] {e} — retrying in {backoff:.0f}s...", end=" ", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
    raise RuntimeError(f"Failed to fetch '{title}' after {MAX_RETRIES} attempts")

def fetch_wikipedia_text(title: str, fallback_titles: list[str] | None = None) -> str:
    """Try primary title, then each fallback title in order."""
    titles_to_try = [title] + (fallback_titles or [])
    last_error: Exception = RuntimeError("No titles to try")
    for t in titles_to_try:
        try:
            return _fetch_one_title(t)
        except (ValueError, RuntimeError) as e:
            last_error = e
            if t != titles_to_try[-1]:
                print(f"\n    [fallback] '{t}' failed ({e}), trying next title...", end=" ", flush=True)
    raise last_error
