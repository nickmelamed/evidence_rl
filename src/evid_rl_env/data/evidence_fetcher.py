import os
import shelve
import hashlib
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

os.makedirs("artifacts/cache", exist_ok=True)
_fetch_cache = shelve.open("artifacts/cache/fetch_cache", writeback=False)


def fetch_evidence(claim: str, search_query: str, max_results: int = 5) -> list[dict]:
    cache_key = hashlib.md5(f"{search_query}:{max_results}".encode()).hexdigest()
    if cache_key in _fetch_cache:
        return _fetch_cache[cache_key]

    api_key = os.getenv("TAVILY_API_KEY")
    client = TavilyClient(api_key=api_key)
    response = client.search(query=search_query, max_results=max_results)
    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "score": r.get("score", 0.0),
        }
        for r in response.get("results", [])
    ]
    _fetch_cache[cache_key] = results
    _fetch_cache.sync()
    return results
