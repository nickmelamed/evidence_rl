import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


def fetch_evidence(claim: str, search_query: str, max_results: int = 5) -> list[dict]:
    api_key = os.getenv("TAVILY_API_KEY")
    client = TavilyClient(api_key=api_key)
    response = client.search(query=search_query, max_results=max_results)
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "score": r.get("score", 0.0),
        }
        for r in response.get("results", [])
    ]
