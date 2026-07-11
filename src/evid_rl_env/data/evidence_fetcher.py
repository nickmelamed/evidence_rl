import hashlib
import json
import os
import pickle
import sqlite3

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

os.makedirs("artifacts/cache", exist_ok=True)

_conn = sqlite3.connect("artifacts/cache/fetch_cache.sqlite3", check_same_thread=False)
_conn.execute("PRAGMA journal_mode=WAL")
_conn.execute(
    "CREATE TABLE IF NOT EXISTS cache "
    "(key TEXT PRIMARY KEY, value BLOB NOT NULL)"
)
_conn.commit()

_TAVILY_TIMEOUT = 8  # seconds; fail fast and fall back to seed data

# Set via use_snapshot() — {claim: {"search_query": ..., "results": [...]}}.
# Keyed on (claim, search_query) together (not claim alone) so a loaded
# snapshot only intercepts the deterministic reset()-time fetch (dataset's
# own search_query) and never QUERY's agent-generated follow-up searches,
# which aren't reproducible/snapshot-able in the first place.
_snapshot: dict | None = None


def use_snapshot(path: str) -> None:
    """Load a snapshot written by export_snapshot() — fetch_evidence() then
    prefers it over the sqlite cache/live Tavily call for matching claims."""
    global _snapshot
    with open(path) as f:
        _snapshot = json.load(f)
    print(f"[Snapshot] Loaded {len(_snapshot)} claims from {path}")


def export_snapshot(claims: list, output_path: str) -> None:
    """Dump this machine's cached Tavily results for `claims` to a portable
    JSON file, so a later run (anywhere) can reproduce the exact same
    evidence pool via use_snapshot() instead of depending on live Tavily
    results (which change over time) or a machine-local sqlite cache."""
    snapshot = {}
    for sample in claims:
        claim = sample["claim"]
        query = sample.get("search_query", claim)
        key = hashlib.md5(f"{query}:5".encode()).hexdigest()
        row = _conn.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        if row is not None:
            snapshot[claim] = {"search_query": query, "results": pickle.loads(row[0])}

    with open(output_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"[Snapshot] Exported {len(snapshot)}/{len(claims)} claims to {output_path}")


def fetch_evidence(claim: str, search_query: str, max_results: int = 5) -> list[dict]:
    if _snapshot is not None:
        entry = _snapshot.get(claim)
        if entry is not None and entry["search_query"] == search_query:
            return entry["results"]

    cache_key = hashlib.md5(f"{search_query}:{max_results}".encode()).hexdigest()

    row = _conn.execute("SELECT value FROM cache WHERE key=?", (cache_key,)).fetchone()
    if row is not None:
        return pickle.loads(row[0])

    api_key = os.getenv("TAVILY_API_KEY")
    client = TavilyClient(api_key=api_key)
    response = client.search(query=search_query, max_results=max_results, timeout=_TAVILY_TIMEOUT)
    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "score": r.get("score", 0.0),
        }
        for r in response.get("results", [])
    ]

    _conn.execute(
        "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
        (cache_key, pickle.dumps(results)),
    )
    _conn.commit()
    return results


def warm_cache(dataset: list) -> None:
    """Pre-fetch evidence for every claim in dataset, populating the SQLite cache.

    Called once at training startup so that env.reset() and _build_examples()
    never block on a cold Tavily call mid-training. Claims already cached are
    skipped. Failures are silently ignored — the env falls back to seed data.
    """
    missing = []
    for sample in dataset:
        query = sample.get("search_query", sample["claim"])
        key = hashlib.md5(f"{query}:5".encode()).hexdigest()
        row = _conn.execute("SELECT 1 FROM cache WHERE key=?", (key,)).fetchone()
        if row is None:
            missing.append(sample)

    if not missing:
        return

    print(f"[Tavily] Warming cache for {len(missing)} uncached claims...")
    for sample in missing:
        query = sample.get("search_query", sample["claim"])
        try:
            fetch_evidence(sample["claim"], query)
        except Exception:
            pass  # env will fall back to seed data for this claim
    print("[Tavily] Cache warm.")
