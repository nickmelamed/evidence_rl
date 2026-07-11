"""
conftest.py's `no_network_fetch` autouse fixture monkeypatches
evidence_fetcher.fetch_evidence itself (so the rest of the suite never hits
Tavily) — these tests are specifically about that function's real behavior,
so they call a reference captured at import time, before any fixture has a
chance to patch the module attribute.
"""
import hashlib
import json
import pickle
import sqlite3

from evid_rl_env.data import evidence_fetcher
from evid_rl_env.data.evidence_fetcher import fetch_evidence as real_fetch_evidence


def _seed_cache_entry(conn, search_query, max_results, results):
    key = hashlib.md5(f"{search_query}:{max_results}".encode()).hexdigest()
    conn.execute(
        "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
        (key, pickle.dumps(results)),
    )
    conn.commit()


def isolated_cache(tmp_path, monkeypatch):
    """Point the module's sqlite connection at a throwaway file so these
    tests never touch the real artifacts/cache/fetch_cache.sqlite3."""
    conn = sqlite3.connect(str(tmp_path / "fetch_cache.sqlite3"), check_same_thread=False)
    conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value BLOB NOT NULL)")
    conn.commit()
    monkeypatch.setattr(evidence_fetcher, "_conn", conn)
    monkeypatch.setattr(evidence_fetcher, "_snapshot", None)
    return conn


def test_export_snapshot_round_trips_cached_results(tmp_path, monkeypatch):
    conn = isolated_cache(tmp_path, monkeypatch)
    claims = [{"claim": "Claim A", "search_query": "query a"}]
    fake_results = [{"title": "t", "url": "u", "content": "c", "score": 0.9}]
    _seed_cache_entry(conn, "query a", 5, fake_results)

    out_path = tmp_path / "snapshot.json"
    evidence_fetcher.export_snapshot(claims, str(out_path))

    with open(out_path) as f:
        snapshot = json.load(f)
    assert snapshot["Claim A"]["search_query"] == "query a"
    assert snapshot["Claim A"]["results"] == fake_results


def test_export_snapshot_skips_uncached_claims(tmp_path, monkeypatch):
    isolated_cache(tmp_path, monkeypatch)
    claims = [{"claim": "Never fetched", "search_query": "nope"}]
    out_path = tmp_path / "snapshot.json"
    evidence_fetcher.export_snapshot(claims, str(out_path))

    with open(out_path) as f:
        snapshot = json.load(f)
    assert snapshot == {}


def test_fetch_evidence_prefers_snapshot_over_cache_and_network(tmp_path, monkeypatch):
    conn = isolated_cache(tmp_path, monkeypatch)
    _seed_cache_entry(
        conn, "query a", 5, [{"title": "from-cache", "url": "", "content": "", "score": 0}]
    )

    snapshot_path = tmp_path / "snapshot.json"
    snapshot_results = [{"title": "from-snapshot", "url": "", "content": "", "score": 0}]
    with open(snapshot_path, "w") as f:
        json.dump({"Claim A": {"search_query": "query a", "results": snapshot_results}}, f)
    evidence_fetcher.use_snapshot(str(snapshot_path))

    def _boom(*a, **kw):
        raise AssertionError("TavilyClient should not be constructed when the snapshot hits")
    monkeypatch.setattr(evidence_fetcher, "TavilyClient", _boom)

    result = real_fetch_evidence("Claim A", "query a")
    assert result == snapshot_results


def test_fetch_evidence_ignores_snapshot_for_a_different_search_query(tmp_path, monkeypatch):
    """QUERY's agent-generated search text must not accidentally hit a
    snapshot entry recorded for the claim's original reset()-time query."""
    conn = isolated_cache(tmp_path, monkeypatch)

    snapshot_path = tmp_path / "snapshot.json"
    with open(snapshot_path, "w") as f:
        json.dump({"Claim A": {"search_query": "original query", "results": []}}, f)
    evidence_fetcher.use_snapshot(str(snapshot_path))

    _seed_cache_entry(
        conn, "a different agent-generated query", 5,
        [{"title": "from-cache", "url": "", "content": "", "score": 0}],
    )

    result = real_fetch_evidence("Claim A", "a different agent-generated query")
    assert result == [{"title": "from-cache", "url": "", "content": "", "score": 0}]
