import os
import pickle
import sqlite3


class SQLiteCache:
    """
    Crash-safe key-value cache backed by sqlite3.

    Replaces shelve/gdbm which creates POSIX semaphores that leak on SIGSEGV
    and leave the database file corrupted for subsequent runs. sqlite3 uses WAL
    mode so the file is always internally consistent even after a hard crash.

    Shared by LLMJudge (judge_cache.sqlite3) and EvidenceLabeler
    (evidence_label_cache.sqlite3) — same cache shape, different content.
    """

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache "
            "(key TEXT PRIMARY KEY, value BLOB NOT NULL)"
        )
        self._conn.commit()

    def __contains__(self, key: str) -> bool:
        cur = self._conn.execute("SELECT 1 FROM cache WHERE key=?", (key,))
        return cur.fetchone() is not None

    def __getitem__(self, key: str):
        cur = self._conn.execute("SELECT value FROM cache WHERE key=?", (key,))
        row = cur.fetchone()
        if row is None:
            raise KeyError(key)
        return pickle.loads(row[0])

    def __setitem__(self, key: str, value) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
            (key, pickle.dumps(value)),
        )
        self._conn.commit()
