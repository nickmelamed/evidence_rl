import json
import logging
import os
import pickle
import re
import hashlib
import sqlite3
import numpy as np

_logger = logging.getLogger(__name__)


class _SQLiteCache:
    """
    Crash-safe key-value cache backed by sqlite3.

    Replaces shelve/gdbm which creates POSIX semaphores that leak on SIGSEGV
    and leave the database file corrupted for subsequent runs. sqlite3 uses WAL
    mode so the file is always internally consistent even after a hard crash.
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

    def sync(self) -> None:
        pass  # every __setitem__ commits immediately

    def close(self) -> None:
        self._conn.close()


class LLMJudge:
    def __init__(self, llm, weight=0.5, cache_scores=True):
        self.llm = llm
        self.weight = weight
        self.cache_scores = cache_scores
        self._cache = _SQLiteCache("artifacts/cache/judge_cache.sqlite3")

    def close(self):
        self._cache.close()

    def _cache_key(self, claim, reasoning, evidence):
        raw = claim + reasoning + "".join(e.text for e in evidence)
        return hashlib.md5(raw.encode()).hexdigest()

    def build_prompt(self, claim, reasoning, evidence):
        evidence_text = "\n".join([f"- {e.text}" for e in evidence])
        return (
            "You are an expert evaluator of reasoning quality. "
            "You must respond with ONLY a JSON object — no explanation, "
            "no markdown, no preamble.\n\n"
            f"CLAIM:\n{claim}\n\n"
            f"EVIDENCE:\n{evidence_text}\n\n"
            f"REASONING:\n{reasoning}\n\n"
            "Score the reasoning on exactly these five dimensions and "
            "return ONLY this JSON object with float values between 0 and 1:\n"
            '{"LCS": 0.0, "ESS": 0.0, "GRS": 0.0, "COMP": 0.0, "BIAS": 0.0, "confidence": 0.0}\n\n'
            "LCS = logical consistency (1=fully coherent)\n"
            "ESS = evidence support (1=correctly uses all evidence)\n"
            "GRS = grounding risk (1=reasoning introduces claims not in evidence, 0=fully grounded)\n"
            "COMP = completeness (1=fully addresses the claim)\n"
            "BIAS = selective citation bias (1=only cites supporting evidence while ignoring contradictions it was given, 0=balanced)\n"
            "confidence = your confidence in these scores (0-1)\n\n"
            "Respond with the JSON object only:"
        )

    def parse(self, response):
        try:
            return json.loads(response.strip())
        except (json.JSONDecodeError, AttributeError):
            pass

        match = re.search(r'\{[^{}]*"LCS"[^{}]*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        scores = {}
        for key in ["LCS", "ESS", "GRS", "COMP", "BIAS", "confidence"]:
            m = re.search(rf'"{key}"\s*:\s*([0-9.]+)', response)
            if m:
                try:
                    scores[key] = float(m.group(1))
                except ValueError:
                    pass

        if len(scores) >= 3:
            for key in ["LCS", "ESS", "GRS", "COMP", "BIAS", "confidence"]:
                scores.setdefault(key, 0.5)
            return scores

        return {"LCS": 0.5, "ESS": 0.5, "GRS": 0.5, "COMP": 0.5, "BIAS": 0.5, "confidence": 0.5}

    def compute_reward(self, claim, reasoning, evidence):
        if not reasoning.strip():
            return 0.0, {"LCS": 0.0, "ESS": 0.0, "GRS": 0.5, "COMP": 0.0, "BIAS": 0.5, "confidence": 0.0}

        if self.cache_scores:
            key = self._cache_key(claim, reasoning, evidence)
            if key in self._cache:
                scores = self._cache[key]
                return self._scores_to_reward(scores), scores

        prompt = self.build_prompt(claim, reasoning, evidence)
        try:
            if hasattr(self.llm, "generate_structured"):
                response, _ = self.llm.generate_structured(prompt)
            else:
                response, _ = self.llm.generate(prompt)
        except Exception as exc:
            _logger.warning(
                "LLMJudge: generation failed (%s) | model=%s | claim='%.80s' — "
                "returning neutral fallback scores.",
                exc,
                getattr(self.llm, "model_name", "unknown"),
                claim,
            )
            scores = {"LCS": 0.5, "ESS": 0.5, "GRS": 0.5, "COMP": 0.5, "BIAS": 0.5, "confidence": 0.0}
            if self.cache_scores:
                self._cache[key] = scores
            return self._scores_to_reward(scores), scores

        scores = self.parse(response)
        if self.cache_scores:
            self._cache[key] = scores

        return self._scores_to_reward(scores), scores

    def _scores_to_reward(self, scores):
        lcs = float(scores.get("LCS", 0.5))
        ess = float(scores.get("ESS", 0.5))
        grs = float(scores.get("GRS", 0.5))
        comp = float(scores.get("COMP", 0.5))
        bias = float(scores.get("BIAS", 0.5))
        conf = float(scores.get("confidence", 0.5))

        reward = (
            0.30 * lcs
            + 0.25 * ess
            + 0.20 * comp
            - 0.25 * grs
            - 0.15 * bias
        )
        if conf < 0.4:
            reward = 0.5 * reward + 0.5 * 0.5
        return float(np.clip(reward, 0.0, 1.0))
