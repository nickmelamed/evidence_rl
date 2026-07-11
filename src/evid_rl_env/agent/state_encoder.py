import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    EMBEDDING_DIM = 384
    SEMANTIC_AVAILABLE = True
except ImportError:
    _MODEL = None
    EMBEDDING_DIM = 0
    SEMANTIC_AVAILABLE = False

# Text-level embedding cache. The same claim and evidence texts are encoded
# repeatedly within an episode (multiple policy calls per step) and across
# episodes (same eval samples reused). Caching at the text level eliminates
# redundant SentenceTransformer inference for any text already seen.
_embed_cache: dict = {}


def _encode_texts(texts: list) -> np.ndarray:
    """Batch-encode texts, using cache for any already seen."""
    missing_idxs = [i for i, t in enumerate(texts) if t not in _embed_cache]
    if missing_idxs:
        missing_texts = [texts[i] for i in missing_idxs]
        embs = _MODEL.encode(missing_texts, normalize_embeddings=True)
        for t, emb in zip(missing_texts, embs):
            _embed_cache[t] = emb
    return np.stack([_embed_cache[t] for t in texts])


def _encode_text(text: str) -> np.ndarray:
    if text not in _embed_cache:
        _embed_cache[text] = _MODEL.encode(text, normalize_embeddings=True)
    return _embed_cache[text]


def embed_text(text: str):
    """Public embedding accessor for callers outside the policy state encoder
    (e.g. RERANK's claim-similarity sort). Returns None if sentence-transformers
    isn't installed rather than raising, so callers can fall back gracefully."""
    if not SEMANTIC_AVAILABLE:
        return None
    return _encode_text(text)


def encode_state_semantic(state):
    """
    Returns a flat numpy float32 vector combining:
      - 4 structural features
      - claim embedding (384-dim)
      - mean of selected evidence embeddings (384-dim), zeros if none selected
      - mean of evidence pool embeddings (384-dim)
      - claim-vs-selected cosine similarity (1 scalar)
      - claim-vs-pool mean cosine similarity (1 scalar)

    Total dim when semantic available: 4 + 384*3 + 2 = 1158
    Falls back to original 4-dim if sentence-transformers not installed.
    """
    structural = np.array([
        len(state.selected_evidence) / 5.0,
        len(state.evidence_pool) / 10.0,
        len(state.debate_history) / 5.0,
        state.steps_taken / state.max_steps,
    ], dtype=np.float32)

    if not SEMANTIC_AVAILABLE or _MODEL is None:
        return structural

    claim_emb = _encode_text(state.claim)

    if state.selected_evidence:
        sel_embs = _encode_texts([e.text for e in state.selected_evidence])
        sel_mean = sel_embs.mean(axis=0)
        sel_sim = float(np.dot(claim_emb, sel_mean))
    else:
        sel_mean = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        sel_sim = 0.0

    if state.evidence_pool:
        pool_embs = _encode_texts([e.text for e in state.evidence_pool])
        pool_mean = pool_embs.mean(axis=0)
        pool_sim = float(np.dot(claim_emb, pool_mean))
    else:
        pool_mean = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        pool_sim = 0.0

    return np.concatenate([
        structural,
        claim_emb.astype(np.float32),
        sel_mean.astype(np.float32),
        pool_mean.astype(np.float32),
        np.array([sel_sim, pool_sim], dtype=np.float32),
    ])


def get_state_dim():
    if SEMANTIC_AVAILABLE:
        return 4 + EMBEDDING_DIM * 3 + 2
    return 4
