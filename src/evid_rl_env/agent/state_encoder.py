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


def encode_state_semantic(state):
    """
    Returns a flat numpy float32 vector combining:
      - 4 structural features (same as original encode_state)
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

    claim_emb = _MODEL.encode(state.claim, normalize_embeddings=True)

    if state.selected_evidence:
        sel_embs = _MODEL.encode(
            [e.text for e in state.selected_evidence],
            normalize_embeddings=True
        )
        sel_mean = sel_embs.mean(axis=0)
        sel_sim = float(np.dot(claim_emb, sel_mean))
    else:
        sel_mean = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        sel_sim = 0.0

    if state.evidence_pool:
        pool_embs = _MODEL.encode(
            [e.text for e in state.evidence_pool],
            normalize_embeddings=True
        )
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
    """Returns the state dimension based on what's available."""
    if SEMANTIC_AVAILABLE:
        return 4 + EMBEDDING_DIM * 3 + 2
    return 4
