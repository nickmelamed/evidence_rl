"""
One-time (but idempotent/re-runnable) data migration: backfills real
SciFact evidence into seed_claims.json.

seed_claims.json's claims were originally extracted from SciFact (Wadden et
al. 2020) outside this repo, carrying over only claim text and label — the
actual evidence abstracts SciFact's human annotators labeled each claim
against were never included, leaving evidence sourced entirely from live
Tavily web search at runtime (see environment/environment.py's ClaimEnv.reset).
This script re-downloads the original SciFact release directly (the same
URL allenai/scifact's own HuggingFace loading script fetches — no `datasets`
package dependency needed) and attaches each claim's real cited-document
evidence, matched by the numeric SciFact claim ID already preserved in each
entry's `notes` field.

Usage: python -m evid_rl_env.data.backfill_scifact_evidence
"""

import json
import tarfile
import tempfile
import urllib.request
from pathlib import Path

_SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
_SEED_CLAIMS_PATH = Path(__file__).parent / "seed_claims.json"


def _scifact_id(claim_id: str) -> int | None:
    """'scifact_306' -> 306; returns None for non-SciFact-shaped ids
    (so this script is safe to re-run against a dataset that later mixes
    in claims from other sources)."""
    if not claim_id.startswith("scifact_"):
        return None
    try:
        return int(claim_id.split("_", 1)[1])
    except ValueError:
        return None


def _load_scifact_claims_and_corpus(data_dir: Path) -> tuple[dict, dict]:
    claims = {}
    for split in ("claims_train.jsonl", "claims_dev.jsonl"):
        with open(data_dir / split) as f:
            for line in f:
                c = json.loads(line)
                claims[c["id"]] = c

    corpus = {}
    with open(data_dir / "corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["doc_id"]] = d

    return claims, corpus


def _build_evidence(scifact_claim: dict, corpus: dict) -> list[dict]:
    """One Evidence-shaped dict per cited document (title + full abstract,
    not just the human-highlighted rationale sentences — the agent should
    still have to find the relevant part itself, same as with a real
    fetched article)."""
    evidence = []
    for doc_id_str, groups in scifact_claim.get("evidence", {}).items():
        doc = corpus[int(doc_id_str)]
        label = groups[0]["label"].lower()  # "SUPPORT"/"CONTRADICT" -> "support"/"contradict"
        text = f"{doc['title']}\n\n{' '.join(doc['abstract'])}"
        evidence.append({"text": text, "label": label})
    return evidence


def main() -> None:
    seed_claims = json.loads(_SEED_CLAIMS_PATH.read_text())

    print(f"Downloading SciFact release from {_SCIFACT_URL} ...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive_path = tmp_path / "data.tar.gz"
        urllib.request.urlretrieve(_SCIFACT_URL, archive_path)
        with tarfile.open(archive_path) as tar:
            tar.extractall(tmp_path, filter="data")

        claims, corpus = _load_scifact_claims_and_corpus(tmp_path / "data")

    matched = 0
    skipped_no_id = 0
    skipped_not_found = 0
    for entry in seed_claims:
        num_id = _scifact_id(entry["id"])
        if num_id is None:
            skipped_no_id += 1
            continue
        scifact_claim = claims.get(num_id)
        if scifact_claim is None or not scifact_claim.get("evidence"):
            skipped_not_found += 1
            continue
        entry["evidence"] = _build_evidence(scifact_claim, corpus)
        matched += 1

    _SEED_CLAIMS_PATH.write_text(json.dumps(seed_claims, indent=2) + "\n")

    print(f"Matched and backfilled evidence for {matched}/{len(seed_claims)} claims.")
    if skipped_no_id:
        print(f"  {skipped_no_id} claims skipped (non-SciFact id, left unchanged).")
    if skipped_not_found:
        print(f"  {skipped_not_found} claims skipped (no matching SciFact evidence found, left unchanged).")


if __name__ == "__main__":
    main()
