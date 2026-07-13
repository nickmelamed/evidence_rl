## seed_claims.json

Claims are derived from the SciFact dataset:

> David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen,
> Arman Cohan, Hannaneh Hajishirzi. "Fact or Fiction: Verifying Scientific
> Claims." EMNLP 2020.

Source: https://github.com/allenai/scifact (train + dev splits), released
under CC BY-NC 2.0 (non-commercial use only).

Only claims with a single, non-conflicting SUPPORT/CONTRADICT label across
all cited evidence documents were kept (693 of 1,109 train+dev claims).
`label` is 1.0 for SUPPORT, 0.0 for CONTRADICT.

Each claim's `evidence` field was backfilled from the same SciFact release
(`corpus.jsonl` + `claims_{train,dev}.jsonl`) via `backfill_scifact_evidence.py`
— one entry per cited document (title + full abstract text), with `label`
set to that document's SUPPORT/CONTRADICT rationale, lowercased. This is
the real evidentiary basis SciFact's human annotators labeled each claim
against, used to seed `ClaimEnv.reset()`'s initial evidence pool; live
Tavily search remains available for the agent's own `QUERY`/
`REQUEST_CLARIFICATION` follow-up searches mid-episode.
