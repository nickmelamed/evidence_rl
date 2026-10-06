---
name: results-table
description: Build the committed results table from artifacts/experiments and check that the README numbers match it. Only the owner starts this.
disable-model-invocation: true
---

Turn finished runs into a committed results table and verify the documents that quote it. The owner names the runs. If none are named, list the folders in `artifacts/experiments/` and ask.

1. For each named run, read `config.json`, `metrics.csv`, and `gold_eval.jsonl` if present. Do not run training or evaluation here.
2. Write one table per kind of result to `results/tables/` as CSV, with run name, method, judge architecture, seed, episode count, and the metric columns. Include the source run folder in every row so each number traces back.
3. Never round, drop, or reorder rows to make a result look better. Report failed or partial runs too.
4. Add or confirm a `numbers` target in the Makefile that runs `python3 scripts/agent/check_numbers.py README.md --sources results/tables`. Changing the Makefile needs the owner's approval.
5. Run the check and show its output. Mark non-result numbers in the README (reward weights, hyperparameters) with `numbers: ok`, after asking.
6. Commit tables and doc changes separately, as `docs(results): ...`.
