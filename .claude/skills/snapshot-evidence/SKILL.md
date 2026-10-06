---
name: snapshot-evidence
description: Export a portable evidence snapshot so runs can be reproduced without live Tavily calls. Only the owner starts this.
disable-model-invocation: true
---

Create an evidence snapshot with `evid-snapshot`. This calls Tavily and spends API quota.

1. Confirm `TAVILY_API_KEY` is set in `.env` without reading its value.
2. Show the command and wait for approval: `evid-snapshot --dataset <path or default> --output <path>`.
3. Write the snapshot outside the tracked source tree unless the owner says it should be committed. Check its size first, and tell the owner if it is over 5 MB.
4. Report how many claims it covers. Claims with gold SciFact evidence do not need Tavily at all.
5. Explain how to use it: pass `--evidence-snapshot <path>` to `evid-train`, `evid-eval`, and `evid-gold-eval`.
