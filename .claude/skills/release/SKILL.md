---
name: release
description: Cut a tagged release. Only the owner starts this.
disable-model-invocation: true
---

Prepare the release version the owner named when invoking this skill (a
semantic version such as v0.2.0). If none was given, ask for it.

1. Confirm the working tree is clean and on `main` with CI green.
2. Run the full check suite and any reproduction target, and show the output.
3. Run `python3 scripts/agent/check_numbers.py <every document that reports
   results> --sources <the generated tables> --require-sources`. At release
   time a missing results directory is a failure, not a skip.
4. Move CHANGELOG.md's "Unreleased" entries under the new version and date.
5. Commit that change as `chore(release): <version>`.
6. Show the owner the changelog entry and the exact `git tag -a` and
   `git push` commands, then wait. The owner approves the tag and push.
