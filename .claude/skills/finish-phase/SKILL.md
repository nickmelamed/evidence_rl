---
name: finish-phase
description: Close out a phase or feature branch. Runs checks, independent review, changelog, and summary before asking to open a PR.
disable-model-invocation: true
---

Close out the current phase or feature branch.

1. Run the full check suite (`make ci` or the repo's equivalent) and show the
   output. Fix failures before going on.
2. Run the spec-reviewer agent on the branch diff. Fix every problem it
   reports as affecting correctness, requirements, or rules. List the
   optional items for the owner without acting on them.
3. Run the style-reviewer agent on the same diff and apply its rewrites.
4. If the phase made a design decision, make sure it is recorded in
   docs/DECISIONS.md.
5. Update CHANGELOG.md under "Unreleased" and tick the finished items in
   PROGRESS.md.
6. Commit the fixes as atomic Conventional Commits.
7. List any procedure that was repeated during this phase and should become
   a skill, and draft it for the owner to approve.
8. Give the owner a short summary covering what was built, the commits, any
   decisions logged, anything the reviewers raised that was left open, and
   the evidence that checks pass.
9. Ask before pushing or opening the pull request.
