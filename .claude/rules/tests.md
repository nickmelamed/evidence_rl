---
paths:
  - "tests/**"
  - "**/test_*.py"
  - "**/*_test.py"
  - "**/conftest.py"
  - "**/testthat/**"
---

# Tests

- Tests are the oracle for the code, so never weaken one to make it pass. Do
  not delete assertions, turn a specific assertion into a vague one, loosen
  tolerances, or add skip or xfail markers without the owner's approval. The
  Stop hook checks all of these against the base branch.
- When fixing a bug, first write a test that fails because of it.
- Prefer small synthetic fixtures with known answers. Tests never touch real
  or private data.
- For numeric code, add property-based tests with Hypothesis (bounds,
  symmetry, invariance) alongside the hand-worked examples.
- Coverage is a floor, not the goal. `mutmut` on core modules is the real
  measure of whether tests catch bugs. Do not write hollow tests to raise a
  number.
- Mark slow tests with `@pytest.mark.slow` so the fast gate stays fast.
