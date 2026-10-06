---
paths:
  - "**/*.py"
  - "**/*.md"
  - "**/*.qmd"
  - "**/*.Rmd"
  - "**/*.R"
  - "**/*.ipynb"
---

# Writing style for comments, docstrings, and docs

The owner writes this code. Everything should read like notes from a careful
engineer, not like generated text. `scripts/agent/check_style.py` checks the
mechanical parts of the lines you change after every edit.

## Comments

- Explain why, never what. If the code is clear, write no comment.
- Never mention CLAUDE.md, AGENTS.md, the agent, or instructions to the agent.
  If a rule matters to a reader, give the reason in plain terms instead.
- Never describe edit history ("now handles", "updated to", "fixed a bug").
  That goes in the commit message.
- No section-banner comments.
- Match the comment density of the file. Do not add comments to lines you did
  not change.

## Docstrings

- NumPy style, only on public modules, classes, and functions. If the repo's
  linter enforces another convention, follow the linter.
- One imperative summary line ("Check that...", "Return...").
- Add more only when behavior would surprise a reader, such as array shapes,
  units, or side effects.
- Skip Parameters and Returns sections when type hints already say it.
- With a single exception, a sentence can replace the Raises section.

Good:

    """Check that ``root`` contains exactly the expected dataset files.

    Hidden files like ``.DS_Store`` are ignored. Any other extra or missing
    file raises ``LayoutError`` listing the offending paths.
    """

Good comment:

    # From the dataset's official page (retrieved 2026-09-26), see D-009.
    # Keep the published wording, since these describe images, not diagnoses.

Bad comment (cites agent instructions and narrates its own diligence):

    # Copied verbatim (checked against the raw page). Never edit the wording
    # because CLAUDE.md section 4 forbids paraphrasing these.

## Prose (README, docs, write-ups, commit messages, PR descriptions)

- Plain sentences. No em dashes, and no en dashes except in numeric ranges.
- No semicolons. Split the sentence instead.
- Colons only to introduce code or a list.
- No bold lead-ins followed by a colon.
- Prefer paragraphs. Bullets only for items that are truly parallel.
- Avoid groups of three adjectives and "not X, but Y" framing.
- Avoid these words and their inflections: seamless, comprehensive, leverage,
  delve, crucial, utilize, meticulous. No emoji.
- Say what the thing does and how to use it. Do not sell it.

Technical terms are fine where they are the precise word, such as "robust
standard errors". To change the word list for this repo, ask the owner to
edit `.claude/style-words`.

Add `style: ok` to a line only when its text is quoted verbatim from an
outside source.
