# Working with Claude Code in this repo

This is the owner's guide to how Claude Code works here. CLAUDE.md and the
files under `.claude/` are what Claude reads. This file is for people.

## Principles

Context is the scarce resource. Claude's performance drops as the context
window fills, and a long CLAUDE.md is the most common way to fill it for no
reason. Keep always-loaded instructions short and load everything else on
demand.

Instructions are advice and hooks are enforcement. If a rule must hold every
time (tests pass, no force-push, no invented numbers), a script checks it and
a hook runs the script. Only judgment calls go in prose.

Verification closes the loop. Claude stops when the work looks done. A check
it can run turns "looks done" into "is done", and that is what makes
unattended work safe.

The author should not grade their own work. Review happens in a fresh context
that sees only the diff and the spec, never the reasoning that produced it.

The owner owns every line. Read each diff before it merges and be able to
explain any line. The decision log is written in the owner's voice.

## Where each kind of instruction goes

| What | Where | Loaded |
|---|---|---|
| Project summary, non-negotiable rules, commands, map of the repo | `CLAUDE.md`, under about 100 lines | Every session |
| Full design and specs | `docs/SPEC.md` | When a task needs it |
| Standards for one part of the code | `.claude/rules/*.md` with `paths:` | When Claude touches matching files |
| Procedures (finish a phase, release, run on Colab) | `.claude/skills/<name>/SKILL.md` | When invoked |
| Status and next steps | `PROGRESS.md` | Its "Now" section at session start and after compaction |
| Why things are the way they are | `docs/DECISIONS.md` | When needed |
| Rules that must always hold | `scripts/agent/`, wired up in `.claude/settings.json` | Enforced automatically |
| Independent review | `.claude/agents/*.md` | When delegated |

For each line in CLAUDE.md, ask whether removing it would make Claude get
something wrong. If not, cut it. If Claude already behaves correctly without a
rule, delete the rule or turn it into a hook.

## What the hooks do

| Hook | Script | Effect |
|---|---|---|
| SessionStart (startup, compact) | `session_context.py` | Shows Claude the current phase, branch, uncommitted files, and last gate result |
| PreToolUse on Bash | `guard_bash.py` | Blocks skipping commit hooks, force-pushes, and reading or printing secrets. Asks you before discarding work, deleting branches, recursive deletes, and shell writes to protected paths |
| PreToolUse on edits | `guard_edit.py` | Asks you before Claude edits a protected path |
| PreToolUse on Read and Grep | `guard_read.py` | Blocks secret files while allowing `.env.example` |
| PostToolUse on edits | `check_style.py --hook` | Sends style findings on the lines Claude changed straight back to Claude |
| Stop | `stop_gate.py` | Runs `.claude/gate-commands` and `check_tests.py` when the tree changed, and keeps Claude working until they pass, for at most three rounds |

"Block" means Claude cannot do it at all. "Ask" means you get a permission
prompt with the reason and decide. `check_numbers.py` fails when a document
reports a number that is not in a generated results file. Pre-commit and CI
run the same checks, so they hold outside Claude Code too.

## Escape hatches

- `AGENT_GATE=off claude` starts a session with the Stop gate off, for when
  the gate itself is broken.
- `AGENT_ALLOW_TEST_CHANGES=1 claude` allows a session to remove assertions,
  add skips, or loosen tolerances on purpose.
- `python3 scripts/agent/smoke_test.py` checks every hook after you change the
  tooling. Add `--gate` to run the Stop gate once.
- Claude cannot set these variables for itself, since hooks read the
  environment Claude Code was started with.

## The workflow for a piece of work

1. Spec. For anything bigger than a day, have Claude interview you and write
   or update `docs/SPEC.md`. Specs name the files and interfaces, say what is
   out of scope, and end with an end-to-end check.
2. Plan. Start a fresh session in plan mode (`claude --permission-mode plan`,
   or Shift+Tab). Have Claude read the relevant code and spec and write a
   plan. Edit it yourself (Ctrl+G opens it in your editor) before approving.
   Skip this for changes you could describe in one sentence.
3. Tests first where it matters. For core logic, have one session write
   failing tests and another write code to pass them.
4. Implement on a branch. Let the Stop hook hold the line. For long unattended
   work use `/goal` with a concrete condition, such as "`make ci` passes and
   PROGRESS.md phase 3 is ticked".
5. Review. Run `/code-review` for bugs, the spec-reviewer agent against the
   spec and plan, and the style-reviewer agent on comments and docs. Fix what
   affects correctness or requirements and treat the rest as optional, since
   a reviewer asked to find problems always finds some.
6. Evidence. Ask for the commands and their output, not "all tests pass".
7. Read the diff yourself. Then commit, push, and open the PR, which Claude
   may prepare but you approve.

## Session hygiene

- One task per session. `/clear` between unrelated tasks, and name sessions
  with `/rename` so you can resume them.
- After two failed corrections on the same issue, `/clear` and restart with a
  better prompt that includes what you learned.
- Use subagents for broad investigation so file dumps stay out of the main
  context. Use `/btw` for side questions.
- Use `/rewind` (Esc Esc) to try something risky and roll back. It does not
  replace git.
- Run `/context` now and then to see what is using the window.

## Scaling up

- Independent workstreams run in parallel git worktrees (`claude -w <name>`
  or a subagent with `isolation: worktree`) so edits never collide.
- `claude -p` runs Claude headless in scripts or CI. Scope it with
  `--allowedTools` and cap it with `--max-turns` and `--max-budget-usd`.
- A Claude review on every PR catches what you miss when tired.

## Test quality

- Coverage is a floor. `mutmut` on core modules measures whether tests catch
  real bugs.
- Property-based tests with Hypothesis for numeric code (bounds, symmetry,
  invariance under irrelevant changes).
- Every bug fix starts with a failing test.
- `check_tests.py` catches removed and weakened assertions, new skips,
  loosened tolerances, and new deselection in conftest.py or pytest config.
  It counts and compares, so the spec-reviewer still judges meaning.

## Writing and authorship

- Comments explain why. Docstrings are short, in the repo's chosen style.
  Prose is plain, with no em dashes or semicolons, colons only before code or
  lists, no bold lead-ins, and no marketing words. The full rules are in
  `.claude/rules/writing-style.md`.
- Never cite CLAUDE.md or the agent in code. Give the human reason instead.
- A one-line README note that the project was built with Claude Code under
  your design and review is professional. Being able to explain every line
  is what actually answers "did you write this?"

## Growing skills

Skills are how a repo's procedures stop living in someone's head or in chat
history. Claude proposes one when it follows the same multi-step routine
twice, or when you give the same instructions a second time. It drafts the
SKILL.md, shows it, and saves it only after you approve.

A skill with side effects (push, deploy, release, training, anything that
spends money) gets `disable-model-invocation: true` so only you start it. A
knowledge skill (a tricky API, a dataset's quirks) stays model-invocable, and
its description says exactly when to use it. A skill that would help in every
repo belongs in `~/.claude/skills/`. Keep the set small, and delete a skill
that nobody has used in a month.

## Maintenance

- When Claude repeats a mistake, add one line to the right rule file, or a
  check if it is mechanical. When a rule stops mattering, delete it.
- Review CLAUDE.md like code, and prune it at every release.
- Keep `.claude/` and `scripts/agent/` in git so the setup travels with the
  repo. `scripts/agent/VERSION` records which version of the standards is
  installed. Ask Claude to upgrade the repo standards to pick up a newer one.
