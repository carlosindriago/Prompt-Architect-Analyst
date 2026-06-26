# Contributing to prompt-architect-analyst

Thank you for your interest in making AI collaboration skills more measurable.
Every contribution, from a typo fix to a new dimension, helps practitioners
understand how they actually build with AI.

This document describes how to set up the project locally, the engineering
invariants we hold, and the workflow we expect from pull requests.

## Code of Conduct

Be kind. Assume good intent. Disagree with arguments, never with people.
The maintainers reserve the right to close threads that violate this.

## Local Setup

```bash
# 1. Clone
git clone https://github.com/carlosindriago/prompt-architect-analyst.git
cd prompt-architect-analyst

# 2. Create a virtual environment (Python 3.12+ recommended)
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install the project with dev extras (pytest, ruff, mypy)
pip install -e ".[dev]"
```

> **Note:** The dev extra installs `pytest`, `pytest-cov`, `ruff>=0.4`,
> `mypy>=1.10`, and `bandit`. Run `pip install -e ".[dev]"` to install
> everything needed to run tests, lint, and type-check.

## Engineering Invariants

We hold the following invariants in production code at all times. Any
pull request that breaks one of them will be asked to fix it before merge.

### 1. Strict TDD Policy

Tests are written **before** the implementation. The cycle is:

1. **Red:** write a failing test in `tests/` that captures the requirement.
2. **Green:** write the minimum code in `src/` to make the test pass.
3. **Refactor:** clean up while keeping tests green.

A pull request is expected to ship tests alongside the logic. If you
discover a bug while implementing a feature, write a test that reproduces
it first, then fix the code.

We never suppress failing tests with `@pytest.mark.skip` or
`@pytest.mark.xfail` to make the suite pass. Refactor the code until the
test passes for real — "making the system a liar is worse than a missing
test."

### 2. Type Checking — Zero `Any`

`mypy` runs in strict mode against `src/` and `tests/`. The pipeline
**must** end with `Success: no issues found`.

- Do not introduce `Any` into `src/`. Use `TypedDict`, `Protocol`, or
  precise unions.
- The handful of `Any` already justified in production are documented in
  `CHANGELOG.md` ("Remaining `Any` instances (justified)"). Do not add
  more without updating that section.
- `# type: ignore` is forbidden unless mathematically necessary, and
  must carry a comment explaining why.

### 3. Immutability by Default

- All domain dataclasses (`Turn`, `Session`, `Corpus`, `ScoreCard`,
  `DimensionScore`) are `@dataclass(frozen=True, slots=True)`. They
  raise on mutation. **Do not** use `object.__setattr__` to bypass
  the frozen guard.
- A pull request that mutates a frozen dataclass will be rejected
  outright. If you need new state, derive a new value with
  `dataclasses.replace(...)`.

### 4. Read-Only Database

The Reader layer opens the SQLite database with `?mode=ro`. It never
writes. If you need to cache derived data, keep it in memory or write
to a separate file outside the source DB.

## Workflow

1. **Branch.** Create a topic branch from `main`:
   `git checkout -b feat/<short-name>` or `fix/<short-name>`.
2. **TDD.** Write the failing test first.
3. **Implement.** Minimum code to pass the test.
4. **Verify locally:**
   ```bash
   ruff format .
   ruff check .
   mypy src/ tests/
   pytest
   ```
5. **Commit.** Use [Conventional Commits](https://www.conventionalcommits.org/):
   `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`.
6. **Push** and open a Pull Request. The GitHub Actions CI must pass
   before a review begins.
7. **Review.** Address feedback with follow-up commits. Avoid force-push
   after a review has started.
8. **Merge.** The maintainer merges via the GitHub UI once CI is green
   and at least one approval is recorded.

## Commit Messages

- Subject line ≤ 72 characters, imperative mood ("add" not "added").
- Body explains *why*, not *what*. The diff shows what.
- Reference any issue ID: `Refs: #42` or `Fixes: #42`.

## Pull Request Checklist

The PR template will remind you of:

- [ ] `pytest`
- [ ] `ruff check .`
- [ ] `ruff format --check .`
- [ ] `mypy src/ tests/`
- [ ] No mutation of frozen dataclasses
- [ ] No new `Any` in `src/`
- [ ] CHANGELOG.md updated if the change is user-visible

## Reporting Security Issues

Please do **not** open a public issue for security findings. Email the
maintainers directly so we can coordinate a fix and disclosure.
