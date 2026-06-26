# Open Source Standardization Plan

## Step 1: Legal Framework
- [x] Create `LICENSE` file using the standard MIT License.
  - *Rule:* Copyright year is 2026. Copyright holder is "Prompt Architect Analyst Contributors".
  - *Verification:* The file exists and contains the exact MIT legal text.

## Step 2: Community Guidelines
- [x] Create `CONTRIBUTING.md`.
  - *Rule:* Must include sections for: "Local Setup" (`pip install -e .[dev]`), "Strict TDD Policy" (tests must be written before logic), and "Type Checking" (Zero `Any` allowed).
  - *Verification:* The markdown is professional, welcoming, and enforces our engineering invariants.
- [x] Create `.github/PULL_REQUEST_TEMPLATE.md`.
  - *Rule:* Must contain a markdown checklist for contributors: `[ ] I ran pytest`, `[ ] I ran ruff check`, `[ ] I ran mypy --strict`, `[ ] I did not mutate any frozen dataclasses`.

## Step 3: The Storefront (README)
- [x] Completely rewrite `README.md`.
  - *Rule:* Must include:
    1. **Badges/Shields** (CI Passing, Python 3.12+, License MIT, Code Style: Ruff).
    2. **Elevator Pitch** (1-paragraph explanation of the AI fluency analyzer).
    3. **Architecture** (Briefly mention the immutable 7-phase pipeline and SQLite read-only mode).
    4. **Installation** (Clear `pip install` instructions).
    5. **Quickstart** (CLI example using `--db-path` and `--api-key`).
  - *Verification:* The documentation reflects the *actual* implemented Typer CLI, not stubs.

## Step 4: Final Quality Gate
- [x] Run `ruff check .` and `ruff format .`
- [x] Run `mypy src/ tests/` (Must be 0 errors)
- [x] Run `pytest` (Must be 177/177 passing)

## Step 5: Atomic Commit
- [x] Stage all modified/created files (including `PLAN_OSS.md`).
- [x] Commit with a conventional message: `docs: implement open source standards and contribution guidelines`
