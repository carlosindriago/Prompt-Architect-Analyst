## Summary

<!-- One or two sentences describing the change. -->

## Type of change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would change existing behavior)
- [ ] Documentation only

## Pre-merge checklist

I have personally verified every item below on my local checkout:

- [ ] I ran `pytest` and all 177 tests pass.
- [ ] I ran `ruff check .` and there are no lint errors.
- [ ] I ran `ruff format .` and the working tree is formatted.
- [ ] I ran `mypy src/ tests/` and there are 0 errors.
- [ ] I did not mutate any frozen dataclass (no `object.__setattr__`, no
      direct assignment to `Turn`, `Session`, `Corpus`, `ScoreCard`,
      or `DimensionScore`).
- [ ] I did not introduce new `Any` types into `src/`.
- [ ] I did not add `# type: ignore` without a justifying comment.
- [ ] I added or updated tests for the new behavior (TDD).
- [ ] I updated `CHANGELOG.md` if the change is user-visible.

## Architecture impact

- [ ] Reader layer remains read-only (`?mode=ro`).
- [ ] Error handling uses the `InsightError` hierarchy, not bare `Exception`.
- [ ] New code follows the existing dataclass(frozen=True, slots=True) pattern.

## Related issues

<!-- Link related issues: Fixes #N, Refs #N. -->
