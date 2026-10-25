## Summary

<!-- What changed and why, in 1–3 sentences. -->

## Changes

<!-- Optional for small PRs. Notable pieces, one bullet each. -->

-

## Testing

- [ ] `uv run pytest`
- [ ] `pre-commit run --all-files` (lint, format, mypy)
- [ ] Manual check (describe below, if any)

## Deploy impact

<!-- Pushes to `main` touching terraform/, docker/, src/, scripts/, pyproject.toml or uv.lock redeploy automatically. -->

- [ ] Changes Terraform or infra
- [ ] Changes the published JSON shape (`prediction/outputs.py` and `server/schemas.py` stay in sync)
- [ ] Changes the model or feature config (needs retrain, publish, promote)
- [ ] None of the above
