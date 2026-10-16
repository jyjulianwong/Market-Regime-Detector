#!/usr/bin/env python3
"""Bump the calendar version, update the root pyproject.toml, and create a git tag.

Usage (run from repo root in CI):
    python3 scripts/bump_calver.py

Outputs (written to $GITHUB_OUTPUT if set):
    version=YYYY.MM.DD.MINOR
    tag=vYYYY.MM.DD.MINOR

Adapted from Investment-News-Analysis-AI's script of the same name — that repo
bumps `lambda/pyproject.toml` because it has a lambda/ vs. server/ split with
independent pyproject.toml files; this project has one root `pyproject.toml`
for the whole monolith (brief section 6: "pyproject.toml single source of
truth"), so there's no "which file" question. One CalVer tag covers both of
this project's deployables (the API zip and the batch image) built from that
same tagged commit — one monorepo release, not two independently-versioned
services.
"""

import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kwargs)


def main() -> None:
    today = datetime.now(tz=UTC).strftime("%Y.%m.%d")

    existing_tags = run(["git", "tag", "--list", f"v{today}.*"]).stdout.splitlines()
    existing_tags = [t.strip() for t in existing_tags if t.strip()]

    if existing_tags:
        minors = [int(t.rsplit(".", 1)[-1]) for t in existing_tags]
        minor = max(minors) + 1
    else:
        minor = 0

    version = f"{today}.{minor}"
    tag = f"v{version}"

    pyproject = Path("pyproject.toml")
    original = pyproject.read_text()
    updated = re.sub(
        r'^version = "[^"]+"', f'version = "{version}"', original, count=1, flags=re.MULTILINE
    )
    if updated == original:
        print(f"ERROR: could not find version field in {pyproject}", file=sys.stderr)
        sys.exit(1)
    pyproject.write_text(updated)

    run(["git", "config", "user.name", "github-actions[bot]"])
    run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"])
    run(["git", "add", str(pyproject)])
    run(["git", "commit", "-m", f"chore: bump version to {version}"])
    run(["git", "tag", "-a", tag, "-m", tag])

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"version={version}\n")
            f.write(f"tag={tag}\n")
    else:
        print(f"version={version}")
        print(f"tag={tag}")

    print(f"Bumped to {tag}", file=sys.stderr)


if __name__ == "__main__":
    main()
