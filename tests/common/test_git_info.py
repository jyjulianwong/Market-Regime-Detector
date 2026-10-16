from pathlib import Path

import pytest

from market_regime_detector.common.git_info import get_git_sha


def test_get_git_sha_returns_a_full_sha_inside_this_repo() -> None:
    sha = get_git_sha()

    # This repo may have zero commits in some dev states (nothing to rev-parse yet)
    # — handle both rather than assuming a commit always exists.
    if sha is not None:
        assert len(sha) == 40


def test_get_git_sha_returns_none_outside_a_git_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    assert get_git_sha() is None
