import subprocess


def get_git_sha() -> str | None:
    """Returns the current commit SHA, or `None` outside a git repo or with no
    commits yet. Used by the dataset manifest (`git_sha`) and by the publisher's
    deterministic `model_id` generation (brief's `mrd-<YYYYMMDD>-<short_git_sha>`
    format).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
