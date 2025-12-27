"""Git utilities for experiment tracking."""

import subprocess
from pathlib import Path


def get_git_info(cwd: Path | str | None = None) -> dict:
    """Get current git SHA and dirty status.

    Args:
        cwd: Working directory for git commands. Defaults to current directory.

    Returns:
        Dict with 'sha' (str) and 'dirty' (bool) keys.
        Returns sha='unknown' and dirty=False if git is not available.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
        ).decode().strip()

        # Check if repo is dirty
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
        ).decode().strip()

        return {"sha": sha, "dirty": len(status) > 0}
    except Exception:
        return {"sha": "unknown", "dirty": False}

