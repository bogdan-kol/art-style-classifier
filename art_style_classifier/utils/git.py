"""Git utilities for tracking code versions."""

from __future__ import annotations

import subprocess
from pathlib import Path


def get_git_commit_id(repo_path: Path | None = None) -> str:
    """Get current git commit ID.

    Args:
        repo_path: Path to git repository root (default: current directory)

    Returns:
        Git commit hash, or 'unknown' if git is not available
    """
    if repo_path is None:
        repo_path = Path.cwd()

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
