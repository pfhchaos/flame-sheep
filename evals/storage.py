"""JSONL storage for eval runs.

One line per run. Append-only. The file is committed into git so
metric evolution is part of the project history.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Co-located with this module so the source-of-truth path doesn't
# require any project-root math from callers.
RESULTS_PATH = Path(__file__).resolve().parent / 'results.jsonl'


@dataclass
class RunEntry:
    """One row in results.jsonl."""
    timestamp: str                  # ISO 8601 UTC
    commit: str                     # short git SHA (7 chars)
    commit_dirty: bool              # True if working tree had uncommitted changes
    branch: str
    notes: str                      # free-form, from --notes
    metrics: dict[str, float] = field(default_factory=dict)
    # Per-metric context (e.g. thresholds, weights hash) — optional.
    # Stored as JSON-serializable values; key is the eval name (NOT
    # the full metric name), value is whatever the eval returns as
    # its `context` (if any).
    contexts: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False) + '\n'

    @classmethod
    def from_json(cls, data: dict) -> RunEntry:
        return cls(**data)


def git_snapshot() -> tuple[str, bool, str]:
    """Return (short_sha, is_dirty, branch). Empty strings on failure
    (e.g. not in a git repo)."""
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = ''
    try:
        # `git status --porcelain` is empty iff clean.
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        dirty = False
    try:
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        branch = ''
    return sha, dirty, branch


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def append_run(entry: RunEntry, path: Path = RESULTS_PATH) -> None:
    """Append one run to the JSONL file. Creates the file if missing.

    Atomic on POSIX since we're appending a single line; the OS
    guarantees single-write atomicity for writes under PIPE_BUF
    (4096 bytes). For our entries (tiny metric dicts) this is fine.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(entry.to_jsonl())


def read_runs(path: Path = RESULTS_PATH) -> list[RunEntry]:
    """Read all runs in order. Empty list if file missing."""
    if not path.exists():
        return []
    runs = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            runs.append(RunEntry.from_json(json.loads(line)))
    return runs
