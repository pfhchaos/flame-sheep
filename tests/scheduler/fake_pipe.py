"""Shared test double for the driver ↔ worker subprocess pipe.

Generalized from step-1's inline ``_FakeStdout`` (test_precompile_flood).
Step 2 removed the worker's stdout protocol, so the double a shim test now
needs is the *write* side: capture what ``PrecompileDriver`` serializes to
the worker's stdin and parse it back as JSON. ``FakeProc`` stands in for the
``subprocess.Popen`` the driver holds in ``self._proc`` — GL-free, no real
process — so protocol/shim tests run in the fast suite.

The scripted-stdout half is kept too (``FakeProc(stdout_lines=...)``) for any
future test that still needs to feed the driver lines, even though step 2's
driver no longer reads stdout.
"""
from __future__ import annotations

import json


class CaptureStdin:
    """A writable file-like that records every newline-framed line the
    driver writes. ``.write``/``.flush``/``.close`` match the subset of the
    text-mode pipe API the driver touches."""

    def __init__(self):
        self._buf = ''
        self.closed = False

    def write(self, s: str) -> int:
        self._buf += s
        return len(s)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def lines(self) -> list[str]:
        """Completed (newline-terminated) lines written so far."""
        return [ln for ln in self._buf.split('\n') if ln]

    def objects(self) -> list[dict]:
        """Each written line parsed as a JSON object, in write order."""
        return [json.loads(ln) for ln in self.lines()]


class ScriptStdout:
    """A readable file-like that yields a fixed list of lines then EOF
    (``''``), mirroring ``pipe.stdout.readline()`` semantics. Also iterable
    (``for line in proc.stdout``) like a real text-mode pipe, so the driver's
    ``_stdout_loop`` (which iterates) can be unit-tested with it."""

    def __init__(self, lines=()):
        self._lines = list(lines)
        self._it = iter(self._lines)

    def readline(self) -> str:
        return next(self._it, '')

    def __iter__(self):
        # Mirror a real pipe: iteration stops at EOF; a scripted trailing ''
        # (explicit EOF marker) also terminates, matching readline semantics.
        for line in self._lines:
            if line == '':
                return
            yield line


class FakeProc:
    """Stand-in for ``subprocess.Popen`` in driver shim tests: a captured
    ``.stdin`` (+ optional scripted ``.stdout``). Assign to ``driver._proc``
    after constructing the driver WITHOUT calling ``start()`` (no real
    subprocess, no GL)."""

    def __init__(self, stdout_lines=()):
        self.stdin = CaptureStdin()
        self.stdout = ScriptStdout(stdout_lines)

    # Convenience pass-throughs so tests read naturally.
    def lines_written(self) -> list[str]:
        return self.stdin.lines()

    def objects_written(self) -> list[dict]:
        return self.stdin.objects()
