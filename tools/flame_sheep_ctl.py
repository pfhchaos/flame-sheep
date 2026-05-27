"""Send a command to the running flame-sheep wallpaper via the control pipe.

Thin wrapper around `echo CMD ARGS > ~/.local/share/flame-sheep/ctl`. Reads
FLAME_SHEEP_CTL env var if set, otherwise the default path. Useful from
shell aliases, voting tools, debug workflows, etc.

Usage:
    python tools/flame_sheep_ctl.py show 1234
    python tools/flame_sheep_ctl.py unshow
    python tools/flame_sheep_ctl.py swap
    python tools/flame_sheep_ctl.py like
    FLAME_SHEEP_CTL=/tmp/test.pipe python tools/flame_sheep_ctl.py show 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_PIPE = Path(os.environ.get(
    'FLAME_SHEEP_CTL',
    os.path.expanduser('~/.local/share/flame-sheep/ctl'),
))


def send(pipe_path: Path, command: str, args: list[str]) -> None:
    if not pipe_path.exists():
        sys.exit(f'error: control pipe does not exist: {pipe_path}\n'
                 f'(is the wallpaper running?)')
    line = ' '.join([command, *args]) + '\n'
    with open(pipe_path, 'w') as f:
        f.write(line)


def main():
    parser = argparse.ArgumentParser(description='Send a command to the wallpaper control pipe')
    parser.add_argument('command', help='command name (show, unshow, swap, like, ...)')
    parser.add_argument('args', nargs='*', help='command arguments')
    parser.add_argument('--pipe', type=Path, default=DEFAULT_PIPE,
                        help=f'pipe path (default: {DEFAULT_PIPE})')
    parsed = parser.parse_args()
    send(parsed.pipe, parsed.command, parsed.args)


if __name__ == '__main__':
    main()
