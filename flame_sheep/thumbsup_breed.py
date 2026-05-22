"""Subprocess entry point for thumbs-up breeding.

Background: `Library.rate` used to call `breed_thumbsup_children`
synchronously, blocking the render thread. c695a12 moved it to a
daemon thread — but Python's GIL means CPU-bound work in
`survey_and_correct` (an attractor simulation) still contends with
the render thread, producing visible stalls even though the thread
itself is "background."

This module runs the breeding in a separate *process* via
`python -m flame_sheep.thumbsup_breed --parent-id N --children N`.
The render thread doesn't share a GIL with the subprocess; the
breeding completes entirely outside the foreground process's
attention.

Cost: ~50-80 ms python startup per upvote (one-time, well below
the watchdog limit). After that, breeding happens completely off
the render-thread critical path.
"""
from __future__ import annotations

import argparse
import logging
import sys

from pathlib import Path

from .storage import Library


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Breed jittered children for a thumbs-upped genome.')
    parser.add_argument('--parent-id', type=int, required=True)
    parser.add_argument('--children', type=int, default=7,
                        help='Number of jittered children to attempt (default 7)')
    parser.add_argument('--jitter-scale', type=float, default=0.1)
    parser.add_argument('--data-dir', type=Path, default=None,
                        help='Library data directory (default: standard location)')
    args = parser.parse_args()

    # Match the main process's log format so the [thumbsup-breed] line
    # shows up alongside other render-process logs in the user's terminal
    # (subprocess stderr is inherited by the parent shell).
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
    )
    log = logging.getLogger(__name__)

    lib = Library(args.data_dir)
    try:
        children = lib.breed_thumbsup_children(
            args.parent_id,
            n_children=args.children,
            jitter_scale=args.jitter_scale,
        )
        log.info('[thumbsup-breed] genome #%d -> %d children %s',
                 args.parent_id, len(children), children)
        return 0
    except Exception:
        log.exception('[thumbsup-breed] failed for genome #%d', args.parent_id)
        return 1
    finally:
        lib.close()


if __name__ == '__main__':
    sys.exit(main())
