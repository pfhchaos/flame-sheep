#!/usr/bin/env python3
"""Pack a corpus of .npz files into mmap-able .npy files per a named schema.

Generic CLI on top of wallpaper_ml.packed_corpus. New corpora plug in by
adding a (key, CorpusSchema) entry to SCHEMAS below; the schema gets
written into the destination directory's manifest so trainers can load
column ranges by named channel.

Usage:
    python tools/pack_corpus.py SRC_DIR DST_DIR --schema NAME
    python tools/pack_corpus.py --list-schemas

Example:
    python tools/pack_corpus.py ~/datasets/beat-labels/ \\
                                ~/datasets/beat-labels-mmap/ \\
                                --schema beat_rnn_3head
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make tools/ importable so we can pull schemas defined alongside trainers
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wallpaper_ml import pack_corpus


def _beat_rnn_3head_schema():
    """Schema for hierarchical 3-head beat-RNN training. Used by
    tools/train_beat_rnn_continuous.py with --n-classes 3."""
    # Imported at call time to avoid loading the trainer module unless
    # this schema is actually requested.
    from train_beat_rnn_continuous import BEAT_RNN_3HEAD_SCHEMA
    return BEAT_RNN_3HEAD_SCHEMA


# Registry of named schemas. Each value is a callable returning a
# CorpusSchema. Add new entries as new trainers gain packed-corpus
# support.
SCHEMAS = {
    'beat_rnn_3head': _beat_rnn_3head_schema,
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('src_dir', type=Path, nargs='?')
    parser.add_argument('dst_dir', type=Path, nargs='?')
    parser.add_argument('--schema', type=str,
                        help='schema name (see --list-schemas)')
    parser.add_argument('--list-schemas', action='store_true',
                        help='print available schema names and exit')
    parser.add_argument('-j', '--jobs', type=int, default=None,
                        help='worker process count (default: CPU count)')
    parser.add_argument('--pattern', type=str, default='*.npz',
                        help='source-file glob pattern (default *.npz)')
    args = parser.parse_args()

    if args.list_schemas:
        for name in sorted(SCHEMAS):
            schema = SCHEMAS[name]()
            print(f'{name}: {schema.total_columns} cols '
                  f'[{" + ".join(f"{c.name}({c.columns})" for c in schema.channels)}]')
        return

    if not args.src_dir or not args.dst_dir or not args.schema:
        parser.error('src_dir, dst_dir, and --schema are required '
                     '(unless --list-schemas)')

    if args.schema not in SCHEMAS:
        parser.error(f'unknown schema {args.schema!r}; '
                     f'try --list-schemas')

    schema = SCHEMAS[args.schema]()
    print(f'Packing {args.src_dir} → {args.dst_dir} '
          f'(schema={args.schema}, {schema.total_columns} cols)')

    summary = pack_corpus(
        args.src_dir, args.dst_dir, schema,
        pattern=args.pattern, n_jobs=args.jobs,
    )

    if summary['error'] > 0:
        print(f'\n{summary["error"]} files failed:', file=sys.stderr)
        for name, msg in summary['errors'][:20]:
            print(f'  {msg}', file=sys.stderr)
        if len(summary['errors']) > 20:
            print(f'  ... and {len(summary["errors"]) - 20} more',
                  file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
