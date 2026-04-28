"""Logging configuration for flame-sheep.

Sets up structured logging with:
  - File handler: ~/.local/share/flame-sheep/flame-sheep.log (rotating)
  - Console handler: stderr (for systemd journal / terminal)
  - Per-component loggers: flame_sheep.audio, flame_sheep.axes.genome, etc.

Usage:
    from flame_sheep.log import setup_logging
    setup_logging(level='INFO')  # call once at startup

    # In any module:
    import logging
    log = logging.getLogger(__name__)
    log.info('something happened')
"""
from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path


LOG_DIR = Path(os.environ.get(
    'FLAME_SHEEP_LOG_DIR',
    os.path.expanduser('~/.local/share/flame-sheep')
))

LOG_FORMAT = '%(asctime)s %(name)s %(levelname)s %(message)s'
LOG_DATE_FORMAT = '%H:%M:%S'


def setup_logging(level: str = 'INFO', log_file: bool = True, quiet: bool = False,
                  component_levels: dict[str, str] | None = None) -> None:
    """Configure logging for the application.

    Args:
        level: Root log level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Write to rotating log file.
        quiet: Suppress console output (useful for systemd where journal captures stderr).
        component_levels: Per-component log level overrides, e.g.
            {'flame_sheep_audio.tempo_acf': 'DEBUG', 'flame_sheep.axes': 'WARNING'}
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Configure both package roots
    for root_name in ('flame_sheep', 'flame_sheep_audio'):
        root = logging.getLogger(root_name)
        root.setLevel(log_level)

        # Console handler (stderr -> systemd journal or terminal)
        if not quiet:
            console = logging.StreamHandler()
            console.setFormatter(formatter)
            root.addHandler(console)

        # File handler with rotation
        if log_file:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            path = LOG_DIR / 'flame-sheep.log'
            file_handler = logging.handlers.RotatingFileHandler(
                path, maxBytes=5 * 1024 * 1024, backupCount=3)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)

    # Per-component overrides
    if component_levels:
        for name, comp_level in component_levels.items():
            logging.getLogger(name).setLevel(
                getattr(logging, comp_level.upper(), logging.INFO))
