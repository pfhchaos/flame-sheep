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


def setup_logging(level: str = 'INFO', log_file: bool = True, quiet: bool = False):
    """Configure logging for the application.

    Args:
        level: Root log level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Write to rotating log file.
        quiet: Suppress console output (useful for systemd where journal captures stderr).
    """
    root = logging.getLogger('flame_sheep')
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Console handler (stderr → systemd journal or terminal)
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
