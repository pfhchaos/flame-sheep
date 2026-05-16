"""
Named pipe control interface for flame-sheep.

Allows external tools (hotkeys, music players) to send commands:
  - swap          Force genome swap
  - song          Song changed, reset tempo tracker
  - tempo <bpm>   Hint current tempo (from player metadata)
  - like          Thumbs up current genome (save for learning)
  - dislike       Thumbs down, swap immediately
  - pause         Player paused, enter ambient mode
  - resume        Player resumed
  - quit          Clean shutdown

Usage from shell:
  echo swap > ~/.local/share/flame-sheep/ctl
  
Integration examples:
  # MPD song change
  mpc idleloop player | while read; do echo song > ~/.local/share/flame-sheep/ctl; done
  
  # MPRIS (Spotify, etc) via playerctl
  playerctl -F metadata -f 'song' >> ~/.local/share/flame-sheep/ctl
  
  # Sway hotkeys
  bindsym $mod+bracketright exec echo like > ~/.local/share/flame-sheep/ctl
  bindsym $mod+bracketleft exec echo dislike > ~/.local/share/flame-sheep/ctl
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


import os
import stat
import threading
import queue
from pathlib import Path
from dataclasses import dataclass


# Default pipe location
DEFAULT_PIPE_PATH = Path(os.environ.get(
    'FLAME_SHEEP_CTL',
    os.path.expanduser('~/.local/share/flame-sheep/ctl')
))


@dataclass
class ControlEvent:
    """Parsed control command."""
    command: str
    args: list[str]


class ControlPipe:
    """
    Reads commands from a named pipe in a background thread.
    
    Commands are queued and can be polled from the main thread.
    Non-blocking, won't stall if no commands are sent.
    """
    
    def __init__(self, pipe_path: Path = DEFAULT_PIPE_PATH) -> None:
        self.pipe_path: Path = pipe_path
        self._queue: queue.Queue[ControlEvent] = queue.Queue()
        self._running: bool = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Create pipe and start listener thread."""
        self._ensure_pipe_exists()
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop listener thread. Pipe remains for future use."""
        self._running = False
        # Write a dummy command to unblock the read
        try:
            with open(self.pipe_path, 'w') as f:
                f.write('\n')
        except:
            pass
        if self._thread:
            self._thread.join(timeout=1.0)
            
    def poll(self) -> ControlEvent | None:
        """
        Non-blocking poll for next command.
        Returns None if no commands pending.
        """
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None
            
    def poll_all(self) -> list[ControlEvent]:
        """Return all pending commands, clearing the queue."""
        events = []
        while True:
            event = self.poll()
            if event is None:
                break
            events.append(event)
        return events
        
    def _ensure_pipe_exists(self) -> None:
        """Create the named pipe if it doesn't exist."""
        self.pipe_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.pipe_path.exists():
            # Check if it's actually a pipe
            if not stat.S_ISFIFO(os.stat(self.pipe_path).st_mode):
                # It's a regular file, remove and recreate
                self.pipe_path.unlink()
                os.mkfifo(self.pipe_path)
        else:
            os.mkfifo(self.pipe_path)
            
        # Make it world-writable so any process can send commands
        os.chmod(self.pipe_path, 0o622)
        
    def _read_loop(self) -> None:
        """Background thread: read lines from pipe, parse, queue."""
        log.info(f'[pipe] reader thread started, path={self.pipe_path}')
        while self._running:
            try:
                log.info('[pipe] waiting for writer...')
                with open(self.pipe_path, 'r') as f:
                    log.info('[pipe] writer connected')
                    for line in f:
                        if not self._running:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        log.info(f'[pipe] raw: {line!r}')
                        event = self._parse_line(line)
                        if event:
                            self._queue.put(event)
                    log.info('[pipe] writer disconnected')
            except Exception as e:
                log.warning(f'[pipe] error: {e}', exc_info=True)
        log.info('[pipe] reader thread exiting (_running=False)')
                    
    def _parse_line(self, line: str) -> ControlEvent | None:
        """Parse a command line into a ControlEvent."""
        parts = line.split()
        if not parts:
            return None
            
        command = parts[0].lower()
        args = parts[1:]
        
        # Validate known commands
        valid_commands = {'swap', 'song', 'tempo', 'like', 'dislike', 'next', 'pause', 'resume', 'seek', 'quit', 'evolve', 'compare', 'left', 'right', 'skip', 'wallpaper'}
        if command not in valid_commands:
            log.warning(f' unknown command: {command}')
            return None
            
        return ControlEvent(command=command, args=args)
