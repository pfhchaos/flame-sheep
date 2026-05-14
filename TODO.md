## Audio Daemon: client reconnect on daemon restart

When the daemon restarts, existing clients hold stale shmem handles. Need:

1. Daemon emits a `DaemonStarted` dbus signal on startup
2. Client subscribes to it, re-opens `/dev/shm/flame-sheep-audio`, rebuilds ShmReader
3. Fallback: client detects seq counter stalling (>1s no increment) and attempts reconnect

Also: daemon should forward `song_started`, `reset_tempo`, `hint_tempo`, `reset_bands` commands from clients via dbus methods (currently no-op stubs in AudioDaemonClient).
