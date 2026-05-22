# Render-Thread Isolation (planned refactor)

Status: filed 2026-05-21 after hitting the watchdog stall when
`Library.rate` ran on the render thread. Hotfix landed in c695a12
(async thumbs-up breed). This doc captures the structural fix that
prevents the next slow handler from doing the same thing.

## Problem

`Orchestrator.tick()` runs on the render thread (called every frame
from `wallpaper.py`). Inside `tick()`:

```python
for cmd in self.control.poll_all():
    for handler in self._command_handlers.get(cmd.command, []):
        handler(cmd)   # ← synchronous, on render thread
```

Any handler that does non-trivial work blocks rendering for the
handler's duration. With a ~4 s watchdog, anything over a couple
hundred milliseconds is at risk.

We hit this once already (thumbs-up breeder, ~3.5 s); patched it by
moving the breeding off-thread *inside the handler*. That works for
`like`/`dislike` but leaves the same trap set for every future slow
handler: evolve, compare-mode load, song-change reset, whatever
comes next.

## The fix: dispatcher thread (option B)

Render thread polls and enqueues; a dedicated dispatcher thread
loops on the queue and runs the handlers.

```python
# Orchestrator.__init__
self._command_queue: queue.Queue = queue.Queue()
self._dispatcher_thread = threading.Thread(
    target=self._dispatch_loop, daemon=True, name='cmd-dispatcher')
self._dispatcher_thread.start()

# Orchestrator.tick() — render thread
for cmd in self.control.poll_all():
    self._command_queue.put(cmd)   # ← non-blocking enqueue

# Orchestrator._dispatch_loop — dispatcher thread
while not self._stop.is_set():
    try:
        cmd = self._command_queue.get(timeout=0.1)
    except queue.Empty:
        continue
    for handler in self._command_handlers.get(cmd.command, []):
        try:
            handler(cmd)
        except Exception:
            log.exception(f'handler crashed: {cmd.command}')
```

Properties:
- Render thread sees only enqueue cost (~microseconds)
- Handlers run sequentially on a single dispatcher thread → no
  surprise concurrency between handlers (the existing assumption)
- One exception in a handler doesn't kill the dispatcher (try/except
  wrapper)
- Library mutations from handlers stay on a single (non-render)
  thread → no SQLite cross-thread issues

## What this is not

- **Not** full main/render thread separation (option C). That's the
  proper architecture but requires explicit "main owns library /
  render owns only frame state" decisions, locks around shared state,
  state-ownership refactor. Filed as future work; option B unblocks
  the symptom in an afternoon.
- **Not** a multi-threaded handler pool. Handlers still run
  sequentially. If we need parallel handling later, that's a third
  refactor on top of B.
- **Not** a fix for handlers that need to *return data to the render
  thread*. Today's handlers mutate library / emit events; the
  dispatcher pattern works because no handler needs synchronous return
  values. If a future handler needs return-to-render-thread, the
  dispatcher pattern needs a result queue.

## When to do this

Before any new handler that does meaningful work ships. The
c695a12 hotfix bought us breathing room only for the specific
thumbs-up case; any other slow handler will recur the bug.

Estimated effort: half a day, including tests that confirm handlers
run on the dispatcher thread (not render thread) and that the queue
doesn't deadlock on shutdown.

## Connection to broader reorg

`docs/reorg_plan.md` discusses worker consolidation (`GpuWorker /
CpuWorkerPool / PeriodicTask`). The dispatcher thread is in the same
spirit — getting work off the render thread when the work doesn't
belong there. Both pieces are part of the same broader architectural
direction: "render thread does rendering; everything else lives
elsewhere."

## Threads vs processes — what Python actually requires

A correction to the framing above: option B (dispatcher thread)
does **not** fix CPU-bound Python handlers. Empirically discovered
2026-05-21 with the thumbs-up breeder:

- c695a12 moved `breed_thumbsup_children` to a daemon thread.
  `rate()` returned in microseconds. **Render stalls persisted.**
- The reason: `Genome.jitter` + `survey_and_correct` are CPU-bound
  Python that hold the GIL between numpy calls. A "background"
  thread still serializes against the render thread on the GIL.
- 491644d moved breeding to a *subprocess*. Render stalls vanished.

The rule, sharpened: **for Python, process isolation is the real
discipline boundary, not thread separation.** Threads protect
against accidentally writing handler code on the render thread,
but they don't protect the render thread from a CPU-bound handler.

Implications for the dispatcher thread (option B):
- Still useful for *I/O-bound* handlers (waiting on disk, network,
  D-Bus, the audio daemon). Those release GIL, threads work fine.
- For CPU-bound work, the dispatcher thread is a routing layer
  that ought to dispatch to subprocesses, not run the handler
  in-thread. "Dispatcher" + "worker subprocesses" together are
  the actual answer.

Practically, this maps to the audio daemon pattern already in use:
the analysis pipeline lives in its own process; the wallpaper
consumes its output via shmem. The same pattern should apply to
library mutations that involve nontrivial Python work — breeding,
evolution, scoring sweeps. Each gets a subprocess (one-shot
spawned via `subprocess.Popen` for episodic work, or a persistent
worker for high-frequency).

Per the configurability-discipline memory, the takeaway is
structural: don't trust a Python "background thread" claim for
CPU-bound work. Look at the inner loop. If it doesn't have
guaranteed GIL release (pure I/O, pure numpy in C), assume thread
boundaries leak.
