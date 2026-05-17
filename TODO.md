## Audio Daemon: client reconnect on daemon restart

When the daemon restarts, existing clients hold stale shmem handles. Need:

1. Daemon emits a `DaemonStarted` dbus signal on startup
2. Client subscribes to it, re-opens `/dev/shm/flame-sheep-audio`, rebuilds ShmReader
3. Fallback: client detects seq counter stalling (>1s no increment) and attempts reconnect

Also: daemon should forward `song_started`, `reset_tempo`, `hint_tempo`, `reset_bands` commands from clients via dbus methods (currently no-op stubs in AudioDaemonClient).

---

## Separate blob storage from genomes table

The genomes table stores render PNGs (render_static, render_swept) and histogram blobs (hist_static, hist_swept, hist_transform, hist_first_hit) inline. At ~350KB per genome compressed, 2400 genomes = ~840MB of blobs in the main table.

This causes:
- Slow table scans (compare mode `_build_candidates()` stalls the render loop scanning 2400 rows of blob metadata)
- SQLite page cache pollution (blob pages evict index/score pages)
- Backup/vacuum operations are slow

Fix: move blobs to a separate `genome_blobs` table joined by genome_id. The main genomes table stays lean (scores, params, metadata). Queries that don't need blobs (fitness ranking, transition lookup, candidate selection) skip the blob table entirely.

```sql
CREATE TABLE genome_blobs (
    genome_id   INTEGER PRIMARY KEY REFERENCES genomes(id) ON DELETE CASCADE,
    render_static BLOB,
    render_swept  BLOB,
    hist_static   BLOB,
    hist_swept    BLOB,
    hist_transform BLOB,
    hist_first_hit BLOB
);
```

Migration: `INSERT INTO genome_blobs SELECT id, render_static, ... FROM genomes`, then drop blob columns from genomes.

---

## Vulkan Migration: GPU issues to fix

### SSBO binding cache (Mesa/Arc)

Mesa caches SSBO bindings per-program at first dispatch. `bind_to_storage_buffer()` is ignored for subsequent dispatches through the same program. This forced the compare mode copy-swap workaround and prevents multiple renderer instances from sharing a GL context cleanly.

**Vulkan fix:** Descriptor sets are explicit per-dispatch. Each dispatch binds its own descriptor set with its own buffer references. No global state, no caching.

### Compare mode blur

Blur uses a cached FBO pool keyed by `(width, height)`. Both compare halves request the same size, get the same FBOs — right side's blur output overwrites left's. Compare mode currently runs without blur.

**Vulkan fix:** Per-dispatch render passes with their own framebuffers. No shared FBO cache.

### Compare mode shader compilation stall

Creating the compare renderer compiles all shaders (~2s on Arc). Pre-creation doesn't help because SSBO bindings from the new renderer clobber the main renderer's bindings.

**Vulkan fix:** Pipeline cache. Compile once, cache to disk, load in <10ms on subsequent runs. Or create pipelines async in a background thread.

### GPU render worker can't coexist with wallpaper

The batch render worker (gpu_render_worker.py) uses 100% GPU and kills desktop performance. Currently runs as a separate CLI tool with the wallpaper stopped.

**Vulkan fix:** Priority queues. Wallpaper renders on a high-priority queue, batch worker on a low-priority queue. GPU hardware scheduler handles preemption. Both run simultaneously without frame drops.

### Two chaos games in compare mode share walkers

Both genomes dispatch through the same walker buffer (65536 walkers). The offset-based histogram approach means both dispatches use the same walkers in the same positions, which means the right genome's walkers start at whatever positions the left genome's chaos game left them in. Not ideal — each genome should have its own converged walker state.

**Vulkan fix:** Separate descriptor sets with separate walker buffers per dispatch. No global binding points to conflict.

### Side screen flicker in compare mode

Side monitors show stale/flickering data during compare mode because the main renderer's histogram isn't being updated (compare renderer owns the dispatch). Currently ignored.

**Vulkan fix:** Multiple render passes — compare renderer handles center monitor, main renderer handles side monitors, each with their own pipeline state.

### Extend leak tracking to images/framebuffers/samplers

`VkCompute` currently tracks `buffer_count` and `pipeline_count` via `_live_buffers` / `_live_pipelines` sets. Used by `tests/visualizer/test_vk_leaks.py` to catch buffer/pipeline leaks in training loops (a real bug that caused a full system lockup before the leak fix).

When the renderer moves to Vulkan, add the same pattern for image-class resources: `_live_images`, `_live_framebuffers`, `_live_samplers`, `_live_descriptor_pools` if not subsumed by pipeline. Each new resource wrapper should accept `ctx=self` in its constructor and call `ctx._live_<type>.discard(id(self))` in `destroy()`. Then add a render-loop leak test mirroring `test_training_step_no_leak`: warm up, snapshot, render N frames, assert counts flat.
