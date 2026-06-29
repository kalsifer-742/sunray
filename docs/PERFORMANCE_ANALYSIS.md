# Sunray — Performance & Structural Analysis

Analysis of the renderer's hot path (`Renderer::render` → render graph → blit/present),
focused on optimization opportunities. Problems are ordered by expected impact.
Each entry: what the problem is, why it hurts, and the fix.

Status legend: ✅ fixed in this branch · 🔲 documented, not yet fixed (larger rework).

---

## 1. The frame loop is fully serialized — zero CPU/GPU overlap ✅

**Where:** `src/lib.rs`, `Renderer::render`.

**Problem.** A single `render()` call contains, in order:

1. `device_wait_idle()` at frame start (drains *all* queues),
2. a synchronous TLAS rebuild (`submit_sync` = submit + fence wait, a full CPU→GPU→CPU round trip),
3. a **second** `device_wait_idle()` immediately before `render_graph.run()`,
4. `render_graph_fence.wait()` immediately **after** `run()` — the CPU blocks until the
   entire frame's GPU work finishes before even submitting the final blit.

**Why it's a problem.** `MAX_FRAMES_IN_FLIGHT = 2` is fictional: the CPU never overlaps
with the GPU at all. The GPU idles while the CPU builds the next frame's graph, and the
CPU idles while the GPU renders. The second `device_wait_idle` is provably redundant
(everything submitted between the frame-start wait and `run()` is itself synchronous).
The post-`run` fence wait is unnecessary for correctness: the pre-recorded blit command
buffer starts with a `COMPUTE_SHADER → TRANSFER` pipeline barrier, which on the same
queue orders it after the graph's passes (the graph's internal barriers chain the RT
stages into the compute stages transitively), and the *next* frame already waits the
graph fence before re-recording the command buffer.

**Fix (applied).**
- Replace the frame-start `device_wait_idle()` with `frame_timeline.wait(previous frame)`
  — same safety guarantee for everything the renderer submitted (the timeline value is
  signaled by the blit, which transitively orders all graph work), without draining
  unrelated queue work.
- Delete the second `device_wait_idle()`.
- Delete the `render_graph_fence.wait()` after `run()`. The fence is still waited at the
  top of the next frame before the command buffer is re-recorded.

**Result:** `render()` now returns right after submitting the GPU work. All CPU work the
app does between frames (game logic, Bevy ECS, input) overlaps GPU rendering, and the
per-frame GPU pipeline drains are gone. (Full 2-frames-in-flight overlap of *renderer*
CPU work landed afterwards — see §8: the timeline wait moved from `N − 1` to
`N − MAX_FRAMES_IN_FLIGHT`.)

---

## 2. Frame-watcher thread: constant polling + up to 100 ms latency on deferred frees ✅

**Where:** `src/lib.rs` (`frame_watcher`, `completed_frame`).

**Problem.** A dedicated thread polls `vkWaitSemaphores` in a 100 ms-timeout loop purely
to publish "last completed frame" into an `AtomicU64`. The render thread reads that to
drain end-of-frame callbacks.

**Why it's a problem.** The thread wakes ~10×/s forever (power/scheduling noise), the
deferred deallocations can lag up to 100 ms behind GPU completion, and it's ~40 lines of
shutdown/ordering machinery — all replaceable by one cheap driver call.

**Fix (applied).** Call `vkGetSemaphoreCounterValue` (`TimelineSemaphore::counter_value`)
directly in `run_due_end_of_frame_callbacks`. The watcher thread, the shutdown flag, and
the atomic are deleted.

---

## 3. Per-frame SPIR-V blob copies and re-hashing ✅

**Where:** `src/lib.rs` (`build_unified_graph`), `src/render_graph/pass_builder.rs`,
`src/render_graph/graph.rs` (`pipeline_cache_key`).

**Problem.** Every frame, while *rebuilding the graph description* (the pipelines
themselves are correctly cached):
- `build_unified_graph` clones every shader blob with `.to_vec()` — 8 RT stage copies +
  1 TAA + 8 denoise + 1 postprocess ≈ **18 full SPIR-V heap copies per frame**;
- `RaytracingRenderPassBuilder::generate_render` copies each stage **again** to build the
  `RayTracingPipelineShaders` value used only for the cache lookup;
- `pipeline_cache_key` hashes the **full byte content** of every shader every frame just
  to find an already-interned pipeline (SipHash over ~1 MB+ of SPIR-V per frame).

**Why it's a problem.** Pure allocator + memcpy + hash churn on the hot path that grows
with shader count; for the 8 a-trous passes the same blob is copied and hashed 8 times.

**Fix (applied).** `ShaderSource::Spirv` now holds a `SpirvBlob`: an `Arc<[u8]>` plus a
content hash computed **once** at construction. The renderer builds its blobs once in
`Renderer::new`; per-frame "copies" are `Arc` clones, and the pipeline-cache key combines
the precomputed per-stage hashes instead of re-hashing the bytes. The actual
`Vec<u8>` copy into `RayTracingPipelineShaders` happens only on a cache miss (i.e. once
per distinct pipeline, at startup).

---

## 4. Per-frame GPU buffer allocation churn (camera/instances/transforms/emissive) ✅

**Where:** `src/lib.rs`, `Renderer::render`.

**Problem.** Every frame allocates four CpuToGpu buffers from `gpu_allocator`
(camera-matrices UBO, TLAS instances, per-instance transforms, emissive indirection),
each of which also lazily allocates a descriptor-heap slot, and then frees all four
through an end-of-frame callback.

**Why it's a problem.** 4 allocations + 4 frees + descriptor-slot alloc/free per frame,
plus the boxed-callback machinery to keep them alive. Allocator churn fragments memory
and shows up directly in frame time.

**Fix (applied).** The four buffers are persistent `Renderer` fields, overwritten in
place each frame (since the §8 rework: one set per frame in flight, and the wait
guarantees the *slot's* last frame completed before its buffers are touched). Growth
policy:
- instances / transforms: grow-only (extra capacity is harmless — the TLAS build takes an
  explicit `instance_count`, transforms are indexed `< count`);
- emissive indirection: recreated when the element **count changes**, because the shader
  derives `num_lights` from the buffer size (`GetDimensions`), so the size is semantic.

---

## 5. Synchronous TLAS path: per-frame driver queries, allocations, and a sync build ✅

**Where:** `src/vulkan_abstraction/resource_manager.rs` (`frame_instance_data`),
`src/vulkan_abstraction/acceleration_structure/*`.

**Problem.**
- `frame_instance_data` calls `vkGetAccelerationStructureDeviceAddressKHR` for every
  BLAS key, every frame, although a BLAS address is immutable. ✅ **fixed:** the address
  is cached at `add_blas` time.
- `TLAS::rebuild_from_buffer` → `AccelerationStructure::rebuild` recreated **everything**
  each frame: a new AS buffer, a new scratch buffer, a new `vkAccelerationStructureKHR`,
  records a one-shot command buffer, and submits it **synchronously** (fence wait). ✅
  **fixed** (together with the §8 frames-in-flight rework):
  - `AccelerationStructure` now splits creation from building (`new_unbuilt` +
    `record_build`) and tracks the `acceleration_structure_size` it was created with, so
    a rebuild whose required size fits **reuses the same handle and buffer** (Vulkan
    permits re-BUILDing into an existing AS of sufficient size). The scratch buffer is a
    persistent grow-only `TLAS` field, and the per-frame
    `vkGetAccelerationStructureBuildSizesKHR` query is cached by instance count (for a
    TLAS the sizes depend only on the count + flags), so a steady-state frame performs
    **zero** driver object creation and zero size queries for the TLAS.
  - The build itself is recorded at the head of the render graph's command buffer
    (`TLAS::prepare_rebuild` returns a plain-data `TlasBuildRecording`; the renderer
    installs it via `RenderGraph::set_head_render`), followed by an
    `ACCELERATION_STRUCTURE_BUILD → RAY_TRACING_SHADER` memory barrier. The last
    synchronous submit in the frame is gone.
  - The arena staging→GPU copy flush (`flush_queued_copies`, previously a sync submit
    on load frames) rides the same path: `ResourceManager::take_queued_copies` drains
    the queue each frame and the copies + their `TRANSFER → RT|COMPUTE` barriers are
    recorded in the same head-of-graph closure.

---

## 6. Render-graph transient resources are destroyed and recreated every frame ✅

**Where:** `src/render_graph/graph.rs`, `TransientResources::populate`.

**Problem.** The graph is rebuilt each frame (intended), but `populate` also freed and
re-created all transient backing state each frame: 5 `vkCreateImage` + views + memory
binds, slot allocations freed and re-allocated through `gpu_allocator`, and
descriptor-heap slots freed/re-reserved — even though the resource descriptions and
lifetimes are identical frame to frame (the code's own TODO noted this).

**Why it's a problem.** Per-frame Vulkan object churn (images, views, descriptor writes)
and allocator traffic with a 100%-predictable outcome.

**Fix (applied).** `populate` fingerprints everything the slot assignment depends on:
each created resource's desc + (first_pass, last_pass), plus the (sorted) pass-component
structure. When the fingerprint matches the previous frame's, phases 1–4 are skipped —
the wrappers, descriptor slots, and slot allocations are reused verbatim and only the
imported-handle maps are rewired. `RenderGraph::reset` now keeps the transient backing
alive (`clear_frame_state`) instead of freeing it; any change (e.g. a resize altering the
extents) misses the fingerprint and takes the old full-rebuild path. The initial
UNDEFINED→first-use discard transition `compile` emits stays valid for reused images
(`oldLayout = UNDEFINED` is always legal). Covered by the new
`transient_reuse_across_recompile` test (same graph twice → same `vk::Image` handles;
changed desc → rebuilt).

---

## 7. O(n²) callback drains ✅

**Where:** `src/lib.rs` (`run_start_of_frame_callbacks`, `run_due_end_of_frame_callbacks`),
`src/vulkan_abstraction/resource_manager.rs` (`start_of_frame`).

**Problem.** Due callbacks are removed with `Vec::remove(i)` inside a scan loop —
quadratic shifting, and `remove` shifts the whole tail on every hit.

**Why it's a problem.** Minor today (small N), but it sits on the per-frame path and the
deferred-deallocation design encourages N to grow (every `render` pushes a callback).

**Fix (applied).** Drain due entries by partitioning (swap-free single pass collecting
due callbacks, `retain`-style) instead of repeated `remove`.

---

## 8. Larger reworks

- **True 2-frames-in-flight.** ✅ **fixed.** Everything per-frame is now replicated per
  in-flight slot and `render()` throttles on the frame timeline at
  `N − MAX_FRAMES_IN_FLIGHT` instead of `N − 1`, so the CPU prepares and records frame
  N+1 while frame N's GPU work is still executing:
  - *Per-frame-in-flight graph command buffers:* `RenderGraph` owns
    `MAX_FRAMES_IN_FLIGHT` primary command buffers, rotated by `compile()`; each is
    guarded by its own submission fence (waited before re-record), so the rotation is
    safe regardless of how the caller paces frames. `run()` no longer takes an external
    fence (the renderer's `render_graph_fence` and its mid-frame wait are deleted).
  - *Double-buffered TLAS + frame-local buffers:* the `ResourceManager` holds one TLAS
    per slot (built inside the slot's own submission, §5b), and the renderer holds one
    `FrameUploadBuffers` set (camera UBO, TLAS instances, transforms, emissive
    indirection) per slot.
  - *Per-frame descriptor versioning:* falls out of the per-slot buffers — each slot's
    buffers own their descriptor-heap slots, resolved into that frame's push constants;
    a buffer is only recreated (freeing + rewriting descriptors) once the timeline says
    its last frame completed. The TLAS heap descriptor is rewritten in place when a
    grow recreates the AS (also fixing a latent stale-descriptor bug in the old rebuild
    path).
  - *GPU-side frame chaining:* consecutive frames share physical resources (the reused
    transient images — whose UNDEFINED-discard init transitions have no source scope —
    the accumulation/denoise/reservoir ping-pongs, the arena buffers). `compile()` opens
    every command buffer with one coarse `ALL_COMMANDS → ALL_COMMANDS` global barrier so
    frame N+1's GPU work chains after frame N's, exactly as before the rework — GPU
    frames stay serialized (they are data-dependent anyway); the overlap won is CPU
    recording vs. GPU execution. A fingerprint miss in `populate` (resize) now drains
    the queue before freeing live transient state. (TODO noted in code: replace the
    coarse barrier with precise per-resource barriers from `resource_end_states`.)
  - Covered by the `frames_in_flight_overlap` test: several frames rendered
    back-to-back with no waits in between (alternating destination images like a
    swapchain), asserting zero validation errors with synchronization validation on.
  - *Driver caveat (presenting path only).* Overlapping renderer CPU work with a busy
    queue **while swapchain presents are in the pipe** crashes the current NVIDIA
    `VK_EXT_descriptor_heap` developer driver (32.0.16.1047) with a NULL deref on an
    internal worker thread — not an API-usage error (sync validation is clean; the crash
    survives every structural workaround). `render()` therefore calls `vkQueueWaitIdle`
    at the start of each frame **when presenting through the internal swapchain**
    (`serialize_frames_workaround`), draining the queue before any CPU driver work.
    Offscreen rendering (no swapchain) keeps the full overlap and is unaffected. Set
    `SUNRAY_FULL_FRAMES_IN_FLIGHT=1` to re-enable full overlap with a swapchain and
    re-test when a fixed driver lands. Full investigation:
    `docs/NVIDIA_DRIVER_CRASH_REPORT.md`.
- **TLAS build inside the graph** (§5b): ✅ **fixed** — see §5.

Documented, intentionally not done here:

- **`load_scene` / `unload_*` use `device_wait_idle`** — acceptable for rare operations,
  but they could key off the frame timeline like everything else.
- **Structure:** `lib.rs` (2k lines) mixes renderer orchestration, graph assembly,
  swapchain/present plumbing, and the blit; `Core` is `Rc`/`RefCell`-single-threaded and
  passed wholesale where only the device is needed (its own TODO). Worth splitting when
  the dust settles, but it does not affect runtime performance.
