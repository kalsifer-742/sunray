# NVIDIA driver crash: presents × CPU frame work on a busy queue (VK_EXT_descriptor_heap dev driver)

Investigation report for the access violation that crashes the `window` example when
the frame loop is pipelined. Bottom line: **a driver-internal NULL-pointer dereference
on an NVIDIA worker thread**, triggered when the renderer does CPU-side frame work
(preparation, recording, submits) while the graphics queue is still executing earlier
work *and* swapchain presents are in the pipe. It is not an API-usage error:
synchronization validation is silent, and the crash survives every structural
workaround attempted (semaphore wait stages, BDA caching, AS-build placement,
submission depth gating, dedicated command pools). Crash frequency scales with how
much the CPU overlaps the busy queue — seconds with full 2-frames-in-flight overlap,
~30 s even with conservative N−1 pacing; only the fully-serialized loop
(queue drained before each frame's CPU work) is soak-stable. The shipped workaround
drains the graphics queue at the start of each presenting frame; offscreen rendering
keeps the full frames-in-flight overlap.

## Environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Ti |
| Driver | 32.0.16.1047 (`nvoglv64.dll`, the **`VK_EXT_descriptor_heap` developer driver**) |
| OS | Windows 11 Pro 10.0.26200 |
| App | `cargo run --example window` (debug), glTF scene `Room.glb`, FIFO present |
| Renderer | sunray @ branch `fable_test` (2-frames-in-flight rework) |

## Symptom

`STATUS_ACCESS_VIOLATION` (0xc0000005) terminating the process within ~1–6 s of
rendering (a few hundred to ~750 frames; under the api_dump layer's slowdown it
reached frame ~749, so it scales with frame count, not wall time). Faulting module is
always the NVIDIA user-mode driver at the **same offset**:

```
Faulting module name: nvoglv64.dll, version: 32.0.16.1047
Exception code:       0xc0000005
Fault offset:         0x000000000015f942
```

(Windows Application event log, Event ID 1000, multiple instances 2026-06-12.)

## Crash dump analysis

Full-memory WER dumps analyzed with `minidump-stackwalk`
(`%LOCALAPPDATA%\CrashDumps\window.exe.*.dmp`):

- **Crashing thread is a driver-internal worker (Thread 12)** — its stack is entirely
  `nvoglv64.dll` + `ntdll.dll`, no application or loader frames.
- The faulting instruction is `mov r8, qword [rsi]` with `rsi = 0` — a **NULL-pointer
  read inside the driver**, deterministic at `nvoglv64.dll+0x15f942`.
- The **main thread** was meanwhile inside a WSI call (in `win32u.dll` syscall invoked
  from `nvoglv64.dll` — the present/acquire path).
- The api_dump trace confirms asynchrony: the last main-thread API call had already
  returned (the dump died mid-line while *printing* its result), i.e. the AV fired on
  another thread while the main thread was in benign query code.

## What triggers it (experiment matrix)

Configurations tested, ~40–180 s soaks; "overlap" = the CPU prepares and records frame
N+1 while frame N's GPU work is still executing (frame-timeline wait at N−2 instead of
N−1):

| # | Configuration | Result |
|---|--------------|--------|
| 1 | Window, full overlap (CPU runs MAX_FRAMES_IN_FLIGHT ahead), TLAS build at head of graph cmd buffer | **crash ≤ ~6 s** |
| 2 | Headless (no swapchain), full overlap, same renderer, same scene, 5000 frames | stable |
| 3 | #1 + acquire-semaphore waited at TRANSFER (spec fix, see below) | **crash** |
| 4 | #3 + `vkGetBufferDeviceAddress` cached (zero per-frame BDA calls) | **crash** |
| 5 | #4 + TLAS build moved to its own async submission (semaphore into graph) | **crash** |
| 6 | #5 + per-frame TLAS builds **disabled entirely** (static TLAS) | **crash** |
| 7 | #5 + submission gated on frame N−1 completion (CPU records ahead, GPU queue depth 1) | **crash** |
| 8 | #7 + dedicated `vkCommandPool` per re-recorded command buffer | **crash** |
| 9 | N−1 timeline pacing (no CPU run-ahead) + the async setup submission of #5 | **crash ≤ ~6 s** |
| 10 | N−1 timeline pacing + head-of-graph TLAS build + all fixes | **crash ~27–40 s** |
| 11 | Committed baseline (fully serialized: `vkDeviceWaitIdle` at frame start, fence wait after submit) | stable (180 s soak) |
| 12 | Rework + `vkQueueWaitIdle` at the start of each presenting frame (shipped) | stable (240 s soak) |

Reading of the matrix:

- The crash needs **presents** (headless full overlap is rock solid for thousands of
  frames) **and** CPU-side driver activity while the queue is busy, but is **not**
  caused by any specific call we make: it persists with no AS builds at all (#6), no
  per-frame BDA queries (#4), isolated command pools (#8), and a GPU queue depth of
  one (#7).
- It is probabilistic per frame and the probability tracks how much CPU driver work
  overlaps the busy queue: full run-ahead ⇒ seconds (#1), N−1 pacing (CPU work still
  overlaps the previous frame's presents/barrier) ⇒ tens of seconds (#10), queue
  drained before CPU work ⇒ stable (#11, #12).
- Splitting the frame into an extra small submission per frame (#5/#9) made it
  *worse* (fast reproduction even without run-ahead), pointing at per-submission
  bookkeeping consumed by the worker thread that also services presents.
- The crashing worker performs a hash-style lookup (splitmix64 mixing constant
  `0xff51afd7ed558ccd` in registers) and dereferences a NULL entry — consistent with
  an internal table being read by the worker while the application thread mutates it
  (insert/remove racing lookup).

## API-correctness findings (fixed, kept regardless)

Two real issues were found and fixed during the investigation — neither was the
trigger, but both are genuine:

1. **Acquire-semaphore wait stage.** The blit that writes the swapchain image waited
   the acquire semaphore at `ALL_GRAPHICS`, which does **not** include the TRANSFER
   stage the blit (and the image's `UNDEFINED` discard transition, which had
   `srcStage=NONE`) executes in — so the swapchain write was never actually ordered
   after the presentation engine released the image. Fixed: wait at `TRANSFER`, and
   chain the discard transition's `srcStage` from `TRANSFER`.
2. **Per-frame `vkGetBufferDeviceAddress` round-trips.** Now cached in `RawBuffer`
   (the address is immutable per buffer object).

## Workaround shipped

`Renderer::serialize_frames_workaround()` (`src/lib.rs`): when the renderer presents
through its internal swapchain, `render()` calls `vkQueueWaitIdle` on the graphics
queue at the **start of the frame**, draining all in-flight GPU work before the CPU
does any driver-side frame preparation, recording, or submission. This reproduces the
one soak-stable condition (no CPU driver activity while the queue is busy) while
keeping the entire structural rework intact: per-slot upload buffers, the
double-buffered TLAS built inside the frame's command buffer, rotated per-slot graph
command buffers, zero synchronous submits, and zero per-frame allocations are all
unchanged — only the run-ahead is removed for the presenting path. Offscreen rendering
(no swapchain) keeps the full `MAX_FRAMES_IN_FLIGHT` overlap and is unaffected.

Set **`SUNRAY_FULL_FRAMES_IN_FLIGHT=1`** to re-enable full overlap with a swapchain —
use this to re-test whenever a newer descriptor-heap driver lands. If it survives a
few minutes of the `window` example, the workaround can be removed.

## Reproducing for a driver report

1. Driver 32.0.16.1047, any descriptor-heap-capable build presumably.
2. `SUNRAY_FULL_FRAMES_IN_FLIGHT=1 cargo run --example window` (validation on or off —
   it crashes either way and validation reports nothing beforehand).
3. Crashes in seconds; WER dump shows the worker-thread NULL deref at
   `nvoglv64.dll+0x15f942`.
