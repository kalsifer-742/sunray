# Sunray — GPU Performance Report

Companion to `PERFORMANCE_ANALYSIS.md` (which covers CPU / submission-side work).
This report focuses on the **device-side** cost of the render: the two RT raygen
passes, temporal accumulation, the 8 a-trous denoise passes, postprocess, and the
acceleration structures they trace against.

Per-frame GPU pipeline today (all at full output resolution):

```
[head: arena copies + TLAS build]
  → RIS raygen      (primary trace + virtual-bounce loop + RIS 16 cand + DI/GI temporal reuse + GI sample)
  → final raygen    (primary trace AGAIN + path trace 10 bounces + DI/GI spatial reuse + NEE)
  → temporal accum  (compute)
  → denoise ×8      (compute, a-trous ping-pong)
  → postprocess     (compute)
```

Each entry below is **Problem / Why it's a problem / Fix**, tagged with rough impact.
Sections: **A** existing waste to remove, **B** features to add, **C** architectural
reworks.

Status legend: 🔲 not yet done.

---

## A. Performance optimizations (waste in the current GPU path)

### A1. The final pass re-traces the primary ray the RIS pass already resolved 🔲 — **High**

**Where:** `shaders/ray_gen_ris.slang` (writes the G-buffer), `shaders/ray_gen_final.slang:74-88`.

**Problem.** The RIS pass already traces the primary ray (including the
glass/mirror virtual-bounce loop) and writes a full G-buffer: depth, normal+roughness,
diffuse/albedo, motion vector. The **final** pass then does `TraceRay(...)` for the
primary ray a *second* time from the camera (bounce 0) and re-walks the same
glass/mirror chain, recomputing `hitPos`, `hit_normal`, `hit_albedo`, roughness,
metallic, transmission — all values that are already sitting in the G-buffer.

**Why it's a problem.** The primary ray is the most expensive ray in the frame
(fully incoherent against the TLAS, plus a closest-hit invocation that fetches three
vertices via BDA and samples up to three textures). Doing it twice per pixel roughly
*doubles* primary-traversal + closest-hit cost, and adds a redundant SBT closest-hit
dispatch per pixel.

**Fix.** Reconstruct the primary hit in the final pass from the G-buffer instead of
re-tracing: world position from `depth` + the camera ray, normal/roughness from
`normal_img`, albedo from `diffuse_img`. The only fields not already in the G-buffer
are `metallic` and `transmission`; add a small material channel (or pack metallic into
`diffuse.a`, which is currently written as `0.0`). Start the final pass's random walk at
bounce 1. This is the single biggest device-side redundancy in the frame.

---

### A2. Frame-chain global barrier is a full pipeline drain 🔲 — **High**

**Where:** `src/render_graph/graph.rs:896` (`ALL_COMMANDS → ALL_COMMANDS`,
`General → General`).

**Problem.** Every command buffer opens with one coarse global barrier that orders the
whole frame after *all* prior GPU work on the queue. The code's own TODO
(`graph.rs:894`) flags it: "replace with precise per-resource barriers derived from the
previous frame's end states."

**Why it's a problem.** `General → General` over `ALL_COMMANDS` is a maximal barrier — it
drains the entire GPU pipeline at frame start and, together with the per-pass barriers,
removes any possibility of overlap between independent work. The graph already computes
`resource_end_states`; the information to do this precisely exists.

**Fix.** Emit per-resource barriers (image layout transition + the specific
src/dst stages) only for the handful of resources actually shared frame-to-frame
(the accumulation/denoise/reservoir ping-pongs and the arena buffers), using the
previous frame's recorded end states. Independent passes can then overlap and the
global drain disappears.

---

### A3. TLAS carries `ALLOW_UPDATE` but is only ever full-rebuilt 🔲 — **Med**

**Where:** `src/vulkan_abstraction/acceleration_structure/tlas.rs:108` and `:209`
(both pass `allow_update = true`); the steady-state path is `prepare_rebuild` →
full `BUILD`, never `update`.

**Problem.** The TLAS is created with `ALLOW_UPDATE`, but the per-frame path always does
a full rebuild (`prepare_rebuild`, see `PERFORMANCE_ANALYSIS.md` §5) — `update()` is
dead code in the hot path.

**Why it's a problem.** `ALLOW_UPDATE` forces the driver to build a larger,
less-trace-optimal TLAS (it must reserve refit headroom) and the build itself is slower —
for a capability that is never exercised. Every ray in the frame pays the
slightly-worse-traversal tax.

**Fix.** Build the TLAS with `PREFER_FAST_TRACE` only (drop `ALLOW_UPDATE`) as long as the
per-frame path is a full rebuild. (If you instead want refit — see C5 — keep the flag but
actually call `update`. The current code is the worst of both: the flag's cost without its
benefit.)

---

### A4. No BLAS compaction 🔲 — **Med**

**Where:** `src/vulkan_abstraction/acceleration_structure/blas.rs:68`,
`mod.rs:flags_for` (no `ALLOW_COMPACTION`, no compacting copy anywhere in the module).

**Problem.** BLASes are built once with `PREFER_FAST_TRACE` but never compacted.

**Why it's a problem.** An uncompacted BLAS commonly uses ~1.5–2× the memory of its
compacted form, and the extra footprint hurts traversal memory locality — every ray that
hits that geometry eats more cache misses. For static scene geometry the compaction is a
pure win.

**Fix.** Build static BLAS with `ALLOW_COMPACTION`, query the compacted size via
`cmd_write_acceleration_structures_properties` (COMPACTED_SIZE), then do a one-time
`cmd_copy_acceleration_structure(... MODE_COMPACT)` into a right-sized buffer at scene
load. Zero runtime cost (it's load-time, static geometry), smaller VRAM, faster traversal.

---

### A5. A-trous denoise re-applies albedo on every one of the 8 passes 🔲 — **High (also a correctness bug)**

**Where:** `shaders/denoise.slang:106` (`... * center_diffuse`), driven 8× by
`src/lib.rs:1781` (`DENOISE_PASSES = 8`).

**Problem.** Each a-trous pass reads the previous pass's output as `temporal_result`,
blurs it, then multiplies the result by `center_diffuse` (albedo) before writing. The
next pass reads *that* and multiplies by albedo again. Albedo is therefore compounded ~8
times across the chain.

**Why it's a problem.** Two issues at once: **(correctness)** the output is modulated by
roughly `albedo^8` instead of `albedo^1`; **(performance)** the entire denoise is run on
*radiance* (albedo already baked in) rather than demodulated *irradiance*, which blurs
across albedo edges and forces more passes / stronger edge-stopping to compensate.

**Fix.** Adopt the standard SVGF demodulation: divide the RT color by albedo *before* TAA
and denoise, run all a-trous passes on the albedo-free irradiance, and re-modulate by
albedo exactly once in postprocess (or in the final pass only). Removes 7 redundant
multiplies, fixes the compounding, and sharpens edges so fewer passes are needed.

---

### A6. Eight hardwired full-res a-trous passes, each re-fetching the whole G-buffer 🔲 — **Med**

**Where:** `shaders/denoise.slang:71-104` (25-tap loop, fetches color + depth + normal +
diffuse per tap), `src/lib.rs:1781` (`DENOISE_PASSES = 8`).

**Problem.** Denoise is 8 sequential full-screen compute dispatches, chained by the
ping-pong dependency (hard barrier between each, no overlap). Every pass re-`Load`s
depth, normal, and diffuse for all 25 taps — up to ~100 texture loads per pixel per pass,
~800 per pixel total — and runs at full output resolution regardless of how converged the
image already is.

**Why it's a problem.** This is a large, fixed, bandwidth-bound cost that doesn't scale
down once TAA has converged the signal (after many static frames the input is already
clean, yet all 8 passes still run).

**Fix.** Three independent wins: **(a)** pack depth+normal+roughness into a single RGBA
"compact G-buffer" texture so each tap is one fetch instead of three; **(b)** drop the
default to ~5 passes (typical for a-trous/SVGF) and measure; **(c)** gate denoise strength
on convergence — skip later passes when per-tile temporal variance is low (needs the
variance estimate from B4).

---

### A7. All texture sampling is forced to mip 0 🔲 — **Med**

**Where:** `shaders/rt_utils.slang:132` (`tex.SampleLevel(smp, uv, 0.0)`), used by every
`sample_texture` call in closest-hit / any-hit.

**Problem.** Every material texture fetch uses LOD 0 unconditionally — there is no mip
selection for minified (distant) surfaces.

**Why it's a problem.** Minified surfaces sample the full-resolution mip, which thrashes
the texture cache (incoherent, poor locality across neighboring rays) and aliases. It
costs both performance (cache misses) and quality (shimmer).

**Fix.** Compute a texture LOD from ray cones (Akenine-Möller ray-cone tracking) or a
cheap distance/spread heuristic and pass it to `SampleLevel`. Minified hits then read
small mips → far better texture-cache hit rate and stable filtering.

---

## B. Features to add (better performance-per-quality)

### B1. Per-pass GPU timestamp profiling 🔲 — **High (enabling)**

**Problem.** There is no per-pass GPU timing anywhere — no `vk::QueryPool` of timestamps
around graph passes.

**Why it's missing matters.** Every optimization in this report is currently a guess about
which pass dominates (RIS vs final vs the 8 denoise passes). You can't prioritize without
numbers, and a path tracer's bottleneck shifts with scene/camera.

**Fix.** Add a timestamp query pool, write a timestamp before/after each pass in
`RenderGraph::compile`/`run`, read it back `MAX_FRAMES_IN_FLIGHT` frames late (no stall),
and surface per-pass milliseconds (egui overlay or log). Cheap, and it turns the rest of
this list into a measured backlog.

---

### B2. Resolution scaling — trace at reduced res, reconstruct to full 🔲 — **High**

**Problem.** The RIS pass, final pass, and all denoise passes run at full output
resolution. Cost scales linearly (RT) and worse (denoise bandwidth) with pixel count.

**Why it's missing matters.** Pixel count is the single largest multiplier on a path
tracer's frame time. Most of the scene doesn't need full-res primary sampling.

**Fix.** Trace RIS/final at half-resolution (or checkerboard) and reconstruct to full
res. The motion-vector + TAA infrastructure already exists, so a temporal upscaler is a
natural fit; alternatively integrate a vendor upscaler (DLSS/FSR) on the denoised output.
~2× fewer rays and ~2× less denoise bandwidth for a modest quality cost.

---

### B3. Light importance sampling structure (alias table / light BVH) 🔲 — **Med/High**

**Where today:** `shaders/ray_gen_ris.slang:191` (`min(rnd*num_lights, ...)` — uniform
light pick), fixed `RIS_CANDIDATES = 16`.

**Problem.** RIS draws candidate lights uniformly over all lights, and NEE does the same.
For scenes with many emitters (or emitters of very different power), uniform selection has
high variance, so you need many candidates — each candidate costs a world-space triangle
transform, area computation, and a full `eval_unshadowed_light` BRDF evaluation.

**Why it's missing matters.** Variance directly drives how many candidates and how much
denoising you need. Importance sampling lets you reach the same noise with fewer
candidates → less per-pixel ALU and less denoise.

**Fix.** Build a power-weighted **alias table** (O(1) sample, probability ∝ emitted power ×
area) on the CPU at scene load, or a light BVH for spatially-aware selection. Sample from
it instead of uniformly; fold the selection pdf into the RIS weight. Fewer candidates for
equal quality.

---

### B4. Variance-aware adaptive sampling + proper firefly rejection 🔲 — **Med**

**Where today:** `SAMPLES = 1`, fixed `SPATIAL_SAMPLES = 5` / `GI_SPATIAL_SAMPLES = 3`
(`ray_gen_final.slang`), fireflies handled by hard `min(..., 5.0)` / `min(..., 10.0)`
clamps.

**Problem.** Smooth/converged regions get exactly the same sampling work as noisy edges,
and the only outlier control is biased hard clamps on radiance.

**Why it's missing matters.** Uniform effort wastes rays on already-clean pixels, and the
hard clamps darken bright highlights (visible energy loss) while still letting moderate
fireflies through into the denoiser, which then smears them.

**Fix.** Maintain a per-pixel/per-tile temporal variance (first + second luminance moment,
SVGF-style). Drive spatial sample counts and denoise step count from it (B2/A6), and
replace the hard clamps with a moment-based outlier rejection. Spends the ray budget where
the image is actually noisy.

---

### B5. Selective geometry opacity (restore alpha-cutout without paying any-hit on opaque) 🔲 — **Med (feature + correctness)**

**Where:** `src/vulkan_abstraction/acceleration_structure/blas.rs:108` — geometry is
flagged `OPAQUE` **unconditionally** (with a `//TODO why always opaque?`), while
`shaders/any_hit.slang` implements the glTF `MASK` alpha test.

**Problem.** Because all geometry is `OPAQUE`, the any-hit shader is never invoked, so
alpha-cutout (`MASK`) materials render as fully solid — a correctness gap. The flag is a
blunt global instrument.

**Why it matters.** You want the *performance* of `OPAQUE` (no any-hit invocation) on the
vast majority of geometry, but *correct* cutouts on the few masked meshes. A global flag
forces an all-or-nothing choice.

**Fix.** Set `GeometryFlagsKHR::OPAQUE` per-geometry: opaque-material primitives keep it
(any-hit skipped, fast), MASK/BLEND primitives drop it so any-hit runs only for them.
Cutouts come back with zero any-hit cost on opaque surfaces. (`NO_DUPLICATE_ANY_HIT_INVOCATION`
is already set, which is correct for the alpha test.)

---

## C. Architectural changes

### C1. Collapse the two RT passes (G-buffer + lighting, or a single fused raygen) 🔲 — **High**

**Where:** `src/lib.rs:1525-1548` (two separate raygen passes with a reservoir hand-off
barrier between them).

**Problem.** RIS and final are two full-screen raygen dispatches, each interning its own
pipeline + SBT, separated by a graph-emitted barrier, and each re-establishes the primary
hit (the structural cause of A1).

**Why it's a problem.** Two pipeline launches, two SBT setups, a full-screen
synchronization point between them, and duplicated primary traversal — all per frame.

**Fix (two options).**
- **(a) Explicit G-buffer pass:** one pass writes the G-buffer (could even be raster for
  primary visibility), then RIS and final become pure lighting passes reading it — no
  primary re-trace in either.
- **(b) Fuse RIS + spatial reuse + final shade into one raygen:** do initial candidate
  generation, temporal+spatial reuse, and shading in a single launch, keeping reservoirs in
  registers/groupshared where possible. Eliminates the inter-pass barrier *and* the second
  primary trace. (Spatial reuse across pixels still needs the written reservoir buffer, so
  a split may remain — but the primary trace need not be duplicated.)

---

### C2. Inline ray queries (`RayQuery`) for all visibility/shadow rays 🔲 — **High**

**Where:** every shadow/visibility `TraceRay` with
`ACCEPT_FIRST_HIT_AND_END_SEARCH | SKIP_CLOSEST_HIT_SHADER` — `ray_gen_ris.slang:338-345`
(GI NEE), `ray_gen_final.slang:203-210` (DI shadow), `:269-276` & `:294-301` (GI spatial /
final visibility), `:343-350` (NEE). These are the *most numerous* rays in the frame.

**Problem.** Each of these goes through the full ray-tracing-pipeline machinery —
SBT lookup, payload read/write, a miss-shader invocation — even though all it needs is a
single occlusion boolean.

**Why it's a problem.** SBT indirection and payload traffic are pure overhead for a
visibility test, multiplied across many such rays per pixel (1 GI-NEE + up to 5 DI-spatial
+ 3 GI-spatial + 1 GI-final + per-bounce NEE).

**Fix.** Use `RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH>` (inline RT) for all
visibility rays: no SBT, no payload, terminate on first hit, read `COMMITTED_NOTHING` for
"unoccluded". Keep closest-hit-based pipeline tracing only where you actually need surface
shading. Removes the SBT/payload tax on the dominant ray type.

---

### C3. Async-compute overlap of denoise/postprocess with the next frame's RT 🔲 — **Med**

**Problem.** RT and the 10 compute dispatches (TAA + 8 denoise + postprocess) all run on a
single queue, fully serialized by per-pass barriers.

**Why it's a problem.** The denoise chain is bandwidth-bound; RT is traversal/ALU-bound.
Run back-to-back on one queue, the RT cores idle during denoise and the memory subsystem
idles during RT. The hardware can do both at once.

**Fix.** Submit denoise + postprocess of frame N on an async compute queue, overlapping
frame N+1's RT, synchronized with a timeline semaphore. **Caveat:** the present path is
currently serialized by the NVIDIA driver-crash workaround (`PERFORMANCE_ANALYSIS.md` §8) —
this lands cleanly on the offscreen path now, and on the present path once a fixed driver
ships. Pairs naturally with A2 (precise barriers are a prerequisite for real overlap).

---

### C4. Shrink and re-lay-out the reservoir buffers 🔲 — **Med**

**Where:** `src/vulkan_abstraction/resources/reservoir.rs` (48 B `Reservoir` + 48 B
`ReservoirGI`), 4 full-res buffers (2 DI + 2 GI ping-pong); incoherent neighbor reads in
`ray_gen_final.slang` (5 DI + 3 GI neighbor reservoirs per pixel).

**Problem.** At 1080p the four buffers are ~400 MB, and the final pass's spatial reuse
does many random-access 48-byte reservoir reads per pixel — incoherent global-memory
traffic in the frame's hottest pass.

**Why it's a problem.** VRAM pressure plus uncoalesced bandwidth exactly where the GPU is
busiest.

**Fix.** Pack reservoirs to ~24–32 B: quantize `light_pos`/`sample_pos` (e.g. fp16 or
relative-to-tile), store the DI sample as `light_idx` + barycentric (the world position is
recomputable from the emissive triangle, removing `light_pos` + `light_normal` entirely),
and pack `sample_radiance` as RGBE/fp16. Consider a structure-of-arrays layout so spatial
reuse fetches only the fields the heuristic needs (normal/depth) before pulling the full
record. Less VRAM, more coalesced reads.

---

### C5. TLAS refit instead of full rebuild for transform-only changes 🔲 — **Med**

**Where:** `src/vulkan_abstraction/acceleration_structure/tlas.rs` (`prepare_rebuild` does
a full `BUILD` every frame; see `PERFORMANCE_ANALYSIS.md` §5).

**Problem.** The TLAS is fully rebuilt every frame even when only a few instance transforms
changed (the common animation case).

**Why it's a problem.** A full TLAS build is markedly more expensive than a refit when
topology (instance count, BLAS set) is unchanged.

**Fix.** Track whether only transforms changed since last frame: if so, do a TLAS *update*
(refit, needs `ALLOW_UPDATE` — interacts with A3, pick one strategy); if instance
count/topology changed, do the full rebuild. Decide A3-vs-C5 deliberately rather than
carrying `ALLOW_UPDATE` and rebuilding anyway.

---

## Suggested order of attack

1. **B1 (timestamps)** first — everything else should be measured, not guessed.
2. **A1 + C1** (kill the duplicate primary trace) and **A5** (denoise demodulation) — the
   two biggest device-side wins, and A5 is also a correctness fix.
3. **C2** (inline visibility rays) and **B2** (resolution scaling) — large, well-understood
   levers.
4. **A2** (precise barriers) — unblocks **C3** (async compute).
5. The AS items (**A3, A4, C5**), reservoir packing (**C4**), and the quality-per-cost
   features (**B3, B4, B5, A6, A7**) as the profiler directs.
