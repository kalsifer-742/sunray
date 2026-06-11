# Bevy integration (`sunray::bevy_integration`)

A Bevy **0.19** plugin that replaces Bevy's stock wgpu `RenderPlugin` with a
backend that drives the sunray hardware ray-tracer directly on `ash`. Bevy keeps
ownership of ECS, windowing (winit), input, time and transforms; only the
rendering is ours.

Gated behind the `bevy` cargo feature. Run the example with:

```text
cargo run --release --features bevy --example bevy_app
```

---

## 1. Locked-in decisions (and why)

| Decision | Choice | Why |
|---|---|---|
| Threading | **Single-threaded render SubApp** (no `PipelinedRenderingPlugin`) | `sunray::Renderer` is `Rc`-based, hence `!Send`. Single-threaded means the render SubApp runs on the **main thread**, so the renderer can live in a **NonSend** resource and the raw window handle is always used in a valid context. |
| Extraction machinery | **Reuse `bevy_render::extract_plugin::ExtractPlugin`** | Free `RenderApp` SubApp, `ExtractSchedule`, `Render` base schedule, `SyncWorldPlugin`, and the `Extract<P>` param. `bevy_render`'s own unit test does exactly `ExtractPlugin::default()` + `update_schedule = Some(Render.intern())`, which is our pattern. |
| Surface / device / swapchain | **Lazy, in a render system** | The window handle only exists after the winit event loop starts. `Renderer::new_with_surface` creates instance+device+surface together from that handle, so the *whole* renderer is created lazily on the first window. |
| Window state in render world | Custom `SunrayWindows` (Send) + `SunrayRenderState` (NonSend) | We never touch wgpu types, so Bevy's `ExtractedWindows` (wgpu-coupled) is unusable. |
| egui | Raw `egui`, our own input layer, our own heap+Slang paint backend | `bevy_egui`'s paint node needs the wgpu `RenderDevice` we removed; `egui-ash-renderer` violates the heap+Slang-only rule. See §6. |

### How this differs from a generic "ash + egui backend"

- A generic backend creates the `Instance` in `finish()` and the device lazily.
  Here, sunray owns instance+device+surface inside `Renderer`, all created in one
  call (`new_with_surface`) that needs the window handle — so **everything** is
  lazy, in `ensure_renderer`.
- The renderer renders a full frame into its own image and **blits** the
  post-process result into the swapchain image; we then flip it to `PRESENT_SRC`.
  There is no per-pass wgpu command encoding to replace.

---

## 2. Per-frame flow

```
main schedule (game logic, fly-cam, egui UI)
   │  ExtractSchedule (reads main world via Extract<P>)
   │    extract_windows  -> SunrayWindows
   │    extract_camera   -> ExtractedCamera
   │    extract_scene    -> ExtractedScene
   │    extract_egui     -> ExtractedEgui
   ▼
RenderApp.update()  (Render schedule, RenderSystems::Render, main thread)
     ensure_renderer   (lazy create / resize / scene (re)load)
     render_frame      (acquire image, set_camera, render_to_image,
                        GENERAL->PRESENT_SRC barrier, present)
     paint_egui        (consumes ExtractedEgui — GPU paint TODO, §6)
```

`render_frame` mirrors `examples/window/main.rs`'s `draw()`: it is the proven
acquire/render/present loop, lifted into ECS systems.

---

## 3. File map

| File | Responsibility |
|---|---|
| `plugin.rs` | `SunrayRenderPlugin` (adds `ExtractPlugin`, sets `update_schedule = Render`, registers systems + NonSend state) and `SunrayEguiPlugin`. |
| `state.rs` | Resources: `SunrayScene` (main world), `SunrayWindows` / `ExtractedCamera` / `ExtractedScene` (render world, Send), `SunrayRenderState` (render world, NonSend). |
| `systems.rs` | `extract_windows` / `extract_camera` / `extract_scene`; `ensure_renderer`; `render_frame`. |
| `camera.rs` | `SunrayCamera` component + `Transform` -> eye/target/fov. |
| `surface.rs` | `raw-window-handle` 0.6 port of surface-extension enumeration + surface creation. |
| `swapchain.rs` | `Surface` (RAII) + `Swapchain` (port of the example helpers; exposes `format()`/`image_views()`). |
| `egui_support.rs` | egui `Context`, input mapping, tessellation, extract. |
| `egui_paint.rs` | egui GPU paint backend: texture manager, vertex/index upload, dynamic-rendering overlay pass (heap + Slang). |

Plus, in the core renderer (not feature-gated): `vulkan_abstraction/graphics_pipeline.rs` —
`GraphicsPipeline::new_heap`, the project's first raster pipeline (null layout +
`DESCRIPTOR_HEAP_EXT`), and `shaders/egui.slang` (vertex + fragment, compiled to
SPIR-V at build time by `build.rs`).

---

## 4. Public API

```rust
use sunray::bevy_integration::{
    SunrayRenderPlugin, SunrayEguiPlugin, // plugins (add egui after render)
    SunrayScene,    // main-world resource: request a glTF load
    SunrayCamera,   // component: put on a Transform entity to be the view
    EguiContext,    // resource: build UI from your own Update systems
};
```

- `SunrayScene::with_gltf(path)` / `scene.request(path)` — (re)load a scene; the
  render world loads it once the device exists, keyed on a generation counter.
- `SunrayCamera { fov_y_degrees }` — the entity's `Transform` is the eye; it looks
  down local `-Z` (Bevy's forward). Only the first matching entity is used.
- `EguiContext::ctx()` — call `egui::Window::new(..).show(egui_ctx.ctx(), ..)` from
  an `Update` system, between `egui_begin` (PreUpdate) and `egui_end` (PostUpdate).

The example builds the **minimal** plugin set (Option B): `TaskPoolPlugin`,
`FrameCountPlugin`, `TimePlugin`, `InputPlugin`, `WindowPlugin`,
`AccessibilityPlugin`, `WinitPlugin`, `TransformPlugin` — **no** `RenderPlugin`.

---

## 5. Bevy 0.19 fork specifics

- Buffered events are **messages**: read with `MessageReader<T>`, not
  `EventReader<T>` (`CursorMoved`, `MouseWheel`, `KeyboardInput`, `MouseMotion`
  all derive `Message`).
- `WinitPlugin` is **not** generic here (`WinitPlugin::default()`).
- NonSend insert is `World::insert_non_send` (`insert_non_send_resource` is
  deprecated).
- `RawHandleWrapper::{get_window_handle, get_display_handle}` are safe `Copy`
  getters; we use them directly from the main-thread NonSend system.

---

## 6. egui — implemented (heap + Slang)

egui is integrated as **Option C**: raw `egui` + a thin `bevy_input`-driven
input layer + tessellation in the main world, extracted into the render world as
`ExtractedEgui { primitives, textures_delta, pixels_per_point }`, then **painted
on the GPU** by `egui_paint.rs`. We deliberately do **not** use `egui-ash-renderer`
(it bundles a descriptor-set + GLSL/SPIR-V raster pipeline — forbidden by the
heap+Slang-only rule — and pulls a crates.io `ash` that clashes with the git one).

What was built:

- `GraphicsPipeline::new_heap` — heap-mode raster pipeline (null layout +
  `DESCRIPTOR_HEAP_EXT`, push constants via `cmd_push_data`), 2D-overlay fixed
  state (triangle list, cull none, premultiplied-alpha blend, dynamic
  viewport/scissor, single color attachment via dynamic rendering).
- `shaders/egui.slang` — vertex (points → clip) + fragment (samples the texture
  through `DescriptorHandle<Texture2D>` + `DescriptorHandle<SamplerState>`,
  applies egui's premultiplied-alpha gamma math). Compiled at build time to
  `egui_vert.spirv` / `egui_frag.spirv`.
- `EguiPaint` — per-`TextureId` texture manager (uploaded via `Image::new_from_data`,
  addressed by `sampled_slot()`; sub-region deltas patch a CPU backing then
  re-upload), grow-only host-visible vertex/index `RawBuffer`s, and per-image
  command buffers.
- Device: `PhysicalDeviceVulkan13Features::dynamic_rendering(true)` (one line in
  `core/device.rs`).

### Where the draw happens

In `systems.rs::render_frame_impl`, right after `render_to_image` the swapchain
image is in `GENERAL`. If `ExtractedEgui` is present (i.e. `SunrayEguiPlugin` is
active), `EguiPaint::paint_frame` runs instead of the plain present barrier:

```
GENERAL -> COLOR_ATTACHMENT_OPTIMAL
begin dynamic rendering (LOAD the existing ray-traced color)
  apply ExtractedEgui.textures_delta (set/free)
  per mesh: scissor = clip_rect * ppp, push texture+sampler slots, draw_indexed
end rendering
COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC
```

Same-queue submission order (after the renderer's blit) + the layout barrier
provide the dependency on the blit, so no extra semaphore is needed.

### Known tuning point

Gamma/premultiplied-alpha handling is the classic egui correctness knob. The
target is the **sRGB** swapchain, so the fragment shader converts its gamma-space
result to linear (`linear_from_srgb`, a `pow(.,2.2)` approximation) before output
and the hardware re-encodes to sRGB on store. If colors look washed out or too
dark on real hardware, that conversion (and the premultiply assumptions on the
font atlas) is the first thing to adjust — this is the one part that needs visual
verification on a GPU.
