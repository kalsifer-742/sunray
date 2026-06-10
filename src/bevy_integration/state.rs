//! Resources for the Bevy integration.
//!
//! Three live in the **render world**:
//! - [`SunrayWindows`] / [`ExtractedCamera`] / [`ExtractedScene`] — plain Send
//!   snapshots written by the extract systems.
//! - [`SunrayRenderState`] — the renderer + window-bound GPU objects. It is a
//!   **NonSend** resource because [`crate::Renderer`] is `Rc`-based (`!Send`);
//!   the single-threaded render SubApp guarantees it's only touched on the main
//!   thread (see `docs/bevy_integration.md`).
//!
//! One lives in the **main world**: [`SunrayScene`], the user-facing handle for
//! requesting a glTF load.

use std::collections::HashMap;

use ash::vk;
use bevy_ecs::prelude::*;
use bevy_window::RawHandleWrapper;

use crate::vulkan_abstraction::image::swapchain::{Surface, Swapchain};
use crate::{Renderer, vulkan_abstraction};

/// Main-world resource: which scene to load. Set [`gltf_path`](Self::request)
/// and the render world will (re)load it when the generation changes.
#[derive(Resource, Default)]
pub struct SunrayScene {
    pub(crate) gltf_path: Option<String>,
    pub(crate) generation: u64,
}

impl SunrayScene {
    /// Request that `path` be loaded (replacing any current scene) on the next
    /// render frame.
    pub fn request<P: Into<String>>(&mut self, path: P) {
        self.gltf_path = Some(path.into());
        self.generation += 1;
    }

    /// Convenience constructor for `App::insert_resource`.
    pub fn with_gltf<P: Into<String>>(path: P) -> Self {
        Self {
            gltf_path: Some(path.into()),
            generation: 1,
        }
    }
}

/// Extracted (Send) snapshot of the Bevy windows the renderer cares about.
#[derive(Resource, Default)]
pub struct SunrayWindows {
    pub windows: HashMap<Entity, SunrayWindowInfo>,
}

pub struct SunrayWindowInfo {
    /// `RawHandleWrapper` is force-`Send + Sync`; only *using* the handle is
    /// thread-restricted, which we honor by creating the surface in a NonSend
    /// (main-thread) system.
    pub handle: RawHandleWrapper,
    pub physical_size: (u32, u32),
    pub size_changed: bool,
    pub is_primary: bool,
}

/// Extracted (Send) camera for the current frame. Stored as plain components so
/// the resource stays `Send` without depending on `sunray::Camera: Clone`.
#[derive(Resource, Default, Clone, Copy)]
pub struct ExtractedCamera {
    pub present: bool,
    pub eye: [f32; 3],
    pub target: [f32; 3],
    pub fov_y_degrees: f32,
}

/// Extracted (Send) copy of the scene-load request.
#[derive(Resource, Default)]
pub struct ExtractedScene {
    pub gltf_path: Option<String>,
    pub generation: u64,
}

/// The renderer + per-window GPU objects. **NonSend** (see module docs).
#[derive(Default)]
pub struct SunrayRenderState {
    pub renderer: Option<Renderer>,
    pub surface: Option<Surface>,
    pub swapchain: Option<Swapchain>,
    /// Window entity that owns the renderer/surface/swapchain (single-window).
    pub owner: Option<Entity>,

    // Per-frame synchronization, mirroring `examples/window/main.rs`.
    pub img_acquired_sems: Vec<vulkan_abstraction::Semaphore>,
    pub img_rendered_fences: Vec<vk::Fence>,
    pub ready_to_present_sems: Vec<vulkan_abstraction::Semaphore>,
    /// One pre-recorded GENERAL -> PRESENT_SRC barrier per swapchain image.
    pub present_barrier_cmd_bufs: Vec<vulkan_abstraction::CmdBuffer>,

    pub frame_count: u64,
    pub loaded_scene_generation: u64,
    pub image_format: vk::Format,

    /// egui GPU paint backend, built lazily on the first frame that has an
    /// `ExtractedEgui` resource (i.e. when `SunrayEguiPlugin` is active).
    pub egui_paint: Option<super::egui_paint::EguiPaint>,
}
