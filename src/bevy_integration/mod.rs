//! Bevy 0.19 integration for the sunray ray-tracer.
//!
//! This module replaces Bevy's stock wgpu `RenderPlugin` with a backend that
//! drives [`crate::Renderer`] directly. It keeps Bevy's ECS, windowing (winit),
//! input, time and transforms, and reuses `bevy_render::extract_plugin::ExtractPlugin`
//! for the `RenderApp` SubApp + extraction bridge. Rendering is **single-threaded**
//! (the renderer is `Rc`-based / `!Send`), so the render SubApp runs on the main
//! thread and the renderer lives in a NonSend resource.
//!
//! See `docs/bevy_integration.md` for the full architecture and
//! `examples/bevy_app` for usage.
//!
//! # Quick start
//! ```ignore
//! App::new()
//!     .add_plugins((/* minimal Bevy plugins, NOT RenderPlugin */))
//!     .add_plugins((SunrayRenderPlugin::default(), SunrayEguiPlugin))
//!     .insert_resource(SunrayScene::with_gltf("examples/assets/Room.glb"))
//!     .add_systems(Startup, |mut commands: Commands| {
//!         commands.spawn((Transform::from_xyz(0.0, 2.0, 10.0), SunrayCamera::default()));
//!     })
//!     .run();
//! ```

mod camera;
mod egui_paint;
mod egui_support;
mod plugin;
mod state;
mod surface;
mod swapchain;
mod systems;

pub use camera::SunrayCamera;
pub use egui_support::{EguiContext, EguiFrameOutput, ExtractedEgui};
pub use plugin::{SunrayEguiPlugin, SunrayRenderPlugin};
pub use state::{ExtractedCamera, ExtractedScene, SunrayRenderState, SunrayScene, SunrayWindows};
