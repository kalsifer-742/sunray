//! The Bevy plugins.
//!
//! [`SunrayRenderPlugin`] replaces `bevy_render::RenderPlugin`: it reuses
//! `ExtractPlugin` for the `RenderApp` SubApp + extraction machinery, then
//! drives the sunray ray-tracer single-threaded on the main thread.
//!
//! [`SunrayEguiPlugin`] adds the egui input/extract layer (paint TODO). It must
//! be added **after** [`SunrayRenderPlugin`] (it extends the `RenderApp` the
//! render plugin creates).

use ash::vk;
use bevy_app::{App, Plugin};
use bevy_ecs::schedule::{IntoScheduleConfigs, ScheduleLabel};
use bevy_render::extract_plugin::ExtractPlugin;
use bevy_render::{ExtractSchedule, Render, RenderApp, RenderSystems};

use super::state::{ExtractedCamera, ExtractedInstances, ExtractedScene, SunrayRenderState, SunrayScene, SunrayWindows};
use super::systems::{ensure_renderer, extract_camera, extract_instances, extract_scene, extract_windows, render_frame};

/// Add this instead of `bevy_render::RenderPlugin` (and without
/// `PipelinedRenderingPlugin` — the renderer is `Rc`-based and runs
/// single-threaded; see `docs/bevy_integration.md`).
pub struct SunrayRenderPlugin {
    /// Format passed to `Renderer::new_with_surface`. Mainly controls the
    /// renderer's sRGB handling; `R8G8B8A8_SRGB` matches the stock examples.
    pub image_format: vk::Format,
}

impl Default for SunrayRenderPlugin {
    fn default() -> Self {
        Self {
            image_format: vk::Format::R8G8B8A8_SRGB,
        }
    }
}

impl Plugin for SunrayRenderPlugin {
    fn build(&self, app: &mut App) {
        // Main-world handle for requesting a scene.
        app.init_resource::<SunrayScene>();

        // Reuse Bevy's extraction machinery: creates the RenderApp SubApp,
        // ExtractSchedule, Render base schedule, SyncWorldPlugin, extract closure.
        app.add_plugins(ExtractPlugin::default());

        let render_app = app.sub_app_mut(RenderApp);

        // Single-threaded: drive `Render` directly as the SubApp's update schedule
        // (exactly as `bevy_render`'s own ExtractPlugin test does).
        render_app.update_schedule = Some(Render.intern());

        render_app.init_resource::<SunrayWindows>();
        render_app.init_resource::<ExtractedCamera>();
        render_app.init_resource::<ExtractedScene>();
        render_app.init_resource::<ExtractedInstances>();

        // The renderer itself is NonSend (Rc-based). Seed it with the chosen format.
        render_app.world_mut().insert_non_send(SunrayRenderState {
            image_format: self.image_format,
            ..Default::default()
        });

        // Extraction: window handles, camera, scene request, entity instances.
        render_app.add_systems(
            ExtractSchedule,
            (extract_windows, extract_camera, extract_scene, extract_instances),
        );

        // Per-frame render work, NonSend → pinned to the main thread, chained.
        render_app.add_systems(Render, (ensure_renderer, render_frame).chain().in_set(RenderSystems::Render));
    }
}

/// egui support (input mapping + tessellation + extract; GPU paint TODO).
///
/// Add **after** [`SunrayRenderPlugin`]. Build UI from your own `Update`
/// systems via [`super::EguiContext`].
pub struct SunrayEguiPlugin;

impl Plugin for SunrayEguiPlugin {
    fn build(&self, app: &mut App) {
        debug_assert!(
            app.get_sub_app(RenderApp).is_some(),
            "SunrayEguiPlugin must be added after SunrayRenderPlugin"
        );
        super::egui_support::register(app);
    }
}
