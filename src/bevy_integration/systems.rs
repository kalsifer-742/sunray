//! Extract + render systems.
//!
//! Extract systems (run in `ExtractSchedule`, read the main world via
//! [`Extract`]): [`extract_windows`], [`extract_camera`], [`extract_scene`].
//!
//! Render systems (run in `Render` / [`bevy_render::RenderSystems::Render`],
//! chained, NonSend → main thread): [`ensure_renderer`] then [`render_frame`].
//!
//! The per-frame logic in [`render_frame`] hands the extracted camera and the
//! scene's instance list to [`Renderer::render_to_swapchain`] — the renderer
//! owns the swapchain and all present plumbing internally.

use ash::vk;
use bevy_ecs::prelude::*;
use bevy_render::Extract;
use bevy_transform::components::Transform;
use bevy_window::{PrimaryWindow, RawHandleWrapper, Window};
use nalgebra as na;

use super::camera::{SunrayCamera, eye_target_fov};
use super::egui_paint::EguiPaint;
use super::egui_support::ExtractedEgui;
use super::state::*;
use super::surface;
use crate::camera::Camera;
use crate::error::{ErrorSource, SrResult};
use crate::{Renderer, SwapchainFrame};

// ---------------------------------------------------------------------------
// Extract (main world -> render world)
// ---------------------------------------------------------------------------

/// Copy window handles + physical sizes into the render world. Only windows
/// that already have a `RawHandleWrapper` (i.e. created by winit after the
/// event loop started) are included.
pub(crate) fn extract_windows(
    mut windows: ResMut<SunrayWindows>,
    query: Extract<Query<(Entity, &Window, &RawHandleWrapper, Option<&PrimaryWindow>)>>,
) {
    let mut seen = Vec::new();
    for (entity, window, handle, primary) in &query {
        seen.push(entity);
        let new_size = (
            window.resolution.physical_width().max(1),
            window.resolution.physical_height().max(1),
        );
        windows
            .windows
            .entry(entity)
            .and_modify(|w| {
                w.size_changed = w.physical_size != new_size;
                w.physical_size = new_size;
                w.is_primary = primary.is_some();
            })
            .or_insert_with(|| SunrayWindowInfo {
                handle: handle.clone(),
                physical_size: new_size,
                size_changed: false,
                is_primary: primary.is_some(),
            });
    }
    // Drop windows that no longer exist in the main world.
    windows.windows.retain(|e, _| seen.contains(e));
}

/// Copy the first `(&Transform, &SunrayCamera)` entity into the render world.
pub(crate) fn extract_camera(mut out: ResMut<ExtractedCamera>, query: Extract<Query<(&Transform, &SunrayCamera)>>) {
    if let Some((transform, cam)) = query.iter().next() {
        let (eye, target, fov_y_degrees) = eye_target_fov(transform, cam);
        *out = ExtractedCamera {
            present: true,
            eye,
            target,
            fov_y_degrees,
        };
    } else {
        out.present = false;
    }
}

/// Copy the scene-load request (path + generation) into the render world.
pub(crate) fn extract_scene(mut out: ResMut<ExtractedScene>, src: Extract<Res<SunrayScene>>) {
    if src.generation != out.generation {
        out.gltf_path = src.gltf_path.clone();
        out.generation = src.generation;
    }
}

// ---------------------------------------------------------------------------
// Render world: lazy init + per-frame
// ---------------------------------------------------------------------------

/// Create the renderer/surface/swapchain on the first window that has a handle,
/// handle resizes, and (re)load the requested scene. Lazy because the window
/// handle only exists after the winit event loop starts.
pub(crate) fn ensure_renderer(mut state: NonSendMut<SunrayRenderState>, windows: Res<SunrayWindows>, scene: Res<ExtractedScene>) {
    if let Err(e) = ensure_renderer_impl(&mut state, &windows, &scene) {
        log::error!("sunray ensure_renderer: {e}");
    }
}

fn ensure_renderer_impl(state: &mut SunrayRenderState, windows: &SunrayWindows, scene: &ExtractedScene) -> SrResult<()> {
    if state.renderer.is_none() {
        // Pick the primary window, else any window with a handle.
        let chosen = windows
            .windows
            .iter()
            .find(|(_, w)| w.is_primary)
            .or_else(|| windows.windows.iter().next());
        if let Some((&entity, info)) = chosen {
            create_renderer_for_window(state, entity, info.physical_size, &info.handle)?;
            log::info!("sunray: renderer created for window {entity:?} @ {:?}", info.physical_size);
        }
    } else if let Some(owner) = state.owner {
        // Resize if the owning window changed size.
        let size_changed = windows.windows.get(&owner).map(|w| w.size_changed).unwrap_or(false);
        if size_changed {
            let size = windows.windows.get(&owner).unwrap().physical_size;
            resize_impl(state, size)?;
        }
    }

    // (Re)load the scene once the renderer exists. The instance list the load
    // returns is kept here (caller side) and passed to the renderer each frame.
    if state.renderer.is_some() && scene.generation != state.loaded_scene_generation {
        if let Some(path) = scene.gltf_path.clone() {
            // Free the previous scene's assets so reloading doesn't leak.
            if let Some(prev_group) = state.scene_group.take() {
                state.scene_instances.clear();
                state.renderer.as_mut().unwrap().unload_scene(prev_group)?;
            }
            match state.renderer.as_mut().unwrap().load_gltf(&path) {
                Ok((group, instances)) => {
                    log::info!("sunray: loaded {} unique BLASes from {path}", instances.len());
                    state.scene_group = Some(group);
                    state.scene_instances = instances;
                }
                Err(e) => log::error!("sunray: load_gltf({path}) failed: {e}"),
            }
        }
        state.loaded_scene_generation = scene.generation;
    }

    Ok(())
}

fn create_renderer_for_window(
    state: &mut SunrayRenderState,
    entity: Entity,
    size: (u32, u32),
    handle: &RawHandleWrapper,
) -> SrResult<()> {
    // Safe getters — only *using* the handle is thread-restricted, and this
    // system runs on the main thread (single-threaded render SubApp).
    let display_handle = handle.get_display_handle();
    let window_handle = handle.get_window_handle();

    let instance_exts = surface::enumerate_required_extensions(display_handle)?;
    let create_surface = move |entry: &ash::Entry, instance: &ash::Instance| -> SrResult<vk::SurfaceKHR> {
        surface::create_surface(entry, instance, display_handle, window_handle)
    };

    let format = if state.image_format == vk::Format::UNDEFINED {
        vk::Format::R8G8B8A8_SRGB
    } else {
        state.image_format
    };

    // The renderer creates and owns the surface + swapchain internally.
    let renderer = Renderer::new_with_surface(size, format, instance_exts, &create_surface)?;

    state.renderer = Some(renderer);
    state.owner = Some(entity);
    state.image_format = format;

    Ok(())
}

fn resize_impl(state: &mut SunrayRenderState, size: (u32, u32)) -> SrResult<()> {
    // The renderer resizes its internal images and rebuilds its swapchain
    // (and everything tied to the swapchain images) itself.
    state.renderer.as_mut().unwrap().resize(size)
}

/// Acquire image, push camera, render, paint egui (if enabled), present.
///
/// `egui` is `Option` so the plugin works with or without `SunrayEguiPlugin`
/// (the latter inserts the `ExtractedEgui` resource).
pub(crate) fn render_frame(
    mut state: NonSendMut<SunrayRenderState>,
    camera: Res<ExtractedCamera>,
    egui: Option<Res<ExtractedEgui>>,
) {
    if state.renderer.is_none() {
        return;
    }
    if let Err(e) = render_frame_impl(&mut state, &camera, egui.as_deref()) {
        match e.get_source() {
            ErrorSource::Vulkan(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                log::warn!("sunray render_frame (out of date — will rebuild on resize): {e}");
            }
            _ => log::error!("sunray render_frame: {e}"),
        }
    }
}

fn render_frame_impl(state: &mut SunrayRenderState, camera: &ExtractedCamera, egui: Option<&ExtractedEgui>) -> SrResult<()> {
    // Split borrows over the distinct fields we touch this frame.
    let SunrayRenderState {
        renderer,
        scene_instances,
        egui_paint,
        ..
    } = &mut *state;
    let renderer = renderer.as_mut().unwrap();

    // The camera is a per-frame input handed to the renderer — nothing is
    // stored on the renderer side.
    let cam = if camera.present {
        Camera::new(
            na::Point3::new(camera.eye[0], camera.eye[1], camera.eye[2]),
            na::Point3::new(camera.target[0], camera.target[1], camera.target[2]),
            camera.fov_y_degrees,
        )
    } else {
        Camera::default()
    };

    if let Some(extracted) = egui {
        // Lazily build the egui GPU backend now that the swapchain (and its
        // color format) exists. The egui pass also performs the PRESENT_SRC
        // transition, so it replaces the renderer's plain present barrier.
        if egui_paint.is_none() {
            let swapchain = renderer.swapchain().unwrap();
            *egui_paint = Some(EguiPaint::new(
                renderer.core().clone(),
                swapchain.format(),
                swapchain.images().len(),
            )?);
        }
        let paint = egui_paint.as_mut().unwrap();
        let mut finalize = |frame: &SwapchainFrame| -> SrResult<()> {
            paint.paint_frame(
                frame.image,
                frame.image_view,
                frame.extent,
                frame.image_index,
                extracted,
                frame.ready_to_present_sem,
            )
        };
        renderer.render_to_swapchain_with(&cam, scene_instances, Some(&mut finalize))?;
    } else {
        renderer.render_to_swapchain(&cam, scene_instances)?;
    }

    Ok(())
}
