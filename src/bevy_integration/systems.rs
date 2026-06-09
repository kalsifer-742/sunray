//! Extract + render systems.
//!
//! Extract systems (run in `ExtractSchedule`, read the main world via
//! [`Extract`]): [`extract_windows`], [`extract_camera`], [`extract_scene`].
//!
//! Render systems (run in `Render` / [`bevy_render::RenderSystems::Render`],
//! chained, NonSend → main thread): [`ensure_renderer`] then [`render_frame`].
//!
//! The per-frame logic in [`render_frame`] mirrors `examples/window/main.rs`'s
//! `draw()`: wait the in-flight fence, acquire a swapchain image, push the
//! camera, `render_to_image`, flip the image to `PRESENT_SRC`, present.

use std::rc::Rc;

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
use super::swapchain::{Surface, Swapchain};
use crate::camera::Camera;
use crate::error::{ErrorSource, SrResult};
use crate::vulkan_abstraction::{self, CmdBuffer, Core, Semaphore};
use crate::{MAX_FRAMES_IN_FLIGHT, Renderer};

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

    // (Re)load the scene once the renderer exists.
    if state.renderer.is_some() && scene.generation != state.loaded_scene_generation {
        if let Some(path) = scene.gltf_path.clone() {
            match state.renderer.as_mut().unwrap().load_gltf(&path) {
                Ok(ids) => log::info!("sunray: loaded {} entities from {path}", ids.len()),
                Err(e) => log::error!("sunray: load_gltf({path}) failed: {e}"),
            }
            // load_gltf() clears image-dependent data, which drops the blit
            // CmdBuffers whose fences we stored; null them so render_frame
            // doesn't wait on destroyed handles.
            for f in state.img_rendered_fences.iter_mut() {
                *f = vk::Fence::null();
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

    let (mut renderer, surface) = Renderer::new_with_surface(size, format, instance_exts, &create_surface)?;
    let surface = Surface::new(renderer.core().entry(), renderer.core().instance(), surface);
    let swapchain = Swapchain::new(Rc::clone(renderer.core()), surface.inner(), size)?;

    renderer.build_image_dependent_data(swapchain.images())?;

    let core = renderer.core().clone();
    let img_acquired_sems = (0..MAX_FRAMES_IN_FLIGHT)
        .map(|_| Semaphore::new(core.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    let img_rendered_fences = vec![vk::Fence::null(); MAX_FRAMES_IN_FLIGHT];
    let ready_to_present_sems = swapchain
        .images()
        .iter()
        .map(|_| Semaphore::new(core.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    let present_barrier_cmd_bufs = build_present_barrier_cmd_bufs(&core, swapchain.images())?;

    state.renderer = Some(renderer);
    state.surface = Some(surface);
    state.swapchain = Some(swapchain);
    state.owner = Some(entity);
    state.image_format = format;
    state.img_acquired_sems = img_acquired_sems;
    state.img_rendered_fences = img_rendered_fences;
    state.ready_to_present_sems = ready_to_present_sems;
    state.present_barrier_cmd_bufs = present_barrier_cmd_bufs;

    Ok(())
}

fn resize_impl(state: &mut SunrayRenderState, size: (u32, u32)) -> SrResult<()> {
    state.renderer.as_mut().unwrap().resize(size)?;

    let core = state.renderer.as_ref().unwrap().core().clone();
    let surface_khr = state.surface.as_ref().unwrap().inner();

    // Refresh cached surface caps, then decide if the swapchain extent changed.
    {
        let surface = state.surface.as_ref().unwrap();
        core.device()
            .update_surface_support_details(surface.inner(), surface.instance());
    }
    let new_extent = Swapchain::get_extent(size, &core.device().surface_support_details());
    if state.swapchain.as_ref().unwrap().extent() == new_extent {
        return Ok(());
    }

    // The present-barrier CmdBuffers reference the old images; null their fences
    // before rebuilding.
    for f in state.img_rendered_fences.iter_mut() {
        *f = vk::Fence::null();
    }

    state.swapchain.as_mut().unwrap().rebuild(surface_khr, size)?;

    let images = state.swapchain.as_ref().unwrap().images().to_vec();
    state.present_barrier_cmd_bufs = build_present_barrier_cmd_bufs(&core, &images)?;
    state.ready_to_present_sems = images
        .iter()
        .map(|_| Semaphore::new(core.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    state.renderer.as_mut().unwrap().build_image_dependent_data(&images)?;

    Ok(())
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
        swapchain,
        img_acquired_sems,
        img_rendered_fences,
        ready_to_present_sems,
        present_barrier_cmd_bufs,
        frame_count,
        egui_paint,
        ..
    } = &mut *state;
    let renderer = renderer.as_mut().unwrap();
    let swapchain = swapchain.as_ref().unwrap();

    let core = renderer.core().clone();

    if camera.present {
        let cam = Camera::new(
            na::Point3::new(camera.eye[0], camera.eye[1], camera.eye[2]),
            na::Point3::new(camera.target[0], camera.target[1], camera.target[2]),
            camera.fov_y_degrees,
        );
        renderer.set_camera(cam)?;
    }
    
    

    let frame_index = (*frame_count as usize) % MAX_FRAMES_IN_FLIGHT;
    let img_acquired_sem = img_acquired_sems[frame_index].inner();
    let img_rendered_fence = img_rendered_fences[frame_index];
    vulkan_abstraction::wait_fence(core.device(), img_rendered_fence)?;

    let img_index = acquire_next_image(swapchain, img_acquired_sem)?;
    let swapchain_image = swapchain.images()[img_index];

    img_rendered_fences[frame_index] = renderer.render_to_image(swapchain_image, img_acquired_sem)?;

    let ready_sem = ready_to_present_sems[img_index].inner();

    // Finalize the swapchain image -> PRESENT_SRC, signaling `ready_sem`. Both
    // paths run on the graphics queue *after* render_to_image's blit (same-queue
    // submission order + the layout barrier provide the dependency on the blit).
    if let Some(extracted) = egui {
        // Lazily build the egui GPU backend now that the swapchain (and its
        // color format) exists. The egui pass also performs the PRESENT_SRC
        // transition, so it replaces the plain present barrier.
        if egui_paint.is_none() {
            *egui_paint = Some(EguiPaint::new(core.clone(), swapchain.format(), swapchain.images().len())?);
        }
        let image_view = swapchain.image_views()[img_index];
        let extent = swapchain.extent();
        egui_paint
            .as_mut()
            .unwrap()
            .paint_frame(swapchain_image, image_view, extent, img_index, extracted, ready_sem)?;
    } else {
        // No egui: the pre-recorded GENERAL -> PRESENT_SRC barrier.
        let barrier_fence = present_barrier_cmd_bufs[img_index].fence_mut().submit()?;
        let barrier_cmd_inner = present_barrier_cmd_bufs[img_index].inner();
        core.graphics_queue()
            .submit_async(barrier_cmd_inner, &[], &[], &[ready_sem], barrier_fence)?;
    }

    present(&core, swapchain, img_index, ready_sem)?;

    *frame_count += 1;
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn acquire_next_image(swapchain: &Swapchain, signal_sem: vk::Semaphore) -> SrResult<usize> {
    let (image_index, suboptimal) = unsafe {
        swapchain
            .device()
            .acquire_next_image(swapchain.inner(), u64::MAX, signal_sem, vk::Fence::null())
    }?;
    if suboptimal {
        log::warn!("VkAcquireNextImageKHR: swapchain suboptimal for the surface");
    }
    Ok(image_index as usize)
}

fn present(core: &Rc<Core>, swapchain: &Swapchain, img_index: usize, wait_sem: vk::Semaphore) -> SrResult<()> {
    let swapchains = [swapchain.inner()];
    let image_indices = [img_index as u32];
    let wait_semaphores = [wait_sem];
    let present_info = vk::PresentInfoKHR::default()
        .wait_semaphores(&wait_semaphores)
        .swapchains(&swapchains)
        .image_indices(&image_indices);

    let queue = core.graphics_queue().inner();
    unsafe { swapchain.device().queue_present(queue, &present_info) }?;
    Ok(())
}

/// One pre-recorded GENERAL -> PRESENT_SRC barrier command buffer per swapchain
/// image (matches `examples/window/main.rs`).
fn build_present_barrier_cmd_bufs(core: &Rc<Core>, images: &[vk::Image]) -> SrResult<Vec<CmdBuffer>> {
    images
        .iter()
        .map(|image| -> SrResult<CmdBuffer> {
            let cmd_buf = CmdBuffer::new(Rc::clone(core))?;
            unsafe {
                let begin_info = vk::CommandBufferBeginInfo::default();
                core.device().inner().begin_command_buffer(cmd_buf.inner(), &begin_info)?;
                vulkan_abstraction::cmd_image_memory_barrier(
                    core,
                    cmd_buf.inner(),
                    *image,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::AccessFlags2::empty(),
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                );
                core.device().inner().end_command_buffer(cmd_buf.inner())?;
            }
            Ok(cmd_buf)
        })
        .collect()
}
