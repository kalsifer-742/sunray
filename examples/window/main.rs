use std::collections::HashSet;
use std::io;
use std::io::Read;
use std::rc::Rc;

use ash::vk;
use log::info;
use nalgebra as na;
use rand::random_range;
use std::time::Instant;
use sunray::vulkan_abstraction::EntityId;
use sunray::{
    MAX_FRAMES_IN_FLIGHT,
    camera::Camera,
    error::{ErrorSource, SrResult},
    vulkan_abstraction,
};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, ElementState, MouseButton, WindowEvent},
    event_loop::{self, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    raw_window_handle_05::{HasRawDisplayHandle, HasRawWindowHandle},
    window::{CursorGrabMode, Window},
};

mod surface;
mod swapchain;
mod utils;

struct AppResources {
    pub swapchain: swapchain::Swapchain,
    #[allow(unused)]
    pub surface: surface::Surface,
    pub img_rendered_fences: Vec<vk::Fence>,
    pub img_barrier_to_present_cmd_bufs: Vec<vulkan_abstraction::CmdBuffer>, // must delete before ready_to_present_sems
    pub img_acquired_sems: Vec<vulkan_abstraction::Semaphore>,

    pub renderer: sunray::Renderer,

    pub ready_to_present_sems: Vec<vulkan_abstraction::Semaphore>,
}

struct App {
    window: Option<Window>,
    resources: Option<AppResources>,

    start_time: Option<std::time::SystemTime>,
    frame_count: u64,
    last_fps_check: Option<Instant>,
    frames_since_check: u32,

    // --- CAMERA STATE ---
    camera_pos: na::Point3<f32>,
    camera_yaw: f32,
    camera_pitch: f32,
    keys_down: HashSet<KeyCode>,
    mouse_captured: bool,
    last_frame_time: Option<Instant>,

    // --- RUNTIME RESOURCE TEST ---
    scene_entities: Vec<sunray::vulkan_abstraction::EntityId>,
    spawned_entity: Option<sunray::vulkan_abstraction::EntityId>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            resources: None,
            start_time: None,
            frame_count: 0,
            last_fps_check: None,
            frames_since_check: 0,

            // Start looking down the -Z axis slightly above the floor
            camera_pos: na::Point3::new(0.0, 2.0, 10.0),
            camera_yaw: -std::f32::consts::FRAC_PI_2,
            camera_pitch: 0.0,
            keys_down: HashSet::new(),
            mouse_captured: false,
            last_frame_time: None,

            scene_entities: Vec::new(),
            spawned_entity: None,
        }
    }
}

impl App {
    fn build_resources(&mut self, size: (u32, u32)) -> SrResult<()> {
        self.resources = None;

        let display_handle = self.window.as_ref().unwrap().raw_display_handle().clone();
        let window_handle = self.window.as_ref().unwrap().raw_window_handle().clone();

        let instance_exts = utils::enumerate_required_extensions(display_handle)?;

        let create_surface = move |entry: &ash::Entry, instance: &ash::Instance| -> SrResult<vk::SurfaceKHR> {
            crate::utils::create_surface(entry, instance, display_handle, window_handle, None)
        };

        // build sunray renderer and surface
        let (mut renderer, surface) =
            sunray::Renderer::new_with_surface(size, vk::Format::R8G8B8A8_SRGB, instance_exts, &create_surface)?;

        self.scene_entities = renderer.load_gltf("examples/assets/Room.glb")?;
        log::info!("Loaded {} entities from scene", self.scene_entities.len());

        //take ownership of the surface
        let surface = surface::Surface::new(renderer.core().entry(), renderer.core().instance(), surface);

        let swapchain = swapchain::Swapchain::new(Rc::clone(renderer.core()), surface.inner(), size)?;

        renderer.build_image_dependent_data(swapchain.images())?;

        let core = renderer.core();

        let img_barrier_to_present_cmd_bufs = swapchain
            .images()
            .iter()
            .map(|image| -> SrResult<vulkan_abstraction::CmdBuffer> {
                let cmd_buf = vulkan_abstraction::CmdBuffer::new(Rc::clone(&core))?;

                unsafe {
                    let cmd_buf_begin_info = vk::CommandBufferBeginInfo::default();

                    core.device()
                        .inner()
                        .begin_command_buffer(cmd_buf.inner(), &cmd_buf_begin_info)?;

                    vulkan_abstraction::cmd_image_memory_barrier(
                        &core,
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
            .collect::<Result<Vec<_>, _>>()?;

        let img_acquired_sems = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| vulkan_abstraction::Semaphore::new(Rc::clone(&core)))
            .collect::<Result<Vec<_>, _>>()?;
        let img_rendered_fences = vec![vk::Fence::null(); MAX_FRAMES_IN_FLIGHT];

        let ready_to_present_sems = swapchain
            .images()
            .iter()
            .map(|_| vulkan_abstraction::Semaphore::new(Rc::clone(&core)))
            .collect::<Result<Vec<_>, _>>()?;

        self.resources = Some(AppResources {
            swapchain,
            surface,
            img_rendered_fences,
            img_acquired_sems,
            ready_to_present_sems,
            img_barrier_to_present_cmd_bufs,
            renderer,
        });

        Ok(())
    }

    fn resize(&mut self, size: (u32, u32)) -> SrResult<()> {
        self.res_mut().renderer.resize(size)?;

        self.res()
            .renderer
            .core()
            .device()
            .update_surface_support_details(self.res().surface.inner(), self.res().surface.instance());

        let curr_size = self.res().swapchain.extent();
        let new_size = swapchain::Swapchain::get_extent(size, &self.res().renderer.core().device().surface_support_details());

        if curr_size == new_size {
            return Ok(());
        }

        let core = Rc::clone(self.res().renderer.core());

        for fence in self.res_mut().img_rendered_fences.iter_mut() {
            *fence = vk::Fence::null();
        }

        let surface = self.res().surface.inner();

        self.res_mut().swapchain.rebuild(surface, size)?;

        self.res_mut().img_barrier_to_present_cmd_bufs = self
            .res()
            .swapchain
            .images()
            .iter()
            .map(|image| -> SrResult<vulkan_abstraction::CmdBuffer> {
                let cmd_buf = vulkan_abstraction::CmdBuffer::new(Rc::clone(&core))?;

                unsafe {
                    let cmd_buf_begin_info = vk::CommandBufferBeginInfo::default();

                    core.device()
                        .inner()
                        .begin_command_buffer(cmd_buf.inner(), &cmd_buf_begin_info)?;

                    vulkan_abstraction::cmd_image_memory_barrier(
                        &core,
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
            .collect::<Result<Vec<_>, _>>()?;

        Ok(())
    }

    fn time_elapsed(&self) -> f32 {
        std::time::SystemTime::now()
            .duration_since(self.start_time.unwrap())
            .unwrap()
            .as_millis() as f32
            / 1000.0
    }

    fn acquire_next_image(&self, signal_sem: vk::Semaphore) -> SrResult<usize> {
        let image_index = {
            let (image_index, swapchain_suboptimal_for_surface) = unsafe {
                self.res().swapchain.device().acquire_next_image(
                    self.res().swapchain.inner(),
                    u64::MAX,
                    signal_sem,
                    vk::Fence::null(),
                )
            }?;

            if swapchain_suboptimal_for_surface {
                log::warn!("VkAcquireNextImageKHR: swapchain is supobtimal for the surface");
            }

            image_index as usize
        };

        Ok(image_index)
    }

    fn present(&self, img_index: usize, ready_to_present_sem: vk::Semaphore) -> SrResult<()> {
        let swapchains = [self.res().swapchain.inner()];
        let image_indices = [img_index as u32];
        let wait_semaphores = [ready_to_present_sem];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let queue = self.res().renderer.core().graphics_queue().inner();

        unsafe { self.res().swapchain.device().queue_present(queue, &present_info) }?;

        Ok(())
    }

    fn draw(&mut self) -> sunray::error::SrResult<()> {
        let now = Instant::now();
        let dt = if let Some(last) = self.last_frame_time {
            now.duration_since(last).as_secs_f32()
        } else {
            0.016
        };
        self.last_frame_time = Some(now);

        // --- UPDATE CAMERA LOGIC ---
        let base_speed = 3.0;
        let speed = if self.keys_down.contains(&KeyCode::ShiftLeft) {
            base_speed * 3.0
        } else {
            base_speed
        };
        let move_dist = speed * dt;

        let forward = na::Vector3::new(
            self.camera_yaw.cos() * self.camera_pitch.cos(),
            self.camera_pitch.sin(),
            self.camera_yaw.sin() * self.camera_pitch.cos(),
        )
        .normalize();

        let right = forward.cross(&na::Vector3::new(0.0, 1.0, 0.0)).normalize();

        if self.keys_down.contains(&KeyCode::KeyW) {
            self.camera_pos += forward * move_dist;
        }
        if self.keys_down.contains(&KeyCode::KeyS) {
            self.camera_pos -= forward * move_dist;
        }
        if self.keys_down.contains(&KeyCode::KeyD) {
            self.camera_pos += right * move_dist;
        }
        if self.keys_down.contains(&KeyCode::KeyA) {
            self.camera_pos -= right * move_dist;
        }
        if self.keys_down.contains(&KeyCode::Space) {
            self.camera_pos += na::Vector3::new(0.0, 1.0, 0.0) * move_dist;
        }
        if self.keys_down.contains(&KeyCode::ControlLeft) {
            self.camera_pos -= na::Vector3::new(0.0, 1.0, 0.0) * move_dist;
        }

        let target = self.camera_pos + forward;

        let camera = Camera::default()
            .set_position(self.camera_pos)
            .set_target(target)
            .set_fov_y(45.0);

        self.res_mut().renderer.set_camera(camera)?;

        //self.update_runtime_test()?;

        let frame_index = self.frame_count as usize % MAX_FRAMES_IN_FLIGHT;

        let img_acquired_sem = self.res().img_acquired_sems[frame_index].inner();
        let img_rendered_fence = self.res().img_rendered_fences[frame_index];
        vulkan_abstraction::wait_fence(self.res().renderer.core().device(), img_rendered_fence)?;
        let img_index = self.acquire_next_image(img_acquired_sem)?;

        let swapchain_image = self.res().swapchain.images()[img_index];

        self.res_mut().img_rendered_fences[frame_index] =
            self.res_mut().renderer.render_to_image(swapchain_image, img_acquired_sem)?;

        let img_barrier_to_present_cmd_buf = &mut self.res_mut().img_barrier_to_present_cmd_bufs[img_index];
        let img_barrier_done_fence = img_barrier_to_present_cmd_buf.fence_mut().submit()?;

        let img_barrier_to_present_cmd_buf_inner = img_barrier_to_present_cmd_buf.inner();
        let ready_to_present_sem = self.res().ready_to_present_sems[img_index].inner();

        self.frames_since_check += 1;

        if let Some(last_check) = self.last_fps_check {
            let elapsed = now.duration_since(last_check);

            if elapsed.as_secs() >= 1 {
                let fps = self.frames_since_check as f32 / elapsed.as_secs_f32();
                if let Some(window) = &self.window {
                    window.set_title(&format!("Sunray Vulkan - FPS: {:.1}", fps));
                }
                self.last_fps_check = Some(now);
                self.frames_since_check = 0;
            }
        }

        self.res().renderer.core().graphics_queue().submit_async(
            img_barrier_to_present_cmd_buf_inner,
            &[],
            &[],
            &[ready_to_present_sem],
            img_barrier_done_fence,
        )?;

        self.present(img_index, ready_to_present_sem)?;

        self.frame_count += 1;
        self.window.as_ref().unwrap().request_redraw();
        Ok(())
    }

    /// Exercises runtime add / move / remove of entities each frame.
    fn update_runtime_test(&mut self) -> SrResult<()> {
        let frame = self.frame_count;

        // if !self.scene_entities.is_empty() {
        //     let id = self.scene_entities.pop().unwrap().0;
        //     return self.res_mut().renderer.destroy_entity(EntityId(id));
        // }

        // Animate the first loaded entity: orbit around Y every frame.
        let src = self.scene_entities.first();
        if let Some(src) = src {
            let src = *src;
            let angle = frame as f32 * 0.0001;
            let (s, c) = angle.sin_cos();
            let radius = 3.0_f32;
            let translation = na::Translation3::new(c * radius, 0.0, s * radius);
            let rotation = na::UnitQuaternion::from_axis_angle(&na::Vector3::y_axis(), angle);
            let transform = (translation * rotation).to_homogeneous();
            self.res_mut().renderer.set_entity_transform(src, transform)?;
            self.res_mut().renderer.rebuild_tlas()?;
        }

        // At frame 120 spawn a duplicate of the first entity offset to the side.
        if frame % 120 == 1 {
            let src = self.scene_entities[random_range(0..self.scene_entities.len())];

            let offset = na::Translation3::new(4.0, 0.0, 0.0).to_homogeneous();
            match self.res_mut().renderer.duplicate_entity(src, offset) {
                Ok(id) => {
                    self.spawned_entity = Some(id);
                    // duplicate_entity calls clear_image_dependent_data(), which drops the
                    // CmdBuffers whose fences were returned by render_to_image. Those
                    // VkFence handles are now destroyed; null them out so draw()'s
                    // wait_fence() skips the stale handles next frame.
                    for fence in self.res_mut().img_rendered_fences.iter_mut() {
                        *fence = vk::Fence::null();
                    }
                    log::info!("[runtime test] spawned duplicate entity {:?}", id);
                }
                Err(e) => log::error!("[runtime test] duplicate_entity failed: {e}"),
            }
        }

        // At frame 240 remove the spawned duplicate.
        if frame % 240 == 1 {
            if let Some(id) = self.spawned_entity.take() {
                match self.res_mut().renderer.destroy_entity(id) {
                    Ok(()) => {
                        // duplicate_entity calls clear_image_dependent_data(), which drops the
                        // CmdBuffers whose fences were returned by render_to_image. Those
                        // VkFence handles are now destroyed; null them out so draw()'s
                        // wait_fence() skips the stale handles next frame.
                        for fence in self.res_mut().img_rendered_fences.iter_mut() {
                            *fence = vk::Fence::null();
                        }
                        log::info!("[runtime test] removed entity {:?}", id)
                    }
                    Err(e) => log::error!("[runtime test] destroy_entity failed: {e}"),
                }
            }
        }

        Ok(())
    }

    fn handle_event(&mut self, event_loop: &event_loop::ActiveEventLoop, event: winit::event::WindowEvent) -> SrResult<()> {
        match event {
            WindowEvent::CloseRequested => {
                let run_time = {
                    let end_time = std::time::SystemTime::now();
                    end_time.duration_since(self.start_time.unwrap()).unwrap().as_millis() as f32 / 1000.0
                };
                let fps = self.frame_count as f32 / run_time;
                log::info!("Frames per second: {fps}");
                unsafe { self.res().renderer.core().device().inner().device_wait_idle() }?;
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.draw()?;
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::Resized(size) => {
                if size.width != 0 && size.height != 0 {
                    self.resize(size.into()).unwrap();
                }
            }
            // --- KEYBOARD TRACKING & MOUSE LOCK EXIT ---
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(keycode) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            if keycode == KeyCode::Escape {
                                self.mouse_captured = false;
                                if let Some(window) = &self.window {
                                    let _ = window.set_cursor_grab(CursorGrabMode::None);
                                    window.set_cursor_visible(true);
                                }
                            } else {
                                self.keys_down.insert(keycode);
                            }
                        }
                        ElementState::Released => {
                            self.keys_down.remove(&keycode);
                        }
                    }
                }
            }
            // --- MOUSE LOCK ACTIVATION ---
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                self.mouse_captured = true;
                if let Some(window) = &self.window {
                    // Confined works best on Windows, Locked works best on Mac/Linux. We fallback if one fails.
                    let _ = window
                        .set_cursor_grab(CursorGrabMode::Confined)
                        .or_else(|_| window.set_cursor_grab(CursorGrabMode::Locked));
                    window.set_cursor_visible(false);
                }
            }
            _ => (),
        }
        Ok(())
    }

    fn handle_srresult(&mut self, event_loop: &event_loop::ActiveEventLoop, result: SrResult<()>) {
        if let Err(e) = result {
            if let ErrorSource::Vulkan(vk::Result::ERROR_OUT_OF_DATE_KHR) = e.get_source() {
                log::warn!("{e}");
            } else {
                log::error!("{e}");
                event_loop.exit();
            }
        }
    }

    fn res_mut(&mut self) -> &mut AppResources {
        self.resources.as_mut().unwrap()
    }

    fn res(&self) -> &AppResources {
        self.resources.as_ref().unwrap()
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        let window = event_loop.create_window(Window::default_attributes()).unwrap();
        let window_size = window.inner_size().into();
        self.window = Some(window);

        if self.resources.is_none() {
            let result = self.build_resources(window_size);
            self.handle_srresult(event_loop, result);
        }

        let result = self.resize(window_size);
        self.handle_srresult(event_loop, result);

        self.start_time = Some(std::time::SystemTime::now());
        self.last_fps_check = Some(Instant::now());
        self.frames_since_check = 0;
        self.last_frame_time = Some(Instant::now());

        self.window.as_ref().unwrap().request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let result = self.handle_event(event_loop, event);
        self.handle_srresult(event_loop, result);
    }

    // --- RAW MOUSE MOTION TRACKING ---
    fn device_event(
        &mut self,
        _event_loop: &event_loop::ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if self.mouse_captured {
            if let DeviceEvent::MouseMotion { delta } = event {
                let sensitivity = 0.002;
                self.camera_yaw += delta.0 as f32 * sensitivity;
                self.camera_pitch -= delta.1 as f32 * sensitivity;

                // Clamp pitch so you don't break your neck looking backward through your legs
                let limit = std::f32::consts::FRAC_PI_2 - 0.01;
                self.camera_pitch = self.camera_pitch.clamp(-limit, limit);
            }
        }
    }
}

fn main() {
    log4rs::config::init_file("examples/log4rs.yaml", log4rs::config::Deserializers::new()).unwrap();

    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        default_hook(info);
        println!("\nPress Enter to exit...");
        let _ = io::stdin().read(&mut [0u8]);
    }));

    if cfg!(debug_assertions) {
        log::set_max_level(log::LevelFilter::Debug);
    } else {
        log::set_max_level(log::LevelFilter::Warn);
    }

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::default();
    let _ = event_loop.run_app(&mut app).unwrap();
}
