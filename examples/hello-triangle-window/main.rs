use std::rc::Rc;

use ash::vk;
use nalgebra as na;
use sunray::{
    camera::Camera,
    error::{ErrorSource, SrResult},
    vulkan_abstraction,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{self, ControlFlow, EventLoop},
    raw_window_handle_05::{HasRawDisplayHandle, HasRawWindowHandle},
    window::Window,
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

#[derive(Default)]
struct App {
    window: Option<Window>,
    resources: Option<AppResources>,

    start_time: Option<std::time::SystemTime>,
    frame_count: u64,
}

/// The number of concurrent frames that are processed (both by CPU and GPU).
///
/// Apparently 2 is the most common choice. Empirically it seems like the performance doesn't really
/// get any better with a higher number, but it does get measurably worse with only 1.
const MAX_FRAMES_IN_FLIGHT: usize = 2;

impl App {
    fn build_resources(&mut self, size: (u32, u32)) -> SrResult<()> {
        self.resources = None;

        let instance_exts = utils::enumerate_required_extensions(
            self.window.as_ref().unwrap().raw_display_handle(),
        )?;

        let display_handle = self.window.as_ref().unwrap().raw_display_handle().clone();
        let window_handle = self.window.as_ref().unwrap().raw_window_handle().clone();
        let create_surface =
            move |entry: &ash::Entry, instance: &ash::Instance| -> SrResult<vk::SurfaceKHR> {
                crate::utils::create_surface(entry, instance, display_handle, window_handle, None)
            };

        // build sunray renderer and surface
        let (mut renderer, surface) = sunray::Renderer::new_with_surface(
            size,
            vk::Format::R8G8B8A8_UNORM,
            instance_exts,
            &create_surface,
        )?;

        renderer.load_gltf("examples/assets/Lantern.glb")?;

        //take ownership of the surface
        let surface =
            surface::Surface::new(renderer.core().entry(), renderer.core().instance(), surface);

        let swapchain =
            swapchain::Swapchain::new(Rc::clone(renderer.core()), surface.inner(), size)?;

        renderer.build_image_dependent_data(swapchain.images())?;

        let core = renderer.core();

        let img_barrier_to_present_cmd_bufs = swapchain
            .images()
            .iter()
            .map(|image| -> SrResult<vulkan_abstraction::CmdBuffer> {
                let cmd_buf = vulkan_abstraction::CmdBuffer::new(Rc::clone(&core))?;

                unsafe {
                    let cmd_buf_begin_info = vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::empty());

                    core.device()
                        .inner()
                        .begin_command_buffer(cmd_buf.inner(), &cmd_buf_begin_info)?;

                    vulkan_abstraction::cmd_image_memory_barrier(
                        &core,
                        cmd_buf.inner(),
                        *image,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::empty(),
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::PRESENT_SRC_KHR,
                    );

                    core.device().inner().end_command_buffer(cmd_buf.inner())?;
                }
                Ok(cmd_buf)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // img_acquired_sems & img_rendered_fences cannot be indexed by image index, since they are sent to gpu before an image index is acquired
        let img_acquired_sems = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| vulkan_abstraction::Semaphore::new(Rc::clone(&core)))
            .collect::<Result<Vec<_>, _>>()?;
        let img_rendered_fences = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| vk::Fence::null())
            .collect::<Vec<_>>();

        let ready_to_present_sems = swapchain
            .images()
            .iter()
            .map(|_| vulkan_abstraction::Semaphore::new(Rc::clone(&core)))
            .collect::<Result<Vec<_>, _>>()?;

        let fmt_handles = |sems: &[vulkan_abstraction::Semaphore]| -> String {
            if log::max_level() < log::LevelFilter::Debug {
                return String::new();
            }
            let mut s = String::from("[ ");
            for sem in sems.iter() {
                s += &format!("{:#x?}, ", sem.inner());
            }
            s += "]";

            s
        };

        log::debug!("img_acquired_sems: {}", fmt_handles(&img_acquired_sems));
        log::debug!(
            "ready_to_present_sems: {}",
            fmt_handles(&ready_to_present_sems)
        );

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

        // necessary so we can create the swapchain with the correct extent
        self.res()
            .renderer
            .core()
            .device()
            .update_surface_support_details(
                self.res().surface.inner(),
                self.res().surface.instance(),
            );

        let curr_size = self.res().swapchain.extent();
        let new_size = swapchain::Swapchain::get_extent(
            size,
            &self
                .res()
                .renderer
                .core()
                .device()
                .surface_support_details(),
        );

        if curr_size == new_size {
            return Ok(());
        }

        log::debug!("resizing swapchain from {curr_size:?} to {new_size:?}");

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
                    let cmd_buf_begin_info = vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::empty());

                    core.device()
                        .inner()
                        .begin_command_buffer(cmd_buf.inner(), &cmd_buf_begin_info)?;

                    vulkan_abstraction::cmd_image_memory_barrier(
                        &core,
                        cmd_buf.inner(),
                        *image,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::empty(),
                        vk::ImageLayout::UNDEFINED,
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

        let queue = self.res().renderer.core().queue().inner();

        unsafe {
            self.res()
                .swapchain
                .device()
                .queue_present(queue, &present_info)
        }?;

        Ok(())
    }

    fn draw(&mut self) -> sunray::error::SrResult<()> {
        // update frame data:
        let time = self.time_elapsed();
        let y = 13.0;
        let dist = 30.0;
        let position = na::Point3::new(dist * time.cos(), y, dist * time.sin());
        let fov_y = 45.0;
        self.res_mut().renderer.set_camera(Camera::new(
            position,
            na::Point3::new(0.0, y, 0.0),
            fov_y,
        )?)?;

        let frame_index = self.frame_count as usize % MAX_FRAMES_IN_FLIGHT;

        //acquire next image
        let img_acquired_sem = self.res().img_acquired_sems[frame_index].inner();
        let img_rendered_fence = self.res().img_rendered_fences[frame_index];
        if img_rendered_fence != vk::Fence::null() {
            unsafe {
                self.res().renderer.core().device().inner().wait_for_fences(
                    &[img_rendered_fence],
                    true,
                    u64::MAX,
                )
            }?
        }
        let img_index = self.acquire_next_image(img_acquired_sem)?;

        let swapchain_image = self.res().swapchain.images()[img_index];

        //render
        self.res_mut().img_rendered_fences[frame_index] = self
            .res_mut()
            .renderer
            .render_to_image(swapchain_image, img_acquired_sem)?;

        // image barrier to transition to PRESENT_SRC
        let img_barrier_to_present_cmd_buf =
            &mut self.res_mut().img_barrier_to_present_cmd_bufs[img_index];
        img_barrier_to_present_cmd_buf.fence_mut().wait()?;
        img_barrier_to_present_cmd_buf.fence_mut().reset()?;
        let img_barrier_done_fence = img_barrier_to_present_cmd_buf.fence_mut().submit();

        let img_barrier_to_present_cmd_buf_inner = img_barrier_to_present_cmd_buf.inner();
        let ready_to_present_sem = self.res().ready_to_present_sems[img_index].inner();

        self.res().renderer.core().queue().submit_async(
            img_barrier_to_present_cmd_buf_inner,
            &[],
            &[],
            &[ready_to_present_sem],
            img_barrier_done_fence,
        )?;

        //present
        self.present(img_index, ready_to_present_sem)?;

        self.frame_count += 1;

        Ok(())
    }

    fn handle_event(
        &mut self,
        event_loop: &event_loop::ActiveEventLoop,
        event: winit::event::WindowEvent,
    ) -> SrResult<()> {
        match event {
            WindowEvent::CloseRequested => {
                let run_time = {
                    let end_time = std::time::SystemTime::now();

                    end_time
                        .duration_since(self.start_time.unwrap())
                        .unwrap()
                        .as_millis() as f32
                        / 1000.0
                };
                let fps = self.frame_count as f32 / run_time;
                log::info!("Frames per second: {fps}");
                unsafe {
                    self.res()
                        .renderer
                        .core()
                        .device()
                        .inner()
                        .device_wait_idle()
                }?;

                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.draw()?;
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::Resized(size) => {
                if size.width != 0 && size.height != 0 {
                    self.resize(size.into()).unwrap(); // TODO: unwrap
                }
            }
            _ => (),
        }
        Ok(())
    }

    fn handle_srresult(&mut self, event_loop: &event_loop::ActiveEventLoop, result: SrResult<()>) {
        match result {
            Ok(()) => {}
            Err(e) => {
                match e.get_source() {
                    Some(ErrorSource::VULKAN(vk::Result::ERROR_OUT_OF_DATE_KHR)) => {
                        log::warn!("{e}"); // we still warn because this isn't really the best behaviour
                    }
                    _ => {
                        log::error!("{e}");

                        event_loop.exit();
                    }
                }
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
        let window = event_loop
            .create_window(Window::default_attributes())
            .unwrap();

        let window_size = window.inner_size().into();

        self.window = Some(window);

        if !self.resources.is_some() {
            let result = self.build_resources(window_size);
            self.handle_srresult(event_loop, result);
        }

        let result = self.resize(window_size);
        self.handle_srresult(event_loop, result);

        self.start_time = Some(std::time::SystemTime::now());
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
}

fn main() {
    log4rs::config::init_file("examples/log4rs.yaml", log4rs::config::Deserializers::new())
        .unwrap();

    if cfg!(debug_assertions) {
        //stdlib unfortunately completely pollutes trace log level, TODO somehow config stdlib/log to fix this?
        log::set_max_level(log::LevelFilter::Debug);
    } else {
        log::set_max_level(log::LevelFilter::Warn);
    }

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::default();
    let _ = event_loop.run_app(&mut app).unwrap();
}
