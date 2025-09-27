use std::rc::Rc;

use ash::vk;
use nalgebra as na;
use sunray::{error::{ErrorSource, SrResult}, vulkan_abstraction};
use winit::{
    application::ApplicationHandler, event::WindowEvent, event_loop::{self, ControlFlow, EventLoop}, raw_window_handle_05::{HasRawDisplayHandle, HasRawWindowHandle}, window::Window
};

mod surface;
mod swapchain;
mod utils;

struct AppResources {
    pub swapchain: swapchain::Swapchain,
    #[allow(unused)]
    pub surface: surface::Surface,
    pub img_rendered_fences: Vec<vulkan_abstraction::Fence>,
    pub img_acquired_sems: Vec<vulkan_abstraction::Semaphore>,
    pub ready_to_present_sems: Vec<vulkan_abstraction::Semaphore>,
    pub img_barrier_to_present_cmd_bufs: Vec<vulkan_abstraction::CmdBuffer>,
    pub renderer: sunray::Renderer,
    pub new_size: Option<(u32,u32)>,
}
impl Drop for AppResources {
    fn drop(&mut self) {
        match self.renderer.core().queue().wait_idle() {
            Ok(()) => {}
            Err(e) => log::warn!("VkQueueWaitIdle returned {e} in AppResources::drop"),
        }
    }
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
const MAX_FRAMES_IN_FLIGHT : usize = 2;

impl App {
    fn rebuild_with_size(&mut self, size: (u32, u32)) -> SrResult<()> {
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
        let (renderer, surface) = sunray::Renderer::new_with_surface(
            size,
            vk::Format::R8G8B8A8_UNORM,
            instance_exts,
            &create_surface,
        )?;

        let core = renderer.core();

        //take ownership of the surface
        let surface = surface::Surface::new(
            core.entry(),
            core.instance(),
            surface,
        );

        let swapchain = swapchain::Swapchain::new(
            Rc::clone(&core),
            &surface,
            size,
        )?;

        // img_acquired_sems & img_rendered_fences cannot be indexed by image index, since they are sent to gpu before an image index is acquired
        let img_acquired_sems = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| vulkan_abstraction::Semaphore::new(Rc::clone(core.device())))
            .collect::<Result<Vec<_>, _>>()?;
        let img_rendered_fences = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| vulkan_abstraction::Fence::new_signaled(Rc::clone(core.device())))
            .collect::<Result<Vec<_>, _>>()?;

        let ready_to_present_sems = swapchain.images().iter()
            .map(|_| vulkan_abstraction::Semaphore::new(Rc::clone(core.device())))
            .collect::<Result<Vec<_>, _>>()?;

        let img_barrier_to_present_cmd_bufs = swapchain.images().iter()
            .map(|image| -> SrResult<vulkan_abstraction::CmdBuffer> {
                let cmd_buf = vulkan_abstraction::CmdBuffer::new(Rc::clone(core))?;

                unsafe {
                    let cmd_buf_begin_info = vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::empty());

                    core.device().inner().begin_command_buffer(cmd_buf.inner(), &cmd_buf_begin_info)?;

                    vulkan_abstraction::cmd_image_memory_barrier(
                        renderer.core(),
                        cmd_buf.inner(),
                        *image,
                        //TODO: ALL_COMMANDS? unnecessary...
                        vk::PipelineStageFlags::ALL_GRAPHICS,
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

        self.resources = Some(AppResources {
            swapchain,
            surface,
            img_rendered_fences,
            img_acquired_sems,
            ready_to_present_sems,
            img_barrier_to_present_cmd_bufs,
            renderer,
            new_size: None

        });

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

        unsafe { self.res().swapchain.device().queue_present(queue, &present_info) }?;

        Ok(())
    }

    fn draw(&mut self) -> sunray::error::SrResult<()> {
        // update frame data:
        let time = self.time_elapsed();

        self.res_mut().renderer.set_camera(sunray::Camera::new(na::Point3::new(0.0, 0.0, 2.0 + time.sin()), na::Point3::origin(), 90.0)?)?;


        let frame_index = self.frame_count as usize % MAX_FRAMES_IN_FLIGHT;

        //acquire next image
        let img_acquired_sem = self.res().img_acquired_sems[frame_index].inner();
        let img_rendered_fence = &mut self.res_mut().img_rendered_fences[frame_index];
        img_rendered_fence.wait()?;
        img_rendered_fence.reset()?;
        let img_rendered_fence = img_rendered_fence.submit();
        let img_index = self.acquire_next_image(img_acquired_sem)?;

        let swapchain_image = self.res().swapchain.images()[img_index];

        //render
        self.res_mut().renderer.render_to_image(swapchain_image, img_acquired_sem, img_rendered_fence)?;

        // image barrier to transition to PRESENT_SRC
        let img_barrier_to_present_cmd_buf = &mut self.res_mut().img_barrier_to_present_cmd_bufs[img_index];
        img_barrier_to_present_cmd_buf.fence_mut().wait()?;
        img_barrier_to_present_cmd_buf.fence_mut().reset()?;
        let img_barrier_done_fence = img_barrier_to_present_cmd_buf.fence_mut().submit();

        let img_barrier_to_present_cmd_buf_inner = img_barrier_to_present_cmd_buf.inner();
        let ready_to_present_sem = self.res().ready_to_present_sems[img_index].inner();


        self.res().renderer.core().queue().submit_async(
            img_barrier_to_present_cmd_buf_inner,
            img_barrier_done_fence,
            &[], &[ready_to_present_sem], &[]
        )?;


        //present
        self.present(img_index, ready_to_present_sem)?;


        self.frame_count += 1;

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

                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let Some(size) = self.res().new_size {
                    self.rebuild_with_size(size)?;
                    self.res_mut().new_size = None;
                }
                self.draw()?;
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::Resized(size) => {
                self.res_mut().new_size = Some(size.into());
            }
            _ => (),
        }
        Ok(())
    }

    fn handle_srresult(&self, event_loop: &event_loop::ActiveEventLoop, result: SrResult<()>) {
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
            },
        }
    }

    fn res_mut(&mut self) -> &mut AppResources { self.resources.as_mut().unwrap() }

    fn res(&self) -> &AppResources { self.resources.as_ref().unwrap() }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(Window::default_attributes())
            .unwrap();

        let window_size = window.inner_size().into();

        self.window = Some(window);

        let result = self.rebuild_with_size(window_size);
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
