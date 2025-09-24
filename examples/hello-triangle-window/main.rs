use std::rc::Rc;

use ash::vk;
use nalgebra as na;
use sunray::{error::{ErrorSource, SrResult}, vulkan_abstraction};
use winit::{
    application::ApplicationHandler, event::WindowEvent, event_loop::{self, ControlFlow, EventLoop}, platform::wayland::WindowAttributesExtWayland, raw_window_handle_05::{HasRawDisplayHandle, HasRawWindowHandle}, window::Window
};

mod surface;
mod swapchain;
mod utils;

#[derive(Default)]
struct App {
    window: Option<Window>,
    swapchain: Option<swapchain::Swapchain>,
    surface: Option<surface::Surface>,
    renderer: Option<sunray::Renderer>,
    start_time: Option<std::time::SystemTime>,
    new_size: Option<(u32,u32)>,
}
impl App {
    fn rebuild_with_size(&mut self, size: (u32, u32)) -> SrResult<()> {
        //ensuring old vulkan resources are dropped in the correct order
        self.swapchain = None;
        self.surface = None;
        self.renderer = None;

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
        self.renderer = Some(renderer);

        let core = self.renderer.as_ref().unwrap().core();

        //take ownership of the surface
        self.surface = Some(surface::Surface::new(
            core.entry(),
            core.instance(),
            surface,
        ));

        self.swapchain = Some(swapchain::Swapchain::new(
            Rc::clone(&core),
            self.surface.as_ref().unwrap(),
            size,
        )?);
        Ok(())
    }

    fn draw(&mut self) -> sunray::error::SrResult<()> {
        let swapchain = self.swapchain.as_ref().unwrap();
        let renderer = self.renderer.as_mut().unwrap();

        //acquire next image
        let image_index = {
            let device = renderer.core().device();
            let fence = vulkan_abstraction::Fence::new_unsignaled(Rc::clone(device))?;

            let (image_index, swapchain_suboptimal_for_surface) = unsafe {
                swapchain.device().acquire_next_image(
                    swapchain.inner(),
                    u64::MAX,
                    vk::Semaphore::null(),
                    fence.inner(),
                )
            }?;

            if swapchain_suboptimal_for_surface {
                log::warn!(
                    "swapchain::Device::acquire_next_image reports that the swapchain is supobtimal for the surface"
                );
            }

            unsafe { device.inner().wait_for_fences(&[fence.inner()], true, u64::MAX) }?;

            image_index as usize
        };

        let swapchain_image = swapchain.images()[image_index];

        let time = std::time::SystemTime::now()
            .duration_since(self.start_time.unwrap())
            .unwrap()
            .as_millis() as f32
            / 1000.0;

        let camera = sunray::Camera::new(na::Point3::new(0.0, 0.0, 2.0 + time.sin()), na::Point3::origin(), 90.0)?;
        renderer.set_camera(camera)?;

        renderer.render_to_image(swapchain_image)?;

        // image barrier to transition to PRESENT_SRC
        {
            let device = renderer.core().device().inner();
            let cmd_buf =
                vulkan_abstraction::cmd_buffer::new(renderer.core().cmd_pool(), device).unwrap();

            unsafe {
                let cmd_buf_begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                device
                    .begin_command_buffer(cmd_buf, &cmd_buf_begin_info)
                    .unwrap();

                vulkan_abstraction::cmd_image_memory_barrier(
                    renderer.core(),
                    cmd_buf,
                    swapchain_image,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::empty(),
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                );

                device.end_command_buffer(cmd_buf).unwrap();
            }

            let queue = renderer.core().queue();
            queue.submit_sync(cmd_buf).unwrap();

            unsafe { device.free_command_buffers(renderer.core().cmd_pool().inner(), &[cmd_buf]) };
        }

        //present
        {
            let swapchains = [swapchain.inner()];
            let image_indices = [image_index as u32];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&[])
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            let queue = renderer.core().queue().inner();

            unsafe { swapchain.device().queue_present(queue, &present_info) }?;
        }

        Ok(())
    }

    fn handle_event(&mut self, event_loop: &event_loop::ActiveEventLoop, event: winit::event::WindowEvent) -> SrResult<()> {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let Some(size) = self.new_size {
                    self.rebuild_with_size(size)?;
                    self.new_size = None;
                }
                self.draw()?;
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::Resized(size) => {
                self.new_size = Some(size.into());
            }
            _ => (),
        }
        Ok(())
    }

    fn handle_srresult(event_loop: &event_loop::ActiveEventLoop, result: SrResult<()>) {
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
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(Window::default_attributes())
            .unwrap();

        let window_size = window.inner_size().into();

        self.window = Some(window);

        self.start_time = Some(std::time::SystemTime::now());

        let result = self.rebuild_with_size(window_size);
        Self::handle_srresult(event_loop, result);
    }

    fn window_event(
        &mut self,
        event_loop: &event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let result = self.handle_event(event_loop, event);
        Self::handle_srresult(event_loop, result);
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
