use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{self, ControlFlow, EventLoop},
    raw_window_handle_05::{HasRawDisplayHandle, HasRawWindowHandle},
    window::Window,
};

#[derive(Default)]
struct App {
    window: Option<Window>,
    renderer: Option<sunray::Renderer>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        self.window = Some(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );

        self.renderer = Some(
            sunray::Renderer::new(
                self.window.as_ref().unwrap().inner_size().into(),
                self.window.as_ref().unwrap().raw_window_handle(),
                self.window.as_ref().unwrap().raw_display_handle(),
            )
            .unwrap(),
        );

        let swapchain = match surface.as_ref() {
            None => None,
            Some(surface) => Some(vulkan_abstraction::Swapchain::new(
                &instance,
                Rc::clone(&device),
                surface,
                create_info.window_extent.unwrap(),
            )?),
        };
    }

    fn window_event(
        &mut self,
        event_loop: &event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                match self.renderer.as_mut().unwrap().render() {
                    Ok(()) => {}
                    Err(error) => {
                        //no need to panic, sunray already takes care of the backtrace
                        eprintln!("Sunray error: {}", error);
                        event_loop.exit();
                    }
                }
            }
            _ => (),
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::default();
    let _ = event_loop.run_app(&mut app).unwrap();
}

pub fn acquire_next_image(&self, swapchain: vk::SwapchainKHR) -> SrResult<u32> {
    let wait_fence = &[self.render_complete_fences[self.current_frame]];
    unsafe {
        self.device
            .inner()
            .wait_for_fences(wait_fence, true, u64::MAX)
    }?;
    unsafe { self.device.inner().reset_fences(wait_fence) }?;

    let image_available_sem = self.img_available_sem[self.current_frame];
    let (index, _suboptimal_surface) = unsafe {
        self.swapchain_device.acquire_next_image(
            swapchain,
            u64::MAX,
            image_available_sem,
            vk::Fence::null(),
        )
    }?;
    Ok(index)
}

pub fn render(&mut self) -> SrResult<()> {
    let img_index = self
        .core
        .queue()
        .acquire_next_image(self.core.swapchain().inner())?;

    let cmd_buf = self.core.cmd_pool().get_buffers()[img_index as usize];

    self.core.queue().submit_async(cmd_buf)?;
    self.core.queue().wait_idle()?;

    self.core
        .queue_mut()
        .present(self.core.swapchain().inner(), img_index)?;
    self.core.queue().wait_idle()?;

    Ok(())
}

pub fn present(&mut self, swapchain: vk::SwapchainKHR, img_idx: u32) -> SrResult<()> {
    let wait_semaphores = &[self.render_complete_sems[self.current_frame]];
    let swapchains = [swapchain];
    let image_indices = [img_idx];
    let present_info = vk::PresentInfoKHR::default()
        .wait_semaphores(wait_semaphores)
        .swapchains(&swapchains)
        .image_indices(&image_indices);

    unsafe {
        self.swapchain_device
            .queue_present(self.queue, &present_info)
    }?;

    self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    Ok(())
}
