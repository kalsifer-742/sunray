use std::{clone, sync::Arc};

use render_context::RenderContext;
use vulkano::{
    command_buffer::allocator::StandardCommandBufferAllocator,
    device::{
        self,
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
    },
    image::{view::ImageView, Image, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    pipeline::graphics::viewport::Viewport,
    render_pass::{self, Framebuffer, FramebufferCreateInfo, RenderPass},
    single_pass_renderpass,
    swapchain::{self, Surface, Swapchain, SwapchainCreateInfo},
    sync, VulkanLibrary,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::EventLoop,
    window::{self, Window},
};

mod render_context;

pub struct App {
    render_context: Option<RenderContext>,
}

impl App {
    //what is () as a generic T which is static?
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        Self {
            render_context: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );

        self.render_context = Some(RenderContext::new(event_loop, window));
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let render_context = self.render_context.as_mut().unwrap(); //i don't remember how box, as_mut etc works

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                () //set swapchain as invalid
            }
            WindowEvent::RedrawRequested => {
                render_context
                    .previous_future
                    .as_mut()
                    .unwrap()
                    .cleanup_finished();

                if render_context.recreate_swapchain {
                    render_context.recreate_swapchain();
                }
            }
            _ => (),
        }
    }
}
