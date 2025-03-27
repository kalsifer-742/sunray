use std::sync::Arc;

use vulkano::{
    device::Device,
    image::{view::ImageView, Image},
    instance::Instance,
    pipeline::graphics::viewport::Viewport,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    single_pass_renderpass,
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
    sync::{self, GpuFuture},
};
use winit::window::Window;

pub struct RenderContext {
    pub window: Arc<Window>,
    pub swapchain: Arc<Swapchain>,
    pub render_pass: Arc<RenderPass>,
    pub framebuffers: Vec<Arc<Framebuffer>>,
    pub recreate_swapchain: bool,
    pub previous_future: Option<Box<dyn GpuFuture>>,
}

impl RenderContext {
    pub fn new(device: Arc<Device>, instance: Arc<Instance>, window: Arc<Window>) -> Self {
        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();
        let (swapchain, images) = Self::get_swapchain(device.clone(), surface, window.clone());
        let render_pass = Self::get_render_pass(device.clone(), swapchain.clone());

        //subpass and graphics pipeline

        let _viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window.inner_size().into(),
            depth_range: 0.0..=1.0,
        };
        let framebuffers = Self::get_framebuffers(&images, render_pass.clone());
        let previous_future = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);

        Self {
            window,
            swapchain,
            render_pass,
            framebuffers,
            recreate_swapchain: false,
            previous_future,
        }
    }

    fn get_swapchain(
        device: Arc<Device>,
        surface: Arc<Surface>,
        window: Arc<Window>,
    ) -> (Arc<Swapchain>, Vec<Arc<Image>>) {
        let caps = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();
        let alpha = caps.supported_composite_alpha.into_iter().next().unwrap(); //another strange type wich i don't understand

        let image_format = device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;

        Swapchain::new(
            device,
            surface,
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count,
                image_format,
                image_extent: window.inner_size().into(),
                image_usage: caps.supported_usage_flags,
                composite_alpha: alpha,
                ..Default::default()
            },
        )
        .unwrap()
    }

    fn get_render_pass(device: Arc<Device>, swapchain: Arc<Swapchain>) -> Arc<RenderPass> {
        single_pass_renderpass!(
            device,
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap()
    }

    fn get_framebuffers(
        images: &Vec<Arc<Image>>,
        render_pass: Arc<RenderPass>,
    ) -> Vec<Arc<Framebuffer>> {
        images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>()
    }

    pub fn recreate_swapchain(&mut self) {
        //omitted match case to handle errors
        let (new_swapchain, new_images) = self
            .swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: self.window.inner_size().into(),
                ..self.swapchain.create_info()
            })
            .unwrap();
        self.swapchain = new_swapchain;
        self.framebuffers = Self::get_framebuffers(&new_images, self.render_pass.clone());
        self.recreate_swapchain = false;
    }
}
