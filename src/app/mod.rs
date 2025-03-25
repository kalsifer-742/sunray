use std::{clone, sync::Arc};

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
    VulkanLibrary,
};
use winit::{
    application::ApplicationHandler,
    event_loop::EventLoop,
    window::{self, Window},
};

pub struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl App {
    //what is () as a generic T which is static?
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        let library = VulkanLibrary::new().unwrap();
        let extensions = Surface::required_extensions(&event_loop).unwrap();

        let instance = Instance::new(
            library,
            // structs are passed as argument to resemble how the Vulkan API works
            InstanceCreateInfo {
                enabled_extensions: extensions,
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY, //this is need to run on macOS trough MoltenVK
                ..Default::default()
            },
        )
        .unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) =
            Self::select_physical_device(&instance, &device_extensions, &event_loop);

        //logical/software device, queues associated to the device
        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap(); //selecting a random queue

        Self {
            instance,
            device,
            queue,
        }
    }

    fn select_physical_device(
        instance: &Arc<Instance>,
        device_extensions: &DeviceExtensions,
        event_loop: &EventLoop<()>,
    ) -> (Arc<PhysicalDevice>, u32) {
        instance
            .enumerate_physical_devices()
            .expect("failed to enumerate physical devices")
            .filter(|p| p.supported_extensions().contains(device_extensions))
            .filter_map(|device| {
                device
                    .queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, queue_family)| {
                        //i'm taking the first queue that satisfies the condition
                        queue_family.queue_flags.contains(QueueFlags::GRAPHICS)
                            && device.presentation_support(i as u32, &event_loop).unwrap()
                    })
                    .map(|i| (device, i as u32))
            })
            .min_by_key(|(device, _i)| match device.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::Cpu => 2,
                PhysicalDeviceType::Other => 3,
                _ => 4,
            })
            .unwrap()
    }

    fn get_swapchain(
        &self,
        surface: Arc<Surface>,
        window: Arc<Window>,
    ) -> (Arc<Swapchain>, Vec<Arc<Image>>) {
        let caps = self
            .device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();
        let alpha = caps.supported_composite_alpha.into_iter().next().unwrap(); //another strange type wich i don't understand

        let image_format = self
            .device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;

        Swapchain::new(
            self.device.clone(),
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

    fn get_render_pass(&self, swapchain: Arc<Swapchain>) -> Arc<RenderPass> {
        single_pass_renderpass!(
            self.device.clone(),
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
        &self,
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
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();
        let (swapchain, images) = self.get_swapchain(surface, window.clone());

        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(self.device.clone(), Default::default());

        let render_pass = self.get_render_pass(swapchain);

        //subpass and graphics pipeline

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window.inner_size().into(),
            depth_range: 0.0..=1.0,
        };

        let framebuffers = self.get_framebuffers(&images, render_pass.clone());
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        todo!()
    }
}
