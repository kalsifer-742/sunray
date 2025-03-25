use std::sync::Arc;

use vulkano::{
    command_buffer::allocator::StandardCommandBufferAllocator,
    device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo},
    image::Image,
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    pipeline::graphics::viewport::Viewport,
    render_pass::{Framebuffer, RenderPass},
    single_pass_renderpass,
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
    sync::{self, GpuFuture},
    VulkanLibrary,
};
use winit::{event_loop::EventLoop, window::Window};

use super::App;

pub struct RenderContext {
    pub window: Arc<Window>,
    pub swapchain: Arc<Swapchain>,
    pub render_pass: Arc<RenderPass>,
    pub framebuffers: Vec<Arc<Framebuffer>>,
    pub recreate_swapchain: bool,
    pub previous_future: Option<Box<dyn GpuFuture>>,
}

impl RenderContext {
    pub fn new(event_loop: &EventLoop<()>, window: Arc<Window>) -> Self {
        let library = VulkanLibrary::new().unwrap();
        let extensions = Surface::required_extensions(event_loop).unwrap();

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

        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();
        let (swapchain, images) = Self::get_swapchain(surface, window.clone());
        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());
        let render_pass = Self::get_render_pass(swapchain.clone());
        //subpass and graphics pipeline
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window.inner_size().into(),
            depth_range: 0.0..=1.0,
        };
        let framebuffers = get_framebuffers(&images, render_pass.clone());
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

    fn get_render_pass(swapchain: Arc<Swapchain>) -> Arc<RenderPass> {
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

    pub fn recreate_swapchain(&self) {
        let (new_swapchain, new_images) = self
            .swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: self.window.inner_size().into(),
                ..self.swapchain.create_info()
            })
            .unwrap();
        self.swapchain = new_swapchain;
        todo!()
    }
}
