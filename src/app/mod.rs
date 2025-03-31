use std::sync::Arc;

use render_context::RenderContext;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
    },
    impl_vertex_member,
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::graphics::vertex_input::Vertex,
    swapchain::{self, Surface, SwapchainPresentInfo},
    sync::{self, GpuFuture},
    Validated, VulkanError, VulkanLibrary,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::EventLoop,
    window::{self, Window},
};

mod render_context;

#[derive(Vertex, BufferContents)]
#[repr(C)] //memory stuff? what is this?
struct MyVertex {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
    #[format(R32G32B32_SFLOAT)] //correct?
    color: [f32; 3],
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "assets/shader.vert"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "assets/shader.frag"
    }
}

pub struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    vertex_buffer: Subbuffer<[MyVertex]>,
    queue: Arc<Queue>,
    render_context: Option<RenderContext>,
}

impl App {
    //what is () as a generic T which is static?
    pub fn new(event_loop: &EventLoop<()>) -> Self {
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

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let vertices = [
            MyVertex {
                position: [-0.5, 0.5, 0.0],
                color: [1.0, 0.0, 0.0],
            },
            MyVertex {
                position: [0.5, 0.5, 0.0],
                color: [0.0, 1.0, 0.0],
            },
            MyVertex {
                position: [0.0, -0.5, 0.0],
                color: [0.0, 0.0, 1.0],
            },
        ];

        let vertex_buffer = Buffer::from_iter(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .unwrap();

        let queue = queues.next().unwrap(); //selecting a random queue

        Self {
            instance,
            device,
            command_buffer_allocator,
            vertex_buffer,
            queue,
            render_context: None,
        }
    }

    fn select_physical_device(
        instance: &Arc<Instance>,
        device_extensions: &DeviceExtensions,
        event_loop: &EventLoop<()>,
    ) -> (Arc<PhysicalDevice>, u32) {
        instance
            .enumerate_physical_devices()
            .unwrap()
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
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );

        self.render_context = Some(RenderContext::new(
            self.device.clone(),
            self.instance.clone(),
            window,
        ));
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: window::WindowId,
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

                let (image_index, suboptimal, acquire_future) = match swapchain::acquire_next_image(render_context.swapchain.clone(), None)
                        .map_err(Validated::unwrap) //unclear syntax
                    {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) => {
                            render_context.recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("{e}"),
                    };

                if suboptimal {
                    render_context.recreate_swapchain = true;
                }

                let clear_values = vec![Some([0.0, 0.0, 0.0, 1.0].into())];

                let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
                    self.command_buffer_allocator.clone(),
                    self.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                //builder rust design pattern
                command_buffer_builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values,
                            ..RenderPassBeginInfo::framebuffer(
                                render_context.framebuffers[image_index as usize].clone(),
                            )
                        },
                        SubpassBeginInfo {
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    .bind_pipeline_graphics(render_context.pipeline.clone())
                    .unwrap()
                    .bind_vertex_buffers(0, self.vertex_buffer.clone())
                    .unwrap();

                unsafe { command_buffer_builder.draw(self.vertex_buffer.len() as u32, 1, 0, 0) }
                    .unwrap();

                command_buffer_builder
                    .end_render_pass(Default::default())
                    .unwrap();

                let command_buffer = command_buffer_builder.build().unwrap();

                let future = render_context
                    .previous_future
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(self.queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        self.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            render_context.swapchain.clone(),
                            image_index,
                        ),
                    )
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    //this part was not explained
                    Ok(future) => {
                        render_context.previous_future = Some(Box::new(future) as Box<_>);
                    }
                    Err(VulkanError::OutOfDate) => {
                        render_context.recreate_swapchain = true;
                        render_context.previous_future =
                            Some(Box::new(sync::now(self.device.clone())) as Box<_>);
                    }
                    Err(e) => {
                        panic!("failed to flush future: {e}");
                    }
                }
            }
            _ => (),
        }
    }
}
