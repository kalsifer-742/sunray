use std::{sync::Arc, time::Instant};

use nalgebra::{Isometry3, Matrix4, Perspective3, Point3, Vector3};

use render_context::RenderContext;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{
        AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryTypeFilter,
        StandardMemoryAllocator,
    },
    pipeline::{graphics::vertex_input::Vertex, Pipeline, PipelineBindPoint},
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
        path: "assets/shader.vert",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "assets/shader.frag"
    }
}

struct MVP {
    model: Matrix4<f32>,
    view: Matrix4<f32>,
    projection: Matrix4<f32>,
}

impl MVP {
    fn new() -> MVP {
        Self {
            model: Matrix4::identity(),
            view: Matrix4::identity(),
            projection: Matrix4::identity(),
        }
    }

    fn get_uniform(&self) -> MyUniform {
        MyUniform {
            model: self.model.into(),
            view: self.view.into(),
            projection: self.projection.into(),
        }
    }
}

#[derive(BufferContents)]
#[repr(C)] //memory stuff? what is this?
struct MyUniform {
    model: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    projection: [[f32; 4]; 4],
}

pub struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    memory_allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    queue: Arc<Queue>,
    vertex_buffer: Subbuffer<[MyVertex]>,
    render_context: Option<RenderContext>,
    start_time: Instant,
    mvp: MVP,
    frame_counter: usize,
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

        let queue = queues.next().unwrap(); //selecting a random queue

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
            memory_allocator.clone(),
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

        let mut mvp = MVP::new();
        let eye = Point3::new(0.0, 0.0, 2.0);
        let target = Point3::new(0.0, 0.0, 0.0);
        mvp.view = Isometry3::look_at_rh(&eye, &target, &Vector3::y()).into();
        // mvp.model = Isometry3::new(Vector3::x(), nalgebra::zero()).into();
        // mvp.projection = Perspective3::new(16.0 / 9.0, 3.14 / 2.0, 0.00, 100.0).into();

        Self {
            instance,
            device,
            command_buffer_allocator,
            memory_allocator,
            queue,
            vertex_buffer,
            render_context: None,
            start_time: Instant::now(),
            mvp,
            frame_counter: 0,
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

        let image_extent: [u32; 2] = window.inner_size().into();
        let aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;
        self.mvp.projection = Perspective3::new(aspect_ratio, 3.14 / 2.0, 0.00, 100.0).into();

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
                let avg_fps = self.frame_counter as f32 / self.start_time.elapsed().as_secs_f32();
                println!(
                    "frames: {} | seconds {} | avg_fps {}",
                    self.frame_counter,
                    self.start_time.elapsed().as_secs_f32(),
                    avg_fps
                );

                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                render_context.recreate_swapchain = true;
            }
            WindowEvent::RedrawRequested => {
                self.frame_counter += 1;

                render_context
                    .previous_future
                    .as_mut()
                    .unwrap()
                    .cleanup_finished();

                if render_context.recreate_swapchain {
                    render_context.recreate_swapchain();

                    let image_extent: [u32; 2] = render_context.window.inner_size().into();
                    let aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;
                    self.mvp.projection =
                        Perspective3::new(aspect_ratio, 3.14 / 2.0, 0.00, 100.0).into();
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

                let elapsed = self.start_time.elapsed().as_secs_f32();
                let rotation = Matrix4::from_axis_angle(&Vector3::y_axis(), elapsed);
                self.mvp.model = rotation;
                let uniform_data = self.mvp.get_uniform();

                let uniform_buffer = Buffer::from_data(
                    self.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::UNIFORM_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    uniform_data,
                )
                .unwrap();

                let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
                    self.command_buffer_allocator.clone(),
                    self.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
                    self.device.clone(),
                    Default::default(),
                ));

                let descriptor_set_layout = render_context
                    .pipeline
                    .layout()
                    .set_layouts()
                    .get(0)
                    .unwrap();

                let descriptor_set = DescriptorSet::new(
                    descriptor_set_allocator,
                    descriptor_set_layout.clone(),
                    [WriteDescriptorSet::buffer(0, uniform_buffer.clone())], // 0 is the binding
                    [],
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
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        render_context.pipeline.layout().clone(),
                        0,
                        descriptor_set,
                    )
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

                render_context.window.request_redraw();
            }
            _ => (),
        }
    }
}
