use vulkano::{
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
        QueueFlags,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    swapchain::Surface,
    VulkanLibrary,
};
use winit::event_loop::EventLoop;

fn main() {
    let library = VulkanLibrary::new().unwrap();
    let event_loop = EventLoop::new().unwrap();
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

    //i don't like this, iterators are cool but at this point it's too much for me
    //When the demo is finished i will probably refactor
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|device| device.supported_extensions().contains(&device_extensions))
        .filter_map(|device| {
            device
                .queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    //i'm taking the first queue that satisfies the condition
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
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
        .unwrap();

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

    let queue = queues.next().unwrap();
}
