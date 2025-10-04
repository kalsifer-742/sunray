pub mod device;
pub mod instance;

pub use device::*;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
pub use instance::*;

use std::cell::{Ref, RefCell, RefMut};
use std::ffi::CStr;
use std::rc::Rc;

use crate::vulkan_abstraction;
use crate::{CreateSurfaceFn, error::*};
use ash::{khr, vk};

#[rustfmt::skip]
#[allow(unused)]
pub struct Core {
    // Note: do not reorder the fields in this struct: they will be dropped in the same order they are declared
    acceleration_structure_device: khr::acceleration_structure::Device,
    ray_tracing_pipeline_device: khr::ray_tracing_pipeline::Device,
    //queue needs mutability for .present()
    queue: RefCell<vulkan_abstraction::Queue>,
    cmd_pool: vulkan_abstraction::CmdPool,

    allocator: RefCell<Allocator>,

    device: Rc<vulkan_abstraction::Device>,
    instance: vulkan_abstraction::Instance,
    entry: ash::Entry,
}

impl Core {
    pub fn new(with_validation_layer: bool, with_gpuav: bool, image_format: vk::Format) -> SrResult<Self> {
        Ok(Self::new_with_surface(with_validation_layer, with_gpuav, image_format, &[], None)?.0)
    }

    // It is necessary to pass a function to create the surface, because surface depends on instance,
    // device depends on surface (if present), and both device and instance are created and owned inside
    // Core so this seems to be the best approach to allow the user to build its own surface.
    pub fn new_with_surface(
        with_validation_layer: bool,
        with_gpuav: bool,
        image_format: vk::Format,
        required_instance_extensions: &[*const i8],
        create_surface: Option<&CreateSurfaceFn>,
    ) -> SrResult<(Self, Option<vk::SurfaceKHR>)> {
        let entry = ash::Entry::linked();

        let instance =
            vulkan_abstraction::Instance::new(&entry, required_instance_extensions, with_validation_layer, with_gpuav)?;

        let surface_support = match create_surface.as_ref() {
            Some(f) => Some((
                f(&entry, instance.inner())?,
                khr::surface::Instance::new(&entry, instance.inner()),
            )),
            None => None,
        };

        let raytracing_device_extensions = [
            khr::ray_tracing_pipeline::NAME,
            khr::acceleration_structure::NAME,
            khr::deferred_host_operations::NAME,
        ]
        .map(CStr::as_ptr);

        let mut device_extensions = raytracing_device_extensions.iter().copied().collect::<Vec<_>>();

        if surface_support.is_some() {
            device_extensions.push(khr::swapchain::NAME.as_ptr());
        }

        let device = Rc::new(device::Device::new(
            &instance,
            &device_extensions,
            image_format,
            &surface_support,
        )?);

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.inner().clone(),
            device: device.inner().clone(),
            physical_device: device.physical_device(),
            debug_settings: Default::default(),
            // NOTE: Ideally, check the BufferDeviceAddressFeatures struct.
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })?;

        let acceleration_structure_device = khr::acceleration_structure::Device::new(&instance.inner(), &device.inner());
        let ray_tracing_pipeline_device = khr::ray_tracing_pipeline::Device::new(&instance.inner(), &device.inner());

        let queue = vulkan_abstraction::Queue::new(Rc::clone(&device), 0)?;

        let cmd_pool = vulkan_abstraction::CmdPool::new(Rc::clone(&device), vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)?;

        Ok((
            Self {
                entry,
                instance,
                device,
                allocator: RefCell::new(allocator),
                acceleration_structure_device,
                ray_tracing_pipeline_device,
                queue: RefCell::new(queue),
                cmd_pool,
            },
            surface_support.map(|(s, _)| s),
        ))
    }

    #[allow(unused)]
    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    #[allow(unused)]
    pub fn instance(&self) -> &ash::Instance {
        self.instance.inner()
    }

    pub fn device(&self) -> &Rc<vulkan_abstraction::Device> {
        &self.device
    }
    pub fn acceleration_structure_device(&self) -> &khr::acceleration_structure::Device {
        &self.acceleration_structure_device
    }
    pub fn rt_pipeline_device(&self) -> &khr::ray_tracing_pipeline::Device {
        &self.ray_tracing_pipeline_device
    }
    pub fn queue(&self) -> Ref<'_, vulkan_abstraction::Queue> {
        self.queue.borrow()
    }
    pub fn queue_mut(&self) -> RefMut<'_, vulkan_abstraction::Queue> {
        self.queue.borrow_mut()
    }
    pub fn allocator(&self) -> Ref<'_, Allocator> {
        self.allocator.borrow()
    }
    pub fn allocator_mut(&self) -> RefMut<'_, Allocator> {
        self.allocator.borrow_mut()
    }
    pub fn cmd_pool(&self) -> &vulkan_abstraction::CmdPool {
        &self.cmd_pool
    }
}
