pub mod device;
pub mod instance;

pub use device::*;
pub use instance::*;

use std::cell::{Ref, RefCell, RefMut};
use std::ffi::CStr;
use std::rc::Rc;

use crate::error::*;
use crate::vulkan_abstraction;
use ash::{khr, vk};
use winit::raw_window_handle_05::{RawDisplayHandle, RawWindowHandle};

#[rustfmt::skip]
#[allow(unused)]
pub struct Core {
    // Note: do not reorder the fields in this struct: they will be dropped in the same order they are declared
    acceleration_structure_device: khr::acceleration_structure::Device,
    ray_tracing_pipeline_device: khr::ray_tracing_pipeline::Device,
    //queue needs mutability for .present()
    queue: RefCell<vulkan_abstraction::Queue>,
    cmd_pool: vulkan_abstraction::CmdPool,

    device: Rc<vulkan_abstraction::Device>,
    instance: vulkan_abstraction::Instance,
    entry: ash::Entry,
}

impl Core {
    pub fn new(with_validation_layer: bool, with_gpuav: bool) -> SrResult<Self> {
        let entry = ash::Entry::linked();

        let instance =
            vulkan_abstraction::Instance::new(&entry, &[], with_validation_layer, with_gpuav)?;

        let raytracing_device_extensions = [
            khr::ray_tracing_pipeline::NAME,
            khr::acceleration_structure::NAME,
            khr::deferred_host_operations::NAME,
        ]
        .map(CStr::as_ptr);

        let device_extensions = raytracing_device_extensions; //for now this are all the needed extensions

        let device = Rc::new(device::Device::new(&instance, &device_extensions)?);

        let acceleration_structure_device =
            khr::acceleration_structure::Device::new(&instance.inner(), &device.inner());
        let ray_tracing_pipeline_device =
            khr::ray_tracing_pipeline::Device::new(&instance.inner(), &device.inner());

        let queue = vulkan_abstraction::Queue::new(Rc::clone(&device), 0)?;

        let cmd_pool = vulkan_abstraction::CmdPool::new(
            Rc::clone(&device),
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        )?;

        Ok(Self {
            entry,
            instance,
            device,
            acceleration_structure_device,
            ray_tracing_pipeline_device,
            queue: RefCell::new(queue),
            cmd_pool,
        })
    }

    pub fn device(&self) -> &vulkan_abstraction::Device {
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
    pub fn cmd_pool(&self) -> &vulkan_abstraction::CmdPool {
        &self.cmd_pool
    }
}
