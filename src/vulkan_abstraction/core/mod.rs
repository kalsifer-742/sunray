pub mod device;
pub mod instance;
pub mod surface;
pub mod swapchain;

pub use device::*;
pub use instance::*;
pub use surface::*;
pub use swapchain::*;

use std::cell::{Ref, RefCell, RefMut};
use std::ffi::CStr;
use std::rc::Rc;

use ash::{ khr, vk, Entry };
use winit::raw_window_handle_05::{RawDisplayHandle, RawWindowHandle};
use crate::error::*;
use crate::vulkan_abstraction;


pub struct CoreCreateInfo<'a> {
    pub instance_exts: &'a [*const i8],
    pub device_exts: &'a [*const i8],

    pub with_swapchain: bool,
    pub window_extent: Option<[u32; 2]>,
    pub raw_window_handle: Option<RawWindowHandle>,
    pub raw_display_handle: Option<RawDisplayHandle>,

    pub with_validation_layer: bool,
    pub with_gpu_assisted_validation: bool,
}

impl Default for CoreCreateInfo<'_> {
    fn default() -> Self {
        Self {
            instance_exts: &[],
            device_exts: &[],

            with_swapchain: false,
            window_extent: None,
            raw_window_handle: None,
            raw_display_handle: None,

            with_validation_layer: false,
            with_gpu_assisted_validation: false,
        }
    }
}


#[allow(unused)]
pub struct Core {
    /* Note: do not reorder the fields in this struct: they will be dropped in the same order they are declared */

    acceleration_structure_device: khr::acceleration_structure::Device,
    ray_tracing_pipeline_device: khr::ray_tracing_pipeline::Device,

    //queue needs mutability for .present()
    queue: RefCell<vulkan_abstraction::Queue>,
    cmd_pool: vulkan_abstraction::CmdPool,

    swapchain: Option<Swapchain>,
    surface: Option<Surface>,

    device: Rc<vulkan_abstraction::Device>,
    instance: vulkan_abstraction::Instance,
    entry: ash::Entry,
}

impl Core {
    pub fn new(create_info: CoreCreateInfo) -> SrResult<Self> {
        let entry = Entry::linked();

        let instance = vulkan_abstraction::Instance::new(&entry, create_info.instance_exts, create_info.with_validation_layer, create_info.with_gpu_assisted_validation)?;


        let raytracing_device_extensions = [
            khr::ray_tracing_pipeline::NAME,
            khr::acceleration_structure::NAME,
            khr::deferred_host_operations::NAME,
        ].map(CStr::as_ptr);


        let surface = if create_info.with_swapchain {
            Some(vulkan_abstraction::Surface::new(&entry, &instance, create_info.raw_display_handle.unwrap(), create_info.raw_window_handle.unwrap())?)
        } else {
            None
        };


        let device_extensions = create_info.device_exts.iter().chain(raytracing_device_extensions.iter()).copied().collect::<Vec<_>>();


        let device = Rc::new(device::Device::new(&instance, &device_extensions, create_info.with_swapchain, &surface)?);

        let swapchain = match surface.as_ref() {
            None => None,
            Some(surface) => Some(vulkan_abstraction::Swapchain::new(&instance, Rc::clone(&device), surface, create_info.window_extent.unwrap())?)
        };

        let acceleration_structure_device = khr::acceleration_structure::Device::new(&instance.inner(), &device.inner());
        let ray_tracing_pipeline_device = khr::ray_tracing_pipeline::Device::new(&instance.inner(), &device.inner());


        //TODO: still takes for granted swapchain exists
        let queue = vulkan_abstraction::Queue::new(
            Rc::clone(&device),
            //TODO: bad clone; swapchain device should be shared
            swapchain.as_ref().unwrap().device().clone(),
            0,
        )?;

        //TODO: still takes for granted swapchain exists
        let cmd_pool = {
            let mut cmd_pool = vulkan_abstraction::CmdPool::new(
                Rc::clone(&device),
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            )?;

            // add render command buffers to cmd_pool
            cmd_pool.append_buffers(vulkan_abstraction::cmd_buffer::new_vec(&cmd_pool, &device, swapchain.as_ref().unwrap().images().len())?);

            cmd_pool
        };



        Ok(Self {
            entry,
            instance,
            device,
            acceleration_structure_device,
            ray_tracing_pipeline_device,
            queue: RefCell::new(queue),
            cmd_pool,

            surface,

            swapchain,
        })
    }


    pub fn device(&self) -> &vulkan_abstraction::Device { &self.device }
    pub fn acceleration_structure_device(&self) -> &khr::acceleration_structure::Device { &self.acceleration_structure_device }
    pub fn rt_pipeline_device(&self) -> &khr::ray_tracing_pipeline::Device { &self.ray_tracing_pipeline_device }

    pub fn queue(&self) -> Ref<'_, vulkan_abstraction::Queue> { self.queue.borrow() }
    pub fn queue_mut(&self) -> RefMut<'_, vulkan_abstraction::Queue> { self.queue.borrow_mut() }
    pub fn cmd_pool(&self) -> &vulkan_abstraction::CmdPool { &self.cmd_pool}

    pub fn swapchain(&self) -> &vulkan_abstraction::Swapchain { self.swapchain.as_ref().unwrap() } // TODO: unwrap
}

