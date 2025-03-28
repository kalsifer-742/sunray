mod buffer;
mod cmd_pool;
mod instance;
mod device;
mod pipeline;
mod render_pass;
mod shader;
mod surface;
mod swapchain;

pub use buffer::*;
pub use cmd_pool::*;
pub use instance::*;
pub use device::*;
pub use pipeline::*;
pub use render_pass::*;
pub use shader::*;
pub use surface::*;
pub use swapchain::*;

use crate::vkal;
use std::rc::Rc;
use ash::{ext, khr, vk};
use winit::raw_window_handle::{RawDisplayHandle, RawWindowHandle};

#[allow(dead_code)]
pub struct VulkanCore {
    /*
        NOTE: Do NOT reorder the fields in this struct.
        The fields are dropped in the same order they are declared,
        and this is taken advantage of to avoid issues with their
        dependencies (listed in the comments to the right), without
        having to use ManuallyDrop<T> & implement Drop.
    */
    queue: vkal::Queue,                      // -> LogicalDevice, CmdPool (soft), Swapchain (soft)
    cmd_pool: vkal::CmdPool,                 // -> LogicalDevice
    swapchain: vkal::Swapchain,              // -> Instance, Surface, LogicalDevice
    device: Rc<vkal::Device>,               // -> Instance, Surface (through PhysicalDevice)
    surface: vkal::Surface,                  // -> Instance
    instance: Rc<vkal::Instance>,           // -> Entry
    entry: ash::Entry,
}

impl VulkanCore {
    pub fn new(instance_params: vkal::InstanceParams, display_handle: RawDisplayHandle, window_handle: RawWindowHandle) -> vkal::Result<Self> {
        let entry = ash::Entry::linked();

        let instance = vkal::Instance::new(instance_params, &entry, display_handle)?;
        let instance = Rc::new(instance);

        let surface = vkal::Surface::new(&entry, Rc::clone(&instance), display_handle, window_handle)?;

        let device = vkal::Device::new(&instance, &surface)?;
        let device = Rc::new(device);

        let swapchain = vkal::Swapchain::new(&surface, Rc::clone(&device))?;

        let cmd_pool = vkal::CmdPool::new(Rc::clone(&device), vk::CommandPoolCreateFlags::empty())?;


        let gfx_qf = device.get_physical_device_info().best_queue_family_for_graphics;
        let q_idx = 0;

        let queue = vkal::Queue::new(Rc::clone(&device), &swapchain, gfx_qf, q_idx)?;

        // Self::print_sizes();

        Ok(VulkanCore { entry, instance, surface, device, swapchain, cmd_pool, queue })
    }

    #[allow(dead_code)]
    pub fn print_sizes() {
        println!("sizeof ash:: Instance:               {}", size_of::<ash ::Instance>());
        println!("sizeof vkal::Instance:               {}", size_of::<vkal::Instance>());

        println!("sizeof vk::  SurfaceKHR:             {}", size_of::<vk::  SurfaceKHR>());
        println!("sizeof vkal::Surface:                {}", size_of::<vkal::Surface>());


        println!("sizeof ash:: Device:                 {}", size_of::<ash:: Device>());
        println!("sizeof vkal::Device:                 {}", size_of::<vkal::Device>());

        println!("sizeof vk::  SwapchainKHR:           {}", size_of::<vk::  SwapchainKHR>());
        println!("sizeof vkal::Swapchain:              {}", size_of::<vkal::Swapchain>());

        println!("sizeof vk::  CommandPool:            {}", size_of::<vk::  CommandPool>());
        println!("sizeof vkal::CmdPool:                {}", size_of::<vkal::CmdPool>());

        println!("sizeof vkal::PhysicalDeviceInfo:     {}", size_of::<vkal::physical_device::PhysicalDeviceInfo>());
        println!("sizeof vkal::DebugUtils:             {}", size_of::<vkal::DebugUtils>());

        println!("sizeof khr::surface::Instance:       {}", size_of::<khr::surface::Instance>());
        println!("sizeof ext::debug_utils::Instance    {}", size_of::<ext::debug_utils::Instance>());
        println!("sizeof vk:Image:                     {}", size_of::<vk::Image>());
        println!("sizeof vk::CommandBuffer:            {}", size_of::<vk::CommandBuffer>());
    }

    pub fn get_cmd_pool_mut(&mut self) -> &mut vkal::CmdPool { &mut self.cmd_pool }

    pub fn get_cmd_pool(&self) -> &vkal::CmdPool { &self.cmd_pool }

    pub fn get_device(&self) -> &Rc<vkal::Device> { &self.device }

    pub fn get_swapchain(&self) -> &vkal::Swapchain { &self.swapchain }

    pub fn get_queue(&self) -> &vkal::Queue { &self.queue }
}