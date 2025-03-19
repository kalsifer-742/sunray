use std::error::Error;
use std::ffi::CStr;
use std::ops::Deref;
use ash::{khr, vk};
use crate::vkal;

pub mod physical_device;
mod queue;

pub use queue::*;

pub struct Device {
    device: ash::Device,
    swapchain_device: ash::khr::swapchain::Device,
    physdev_info: physical_device::PhysicalDeviceInfo,
}
impl Device {
    pub fn new(instance: &vkal::Instance, surface: &vkal::Surface) -> Result<Device, Box<dyn Error>> {
        let physdev_info = physical_device::PhysicalDeviceInfo::new(instance, surface)?;
        let physdev = physdev_info.get_physical_device(instance)?;

        let device_q_create_flags = vk::DeviceQueueCreateFlags::empty();

        let priorities = vec![1.0; physdev_info.number_of_queues as usize];

        //currently only one queue family is used
        let device_q_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(physdev_info.best_queue_family_for_graphics as u32)
            .flags(device_q_create_flags)
            .queue_priorities(&priorities);
        let device_q_create_infos = [device_q_create_info];

        let extensions = [
            ash::khr::swapchain::NAME,
        ].map(CStr::as_ptr);


        let logical_dev_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&device_q_create_infos)
            .enabled_extension_names(&extensions)
        ;
        let device = unsafe { instance.create_device(physdev, &logical_dev_create_info, vkal::NO_ALLOCATOR) }?;

        let swapchain_device = ash::khr::swapchain::Device::new(instance, &device);

        Ok(Device { device, swapchain_device, physdev_info })
    }

    pub fn get_swapchain_device(&self) -> &khr::swapchain::Device { &self.swapchain_device }
    pub fn get_physical_device_info(&self) -> &physical_device::PhysicalDeviceInfo { &self.physdev_info }
}
impl Drop for Device {
    fn drop(&mut self) {
        unsafe { self.device.destroy_device(vkal::NO_ALLOCATOR) };
    }
}
impl Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target { &self.device }
}