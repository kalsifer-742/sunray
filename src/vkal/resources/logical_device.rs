use std::error::Error;
use std::ffi::CStr;
use std::ops::Deref;
use ash::vk;
use crate::vkal;

pub struct LogicalDevice {
    _dev: ash::Device,
}
impl LogicalDevice {
    pub fn new(instance: &ash::Instance, physdev_info: vkal::PhysicalDeviceInfo) -> Result<LogicalDevice, Box<dyn Error>> {
        let allocator = None; // currently no support for custom allocator
        let physdev = physdev_info.get_physical_device(instance)?;

        let device_q_create_flags = vk::DeviceQueueCreateFlags::empty();

        let priorities = vec![1.0; physdev_info.number_of_queues as usize];

        let device_q_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(physdev_info.best_queue_family_for_graphics as u32)
            .flags(device_q_create_flags)
            .queue_priorities(&priorities);
        let device_q_create_infos = [device_q_create_info];

        let extensions = [
            ash::khr::swapchain::NAME
        ].map(CStr::as_ptr);


        let logical_dev_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&device_q_create_infos)
            .enabled_extension_names(&extensions)
        ;
        let logical_dev = unsafe { instance.create_device(physdev, &logical_dev_create_info, allocator) }?;

        Ok(LogicalDevice { _dev : logical_dev })
    }
}
impl Drop for LogicalDevice {
    fn drop(&mut self) {
        let allocator = None; // currently no support for custom allocator
        unsafe { self._dev.destroy_device(allocator) };
    }
}
impl Deref for LogicalDevice {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target { &self._dev }
}