pub mod owned_instance;
pub mod physical_device;
pub mod logical_device;
mod swapchain;

pub use owned_instance::*;
pub use physical_device::*;
pub use logical_device::*;
pub use swapchain::*;

use crate::vkal;
use std::error::Error;
use std::mem::ManuallyDrop;
use ash::vk;
use winit::raw_window_handle::{RawDisplayHandle, RawWindowHandle};

pub struct VulkanResources {
    _entry: ash::Entry,
    instance: ManuallyDrop<vkal::OwnedInstance>,
    surface_instance: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    _physdev_info: vkal::PhysicalDeviceInfo,
    logical_device: ManuallyDrop<vkal::LogicalDevice>,
    swapchain: ManuallyDrop<vkal::Swapchain>
}
impl VulkanResources {
    pub fn new(instance_params: vkal::OwnedInstanceParams, display_handle: RawDisplayHandle, window_handle: RawWindowHandle) -> Result<VulkanResources, Box<dyn Error>> {
        let allocator = None; // currently no support for custom allocator

        let entry = ash::Entry::linked();
        let instance = vkal::OwnedInstance::new(instance_params, &entry, display_handle).map(ManuallyDrop::new)?;

        let surface_instance = ash::khr::surface::Instance::new(&entry, &instance);
        let surface = unsafe { ash_window::create_surface(&entry, &instance, display_handle, window_handle, allocator)? };

        let physdev_info = vkal::PhysicalDeviceInfo::new(&instance, &surface_instance, surface)?;

        let logical_device = vkal::LogicalDevice::new(&instance, physdev_info).map(ManuallyDrop::new)?;

        let swapchain = vkal::Swapchain::new(surface, &instance, &logical_device, &physdev_info).map(ManuallyDrop::new)?;

        Ok(VulkanResources {
            _entry: entry, instance, surface_instance, surface,
            _physdev_info: physdev_info, logical_device, swapchain
        })
    }


}
impl Drop for VulkanResources {
    fn drop(&mut self) {
        let allocator = None; // currently no support for custom allocator

        // swapchain needs to be dropped before surface & logical_device
        unsafe { ManuallyDrop::drop(&mut self.swapchain); }

        // logical_device must be dropped before dropping instance
        unsafe { ManuallyDrop::drop(&mut self.logical_device); }

        // surface must be destroyed before dropping instance
        unsafe { self.surface_instance.destroy_surface(self.surface, allocator) };

        // instance should be dropped before entry
        unsafe { ManuallyDrop::drop(&mut self.instance); }
    }
}
