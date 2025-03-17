use std::error::Error;
use ash::vk;
use crate::vkal;

pub struct Swapchain {
    swapchain: vk::SwapchainKHR,
    swapchain_device: ash::khr::swapchain::Device,
}
impl Swapchain {
    pub fn new(surface: vk::SurfaceKHR, instance: &ash::Instance, device: &ash::Device, physdev_info: &vkal::PhysicalDeviceInfo) -> Result<Swapchain, Box<dyn Error>> {
        let allocator = None; // currently no support for custom allocator
        let mut number_of_images = physdev_info.surface_capabilities.min_image_count+1;
        if physdev_info.surface_capabilities.max_image_count != 0 {
            number_of_images = number_of_images.min(physdev_info.surface_capabilities.max_image_count)
        }
        let qf_indices = [physdev_info.best_queue_family_for_graphics as u32];
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .image_format(physdev_info.format.format)
            .image_color_space(physdev_info.format.color_space)
            .image_extent(physdev_info.surface_capabilities.current_extent)
            .pre_transform(physdev_info.surface_capabilities.current_transform)
            .min_image_count(number_of_images)
            .present_mode(physdev_info.presentation_mode)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT) //enable TRANSFER_DST for post-processing
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&qf_indices)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE) // ignore the alpha channel for now
            .clipped(true)
        ;

        let swapchain_device = ash::khr::swapchain::Device::new(instance, device);
        let swapchain = unsafe { swapchain_device.create_swapchain(&swapchain_create_info, allocator) }?;

        Ok(Self { swapchain, swapchain_device, })
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        let allocator = None; // currently no support for custom allocator
        unsafe { self.swapchain_device.destroy_swapchain(self.swapchain, allocator) };
    }
}