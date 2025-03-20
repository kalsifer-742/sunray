use std::error::Error;
use std::ops::Deref;
use std::rc::Rc;
use ash::vk;
use ash::vk::{ComponentMapping};
use crate::vkal;

pub struct Swapchain {
    swapchain: vk::SwapchainKHR,
    device: Rc<vkal::Device>, // needed for destruction of image_views
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
}
impl Swapchain {
    pub fn new(surface: &vkal::Surface, device: Rc<vkal::Device>) -> Result<Swapchain, Box<dyn Error>> {
        let physdev_info = device.get_physical_device_info();

        let mut number_of_images = physdev_info.surface_capabilities.min_image_count+1;
        if physdev_info.surface_capabilities.max_image_count != 0 {
            number_of_images = number_of_images.min(physdev_info.surface_capabilities.max_image_count)
        }
        let qf_indices = [physdev_info.best_queue_family_for_graphics as u32];
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.inner())
            .image_format(physdev_info.format.format)
            .image_color_space(physdev_info.format.color_space)
            .image_extent(physdev_info.surface_capabilities.current_extent)
            .pre_transform(physdev_info.surface_capabilities.current_transform)
            .min_image_count(number_of_images)
            .present_mode(physdev_info.presentation_mode)
            .image_array_layers(1)
            // TRANSFER_DST is necessary for clearing, COLOR_ATTACHMENT (or one a few others) is necessary for creating image views
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE) // only 1 qf can work on an image of the swapchain
            .queue_family_indices(&qf_indices)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE) // ignore the alpha channel for now
            .clipped(true) //allow discarding render operations on non-visible regions of the surface
        ;

        let swapchain = unsafe { device.get_swapchain_device().create_swapchain(&swapchain_create_info, vkal::NO_ALLOCATOR) }?;

        let images = unsafe { device.get_swapchain_device().get_swapchain_images(swapchain) }?;
        let image_views : Vec<vk::ImageView> = images.iter().map(|img| -> ash::prelude::VkResult<vk::ImageView> {
            let flags = vk::ImageViewCreateFlags::empty();
            let info = vk::ImageViewCreateInfo::default()
                .image(img.clone())
                .flags(flags)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(physdev_info.format.format)
                .components(ComponentMapping::default()) // no swizzling
                .subresource_range(vk::ImageSubresourceRange{
                    aspect_mask: vk::ImageAspectFlags::COLOR, // which aspects are included in the view
                    base_mip_level: 0,
                    level_count: 1, // no mipmapping (for now), so only 1 level
                    base_array_layer: 0,
                    layer_count: 1, // no layering (for now), so only 1 layer
                })
            ;

            unsafe { device.create_image_view(&info, vkal::NO_ALLOCATOR) }

        }).collect::<Result<Vec<_>, _>>()?;

        Ok(Self { swapchain, device, images, image_views })
    }
    #[allow(dead_code)]
    pub fn get_image_views(&self) -> &Vec<vk::ImageView> { &self.image_views }
    pub fn get_images(&self) -> &Vec<vk::Image> { &self.images }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        let views = std::mem::replace(&mut self.image_views, Vec::new()); // to take ownership of the views, so we can use into_iter
        for img_view in views {
            unsafe { self.device.destroy_image_view(img_view, vkal::NO_ALLOCATOR) };
        }
        let sc_dev = self.device.get_swapchain_device();
        unsafe { sc_dev.destroy_swapchain(self.swapchain, vkal::NO_ALLOCATOR) };
    }
}
impl Deref for Swapchain {
    type Target = vk::SwapchainKHR;
    fn deref(&self) -> &Self::Target { &self.swapchain }
}