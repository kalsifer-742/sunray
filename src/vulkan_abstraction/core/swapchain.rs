use std::rc::Rc;

use ash::{khr, vk};

use crate::{error::*, vulkan_abstraction};

pub struct Swapchain {
    device: Rc<vulkan_abstraction::Device>,
    swapchain_device: khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_image_extent: vk::Extent2D,
}

impl Swapchain {
    pub fn new(instance: &vulkan_abstraction::Instance, device: Rc<vulkan_abstraction::Device>, surface: &vulkan_abstraction::Surface, window_extent: [u32;2]) -> SrResult<Self> {
        let instance = instance.inner();
        let swapchain_device = khr::swapchain::Device::new(instance, &device.inner());

        // for creating swapchain and swapchain_image_views
        let surface_format = {
            let formats = &device.swapchain_support_details().surface_formats;

            //find the BGRA8 SRGB nonlinear surface format
            let bgra8_srgb_nonlinear = formats.iter().find(|surface_format| {
                surface_format.format == vk::Format::B8G8R8A8_SRGB
                    && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            });

            if let Some(format) = bgra8_srgb_nonlinear {
                *format
            } else {
                //or else get the first format the device offers
                let format = *formats.first().ok_or(SrError::new(
                    None,
                    String::from("Physical device does not support any surface formats"),
                ))?;

                log::warn!("the BGRA8 SRGB format is not supported by the current physical device; falling back to {format:?}");

                format
            }
        };

        let swapchain_image_extent = if device.swapchain_support_details().surface_capabilities.current_extent.width != u32::MAX {
            device.swapchain_support_details().surface_capabilities.current_extent
        } else {
            vk::Extent2D {
                width: window_extent[0].clamp(
                    device.swapchain_support_details().surface_capabilities.min_image_extent.width,
                    device.swapchain_support_details().surface_capabilities.max_image_extent.width,
                ),
                height: window_extent[1].clamp(
                    device.swapchain_support_details().surface_capabilities.min_image_extent.height,
                    device.swapchain_support_details().surface_capabilities.max_image_extent.height,
                ),
            }
        };

        let swapchain = {
            let present_modes = &device.swapchain_support_details().surface_present_modes;
            let present_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
                vk::PresentModeKHR::MAILBOX
            } else if present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
                vk::PresentModeKHR::IMMEDIATE
            } else {
                vk::PresentModeKHR::FIFO // fifo is guaranteed to exist
            };

            let surface_capabilities = &device.swapchain_support_details().surface_capabilities;

            // Sticking to this minimum means that we may sometimes have to wait on the driver to
            // complete internal operations before we can acquire another image to render to.
            // Therefore it is recommended to request at least one more image than the minimum
            let mut image_count = surface_capabilities.min_image_count + 1;

            if surface_capabilities.max_image_count > 0
                && image_count > surface_capabilities.max_image_count
            {
                image_count = surface_capabilities.max_image_count;
            }

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
                .surface(surface.inner())
                .min_image_count(image_count)
                .image_format(surface_format.format)
                .image_color_space(surface_format.color_space)
                .image_extent(swapchain_image_extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(surface_capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .old_swapchain(vk::SwapchainKHR::null());

            unsafe { swapchain_device.create_swapchain(&swapchain_create_info, None) }?
        };

        let swapchain_images = unsafe { swapchain_device.get_swapchain_images(swapchain) }?;

        let swapchain_image_views = swapchain_images
            .iter()
            .map(|image| {
                let image_view_create_info = vk::ImageViewCreateInfo::default()
                    .image(*image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );

                unsafe { device.inner().create_image_view(&image_view_create_info, None) }
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            device,
            swapchain_device,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            swapchain_image_extent,
        })
    }
    pub fn inner(&self) -> vk::SwapchainKHR { self.swapchain }
    pub fn device(&self) -> &khr::swapchain::Device { &self.swapchain_device }
    pub fn image_extent(&self) -> vk::Extent2D { self.swapchain_image_extent }
    pub fn images(&self) -> &[vk::Image]{ &self.swapchain_images }
    #[allow(unused)]
    pub fn image_views(&self) -> &[vk::ImageView]{ &self.swapchain_image_views }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        for img_view in self.swapchain_image_views.iter() {
            unsafe { self.device.inner().destroy_image_view(*img_view, None) };
        }
        //"swapchain and all associated VkImage handles are destroyed" by calling VkDestroySwapchainKHR

        unsafe { self.swapchain_device.destroy_swapchain(self.swapchain, None) };
    }
}
