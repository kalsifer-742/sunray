use std::rc::Rc;

use ash::{khr, vk};

use sunray::{error::*, vulkan_abstraction};

pub struct Swapchain {
    core: Rc<vulkan_abstraction::Core>,
    swapchain_device: khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    image_extent: vk::Extent2D,
}

impl Swapchain {
    pub fn get_extent(
        window_extent: (u32, u32),
        surface_support_details: &vulkan_abstraction::SurfaceSupportDetails,
    ) -> vk::Extent2D {
        if surface_support_details
            .surface_capabilities
            .current_extent
            .width
            != u32::MAX
        {
            surface_support_details.surface_capabilities.current_extent
        } else {
            vk::Extent2D {
                width: window_extent.0.clamp(
                    surface_support_details
                        .surface_capabilities
                        .min_image_extent
                        .width,
                    surface_support_details
                        .surface_capabilities
                        .max_image_extent
                        .width,
                ),
                height: window_extent.1.clamp(
                    surface_support_details
                        .surface_capabilities
                        .min_image_extent
                        .height,
                    surface_support_details
                        .surface_capabilities
                        .max_image_extent
                        .height,
                ),
            }
        }
    }

    fn build_swapchain(
        core: &Rc<vulkan_abstraction::Core>,
        surface: vk::SurfaceKHR,
        window_extent: (u32, u32),
        old_swapchain: Option<vk::SwapchainKHR>,
    ) -> SrResult<(
        vk::SwapchainKHR,
        Vec<vk::Image>,
        Vec<vk::ImageView>,
        vk::Extent2D,
    )> {
        let instance = core.instance();
        let device = core.device();
        let swapchain_device = khr::swapchain::Device::new(instance, device.inner());

        // for creating swapchain and swapchain_image_views
        let surface_format = {
            let formats = &device.surface_support_details().surface_formats;

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

                log::warn!(
                    "the BGRA8 SRGB format is not supported by the current physical device; falling back to {format:?}"
                );

                format
            }
        };

        let image_extent = Self::get_extent(window_extent, &device.surface_support_details());

        let swapchain = {
            let present_modes = &device.surface_support_details().surface_present_modes;
            let present_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
                vk::PresentModeKHR::MAILBOX
            } else if present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
                vk::PresentModeKHR::IMMEDIATE
            } else {
                vk::PresentModeKHR::FIFO // fifo is guaranteed to exist
            };

            let surface_capabilities = &device.surface_support_details().surface_capabilities;

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
                .surface(surface)
                .min_image_count(image_count)
                .image_format(surface_format.format)
                .image_color_space(surface_format.color_space)
                .image_extent(image_extent)
                .image_array_layers(1)
                .image_usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
                )
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(surface_capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .old_swapchain(old_swapchain.unwrap_or(vk::SwapchainKHR::null()));

            unsafe { swapchain_device.create_swapchain(&swapchain_create_info, None) }?
        };

        let images = unsafe { swapchain_device.get_swapchain_images(swapchain) }?;

        let fmt_handles = |imgs: &[vk::Image]| -> String {
            if log::max_level() < log::LevelFilter::Debug {
                return String::new();
            }
            let mut s = String::from("[ ");
            for img in imgs.iter() {
                s += &format!("{:#x?}, ", img);
            }
            s += "]";

            s
        };

        log::debug!(
            "built swapchain {{ handle: {swapchain:#x?}, extent: {image_extent:?}, images: {} }}",
            fmt_handles(&images)
        );

        let image_views = images
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

                unsafe {
                    device
                        .inner()
                        .create_image_view(&image_view_create_info, None)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok((swapchain, images, image_views, image_extent))
    }

    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        surface: vk::SurfaceKHR,
        window_extent: (u32, u32),
    ) -> SrResult<Self> {
        let swapchain_device = khr::swapchain::Device::new(core.instance(), core.device().inner());
        let (swapchain, images, image_views, image_extent) =
            Self::build_swapchain(&core, surface, window_extent, None)?;

        Ok(Self {
            core,
            swapchain_device,
            swapchain,
            images,
            image_views,
            image_extent,
        })
    }

    pub fn inner(&self) -> vk::SwapchainKHR {
        self.swapchain
    }
    pub fn device(&self) -> &khr::swapchain::Device {
        &self.swapchain_device
    }
    #[allow(unused)]
    pub fn extent(&self) -> vk::Extent2D {
        self.image_extent
    }
    pub fn images(&self) -> &[vk::Image] {
        &self.images
    }
    #[allow(unused)]
    pub fn image_views(&self) -> &[vk::ImageView] {
        &self.image_views
    }

    pub fn rebuild(&mut self, surface: vk::SurfaceKHR, window_extent: (u32, u32)) -> SrResult<()> {
        for img_view in self.image_views.iter() {
            unsafe {
                self.core
                    .device()
                    .inner()
                    .destroy_image_view(*img_view, None)
            };
        }

        self.image_views = vec![];
        self.images = vec![];

        let (swapchain, images, image_views, image_extent) =
            Self::build_swapchain(&self.core, surface, window_extent, Some(self.swapchain))?;

        unsafe {
            self.swapchain_device
                .destroy_swapchain(self.swapchain, None)
        };

        self.swapchain = swapchain;
        self.images = images;

        self.image_views = image_views;
        self.image_extent = image_extent;

        Ok(())
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        for img_view in self.image_views.iter() {
            unsafe {
                self.core
                    .device()
                    .inner()
                    .destroy_image_view(*img_view, None)
            };
        }

        //"swapchain and all associated VkImage handles are destroyed" by calling VkDestroySwapchainKHR
        if self.swapchain != vk::SwapchainKHR::null() {
            unsafe {
                self.swapchain_device
                    .destroy_swapchain(self.swapchain, None)
            };
        }
    }
}
