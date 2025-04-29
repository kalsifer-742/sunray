use std::ffi::CStr;

use ash::{
    Entry, Instance,
    khr::{self, surface, swapchain},
    vk::{
        ApplicationInfo, ColorSpaceKHR, ComponentMapping, ComponentSwizzle, CompositeAlphaFlagsKHR,
        DeviceCreateInfo, DeviceQueueCreateInfo, Extent2D, Format, ImageAspectFlags,
        ImageSubresourceRange, ImageUsageFlags, ImageView, ImageViewCreateInfo, ImageViewType,
        InstanceCreateInfo, PhysicalDevice, PhysicalDeviceType, PresentModeKHR, QueueFlags,
        SharingMode, SurfaceCapabilitiesKHR, SurfaceFormatKHR, SurfaceKHR, SwapchainCreateInfoKHR,
        SwapchainKHR, make_api_version,
    },
};
use error::{SrError, SrResult};
use winit::raw_window_handle_05::{RawDisplayHandle, RawWindowHandle};

pub mod error;
pub mod utils;

struct SwapchainSupportDetails {
    surface_capabilities: SurfaceCapabilitiesKHR,
    surface_formats: Vec<SurfaceFormatKHR>,
    surface_present_modes: Vec<PresentModeKHR>,
}

impl SwapchainSupportDetails {
    /*
    TODO: bad error handling.
    many different phases can cause vulkan errors in choosing a physical device.
    we don't keep track of which errors occur because if no suitable device is found we'd need to tell the user what error makes each device unsuitable
    */
    fn new(
        surface: SurfaceKHR,
        surface_instance: &surface::Instance,
        physical_device: PhysicalDevice,
    ) -> Self {
        let surface_capabilities = unsafe {
            surface_instance.get_physical_device_surface_capabilities(physical_device, surface)
        }
        .unwrap();

        let surface_formats = unsafe {
            surface_instance.get_physical_device_surface_formats(physical_device, surface)
        }
        .unwrap();

        let surface_present_modes = unsafe {
            surface_instance.get_physical_device_surface_present_modes(physical_device, surface)
        }
        .unwrap();

        Self {
            surface_capabilities,
            surface_formats,
            surface_present_modes,
        }
    }

    fn check_swapchain_support(&self) -> bool {
        !self.surface_formats.is_empty() && !self.surface_present_modes.is_empty()
    }
}

//TODO: impl Drop

pub struct Core {
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: ash::vk::Queue,
    surface: ash::vk::SurfaceKHR,
}

impl Core {
    const VALIDATION_LAYER_NAME: &CStr = c"VK_LAYER_KHRONOS_validation";

    // TODO: currently take for granted that the user has a window, no support for offline rendering
    pub fn new(
        window_extent: [u32; 2],
        raw_window_handle: RawWindowHandle,
        raw_display_handle: RawDisplayHandle,
    ) -> SrResult<Self> {
        let entry = Entry::linked();
        let application_info = ApplicationInfo::default().api_version(make_api_version(0, 1, 4, 0));

        let enabled_layer_names =
            if cfg!(debug_assertions) && Self::check_validation_layer_support(&entry)? {
                &[c"VK_LAYER_KHRONOS_validation".as_ptr()]
            } else {
                [].as_slice()
            };

        let required_extensions = crate::utils::enumerate_required_extensions(raw_display_handle)?;

        let instance_create_info = InstanceCreateInfo::default()
            .application_info(&application_info)
            .enabled_layer_names(enabled_layer_names)
            .enabled_extension_names(required_extensions);

        let instance = unsafe { entry.create_instance(&instance_create_info, None) }
            .map_err(|e| SrError::from_vk_result(e))?;

        let surface_instance = ash::khr::surface::Instance::new(&entry, &instance);

        let surface = unsafe {
            crate::utils::create_surface(
                &entry,
                &instance,
                raw_display_handle,
                raw_window_handle,
                None,
            )
        }?;

        let required_device_extensions = &[
            khr::ray_tracing_pipeline::NAME,
            khr::acceleration_structure::NAME,
            khr::deferred_host_operations::NAME,
            khr::swapchain::NAME, // TODO: not needed for offline rendering
        ]
        .map(CStr::as_ptr);

        let physical_devices =
            unsafe { instance.enumerate_physical_devices() }.map_err(SrError::from_vk_result)?;

        let (physical_device, queue_family_index, swapchain_support_details) = physical_devices
            .into_iter()
            .filter(|physical_device| {
                let device_type =
                    unsafe { instance.get_physical_device_properties(*physical_device) }
                        .device_type;

                device_type == PhysicalDeviceType::DISCRETE_GPU
                    && Self::check_device_extension_support(
                        &instance,
                        *physical_device,
                        required_device_extensions,
                    )
            })
            .filter_map(|physical_device| {
                let swapchain_support_details =
                    SwapchainSupportDetails::new(surface, &surface_instance, physical_device);

                if swapchain_support_details.check_swapchain_support() {
                    Some((physical_device, swapchain_support_details))
                } else {
                    None
                }
            })
            .filter_map(|(physical_device, swapchain_support_details)| {
                Some((
                    physical_device,
                    Self::select_queue_family(
                        &instance,
                        &surface_instance,
                        physical_device,
                        surface,
                    )?,
                    swapchain_support_details,
                ))
            })
            .next()
            .unwrap(); //TODO return error

        let queue_priorities = vec![1.0; 1]; // TODO: use more than 1 queue
        let queue_create_infos = [DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities)];

        let device_create_info = DeviceCreateInfo::default()
            .enabled_extension_names(required_device_extensions)
            .queue_create_infos(&queue_create_infos);

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }
            .map_err(SrError::from_vk_result)?; //TODO manage errors

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let surface_format =
            swapchain_support_details
                .surface_formats
                .iter()
                .find(|surface_format| {
                    surface_format.format == Format::B8G8R8A8_SRGB
                        && surface_format.color_space == ColorSpaceKHR::SRGB_NONLINEAR
                })
                .unwrap_or(swapchain_support_details.surface_formats.first().ok_or(
                    SrError::new(
                        None,
                        String::from("Physical device does not support any surface formats"),
                    ),
                )?);

        let presentation_mode = if swapchain_support_details
            .surface_present_modes
            .contains(&PresentModeKHR::MAILBOX)
        {
            PresentModeKHR::MAILBOX
        } else if swapchain_support_details
            .surface_present_modes
            .contains(&PresentModeKHR::IMMEDIATE)
        {
            PresentModeKHR::IMMEDIATE
        } else {
            PresentModeKHR::FIFO // fifo is guaranteed to exist
        };

        let capabilities = swapchain_support_details.surface_capabilities;
        let image_extent = if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            Extent2D {
                width: window_extent[0].clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: window_extent[1].clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        };

        //sticking to this minimum means that we may sometimes have to wait on the driver to complete internal operations before we can acquire another image to render to.
        // Therefore it is recommended to request at least one more image than the minimum
        let mut image_count = swapchain_support_details
            .surface_capabilities
            .min_image_count
            + 1;

        if swapchain_support_details
            .surface_capabilities
            .max_image_count
            > 0
            && image_count
                > swapchain_support_details
                    .surface_capabilities
                    .max_image_count
        {
            image_count = swapchain_support_details
                .surface_capabilities
                .max_image_count;
        }

        let swapchain_create_info = SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(image_extent)
            .image_array_layers(1)
            .image_usage(ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(SharingMode::EXCLUSIVE)
            .pre_transform(
                swapchain_support_details
                    .surface_capabilities
                    .current_transform,
            )
            .composite_alpha(CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(presentation_mode)
            .clipped(true)
            .old_swapchain(SwapchainKHR::null());

        let swapchain_device = swapchain::Device::new(&instance, &device);
        let swapchain = unsafe { swapchain_device.create_swapchain(&swapchain_create_info, None) }
            .map_err(SrError::from_vk_result)?;

        let images = unsafe { swapchain_device.get_swapchain_images(swapchain) }
            .map_err(SrError::from_vk_result)?;

        let image_views = images
            .iter()
            .map(|image| {
                let image_view_create_info = ImageViewCreateInfo::default()
                    .image(*image)
                    .view_type(ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .components(ComponentMapping {
                        r: ComponentSwizzle::IDENTITY,
                        g: ComponentSwizzle::IDENTITY,
                        b: ComponentSwizzle::IDENTITY,
                        a: ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(
                        ImageSubresourceRange::default()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );

                unsafe { device.create_image_view(&image_view_create_info, None) }
                    .map_err(SrError::from_vk_result)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            entry,
            instance,
            device,
            queue,
            surface,
        })
    }

    fn check_validation_layer_support(entry: &Entry) -> SrResult<bool> {
        let layers_props = unsafe { entry.enumerate_instance_layer_properties() }
            .map_err(SrError::from_vk_result)?;

        Ok(layers_props
            .iter()
            .any(|p| p.layer_name_as_c_str().unwrap() == Self::VALIDATION_LAYER_NAME)) //TODO unwrap
    }

    fn select_queue_family(
        instance: &Instance,
        surface_instance: &surface::Instance,
        physical_device: PhysicalDevice,
        surface: SurfaceKHR,
    ) -> Option<u32> {
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
            .into_iter()
            .enumerate()
            .filter(|(queue_family_index, queue_family_props)| {
                queue_family_props
                    .queue_flags
                    .contains(QueueFlags::GRAPHICS)
                    && unsafe {
                        surface_instance.get_physical_device_surface_support(
                            physical_device,
                            *queue_family_index as u32,
                            surface,
                        )
                    }
                    .unwrap_or(false)
            })
            .map(|(queue_family_index, _)| queue_family_index as u32)
            .next()
    }

    // TODO: make more readable (create 2 sets, call .is_subset or whatever)
    //      example: https://docs.vulkan.org/tutorial/latest/03_Drawing_a_triangle/01_Presentation/01_Swap_chain.html
    fn check_device_extension_support(
        instance: &Instance,
        physical_device: PhysicalDevice,
        required_device_extensions: &[*const i8],
    ) -> bool {
        required_device_extensions.iter().all(|ext| {
            let available_extensions =
                unsafe { instance.enumerate_device_extension_properties(physical_device) }.unwrap(); // TODO: manage error

            available_extensions.iter().any(|ext_props| {
                let ext_cstr = unsafe { CStr::from_ptr(*ext) };

                ext_props.extension_name_as_c_str().unwrap() == ext_cstr
            })
        })
    }
}
