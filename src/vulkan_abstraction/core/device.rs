use std::{collections::HashSet, ffi::CStr};

use ash::{
    khr,
    vk::{self, FormatFeatureFlags},
};

use crate::{error::*, vulkan_abstraction};

pub struct Device {
    device: ash::Device,
    physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    physical_device_rt_pipeline_properties:
        vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>,
    queue_family_index: u32,
    surface_support_details: Option<SurfaceSupportDetails>,
}

impl Device {
    pub fn new(
        instance: &vulkan_abstraction::Instance,
        device_extensions: &[*const i8],
        image_format: vk::Format,
        surface_to_support: &Option<(vk::SurfaceKHR, khr::surface::Instance)>
    ) -> SrResult<Self> {
        let instance = instance.inner();
        let physical_devices = unsafe { instance.enumerate_physical_devices() }?;

        let (physical_device, surface_support_details, queue_family_index) = physical_devices
            .into_iter()
            //only allow devices which support all required extensions
            .filter(|physical_device| {
                Self::check_device_extension_support(
                    &instance,
                    *physical_device,
                    &device_extensions,
                )
                .unwrap_or(false)
            })
            //check for blit support
            .filter(|physical_device| {
                let format_properties = unsafe {
                    instance.get_physical_device_format_properties(*physical_device, image_format)
                };

                format_properties
                    .optimal_tiling_features
                    .contains(FormatFeatureFlags::BLIT_SRC)
                    && format_properties
                        .linear_tiling_features
                        .contains(FormatFeatureFlags::BLIT_DST)
            })
            // filter out devices without swapchain support if necessary
            .filter_map(|physical_device| {
                if let Some((surface, surface_instance)) = &surface_to_support {
                    let surface_support_details = SurfaceSupportDetails::new(*surface, surface_instance, physical_device).unwrap();
                    if surface_support_details.check_swapchain_support() {
                        Some((physical_device, Some(surface_support_details)))
                    } else {
                        None
                    }
                } else {
                    Some((physical_device, None))
                }
            })
            //choose a suitable queue family, and filter out devices without one
            .filter_map(|(physical_device, surface_support_details)| {
                Some((
                    physical_device,
                    surface_support_details,
                    Self::select_queue_family(&instance, physical_device)?,
                ))
            })
            // try to get a discrete or at least integrated gpu
            .max_by_key(|(physical_device, _surface_support_details, _queue_family_index)| {
                let device_type =
                    unsafe { instance.get_physical_device_properties(*physical_device) }
                        .device_type;

                match device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 2,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                    _ => 0,
                }
            })
            .ok_or(SrError::new(None, String::from("No suitable GPU found!")))?;

        let device = {
            let queue_priorities = vec![1.0; 1]; // TODO: use more than 1 queue?
            let queue_create_infos = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&queue_priorities)];

            // enable some device features necessary for ray-tracing
            let mut vk12_features = vk::PhysicalDeviceVulkan12Features::default()
                .buffer_device_address(true) // necessary for ray-tracing
                .timeline_semaphore(true)
                .vulkan_memory_model(true)
                .vulkan_memory_model_device_scope(true)
                .storage_buffer8_bit_access(true);
            let mut physical_device_rt_pipeline_features =
                vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default()
                    .ray_tracing_pipeline(true);
            let mut physical_device_acceleration_structure_features =
                vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
                    .acceleration_structure(true);

            let device_create_info = vk::DeviceCreateInfo::default()
                .enabled_extension_names(&device_extensions)
                .push_next(&mut vk12_features)
                .push_next(&mut physical_device_rt_pipeline_features)
                .push_next(&mut physical_device_acceleration_structure_features)
                .queue_create_infos(&queue_create_infos);

            unsafe { instance.create_device(physical_device, &device_create_info, None) }?
        };
        //necessary for memory allocations
        let physical_device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let physical_device_rt_pipeline_properties = {
            let mut physical_device_rt_pipeline_properties =
                vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();

            let mut physical_device_properties = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut physical_device_rt_pipeline_properties);

            unsafe {
                instance.get_physical_device_properties2(
                    physical_device,
                    &mut physical_device_properties,
                )
            };

            physical_device_rt_pipeline_properties
        };

        Ok(Self {
            device,
            physical_device_memory_properties,
            physical_device_rt_pipeline_properties,
            queue_family_index,
            surface_support_details,
        })
    }

    fn select_queue_family(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Option<u32> {
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
            .into_iter()
            .enumerate()
            .filter(|(_queue_family_index, queue_family_props)| {
                queue_family_props
                    .queue_flags
                    .contains(vk::QueueFlags::GRAPHICS)
            })
            .map(|(queue_family_index, _)| queue_family_index as u32)
            .next()
    }

    fn check_device_extension_support(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        required_device_extensions: &[*const i8],
    ) -> SrResult<bool> {
        let required_exts_set: HashSet<&CStr> = required_device_extensions
            .iter()
            .map(|p| unsafe { CStr::from_ptr(*p) })
            .collect();

        let available_exts =
            unsafe { instance.enumerate_device_extension_properties(physical_device) }?;

        let available_exts_set: HashSet<&CStr> = available_exts
            .iter()
            .map(|props| props.extension_name_as_c_str())
            .collect::<Result<_,_>>()
            .map_err(|e| {
                SrError::new(None, format!("Error while checking device extension support. Could not convert extension name to CStr with message: {e}"))
            })?;

        let all_exts_are_available = required_exts_set.is_subset(&available_exts_set);

        Ok(all_exts_are_available)
    }

    pub fn inner(&self) -> &ash::Device {
        &self.device
    }
    pub fn rt_pipeline_properties(
        &self,
    ) -> &vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static> {
        &self.physical_device_rt_pipeline_properties
    }
    pub fn memory_properties(&self) -> &vk::PhysicalDeviceMemoryProperties {
        &self.physical_device_memory_properties
    }
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }
    pub fn surface_support_details(&self) -> &SurfaceSupportDetails {
        self.surface_support_details.as_ref().unwrap()
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}

pub struct SurfaceSupportDetails {
    pub surface_capabilities: vk::SurfaceCapabilitiesKHR,
    pub surface_formats: Vec<vk::SurfaceFormatKHR>,
    pub surface_present_modes: Vec<vk::PresentModeKHR>,
}

impl SurfaceSupportDetails {
    /*
    TODO: bad error handling.
    many different phases can cause vulkan errors in choosing a physical device.
    we don't keep track of which errors occur because if no suitable device is found we'd need to tell the user what error makes each device unsuitable
    */
    fn new(
        surface: vk::SurfaceKHR,
        surface_instance: &khr::surface::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> SrResult<Self> {
        let surface_capabilities = unsafe {
            surface_instance.get_physical_device_surface_capabilities(physical_device, surface)
        }?;

        let surface_formats = unsafe {
            surface_instance.get_physical_device_surface_formats(physical_device, surface)
        }?;

        let surface_present_modes = unsafe {
            surface_instance.get_physical_device_surface_present_modes(physical_device, surface)
        }?;

        Ok(Self {
            surface_capabilities,
            surface_formats,
            surface_present_modes,
        })
    }

    fn check_swapchain_support(&self) -> bool {
        !self.surface_formats.is_empty() && !self.surface_present_modes.is_empty()
    }
}
