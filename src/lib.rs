use std::ffi::{CStr, c_char};

use ash::{
    khr, vk::{
        make_api_version, ApplicationInfo, DeviceCreateInfo, InstanceCreateInfo, PhysicalDevice, PhysicalDeviceType, QueueFlags
    }, Entry, Instance
};
use error::{SrError, SrResult};

pub mod error;
pub mod utils;

//TODO: impl Drop

pub struct Core {
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
}

impl Core {
    const VALIDATION_LAYER_NAME: &CStr = c"VK_LAYER_KHRONOS_validation";

    pub fn new(required_extensions: &[*const c_char]) -> SrResult<Self> {
        let entry = Entry::linked();
        let application_info = ApplicationInfo::default().api_version(make_api_version(0, 1, 4, 0));

        let enabled_layer_names =
            if cfg!(debug_assertions) && Self::check_validation_layer_support(&entry)? {
                &[c"VK_LAYER_KHRONOS_validation".as_ptr()]
            } else {
                [].as_slice()
            };

        let instance_create_info = InstanceCreateInfo::default()
            .application_info(&application_info)
            .enabled_layer_names(enabled_layer_names)
            .enabled_extension_names(required_extensions);

        let instance = unsafe { entry.create_instance(&instance_create_info, None) }
            .map_err(|e| SrError::from_vk_result(e))?;

        let physical_devices =
            unsafe { instance.enumerate_physical_devices() }.map_err(SrError::from_vk_result)?;

        let (physical_device, _queue_family_index) = physical_devices
        .into_iter()
        .filter(|device| unsafe {instance.get_physical_device_properties(*device)}.device_type == PhysicalDeviceType::DISCRETE_GPU )
        .filter_map(|physical_device| { 
            Some((physical_device, Self::select_queue_family(&instance, physical_device)?))
        })
        .next()
        .unwrap(); //TODO return error

        let extensions = &[khr::ray_tracing_pipeline::NAME, khr::acceleration_structure::NAME, khr::deferred_host_operations::NAME].map(CStr::as_ptr);
        let device_create_info = DeviceCreateInfo::default()
        .enabled_extension_names(extensions);

        //TODO get queue for device creation

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }.map_err(SrError::from_vk_result)?; //TODO manage errors

        Ok(Self { entry, instance, device })
    }

    fn check_validation_layer_support(entry: &Entry) -> SrResult<bool> {
        let layers_props = unsafe { entry.enumerate_instance_layer_properties() }
            .map_err(SrError::from_vk_result)?;

        Ok(layers_props
            .iter()
            .any(|p| p.layer_name_as_c_str().unwrap() == Self::VALIDATION_LAYER_NAME)) //TODO unwrap
    }

    fn select_queue_family(instance: &Instance, physical_device: PhysicalDevice) -> Option<usize> {
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
            .into_iter()
            .enumerate()
            .filter(|(_queue_family_index, queue_family_props)| queue_family_props.queue_flags.contains(QueueFlags::GRAPHICS))
            .map(|(queue_family_index, _)| queue_family_index)
            .next()
    }
}
