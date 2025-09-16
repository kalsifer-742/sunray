use std::ffi::CStr;

use ash::{ext, vk, Entry};

use crate::error::SrResult;

pub struct Instance {
    instance: ash::Instance,
}

impl Instance {
    pub const VALIDATION_LAYER_NAME: &'static CStr = c"VK_LAYER_KHRONOS_validation";

    fn check_validation_layer_support(entry: &Entry) -> SrResult<bool> {
        let layers_props = unsafe { entry.enumerate_instance_layer_properties() }?;

        let supports_validation_layer = layers_props
            .iter()
            .any(|p| p.layer_name_as_c_str().unwrap() == Self::VALIDATION_LAYER_NAME); //TODO unwrap

        Ok(supports_validation_layer)
    }

    pub fn new(entry: &ash::Entry, instance_exts: &[*const i8], with_validation_layer: bool, with_gpuav: bool) -> SrResult<Self> {
        let application_info = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 4, 0));

        let (enable_validation_layer, layer_names) = {
            let validation_layer_supported = Self::check_validation_layer_support(&entry)?;

            let enable_validation_layer = if with_validation_layer{
                if validation_layer_supported {
                    println!("Validation layer enabled"); //TODO: use a logging library
                    true
                } else {
                    println!("No validation layer support; continuing without validation layer...");
                    false
                }
            } else {
                println!("validation layer disabled");
                false
            };

            let layer_names = if enable_validation_layer {
                vec![ Self::VALIDATION_LAYER_NAME.as_ptr() ]
            } else {
                vec![]
            };

            (enable_validation_layer, layer_names)
        };

        let instance_extensions = {
            if enable_validation_layer {
            // add VK_EXT_layer_settings for configuring the validation layer
                instance_exts.iter()
                    .copied()
                    .chain(
                        std::iter::once(ext::layer_settings::NAME.as_ptr())
                    )
                    .collect::<Vec<_>>()
            } else {
                instance_exts.iter().copied().collect::<Vec<_>>()
            }
        };


        // use VK_EXT_layer_settings to configure the validation layer
        let validation_setting = |name: &'static CStr, v: bool| {
            const TRUE_BYTES : [u8; 4] = 1_u32.to_le_bytes();
            const FALSE_BYTES : [u8; 4] = 0_u32.to_le_bytes();
            vk::LayerSettingEXT::default()
                .layer_name(Self::VALIDATION_LAYER_NAME)
                .setting_name(name)
                .ty(vk::LayerSettingTypeEXT::BOOL32)
                .values(if v {&TRUE_BYTES} else {&FALSE_BYTES})
        };
        let settings = [
            // Khronos Validation layer recommends not to enable both GPU Assisted Validation (gpuav_enable) and Normal Core Check Validation (validate_core), as it will be very slow.
            // Once all errors in Core Check are solved it recommends to disable validate_core, then only use GPU-AV for best performance.
            validation_setting(c"validate_core", !with_gpuav),
            validation_setting(c"gpuav_enable", with_gpuav), // gpu assisted validation
            validation_setting(c"gpuav_shader_instrumentation", with_gpuav), // instrument shaders to validate descriptors (aka shader instrumentality project)
            validation_setting(c"validate_sync", true),
            validation_setting(c"validate_best_practices", true),
        ];
        let mut layer_settings_create_info = if enable_validation_layer {
            Some(
                vk::LayerSettingsCreateInfoEXT::default()
                    .settings(&settings)
            )
        } else {
            None
        };

        let instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&application_info)
            .enabled_layer_names(&layer_names)
            .enabled_extension_names(&instance_extensions);

        let instance_create_info = if enable_validation_layer {
            instance_create_info.push_next(layer_settings_create_info.as_mut().unwrap())
        } else {
            instance_create_info
        };

        let instance = unsafe { entry.create_instance(&instance_create_info, None) }?;
        Ok(Self { instance })
    }
    pub fn inner(&self) -> &ash::Instance { &self.instance }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}
