use std::{
    collections::HashSet,
    ffi::{CStr, c_void},
};

use ash::{ext, vk};

use crate::error::SrResult;

pub struct Instance {
    instance: ash::Instance,
    debug_utils_instance: Option<ext::debug_utils::Instance>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

impl Instance {
    pub const VALIDATION_LAYER_NAME: &'static CStr = c"VK_LAYER_KHRONOS_validation";
    pub const DEBUG_EXTENSIONS: [&'static CStr; 1] = [
        // VK_EXT_debug_utils for setting up a debug messenger to log validation layer errors
        // NOTE: VK_EXT_debug_utils also allows tagging objects or command buffer commands to make them more recognizable in gpu debugging softwares like renderdoc or nsight graphics
        ext::debug_utils::NAME,
    ];

    fn check_validation_layer_support(entry: &ash::Entry) -> SrResult<bool> {
        let layers_props = unsafe { entry.enumerate_instance_layer_properties() }?;

        let supports_validation_layer = layers_props
            .iter()
            .any(|p| p.layer_name_as_c_str().unwrap() == Self::VALIDATION_LAYER_NAME); //TODO unwrap

        Ok(supports_validation_layer)
    }

    fn filter_supported_exts<'a>(
        entry: &ash::Entry,
        layer: Option<&CStr>,
        required_exts: &[&'a CStr],
    ) -> SrResult<Vec<&'a CStr>> {
        match layer {
            Some(layer) => log::info!(
                "Attempting to enable some instance extensions for layer {layer:?}: {required_exts:?}"
            ),
            None => log::info!("Attempting to enable some instance extensions: {required_exts:?}"),
        };

        let extension_support = unsafe { entry.enumerate_instance_extension_properties(layer) }?;
        let extension_support_u8 = extension_support
            .iter()
            .map(|ext| ext.extension_name.map(|c| c as u8))
            .collect::<Vec<_>>();
        let extension_support_hashset = HashSet::<&CStr>::from_iter(
            extension_support_u8
                .iter()
                .map(|ext_name_u8| CStr::from_bytes_until_nul(ext_name_u8).unwrap()),
        );

        let mut supported = Vec::new();
        let mut unsupported = Vec::new();

        for ext in required_exts.iter().copied() {
            if extension_support_hashset.contains(ext) {
                supported.push(ext);
            } else {
                unsupported.push(ext);
            }
        }

        if !unsupported.is_empty() {
            match layer {
                Some(layer) => log::warn!(
                    "Some instance extensions for layer {layer:?} required by sunray were unavailable: {:?}",
                    unsupported
                ),
                None => log::warn!(
                    "Some instance extensions required by sunray were unavailable: {:?}",
                    unsupported
                ),
            }
        }

        Ok(supported)
    }

    unsafe extern "system" fn debug_utils_callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
        _user_data: *mut c_void,
    ) -> vk::Bool32 {
        let callback_data = unsafe { *callback_data };
        let msg_id_number = callback_data.message_id_number;
        let msg_id_name = unsafe { CStr::from_ptr(callback_data.p_message_id_name) }
            .to_str()
            .unwrap();
        let msg_text = unsafe { CStr::from_ptr(callback_data.p_message) }
            .to_str()
            .unwrap();

        match (message_severity, message_type, msg_id_number) {
            (
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                0x675dc32e,
            ) => return vk::FALSE,
            (
                vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
                0x0,
            )
            | (
                vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
                0x0,
            ) => {
                if msg_id_name == "Loader Message" {
                    return vk::FALSE;
                }
            }
            _ => {}
        }

        let level = match message_severity {
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => log::Level::Debug,
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::Level::Info,
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::Level::Warn,
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::Level::Error,
            _ => {
                log::warn!("unexpected message severity, got {message_severity:?}");
                log::Level::Warn
            }
        };

        // some messages have id_number=0, don't print it in that case
        if msg_id_number != 0 {
            // print the id num in lowercase hex, padding up to 10 characters. 10 characters = 0x_ where _ are 8 hex digits => 8 nibbles = 32 bits
            log::log!(
                level,
                "{message_type:?} {msg_id_number:#010x} - {msg_id_name}: {msg_text}"
            );
        } else {
            log::log!(level, "{message_type:?} {msg_id_name}: {msg_text}");
        }

        vk::FALSE
    }

    pub fn new(
        entry: &ash::Entry,
        instance_exts: &[*const i8],
        with_validation_layer: bool,
        with_gpuav: bool,
    ) -> SrResult<Self> {
        let application_info =
            vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 4, 0));

        let (enable_validation_layer, layer_names) = {
            let enable_validation_layer = if with_validation_layer {
                if Self::check_validation_layer_support(&entry)? {
                    log::info!("Validation layer enabled");
                    true
                } else {
                    log::warn!(
                        "No validation layer support; continuing without validation layer..."
                    );
                    false
                }
            } else {
                log::info!("validation layer disabled");
                false
            };

            let layer_names = if enable_validation_layer {
                vec![Self::VALIDATION_LAYER_NAME.as_ptr()]
            } else {
                vec![]
            };

            (enable_validation_layer, layer_names)
        };

        let supported_debug_extensions =
            Self::filter_supported_exts(entry, None, &Self::DEBUG_EXTENSIONS)?;
        let enable_layer_settings = enable_validation_layer
            && Self::filter_supported_exts(
                entry,
                Some(Self::VALIDATION_LAYER_NAME),
                &[ext::layer_settings::NAME],
            )?
            .len()
                > 0;
        let enable_debug_utils =
            enable_validation_layer && supported_debug_extensions.contains(&ext::debug_utils::NAME);

        let instance_extensions = {
            if enable_validation_layer {
                instance_exts
                    .iter()
                    .copied()
                    .chain(supported_debug_extensions.iter().map(|arr| arr.as_ptr()))
                    .collect::<Vec<*const i8>>()
            } else {
                instance_exts.iter().copied().collect::<Vec<_>>()
            }
        };

        // use VK_EXT_layer_settings to configure the validation layer
        let validation_setting = |name: &'static CStr, v: bool| {
            const TRUE_BYTES: [u8; 4] = 1_u32.to_le_bytes();
            const FALSE_BYTES: [u8; 4] = 0_u32.to_le_bytes();
            vk::LayerSettingEXT::default()
                .layer_name(Self::VALIDATION_LAYER_NAME)
                .setting_name(name)
                .ty(vk::LayerSettingTypeEXT::BOOL32)
                .values(if v { &TRUE_BYTES } else { &FALSE_BYTES })
        };
        if with_gpuav {
            log::info!("Enabling GPU assisted validation");
        }
        let settings = [
            // Khronos Validation layer recommends not to enable both GPU Assisted Validation (gpuav_enable) and Normal Core Check Validation (validate_core), as it will be very slow.
            // Once all errors in Core Check are solved it recommends to disable validate_core, then only use GPU-AV for best performance.
            validation_setting(c"validate_core", !with_gpuav),
            validation_setting(c"gpuav_enable", with_gpuav), // gpu assisted validation
            validation_setting(c"gpuav_shader_instrumentation", with_gpuav), // instrument shaders to validate descriptors (aka shader instrumentality project)
            validation_setting(c"gpuav_validate_ray_query", false), // ignore gpuav warning that rayQuery feature is not supported
            validation_setting(c"validate_sync", true),
            validation_setting(c"validate_best_practices", true),
        ];
        let mut layer_settings_create_info = if enable_layer_settings {
            Some(vk::LayerSettingsCreateInfoEXT::default().settings(&settings))
        } else {
            None
        };

        let mut debug_messenger_create_info = if enable_debug_utils {
            use vk::DebugUtilsMessageSeverityFlagsEXT as Severity;
            use vk::DebugUtilsMessageTypeFlagsEXT as MsgType;
            Some(
                vk::DebugUtilsMessengerCreateInfoEXT::default()
                    .message_severity(
                        Severity::VERBOSE | Severity::INFO | Severity::WARNING | Severity::ERROR,
                    )
                    .message_type(MsgType::VALIDATION | MsgType::GENERAL | MsgType::PERFORMANCE)
                    .pfn_user_callback(Some(Self::debug_utils_callback)),
            )
        } else {
            None
        };

        let instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&application_info)
            .enabled_layer_names(&layer_names)
            .enabled_extension_names(&instance_extensions);

        let instance_create_info = if enable_layer_settings {
            instance_create_info.push_next(layer_settings_create_info.as_mut().unwrap())
        } else {
            instance_create_info
        };

        let instance_create_info = if enable_debug_utils {
            instance_create_info.push_next(debug_messenger_create_info.as_mut().unwrap())
        } else {
            instance_create_info
        };

        let instance = unsafe { entry.create_instance(&instance_create_info, None) }?;

        let (debug_utils_instance, debug_messenger) = if enable_debug_utils {
            let debug_utils_instance = ext::debug_utils::Instance::new(&entry, &instance);
            let debug_messenger = unsafe {
                debug_utils_instance.create_debug_utils_messenger(
                    debug_messenger_create_info.as_ref().unwrap(),
                    None,
                )
            }?;

            (Some(debug_utils_instance), Some(debug_messenger))
        } else {
            (None, None)
        };

        Ok(Self {
            instance,
            debug_utils_instance,
            debug_messenger,
        })
    }
    pub fn inner(&self) -> &ash::Instance {
        &self.instance
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            if let Some(debug_messenger) = self.debug_messenger {
                self.debug_utils_instance
                    .as_ref()
                    .unwrap()
                    .destroy_debug_utils_messenger(debug_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
