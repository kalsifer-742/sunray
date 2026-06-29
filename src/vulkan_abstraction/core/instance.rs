use std::{
    collections::HashSet,
    ffi::{CStr, c_void},
};

use ash::vk::TaggedStructure;
use ash::{ext, vk};

use crate::error::SrResult;
use crate::vulkan_abstraction::diagnostics::{DiagnosticTool, DiagnosticsContext};

pub struct Instance {
    instance: ash::Instance,
    debug_utils_instance: Option<ext::debug_utils::Instance>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    diagnostics: DiagnosticsContext,
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

    /// Log every instance layer the loader can *discover*, plus the loader env vars
    /// that decide which implicit layers actually load.
    ///
    /// CAVEAT: `vkEnumerateInstanceLayerProperties` returns *installed* layers, not
    /// the ones active in this process — implicit layers (Nsight Graphics, RenderDoc,
    /// ReShade, Steam/OBS overlays, …) are activated separately at `vkCreateInstance`
    /// from registry/env conditions. So this list looks identical whether or not a
    /// tool is attached and can't, on its own, name who injects the stray
    /// `vkCmdPushDescriptorSetKHR` calls that bind our BDA-only vertex/index buffers
    /// as storage buffers (sunray uses a descriptor heap + BDA and records no
    /// descriptor-set writes). To see the *active* layer chain, run with
    /// `VK_LOADER_DEBUG=layer`; to rule implicit injectors out entirely, run with
    /// `VK_LOADER_LAYERS_DISABLE=~implicit~`. Suspected capture/instrumentation
    /// layers are surfaced as warnings, and any set loader env vars are echoed below.
    fn log_available_layers(entry: &ash::Entry) -> SrResult<()> {
        // Substrings (case-insensitive) of layer names that record/instrument the
        // command stream and are a plausible source of injected GPU work.
        const CAPTURE_LAYER_HINTS: [&str; 7] = [
            "nomad",
            "gpu_trace",
            "renderdoc",
            "gfxreconstruct",
            "fossilize",
            "obs",
            "reshade",
        ];

        let layers_props = unsafe { entry.enumerate_instance_layer_properties() }?;
        log::info!(
            "Discoverable Vulkan instance layers ({}) — INSTALLED, not necessarily active; run with VK_LOADER_DEBUG=layer to see the active chain:",
            layers_props.len()
        );
        for p in &layers_props {
            let name = p
                .layer_name_as_c_str()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let desc = p
                .description_as_c_str()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let lname = name.to_ascii_lowercase();
            if CAPTURE_LAYER_HINTS.iter().any(|h| lname.contains(h)) {
                log::warn!("  [capture/instrumentation] {name} — {desc}");
            } else {
                log::info!("  {name} — {desc}");
            }
        }

        // These env vars decide which implicit layers actually load (and let you
        // force/suppress layers). Echoing them is the cheap way to spot, e.g., an
        // Nsight launcher that flipped an activation var for this process.
        const LOADER_ENV_VARS: [&str; 5] = [
            "VK_INSTANCE_LAYERS",
            "VK_LOADER_LAYERS_ENABLE",
            "VK_LOADER_LAYERS_DISABLE",
            "VK_ADD_LAYER_PATH",
            "VK_LOADER_DEBUG",
        ];
        for var in LOADER_ENV_VARS {
            if let Ok(val) = std::env::var(var) {
                log::warn!("  loader env: {var}={val}");
            }
        }
        Ok(())
    }

    fn filter_supported_exts<'a>(
        entry: &ash::Entry,
        layer: Option<&CStr>,
        required_exts: &[&'a CStr],
    ) -> SrResult<Vec<&'a CStr>> {
        match layer {
            Some(layer) => log::info!("Attempting to enable some instance extensions for layer {layer:?}: {required_exts:?}"),
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
        let msg_id_name = unsafe { CStr::from_ptr(callback_data.p_message_id_name) }.to_str().unwrap();
        let msg_text = unsafe { CStr::from_ptr(callback_data.p_message) }.to_str().unwrap();

        match (message_severity, message_type, msg_id_number) {
            (vk::DebugUtilsMessageSeverityFlagsEXT::WARNING, vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION, 0x675dc32e) => {
                return vk::FALSE;
            }
            (vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE, vk::DebugUtilsMessageTypeFlagsEXT::GENERAL, 0x0)
            | (vk::DebugUtilsMessageSeverityFlagsEXT::INFO, vk::DebugUtilsMessageTypeFlagsEXT::GENERAL, 0x0)
                if msg_id_name == "Loader Message" =>
            {
                return vk::FALSE;
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
            log::log!(level, "{message_type:?} {msg_id_number:#010x} - {msg_id_name}: {msg_text}");
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
        diagnostics: DiagnosticTool,
    ) -> SrResult<Self> {
        let application_info = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 4, 0));

        // Dump all visible layers (incl. implicit ones injected by capture tools) so
        // it's obvious at attach time who, if anyone, is instrumenting the command
        // stream. sunray records no descriptor-set writes itself.
        Self::log_available_layers(entry)?;

        // Names the render thread for NVTX and logs which profiling backend is
        // live (the "move vendor logging data" switch is `profiling-nvtx`).
        // No-op without the `profiling` feature.
        crate::profiling::init();

        let (enable_validation_layer, layer_names) = {
            let enable_validation_layer = if with_validation_layer {
                if Self::check_validation_layer_support(entry)? {
                    log::info!("Validation layer enabled");
                    true
                } else {
                    log::warn!("No validation layer support; continuing without validation layer...");
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

        let supported_debug_extensions = Self::filter_supported_exts(entry, None, &Self::DEBUG_EXTENSIONS)?;
        let enable_layer_settings = enable_validation_layer
            && !Self::filter_supported_exts(entry, Some(Self::VALIDATION_LAYER_NAME), &[ext::layer_settings::NAME])?.is_empty();
        // VK_EXT_debug_utils backs both the validation messenger and the
        // `profiling` feature's GPU-timeline pass labels, so enable it whenever
        // either wants it and the loader supports it.
        let want_debug_utils = enable_validation_layer || cfg!(feature = "profiling");
        let enable_debug_utils = want_debug_utils && supported_debug_extensions.contains(&ext::debug_utils::NAME);

        let instance_extensions = {
            let mut exts = instance_exts.to_vec();
            if enable_debug_utils {
                // `supported_debug_extensions` is the loader-supported subset of
                // `DEBUG_EXTENSIONS` (currently just VK_EXT_debug_utils).
                exts.extend(supported_debug_extensions.iter().map(|arr| arr.as_ptr()));
            }
            exts
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
                    .message_severity(Severity::VERBOSE | Severity::INFO | Severity::WARNING | Severity::ERROR)
                    .message_type(MsgType::VALIDATION | MsgType::GENERAL | MsgType::PERFORMANCE)
                    .pfn_user_callback(Some(Self::debug_utils_callback)),
            )
        } else {
            None
        };

        // Boot the diagnostic backend (e.g. Aftermath) *before* vkCreateInstance so any
        // device loss during initialization still produces a crash dump.
        let diagnostics_ctx = DiagnosticsContext::new(diagnostics);

        let instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&application_info)
            .enabled_layer_names(&layer_names)
            .enabled_extension_names(&instance_extensions);

        let instance_create_info = if enable_layer_settings {
            instance_create_info.push(layer_settings_create_info.as_mut().unwrap())
        } else {
            instance_create_info
        };

        let instance_create_info = if enable_debug_utils {
            instance_create_info.push(debug_messenger_create_info.as_mut().unwrap())
        } else {
            instance_create_info
        };

        // Layers sunray explicitly requests — anything active beyond these (see the
        // available-layers dump above) is an implicit layer the loader injected.
        log::info!(
            "Explicitly enabling {} instance layer(s){}",
            layer_names.len(),
            if enable_validation_layer {
                format!(" ({})", Self::VALIDATION_LAYER_NAME.to_string_lossy())
            } else {
                String::new()
            }
        );

        let instance = unsafe { entry.create_instance(&instance_create_info, None) }?;

        let (debug_utils_instance, debug_messenger) = if enable_debug_utils {
            let debug_utils_instance = ext::debug_utils::Instance::load(entry, &instance);
            let debug_messenger = unsafe {
                debug_utils_instance.create_debug_utils_messenger(debug_messenger_create_info.as_ref().unwrap(), None)
            }?;

            (Some(debug_utils_instance), Some(debug_messenger))
        } else {
            (None, None)
        };

        Ok(Self {
            instance,
            debug_utils_instance,
            debug_messenger,
            diagnostics: diagnostics_ctx,
        })
    }
    pub fn inner(&self) -> &ash::Instance {
        &self.instance
    }
    /// Whether `VK_EXT_debug_utils` was enabled (the messenger is only created
    /// when it is). Used to gate GPU-timeline labels in
    /// [`crate::profiling::GpuProfiler`].
    pub fn debug_utils_enabled(&self) -> bool {
        self.debug_utils_instance.is_some()
    }
    pub fn diagnostics(&self) -> &DiagnosticsContext {
        &self.diagnostics
    }
    pub fn diagnostics_mut(&mut self) -> &mut DiagnosticsContext {
        &mut self.diagnostics
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
