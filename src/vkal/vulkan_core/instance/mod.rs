use std::error::Error;
use std::ffi::{c_char, CStr, CString};
use std::ops::Deref;
use ash::vk;
use winit::raw_window_handle::{RawDisplayHandle};
use crate::vkal;

//VulKan Abstraction Layer
mod debug_utils;
pub use debug_utils::*;

pub struct InstanceParams<'a> {
    pub app_name: &'a str, // if set to "" (default) uses the crate name
    pub vk_api_version: u32, // see ash::vk::definitions::make_api_version; 1.0.0 by default
    pub enable_debug_utils: bool, // enables validation layer and debug utilities
}

impl<'a> Default for InstanceParams<'a> {
    fn default() -> Self {
        Self {
            app_name: "",
            vk_api_version: vk::make_api_version(0,1,0,0),
            enable_debug_utils: true,
        }
    }
}



// RAII type to handle init and deinit for ash::vk::Instance, other Instance types and vkal::DebugUtils
pub struct Instance {
    instance: ash::Instance,
    surface_instance: ash::khr::surface::Instance,

    debug_utils: Option<vkal::DebugUtils>,
}
impl Instance {
    pub fn new(params: InstanceParams, entry: &ash::Entry, display_handle: RawDisplayHandle) -> Result<Instance, Box<dyn Error>> {
        let app_name =
            if params.app_name.is_empty() { env!("CARGO_PKG_NAME") }
            else { &params.app_name };
        let app_name_cstring = CString::new(app_name)?;
        let app_version = parse_crate_version().unwrap_or(vk::make_api_version(0,0,0,0));

        let application_info = vk::ApplicationInfo::default()
            .application_name(app_name_cstring.as_c_str())
            .application_version(app_version)
            .engine_name(app_name_cstring.as_c_str())
            .engine_version(app_version)
            .api_version(params.vk_api_version);

        // no layers other than (if requested) validation at the moment
        let layer_names_no_validation : &[&CStr] = &[];
        let validation_layer_name = c"VK_LAYER_KHRONOS_validation";

        let mut layers_names_raw: Vec<*const c_char> = layer_names_no_validation
            .iter()
            .map(|name| name.as_ptr())
            .collect();

        if params.enable_debug_utils {
            layers_names_raw.push(validation_layer_name.as_ptr());
        }

        // required extensions for window surface and, if requested, debug_utils
        let mut extension_names =
            ash_window::enumerate_required_extensions(display_handle)
                .unwrap()
                .to_vec();
        if params.enable_debug_utils {
            extension_names.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        let instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&application_info)
            .enabled_layer_names(&layers_names_raw)
            .enabled_extension_names(&extension_names);

        let instance = unsafe { entry.create_instance(&instance_create_info, vkal::NO_ALLOCATOR) }?;

        let surface_instance = ash::khr::surface::Instance::new(&entry, &instance);

        let debug_utils =
            if params.enable_debug_utils { Some(vkal::DebugUtils::new(entry, &instance)?) }
            else { None };

        Ok(Instance { instance, surface_instance, debug_utils })
    }

    pub fn surface_instance(&self) -> &ash::khr::surface::Instance { &self.surface_instance }
}
impl Drop for Instance {
    fn drop(&mut self) {
        // debug utils must be dropped manually BEFORE dropping instance;
        // to preserve the debug functionality it should be dropped as late as possible
        drop(self.debug_utils.take());

        unsafe { self.instance.destroy_instance(vkal::NO_ALLOCATOR) }
    }
}
impl Deref for Instance {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target { &self.instance }
}
fn parse_crate_version() -> Option<u32> {
    let major_version = u32::from_str_radix(env!("CARGO_PKG_VERSION_MAJOR"), 10).ok()?;
    let minor_version = u32::from_str_radix(env!("CARGO_PKG_VERSION_MINOR"), 10).ok()?;
    let patch_version = u32::from_str_radix(env!("CARGO_PKG_VERSION_PATCH"), 10).ok()?;

    Some(vk::make_api_version(0, major_version, minor_version, patch_version))
}
