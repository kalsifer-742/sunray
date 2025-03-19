use std::borrow::Cow;
use std::error::Error;
use std::ffi::{c_char, CStr};
use ash::vk;
use crate::vkal;
//DebugUtils is just a RAII type to init and deinit debug functionality

pub struct DebugUtils {
    debug_utils_instance: ash::ext::debug_utils::Instance,
    debug_messenger: vk::DebugUtilsMessengerEXT,
}
impl DebugUtils {
    pub fn new(entry: &ash::Entry, instance: &ash::Instance) -> Result<Self, Box<dyn Error>> {
        let debug_utils_instance = ash::ext::debug_utils::Instance::new(entry, &instance);
        let debug_messenger = Self::create_dbg_messenger(&debug_utils_instance)?;
        Ok(Self{ debug_utils_instance, debug_messenger })
    }
    fn create_dbg_messenger(i: &ash::ext::debug_utils::Instance) -> Result<vk::DebugUtilsMessengerEXT, Box<dyn Error>> {
        let severity =
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                // | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                // | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
            ;

        let msg_type =
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE;

        let dbg_messenger_create_info =
            vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(severity)
                .message_type(msg_type)
                .pfn_user_callback(Some(vulkan_debug_callback));

        Ok(unsafe { i.create_debug_utils_messenger(&dbg_messenger_create_info, vkal::NO_ALLOCATOR)? })
    }
}
impl Drop for DebugUtils {
    fn drop(&mut self) {
        unsafe { self.debug_utils_instance.destroy_debug_utils_messenger(self.debug_messenger, vkal::NO_ALLOCATOR); }
    }
}


unsafe extern "system" fn vulkan_debug_callback(
    msg_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let msg_id_number = callback_data.message_id_number;

    let ptr_to_string = |p : *const c_char| {
        if p.is_null() { Cow::from("") }
        else { CStr::from_ptr(p).to_string_lossy() }
    };

    let msg_id_name = ptr_to_string(callback_data.p_message_id_name);
    let msg = ptr_to_string(callback_data.p_message);

    println!("{msg_severity:?}:\n{msg_type:?} [{msg_id_name} ({msg_id_number})] : {msg}\n",);

    vk::FALSE
}
