use ash::{
    Entry,
    vk::{ApplicationInfo, InstanceCreateInfo, make_api_version},
};
use error::{SrError, SrResult};

mod error;

pub struct Core {
    entry: ash::Entry,
    instance: ash::Instance,
}

impl Core {
    pub fn new() -> SrResult<Self> {
        let entry = Entry::linked();
        let application_info = ApplicationInfo {
            api_version: make_api_version(0, 1, 4, 0), //what is the variant?
            ..Default::default()
        };
        let create_info = InstanceCreateInfo {
            p_application_info: &application_info,
            ..Default::default()
        };

        let instance = unsafe { entry.create_instance(&create_info, None) }
            .map_err(|e| SrError::from_vk_result(e))?;

        Ok(Self { entry, instance })
    }
}
