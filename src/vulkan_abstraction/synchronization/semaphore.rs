use std::rc::Rc;

use ash::vk;

use crate::{error::SrResult, vulkan_abstraction};


pub struct Semaphore {
    core: Rc<vulkan_abstraction::Core>,
    handle: vk::Semaphore,
}
impl Semaphore {
    pub fn new(core: Rc<vulkan_abstraction::Core>) -> SrResult<Self> {
        let handle = unsafe { core.device().inner().create_semaphore(
            &vk::SemaphoreCreateInfo::default()
                // there are no fields in info besides flags and flags has (currently) no valid values besides empty
                .flags(vk::SemaphoreCreateFlags::empty()),
            None
        ) }?;

        Ok(Self {
            core,
            handle,
        })
    }

    pub fn inner(&self) -> vk::Semaphore { self.handle }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe {
            self.core.device().inner().destroy_semaphore(self.handle, None);
        }
    }
}
