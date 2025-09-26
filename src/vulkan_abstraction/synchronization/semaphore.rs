use std::rc::Rc;

use ash::vk;

use crate::{error::SrResult, vulkan_abstraction};


pub struct Semaphore {
    device: Rc<vulkan_abstraction::Device>,
    handle: vk::Semaphore,
}
impl Semaphore {
    pub fn new(device: Rc<vulkan_abstraction::Device>) -> SrResult<Self> {
        let handle = unsafe { device.inner().create_semaphore(
            &vk::SemaphoreCreateInfo::default()
                // there are no fields in info besides flags and flags has (currently) no valid values besides empty
                .flags(vk::SemaphoreCreateFlags::empty()),
            None
        ) }?;

        Ok(Self {
            device,
            handle,
        })
    }

    pub fn inner(&self) -> vk::Semaphore { self.handle }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe {
            self.device.inner().destroy_semaphore(self.handle, None);
        }
    }
}
