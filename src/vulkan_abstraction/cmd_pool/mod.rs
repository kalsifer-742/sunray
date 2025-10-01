pub mod cmd_buffer;

pub use cmd_buffer::*;

use crate::error::*;
use crate::vulkan_abstraction;

use ash::vk;
use std::ops::Deref;
use std::rc::Rc;

pub struct CmdPool {
    cmd_pool: vk::CommandPool,
    device: Rc<vulkan_abstraction::Device>,
}

impl CmdPool {
    pub fn new(
        device: Rc<vulkan_abstraction::Device>,
        flags: vk::CommandPoolCreateFlags,
    ) -> SrResult<Self> {
        let info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.queue_family_index())
            .flags(flags);

        let cmd_pool = unsafe { device.inner().create_command_pool(&info, None) }?;

        Ok(Self { cmd_pool, device })
    }

    pub fn inner(&self) -> vk::CommandPool {
        self.cmd_pool
    }
}
impl Drop for CmdPool {
    fn drop(&mut self) {
        unsafe {
            self.device
                .inner()
                .destroy_command_pool(self.cmd_pool, None)
        };
    }
}
impl Deref for CmdPool {
    type Target = vk::CommandPool;
    fn deref(&self) -> &Self::Target {
        &self.cmd_pool
    }
}
