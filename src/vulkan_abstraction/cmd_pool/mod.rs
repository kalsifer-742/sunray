#![allow(dead_code)]
pub mod cmd_buffer;

use crate::error::*;
use crate::vulkan_abstraction;

use ash::vk;
use std::ops::Deref;
use std::rc::Rc;

pub struct CmdPool {
    cmd_pool: vk::CommandPool,
    device: Rc<vulkan_abstraction::Device>,
    cmd_buf: vk::CommandBuffer,
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

        let mut ret = Self {
            cmd_pool,
            device,
            cmd_buf: vk::CommandBuffer::default(),
        };

        ret.cmd_buf = cmd_buffer::new(&ret, &ret.device.inner())?;

        Ok(ret)
    }

    pub fn inner(&self) -> vk::CommandPool {
        self.cmd_pool
    }

    pub fn get_buffer(&self) -> vk::CommandBuffer {
        self.cmd_buf
    }
}
impl Drop for CmdPool {
    fn drop(&mut self) {
        match unsafe { self.device.inner().device_wait_idle() } {
            // do not panic: drop should not panic, since it is invoked for all objects after a panic; for example if the logical device
            // is lost all CmdPool will be dropped on panic and they will all panic themselves and make the backtrace unreadable
            Err(e) => {
                log::error!("Device::device_wait_idle (inside CmdPool::drop) returned '{e}'");
                //if device was lost do not attempt to free/destroy objects
                // if e == ash::vk::Result::ERROR_DEVICE_LOST {
                //     return;
                // }
            }
            Ok(()) => {}
        }

        // cmd_buf must be destroyed before cmd_pool
        unsafe {
            self.device
                .inner()
                .free_command_buffers(self.cmd_pool, &[self.cmd_buf])
        };

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
