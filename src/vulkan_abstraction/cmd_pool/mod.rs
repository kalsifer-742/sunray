#![allow(dead_code)]
pub mod cmd_buffer;

use crate::error::*;

use std::ops::Deref;
use ash::vk;
use ash::Device;


pub struct CmdPool {
    cmd_pool: vk::CommandPool,
    device: Device,
    cmd_bufs: Vec<vk::CommandBuffer>,
}
impl CmdPool {
    pub fn new(device: Device, flags: vk::CommandPoolCreateFlags, queue_family: u32) -> SrResult<Self> {
        let info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(flags);

        let cmd_pool = unsafe { device.create_command_pool(&info, None)  }?;

        Ok(Self { cmd_pool, device, cmd_bufs: vec![] })
    }

    pub fn as_raw(&self) -> vk::CommandPool { self.cmd_pool }

    #[allow(dead_code)]
    pub fn get_buffers_mut(&mut self) -> &mut Vec<vk::CommandBuffer> { &mut self.cmd_bufs }
    #[allow(dead_code)]
    pub fn get_buffers(&self) -> &Vec<vk::CommandBuffer> { &self.cmd_bufs }

    pub fn append_buffers(&mut self, bufs: Vec<vk::CommandBuffer>) {
        let mut bufs = bufs;
        self.cmd_bufs.append(&mut bufs);
    }
}
impl Drop for CmdPool {
    fn drop(&mut self) {
        match unsafe { self.device.device_wait_idle() } {
            // do not panic: drop should not panic, since it is invoked for all objects after a panic; for example if the logical device
            // is lost all CmdPool will be dropped on panic and they will all panic themselves and make the backtrace unreadable
            Err(e) => {
                eprintln!("Device::device_wait_idle (inside CmdPool::drop) returned '{e}'");
                //if device was lost do not attempt to free/destroy objects
                if e == ash::vk::Result::ERROR_DEVICE_LOST {
                    return;
                }
            }
            Ok(()) => {}

        }

        if self.cmd_bufs.len() != 0 {
            // cmd_bufs must be destroyed before cmd_pool
            unsafe { self.device.free_command_buffers(self.cmd_pool, &self.cmd_bufs) };
        }

        unsafe { self.device.destroy_command_pool(self.cmd_pool, None) };
    }
}
impl Deref for CmdPool {
    type Target = vk::CommandPool;
    fn deref(&self) -> &Self::Target { &self.cmd_pool }
}
