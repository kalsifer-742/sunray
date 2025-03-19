pub mod cmd_buffer;

use std::error::Error;
use std::ops::Deref;
use std::rc::Rc;
use ash::vk;
use crate::vkal;


pub struct CmdPool {
    cmd_pool: vk::CommandPool,
    device: Rc<vkal::Device>,
    cmd_bufs: Vec<vk::CommandBuffer>,
}
impl CmdPool {
    pub fn new(device: Rc<vkal::Device>, flags: vk::CommandPoolCreateFlags) -> Result<Self, Box<dyn Error>> {
        let info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.get_physical_device_info().best_queue_family_for_graphics as u32)
            .flags(flags);

        let cmd_pool = unsafe { device.create_command_pool(&info, vkal::NO_ALLOCATOR) }?;

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
        // cmd_bufs must be destroyed before cmd_pool
        unsafe { self.device.free_command_buffers(self.cmd_pool, &self.cmd_bufs) };

        unsafe { self.device.destroy_command_pool(self.cmd_pool, vkal::NO_ALLOCATOR) };
    }
}
impl Deref for CmdPool {
    type Target = vk::CommandPool;
    fn deref(&self) -> &Self::Target { &self.cmd_pool }
}