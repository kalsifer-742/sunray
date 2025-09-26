use std::rc::Rc;

use ash::vk;
use crate::{error::*, vulkan_abstraction};

// Device::free_command_buffers must be called on vk::CommandBuffer before it is dropped
pub fn new_command_buffer_vec(cmd_pool: &vulkan_abstraction::CmdPool, device: &ash::Device, len: usize) -> SrResult<Vec<vk::CommandBuffer>> {
    new_command_buffer_vec_impl(cmd_pool, device, vk::CommandBufferLevel::PRIMARY, len)
}
pub fn new_command_buffer_vec_secondary(cmd_pool: &vulkan_abstraction::CmdPool, device: &ash::Device, len: usize) -> SrResult<Vec<vk::CommandBuffer>> {
    new_command_buffer_vec_impl(cmd_pool, device, vk::CommandBufferLevel::SECONDARY, len)
}
pub fn new_command_buffer(cmd_pool: &vulkan_abstraction::CmdPool, device: &ash::Device) -> SrResult<vk::CommandBuffer> {
    let v = new_command_buffer_vec_impl(cmd_pool, device, vk::CommandBufferLevel::PRIMARY,1)?;
    v.into_iter().next().ok_or_else(|| SrError::new(None, String::from("Error in new_command_buffer")))
}
pub fn new_command_buffer_secondary(cmd_pool: &vulkan_abstraction::CmdPool, device: &ash::Device) -> SrResult<vk::CommandBuffer> {
    let v = new_command_buffer_vec_impl(cmd_pool, device, vk::CommandBufferLevel::SECONDARY,1)?;
    v.into_iter().next().ok_or_else(|| SrError::new(None, String::from("Error in new_command_buffer_secondary")))
}

fn new_command_buffer_vec_impl(cmd_pool: &vulkan_abstraction::CmdPool, device: &ash::Device, level: vk::CommandBufferLevel, len: usize) -> SrResult<Vec<vk::CommandBuffer>> {
    let info = vk::CommandBufferAllocateInfo::default()
        .command_pool(cmd_pool.inner())
        .level(level)
        .command_buffer_count(len as u32);

    let ret = unsafe { device.allocate_command_buffers(&info) }?;

    Ok(ret)
}


// this is for now assumed to be a non-ONE_TIME_SUBMIT command buffer
pub struct CmdBuffer {
    handle: vk::CommandBuffer,
    fence: vulkan_abstraction::Fence,
    core: Rc<vulkan_abstraction::Core>,
}

impl CmdBuffer {
    pub fn new(core: Rc<vulkan_abstraction::Core>) -> SrResult<Self> {
        let handle = vulkan_abstraction::cmd_buffer::new_command_buffer(core.cmd_pool(), core.device().inner())?;
        let fence = vulkan_abstraction::Fence::new_signaled(Rc::clone(core.device()))?;
        Ok(Self { core, handle, fence })
    }
    pub fn inner(&self) -> vk::CommandBuffer { self.handle }

    pub fn fence(&self) -> &vulkan_abstraction::Fence { &self.fence }
    pub fn fence_mut(&mut self) -> &mut vulkan_abstraction::Fence { &mut self.fence }
}

impl Drop for CmdBuffer {
    fn drop(&mut self) {
        unsafe {
            match self.fence.wait() {
                Ok(()) => {}
                Err(e) => log::warn!("device.wait_for_fences returned {e} inside CommandBuffer::drop"),
            };
            self.core.device().inner().free_command_buffers(self.core.cmd_pool().inner(), &[self.handle]);
        }
    }
}
