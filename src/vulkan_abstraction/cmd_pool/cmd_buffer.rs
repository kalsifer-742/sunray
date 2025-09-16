#![allow(dead_code)]
use ash::vk;
use crate::{error::*, vulkan_abstraction};

// Device::free_command_buffers must be called on vk::CommandBuffer before it is dropped
pub fn new_vec(cmd_pool: &vulkan_abstraction::CmdPool, device: &vulkan_abstraction::Device, len: usize) -> SrResult<Vec<vk::CommandBuffer>> {
    new_vec_impl(cmd_pool, device, vk::CommandBufferLevel::PRIMARY, len)
}
pub fn new_vec_secondary(cmd_pool: &vulkan_abstraction::CmdPool, device: &vulkan_abstraction::Device, len: usize) -> SrResult<Vec<vk::CommandBuffer>> {
    new_vec_impl(cmd_pool, device, vk::CommandBufferLevel::SECONDARY, len)
}
pub fn new(cmd_pool: &vulkan_abstraction::CmdPool, device: &vulkan_abstraction::Device) -> SrResult<vk::CommandBuffer> {
    let v = new_vec_impl(cmd_pool, device, vk::CommandBufferLevel::PRIMARY,1)?;
    v.into_iter().next().ok_or_else(|| SrError::new(None, String::from("Error in CmdBuffer::new")))
}
pub fn new_secondary(cmd_pool: &vulkan_abstraction::CmdPool, device: &vulkan_abstraction::Device) -> SrResult<vk::CommandBuffer> {
    let v = new_vec_impl(cmd_pool, device, vk::CommandBufferLevel::SECONDARY,1)?;
    v.into_iter().next().ok_or_else(|| SrError::new(None, String::from("Error in CmdBuffer::new_secondary")))
}

fn new_vec_impl(cmd_pool: &vulkan_abstraction::CmdPool, device: &vulkan_abstraction::Device, level: vk::CommandBufferLevel, len: usize) -> SrResult<Vec<vk::CommandBuffer>> {
    let info = vk::CommandBufferAllocateInfo::default()
        .command_pool(cmd_pool.inner())
        .level(level)
        .command_buffer_count(len as u32);

    let ret = unsafe { device.inner().allocate_command_buffers(&info) }?;

    Ok(ret)
}
