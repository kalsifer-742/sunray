#![allow(dead_code)]

use ash::vk;
use ash::vk::CommandBufferAllocateInfo;
use crate::vkal;

// Device::free_command_buffers must be called on vk::CommandBuffer before it is dropped
pub fn new_vec(cmd_pool: vk::CommandPool, device: &vkal::Device, len: usize) -> vkal::Result<Vec<vk::CommandBuffer>> {
    new_vec_impl(cmd_pool, device, vk::CommandBufferLevel::PRIMARY, len)
}
pub fn new_vec_secondary(cmd_pool: vk::CommandPool, device: &vkal::Device, len: usize) -> vkal::Result<Vec<vk::CommandBuffer>> {
    new_vec_impl(cmd_pool, device, vk::CommandBufferLevel::SECONDARY, len)
}
pub fn new(cmd_pool: vk::CommandPool, device: &vkal::Device) -> vkal::Result<vk::CommandBuffer> {
    let v = new_vec_impl(cmd_pool, device, vk::CommandBufferLevel::PRIMARY,1)?;
    Ok(v.into_iter().next().ok_or("Error in CmdBuffer::new")?)
}
pub fn new_secondary(cmd_pool: vk::CommandPool, device: &vkal::Device) -> vkal::Result<vk::CommandBuffer> {
    let v = new_vec_impl(cmd_pool, device, vk::CommandBufferLevel::SECONDARY,1)?;
    Ok(v.into_iter().next().ok_or("Error in CmdBuffer::new")?)
}

fn new_vec_impl(cmd_pool: vk::CommandPool, device: &vkal::Device, level: vk::CommandBufferLevel, len: usize) -> vkal::Result<Vec<vk::CommandBuffer>> {
    let info = CommandBufferAllocateInfo::default()
        .command_pool(cmd_pool)
        .level(level)
        .command_buffer_count(len as u32);

    let ret = unsafe { device.allocate_command_buffers(&info) }?;

    Ok(ret)
}