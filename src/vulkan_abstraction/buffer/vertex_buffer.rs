use std::ops::Deref;

use ash::{vk::{BufferUsageFlags, MemoryAllocateFlags, MemoryPropertyFlags, PhysicalDeviceMemoryProperties}, Device};

use crate::error::*;
use crate::vulkan_abstraction::buffer::Buffer;

pub struct VertexBuffer {
    buffer: Buffer,
    len: usize,
    stride: usize,
}
impl VertexBuffer {
    //build a vertex buffer with flags for usage in a blas
    pub fn new_for_blas<T>(device: Device, len: usize, mem_props : &PhysicalDeviceMemoryProperties) -> SrResult<Self> {
        let mem_flags = MemoryPropertyFlags::DEVICE_LOCAL;
        let mem_alloc_flags = MemoryAllocateFlags::DEVICE_ADDRESS;
        let usage_flags = 
            BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::VERTEX_BUFFER
            | BufferUsageFlags::SHADER_DEVICE_ADDRESS | BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;

        Ok(Self {
            buffer: Buffer::new::<T>(device, len, mem_flags, mem_alloc_flags, usage_flags, mem_props)?,
            len,
            stride: std::mem::size_of::<T>(),
        })
    }

    #[allow(dead_code)]
    pub fn buffer(&self) -> &Buffer { &self.buffer }
    pub fn len(&self) -> usize { self.len }
    pub fn stride(&self) -> usize { self.stride }
}
impl Deref for VertexBuffer {
    type Target = Buffer;
    fn deref(&self) -> &Self::Target { &self.buffer }
}
