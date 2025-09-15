use std::{ops::Deref, rc::Rc};

use ash::{vk::{BufferUsageFlags, MemoryAllocateFlags, MemoryPropertyFlags}};

use crate::{error::*, vulkan_abstraction};

pub struct VertexBuffer {
    buffer: vulkan_abstraction::Buffer,
    len: usize,
    stride: usize,
}
impl VertexBuffer {
    //build a vertex buffer with flags for usage in a blas
    pub fn new_for_blas<T>(core: Rc<vulkan_abstraction::Core>, len: usize) -> SrResult<Self> {
        let mem_flags = MemoryPropertyFlags::DEVICE_LOCAL;
        let mem_alloc_flags = MemoryAllocateFlags::DEVICE_ADDRESS;
        let usage_flags = 
            BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::VERTEX_BUFFER
            | BufferUsageFlags::SHADER_DEVICE_ADDRESS | BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;

        Ok(Self {
            buffer: vulkan_abstraction::Buffer::new::<T>(core, len, mem_flags, mem_alloc_flags, usage_flags)?,
            len,
            stride: std::mem::size_of::<T>(),
        })
    }

    #[allow(dead_code)]
    pub fn buffer(&self) -> &vulkan_abstraction::Buffer { &self.buffer }
    pub fn len(&self) -> usize { self.len }
    pub fn stride(&self) -> usize { self.stride }
}
impl Deref for VertexBuffer {
    type Target = vulkan_abstraction::Buffer;
    fn deref(&self) -> &Self::Target { &self.buffer }
}
