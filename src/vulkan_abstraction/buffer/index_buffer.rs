use std::{any::TypeId, ops::Deref};

use ash::{vk::{BufferUsageFlags, IndexType, MemoryAllocateFlags, MemoryPropertyFlags, PhysicalDeviceMemoryProperties}, Device};

use crate::error::*;
use crate::vulkan_abstraction::buffer::Buffer;

pub struct IndexBuffer {
    buffer: Buffer,
    len: usize,
    idx_type: IndexType,
}
impl IndexBuffer {
    //build an index buffer with flags for usage in a blas
    pub fn new_for_blas<T : 'static>(device: Device, len: usize, mem_props : &PhysicalDeviceMemoryProperties) -> SrResult<Self> {
        let mem_flags = MemoryPropertyFlags::DEVICE_LOCAL;
        let alloc_flags = MemoryAllocateFlags::DEVICE_ADDRESS;
        let usage_flags = 
            BufferUsageFlags::TRANSFER_DST 
            | BufferUsageFlags::INDEX_BUFFER
            | BufferUsageFlags::SHADER_DEVICE_ADDRESS 
            | BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
        
        let idx_type = match get_index_type::<T>() {
            Some(idx_type) => idx_type,
            None => {
                return Err(SrError::new(None, String::from("attempting to construct IndexBuffer from invalid type")))
            },
        };

        let buffer = Buffer::new::<T>(device, len, mem_flags, alloc_flags, usage_flags, mem_props)?;

        Ok(Self { buffer, len, idx_type })
    }
    #[allow(dead_code)]
    pub fn buffer(&self) -> &Buffer { &self.buffer }
    pub fn len(&self) -> usize { self.len }
    pub fn index_type(&self) -> IndexType { self.idx_type }

}
impl Deref for IndexBuffer {
    type Target = Buffer;
    fn deref(&self) -> &Self::Target { &self.buffer }
}

fn get_index_type<T: 'static>() -> Option<IndexType> {
    let idx_type = if TypeId::of::<T>() == TypeId::of::<u32>() {
        IndexType::UINT32
    } else if TypeId::of::<T>() == TypeId::of::<u16>() {
        IndexType::UINT16
    } else if TypeId::of::<T>() == TypeId::of::<u8>() {
        assert_eq!(IndexType::UINT8_KHR, IndexType::UINT8_EXT);
        IndexType::UINT8_KHR 
    } else {
        return None;
    };

    Some(idx_type)
}