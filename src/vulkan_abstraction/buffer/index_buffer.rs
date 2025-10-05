use std::{any::TypeId, ops::Deref, rc::Rc};

use ash::vk;

use crate::vulkan_abstraction::buffer::Buffer;
use crate::{error::*, vulkan_abstraction};

pub struct IndexBuffer {
    buffer: Buffer,
    len: usize,
    idx_type: vk::IndexType,
}
impl IndexBuffer {
    //build an index buffer with flags for usage in a blas
    pub fn new_for_blas_from_data<T: 'static + Copy>(core: Rc<vulkan_abstraction::Core>, data: &[T]) -> SrResult<Self> {
        let usage_flags = vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;

        let idx_type = match get_index_type::<T>() {
            Some(idx_type) => idx_type,
            None => {
                return Err(SrError::new_custom(
                    "attempting to construct IndexBuffer from invalid type".to_string(),
                ));
            }
        };

        let buffer = Buffer::new_from_data(
            core,
            data,
            gpu_allocator::MemoryLocation::GpuOnly,
            usage_flags,
            "index buffer for BLAS usage",
        )?;

        Ok(Self {
            buffer,
            len: data.len(),
            idx_type,
        })
    }
    pub fn new_for_blas<T: 'static>(core: Rc<vulkan_abstraction::Core>, len: usize) -> SrResult<Self> {
        let usage_flags = vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;

        let idx_type = match get_index_type::<T>() {
            Some(idx_type) => idx_type,
            None => {
                return Err(SrError::new_custom(
                    "attempting to construct IndexBuffer from invalid type".to_string(),
                ));
            }
        };

        let buffer = Buffer::new::<T>(
            core,
            len,
            gpu_allocator::MemoryLocation::GpuOnly,
            usage_flags,
            "index buffer for BLAS usage",
        )?;

        Ok(Self { buffer, len, idx_type })
    }
    #[allow(dead_code)]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn index_type(&self) -> vk::IndexType {
        self.idx_type
    }
}
impl Deref for IndexBuffer {
    type Target = Buffer;
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

fn get_index_type<T: 'static>() -> Option<vk::IndexType> {
    let idx_type = if TypeId::of::<T>() == TypeId::of::<u32>() {
        vk::IndexType::UINT32
    } else if TypeId::of::<T>() == TypeId::of::<u16>() {
        vk::IndexType::UINT16
    } else if TypeId::of::<T>() == TypeId::of::<u8>() {
        assert_eq!(vk::IndexType::UINT8_KHR, vk::IndexType::UINT8_EXT);
        vk::IndexType::UINT8_KHR
    } else {
        return None;
    };

    Some(idx_type)
}
