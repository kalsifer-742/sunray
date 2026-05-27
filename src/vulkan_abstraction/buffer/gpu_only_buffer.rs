use crate::error::SrResult;
use crate::vulkan_abstraction;
use crate::vulkan_abstraction::Buffer;
use crate::vulkan_abstraction::{RawBuffer, StagingBuffer};
use ash::vk;
use std::rc::Rc;

pub struct GpuOnlyBuffer {
    raw: RawBuffer,
}
crate::impl_buffer_trait!(GpuOnlyBuffer);

impl GpuOnlyBuffer {
    pub fn new<T>(
        core: Rc<vulkan_abstraction::Core>,
        len: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        let byte_size = len * std::mem::size_of::<T>() as vk::DeviceSize;
        let raw = RawBuffer::new_aligned(
            core,
            byte_size,
            1,
            gpu_allocator::MemoryLocation::GpuOnly,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            name,
        )?;
        // log::debug!("New Gpu Buffer with these usage flags {usage:?}");
        Ok(Self { raw })
    }

    pub fn new_aligned<T>(
        core: Rc<vulkan_abstraction::Core>,
        len: vk::DeviceSize,
        alignment: u64,
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        let byte_size = len * std::mem::size_of::<T>() as vk::DeviceSize;
        let raw = RawBuffer::new_aligned(
            core,
            byte_size,
            alignment,
            gpu_allocator::MemoryLocation::GpuOnly,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            name,
        )?;
        Ok(Self { raw })
    }

    pub fn len<T>(&self) -> usize {
        (self.raw.byte_size as usize) / std::mem::size_of::<T>()
    }

    pub fn new_from_data<T>(
        core: Rc<vulkan_abstraction::Core>,
        data: &[T],
        buffer_usage_flags: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self>
    where
        T: Copy,
    {
        if data.is_empty() {
            return Ok(Self::new_null(core));
        }

        let staging_buffer = StagingBuffer::new_temp_from_data(Rc::clone(&core), data)?;
        let gpu_buffer =
            staging_buffer.new_cloned_to_gpu_only_buffer(buffer_usage_flags | vk::BufferUsageFlags::TRANSFER_DST, name)?;

        Ok(gpu_buffer)
    }
}
