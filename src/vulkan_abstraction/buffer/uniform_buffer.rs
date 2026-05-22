use crate::error::SrResult;
use crate::vulkan_abstraction::Buffer;
use crate::vulkan_abstraction::{HostAccessibleBuffer, RawBuffer};
use crate::{impl_buffer_trait, vulkan_abstraction};
use ash::vk;
use std::marker::PhantomData;
use std::rc::Rc;

pub struct UniformBuffer<T> {
    raw: RawBuffer,
    _marker: PhantomData<T>,
}
impl_buffer_trait!(UniformBuffer<T>);

impl<T> UniformBuffer<T> {
    pub fn new(core: Rc<vulkan_abstraction::Core>, len: vk::DeviceSize) -> SrResult<Self> {
        let byte_size = len * std::mem::size_of::<T>() as vk::DeviceSize;
        let raw = RawBuffer::new_aligned(
            core,
            byte_size,
            1,
            gpu_allocator::MemoryLocation::CpuToGpu,
            // STORAGE_BUFFER too so the heap path can expose this buffer via
            // `storage_slot()` — the Slang RT shaders read matrices through a
            // `StructuredBuffer<Matrices>` because `DescriptorHandle<ConstantBuffer<T>>`
            // doesn't forward field access on the Slang version we use.
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
            "uniform buffer",
        )?;
        Ok(Self {
            raw,
            _marker: PhantomData,
        })
    }
}

impl<T> HostAccessibleBuffer<T> for UniformBuffer<T> {
    fn map_mut(&mut self) -> SrResult<&mut [T]> {
        self.raw.map_mut::<T>()
    }

    fn map(&self) -> SrResult<&[T]> {
        self.raw.map::<T>()
    }

    fn len(&self) -> usize {
        (self.raw.byte_size as usize) / std::mem::size_of::<T>()
    }
}
