use std::rc::Rc;

use ash::vk;

use crate::error::*;
use crate::vulkan_abstraction;

use super::{ArenaRingCore, impl_arena_ring_buffer};

/// Direct-indexed arena buffer (like a `Vec` with stable slot indices).
/// Keeps a ring-buffered staging buffer for per-frame writes and a GPU-only
/// buffer for shader access. Slots are allocated from a free-list and freed
/// with deferred deallocation.
pub struct ArenaGpuBuffer<T: Copy> {
    inner: ArenaRingCore<T>,
}

impl_arena_ring_buffer!(ArenaGpuBuffer, inner);

impl<T: Copy> ArenaGpuBuffer<T> {
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        capacity: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        Ok(Self {
            inner: ArenaRingCore::new(core, capacity, usage, name)?,
        })
    }

    pub(crate) fn new_from_data(
        core: Rc<vulkan_abstraction::Core>,
        data: &[T],
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        Ok(Self {
            inner: ArenaRingCore::new_from_data(core, data, usage, name)?,
        })
    }

    /// Allocates a slot for new data. Returns the assigned index and the
    /// `BufferCopy` region that needs to be submitted on a command buffer.
    pub fn allocate_and_update(&mut self, data: &T) -> SrResult<(usize, vk::BufferCopy)> {
        self.inner.allocate_and_write(data)
    }

    /// Write new data to an existing slot. Returns the `BufferCopy` region
    /// that needs to be submitted on a command buffer.
    pub fn update(&mut self, slot: usize, data: &T) -> SrResult<vk::BufferCopy> {
        let (_, copy) = self.inner.write_to_slot(slot, data)?;
        Ok(copy)
    }

    /// Frees an index so it can be reused by future allocations.
    pub fn free_index(&mut self, index: usize) {
        self.inner.free_slot(index);
    }

    pub fn inner_staging(&self) -> vk::Buffer {
        self.inner.inner_staging()
    }
}
