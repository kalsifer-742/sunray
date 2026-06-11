use std::collections::VecDeque;
use std::rc::Rc;

use ash::vk;

use crate::MAX_FRAMES_IN_FLIGHT;
use crate::error::*;
use crate::vulkan_abstraction;
use crate::vulkan_abstraction::RawBuffer;

use super::{ArenaBuffer, Buffer, HostAccessibleBuffer, StagingBuffer};

/// Direct-indexed, host-visible arena buffer.
/// No GPU-only copy — the staging buffer is directly accessible from shaders.
/// Slots are allocated from a free-list and freed with deferred deallocation.
pub struct ArenaHostBuffer<T: Copy> {
    staging: StagingBuffer<T>,
    capacity: vk::DeviceSize,
    free_slots: Vec<usize>,
    pending_free_slots: VecDeque<(u64, usize)>,
    len: usize,
    core: Rc<vulkan_abstraction::Core>,
}

impl<T: Copy> ArenaHostBuffer<T> {
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        capacity: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        let staging = StagingBuffer::new(core.clone(), capacity, usage, name)?;
        let free_slots = (0..capacity as usize).rev().collect();

        Ok(Self {
            staging,
            capacity,
            free_slots,
            pending_free_slots: VecDeque::new(),
            len: 0,
            core,
        })
    }

    /// Allocate a slot and write data. Returns the assigned index.
    pub fn allocate_and_update(&mut self, data: &T) -> SrResult<usize> {
        let slot = self
            .free_slots
            .pop()
            .ok_or_else(|| SrError::new_custom("ArenaHostBuffer out of capacity!".to_string()))?;

        let mapped = self.staging.map_mut()?;
        mapped[slot] = *data;
        self.len += 1;

        Ok(slot)
    }

    /// Update data at an existing slot.
    pub fn update(&mut self, slot: usize, data: &T) -> SrResult<()> {
        let mapped = self.staging.map_mut()?;
        mapped[slot] = *data;
        Ok(())
    }

    /// Free a slot for deferred recycling.
    pub fn free_index(&mut self, slot: usize) {
        let current_frame = *self.core.absolute_frame_count.borrow() as u64;
        self.pending_free_slots.push_back((current_frame, slot));
        self.len -= 1;
    }

    /// Number of currently allocated slots.
    pub fn allocated_len(&self) -> usize {
        self.len
    }
}

impl<T: Copy> Buffer for ArenaHostBuffer<T> {
    fn inner(&self) -> vk::Buffer {
        self.staging.inner()
    }
    fn usage(&self) -> vk::BufferUsageFlags {
        self.staging.usage()
    }
    fn raw(&self) -> &RawBuffer {
        self.staging.raw()
    }
    fn raw_mut(&mut self) -> &mut RawBuffer {
        self.staging.raw_mut()
    }
    fn byte_size(&self) -> vk::DeviceSize {
        self.staging.byte_size()
    }
    fn is_null(&self) -> bool {
        self.staging.is_null()
    }
    fn get_device_address(&self) -> vk::DeviceAddress {
        self.staging.get_device_address()
    }
    fn new_null(core: Rc<vulkan_abstraction::Core>) -> Self {
        Self {
            staging: StagingBuffer::new_null(core.clone()),
            capacity: 0,
            free_slots: vec![],
            pending_free_slots: VecDeque::new(),
            len: 0,
            core,
        }
    }
}

impl<T: Copy> HostAccessibleBuffer<T> for ArenaHostBuffer<T> {
    fn map_mut(&mut self) -> SrResult<&mut [T]> {
        self.staging.map_mut()
    }

    fn map(&self) -> SrResult<&[T]> {
        self.staging.map()
    }

    fn len(&self) -> usize {
        self.capacity as usize
    }
}

impl<T: Copy> ArenaBuffer for ArenaHostBuffer<T> {
    fn capacity(&self) -> vk::DeviceSize {
        self.capacity
    }

    fn process_pending_frees(&mut self) {
        let current_frame = *self.core.absolute_frame_count.borrow() as u64;

        while let Some(&(frame_freed, slot)) = self.pending_free_slots.front() {
            if current_frame >= frame_freed + MAX_FRAMES_IN_FLIGHT as u64 {
                self.free_slots.push(slot);
                self.pending_free_slots.pop_front();
            } else {
                break;
            }
        }
    }
}
