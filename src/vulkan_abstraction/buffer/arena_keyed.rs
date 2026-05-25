use std::collections::HashMap;
use std::rc::Rc;

use ash::vk;

use crate::error::*;
use crate::vulkan_abstraction;
use crate::vulkan_abstraction::{Buffer, HostAccessibleBuffer};

use super::{ArenaRingCore, GpuOnlyBuffer, StagingBuffer, impl_arena_ring_buffer};

/// Implements the common keyed-arena helpers that every `HashMap<u64,usize>`
/// + `ArenaRingCore`-backed struct shares.
macro_rules! impl_keyed_arena_common {
    ($struct_name:ident, $ring_field:ident, $id_map_field:ident) => {
        impl<T: Copy> $struct_name<T> {
            pub fn get_slot(&self, id: u64) -> Option<usize> {
                self.$id_map_field.get(&id).copied()
            }

            pub fn contains(&self, id: u64) -> bool {
                self.$id_map_field.contains_key(&id)
            }

            // pub fn inner_gpu(&self) -> vk::Buffer {
            //     self.$ring_field.inner_gpu()
            // }

            pub fn inner_staging(&self) -> vk::Buffer {
                self.$ring_field.inner_staging()
            }

            /// Resolve-or-allocate: returns the existing slot for `id`, or
            /// allocates a fresh one and records the mapping.
            fn resolve_or_allocate_slot(&mut self, id: u64) -> SrResult<usize> {
                if let Some(&existing) = self.$id_map_field.get(&id) {
                    Ok(existing)
                } else {
                    let slot = self.$ring_field.allocate_slot()?;
                    self.$id_map_field.insert(id, slot);
                    Ok(slot)
                }
            }
        }
    };
}

// ─── 1. ArenaKeyMappedBuffer ────────────────────────────────────────────────

/// Keyed arena buffer (like a `HashMap` with stable indices).
/// The id→slot mapping lives only on CPU.
/// The GPU sees a flat storage buffer indexed by physical slot.
pub struct ArenaKeyMappedBuffer<T: Copy> {
    ring: ArenaRingCore<T>,
    id_map: HashMap<u64, usize>,
}

impl_arena_ring_buffer!(ArenaKeyMappedBuffer, ring, _core => { id_map: HashMap::new() });
impl_keyed_arena_common!(ArenaKeyMappedBuffer, ring, id_map);

impl<T: Copy> ArenaKeyMappedBuffer<T> {
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        capacity: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        Ok(Self {
            ring: ArenaRingCore::new(core, capacity, usage, name)?,
            id_map: HashMap::new(),
        })
    }

    /// Insert or update a value by key. Returns (slot, data_copy).
    pub fn insert(&mut self, id: u64, data: &T) -> SrResult<(usize, vk::BufferCopy)> {
        let slot = self.resolve_or_allocate_slot(id)?;
        self.ring.write_to_slot(slot, data)
    }

    /// Remove a key. The slot is deferred-freed.
    pub fn remove(&mut self, id: u64) {
        if let Some(slot) = self.id_map.remove(&id) {
            self.ring.free_slot(slot);
        }
    }
}

// ─── 2. ArenaGpuKeyMappedBuffer ─────────────────────────────────────────────

/// Keyed arena buffer with a GPU-visible id→slot mapping buffer.
/// `mapping_gpu[slot] = external_id_u32`.
pub struct ArenaGpuKeyMappedBuffer<T: Copy> {
    ring: ArenaRingCore<T>,
    id_map: HashMap<u64, usize>,
    mapping_gpu: GpuOnlyBuffer,
    mapping_staging: StagingBuffer<u32>,
}

impl_arena_ring_buffer!(ArenaGpuKeyMappedBuffer, ring, core => {
    id_map: HashMap::new(),
    mapping_gpu: GpuOnlyBuffer::new_null(core.clone()),
    mapping_staging: StagingBuffer::new_null(core.clone()),
});
impl_keyed_arena_common!(ArenaGpuKeyMappedBuffer, ring, id_map);

impl<T: Copy> ArenaGpuKeyMappedBuffer<T> {
    const EMPTY_SLOT: u32 = u32::MAX;

    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        capacity: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        let mapping_staging = StagingBuffer::new_from_data(
            core.clone(),
            &vec![Self::EMPTY_SLOT; capacity as usize],
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
            "arena keyed mapping staging",
        )?;
        let mapping_gpu =
            mapping_staging.new_cloned_to_gpu_only_buffer(vk::BufferUsageFlags::STORAGE_BUFFER, "arena keyed mapping gpu")?;

        Ok(Self {
            ring: ArenaRingCore::new(core, capacity, usage, name)?,
            id_map: HashMap::new(),
            mapping_gpu,
            mapping_staging,
        })
    }

    /// Insert or update. Returns (slot, data_copy, mapping_copy) — both
    /// BufferCopy regions must be submitted to sync data and mapping to GPU.
    pub fn insert(&mut self, id: u64, data: &T) -> SrResult<(usize, vk::BufferCopy, vk::BufferCopy)> {
        let slot = self.resolve_or_allocate_slot(id)?;
        let (_, data_copy) = self.ring.write_to_slot(slot, data)?;
        let mapping_copy = self.write_mapping_slot(slot, id as u32)?;
        Ok((slot, data_copy, mapping_copy))
    }

    /// Remove a key. Returns the mapping BufferCopy to clear the slot on GPU.
    pub fn remove(&mut self, id: u64) -> Option<vk::BufferCopy> {
        if let Some(slot) = self.id_map.remove(&id) {
            self.ring.free_slot(slot);
            self.write_mapping_slot(slot, Self::EMPTY_SLOT).ok()
        } else {
            None
        }
    }

    fn write_mapping_slot(&mut self, slot: usize, value: u32) -> SrResult<vk::BufferCopy> {
        let mapped = self.mapping_staging.map_mut()?;
        mapped[slot] = value;

        let size = std::mem::size_of::<u32>() as vk::DeviceSize;
        Ok(vk::BufferCopy::default()
            .src_offset((slot as vk::DeviceSize) * size)
            .dst_offset((slot as vk::DeviceSize) * size)
            .size(size))
    }

    pub fn mapping_staging_buffer(&self) -> vk::Buffer {
        self.mapping_staging.inner()
    }

    pub fn mapping_gpu_buffer(&self) -> vk::Buffer {
        self.mapping_gpu.inner()
    }
}

// ─── 3. ArenaKeyMappedHostVisibleBuffer ─────────────────────────────────────

/// Keyed arena buffer with a host-visible mapping buffer (accessible from
/// both CPU and GPU without an extra copy command for the mapping).
pub struct ArenaKeyMappedHostVisibleBuffer<T: Copy> {
    ring: ArenaRingCore<T>,
    id_map: HashMap<u64, usize>,
    mapping_staging: StagingBuffer<u32>,
}

impl_arena_ring_buffer!(ArenaKeyMappedHostVisibleBuffer, ring, core => {
    id_map: HashMap::new(),
    mapping_staging: StagingBuffer::new_null(core.clone()),
});
impl_keyed_arena_common!(ArenaKeyMappedHostVisibleBuffer, ring, id_map);

impl<T: Copy> ArenaKeyMappedHostVisibleBuffer<T> {
    const EMPTY_SLOT: u32 = u32::MAX;

    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        capacity: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        let mapping_staging = StagingBuffer::new(core.clone(), capacity, usage, name)?;

        let mut arena = Self {
            ring: ArenaRingCore::new(core, capacity, usage, name)?,
            id_map: HashMap::new(),
            mapping_staging,
        };

        // Initialize all mapping slots to EMPTY
        let mapped = arena.mapping_staging.map_mut()?;
        mapped.iter_mut().for_each(|v| *v = Self::EMPTY_SLOT);

        Ok(arena)
    }

    /// Insert or update. Returns (slot, data_copy). The mapping is updated
    /// in-place on the host-visible buffer (no copy command needed for mapping).
    pub fn insert(&mut self, id: u64, data: &T) -> SrResult<(usize, vk::BufferCopy)> {
        let slot = self.resolve_or_allocate_slot(id)?;
        let (_, data_copy) = self.ring.write_to_slot(slot, data)?;

        let mapped = self.mapping_staging.map_mut()?;
        mapped[slot] = id as u32;

        Ok((slot, data_copy))
    }

    /// Remove a key. The mapping is cleared in-place on the host-visible buffer.
    pub fn remove(&mut self, id: u64) {
        if let Some(slot) = self.id_map.remove(&id) {
            if let Ok(mapped) = self.mapping_staging.map_mut() {
                mapped[slot] = Self::EMPTY_SLOT;
            }
            self.ring.free_slot(slot);
        }
    }

    pub fn mapping_buffer(&self) -> vk::Buffer {
        self.mapping_staging.inner()
    }
}
