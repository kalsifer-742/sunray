use ash::vk;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum HeapKind {
    Resource,
    Sampler,
}

/// Index of a single descriptor inside a heap. Not RAII — the owning resource
/// must call [`super::DescriptorHeap::free`] in its Drop impl.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DescriptorSlot {
    pub kind: HeapKind,
    pub index: u32,
}

impl DescriptorSlot {
    /// Index as it would appear in `ResourceDescriptorHeap[i]` / `SamplerDescriptorHeap[i]` in the shader.
    pub fn shader_index(self) -> u32 {
        self.index
    }
}

/// Bump + free-list allocator over a contiguous range of indices.
pub(crate) struct SlotAllocator {
    capacity: u32,
    high_water: u32,
    free_list: Vec<u32>,
}

impl SlotAllocator {
    pub fn new(capacity: u32) -> Self {
        Self {
            capacity,
            high_water: 0,
            free_list: Vec::new(),
        }
    }

    pub fn alloc(&mut self) -> Option<u32> {
        if let Some(i) = self.free_list.pop() {
            return Some(i);
        }
        if self.high_water >= self.capacity {
            return None;
        }
        let i = self.high_water;
        self.high_water += 1;
        Some(i)
    }

    pub fn free(&mut self, index: u32) {
        debug_assert!(index < self.high_water);
        self.free_list.push(index);
    }
}

/// What kind of resource descriptor a slot holds. Used to pick the right
/// stride bucket inside the resource heap.
#[derive(Copy, Clone, Debug)]
pub enum ResourceDescriptorKind {
    SampledImage,
    StorageImage,
    UniformBuffer,
    StorageBuffer,
    AccelerationStructure,
}

impl ResourceDescriptorKind {
    pub fn descriptor_type(self) -> vk::DescriptorType {
        match self {
            Self::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
            Self::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
            Self::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
            Self::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
            Self::AccelerationStructure => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
        }
    }
}
