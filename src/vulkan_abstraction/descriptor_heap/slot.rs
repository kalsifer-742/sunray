use ash::vk;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum HeapKind {
    Resource,
    Sampler,
}

/// What kind of resource descriptor a slot holds. The page allocator partitions the
/// resource heap by *page class* (image-like vs buffer-like) so that all descriptors
/// in a page share the same byte stride and cannot alias each other in byte space.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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

    /// Page class: image-like descriptors share image-sized pages, everything else
    /// (uniform / storage buffer, AS device-address) shares buffer-sized pages.
    pub fn page_class(self) -> PageClass {
        match self {
            Self::SampledImage | Self::StorageImage => PageClass::Image,
            Self::UniformBuffer | Self::StorageBuffer | Self::AccelerationStructure => PageClass::Buffer,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PageClass {
    Image,
    Buffer,
}

/// Index of a single descriptor inside a heap. Not RAII — the owning resource
/// must call [`super::DescriptorHeap::free`] in its Drop impl.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DescriptorSlot {
    pub kind: HeapKind,
    /// `ResourceDescriptorHeap[index]` / `SamplerDescriptorHeap[index]` in the shader.
    /// For resource slots this is `byte_offset / type_descriptor_size`; the byte offset
    /// is recoverable as `index * type_descriptor_size` and `(page_idx, slot_in_page)`
    /// as `(index / per_page, index % per_page)` using the page class's per-page count.
    pub index: u32,
    /// Resource page-class for free()'s page lookup. Ignored for samplers.
    pub class: PageClass,
}

impl DescriptorSlot {
    pub fn shader_index(self) -> u32 {
        self.index
    }
}

/// Uniform-stride bump + free-list allocator. Used for the sampler heap (single type).
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

/// Page-based allocator for the resource heap. Each page is a fixed byte range that
/// only ever holds one [`PageClass`] for its lifetime — this avoids the cross-type
/// byte-aliasing problem (where the same byte offset maps to different shader indices
/// depending on type) that single-stride heterogeneous allocation would create.
///
/// On allocation we look for an existing page of the requested class with a free slot
/// before claiming a fresh page from the free pool. On free we mark the slot available;
/// pages are not currently returned to the free pool even when fully empty (keeps page
/// classes stable; trivial to add later if fragmentation becomes an issue).
pub(crate) struct PagedSlotAllocator {
    pages: Vec<Option<Page>>,
    free_pages: Vec<u32>,
    image_per_page: u32,
    buffer_per_page: u32,
}

struct Page {
    class: PageClass,
    /// Returned slot indices, intra-page.
    free_slots: Vec<u32>,
    /// Next never-allocated intra-page slot.
    bump: u32,
    capacity: u32,
}

impl PagedSlotAllocator {
    pub fn new(num_pages: u32, image_per_page: u32, buffer_per_page: u32) -> Self {
        Self {
            pages: (0..num_pages).map(|_| None).collect(),
            // Pop from the back, so pages get used 0..n in order — easier to debug.
            free_pages: (0..num_pages).rev().collect(),
            image_per_page,
            buffer_per_page,
        }
    }

    pub fn per_page(&self, class: PageClass) -> u32 {
        match class {
            PageClass::Image => self.image_per_page,
            PageClass::Buffer => self.buffer_per_page,
        }
    }

    /// Allocate a slot for the given class. Returns `(page_idx, slot_in_page)`.
    pub fn alloc(&mut self, class: PageClass) -> Option<(u32, u32)> {
        for (pi, slot) in self.pages.iter_mut().enumerate() {
            if let Some(p) = slot {
                if p.class == class {
                    if let Some(s) = p.alloc_slot() {
                        return Some((pi as u32, s));
                    }
                }
            }
        }
        let pi = self.free_pages.pop()?;
        let cap = match class {
            PageClass::Image => self.image_per_page,
            PageClass::Buffer => self.buffer_per_page,
        };
        let mut page = Page {
            class,
            free_slots: Vec::new(),
            bump: 0,
            capacity: cap,
        };
        let s = page.alloc_slot().expect("freshly-created page must have a free slot");
        self.pages[pi as usize] = Some(page);
        Some((pi, s))
    }

    pub fn free(&mut self, page_idx: u32, slot_in_page: u32) {
        if let Some(p) = self.pages.get_mut(page_idx as usize).and_then(|s| s.as_mut()) {
            debug_assert!(slot_in_page < p.bump);
            p.free_slots.push(slot_in_page);
        }
    }
}

impl Page {
    fn alloc_slot(&mut self) -> Option<u32> {
        if let Some(s) = self.free_slots.pop() {
            return Some(s);
        }
        if self.bump < self.capacity {
            let s = self.bump;
            self.bump += 1;
            return Some(s);
        }
        None
    }
}
