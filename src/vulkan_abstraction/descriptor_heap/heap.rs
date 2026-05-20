use std::ffi::c_void;
use std::ptr::NonNull;

use ash::vk::TaggedStructure;
use ash::{ext, vk};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use log::info;
use num::integer::lcm;
use crate::error::SrResult;
use crate::vulkan_abstraction::descriptor_heap::slot::{
    DescriptorSlot, HeapKind, PageClass, PagedSlotAllocator, ResourceDescriptorKind, SlotAllocator,
};

/// Default capacities. These are *page* counts for the resource heap, and a slot count
/// for the sampler heap. With the page allocator the per-type slot count is
/// `RESOURCE_PAGES * descriptors_per_page_for_that_type`, so 64 pages × ~64 descriptors
/// per page = ~4k slots per type, comfortably inside u32.
pub const DEFAULT_RESOURCE_PAGES: u32 = 64;
pub const DEFAULT_SAMPLER_CAPACITY: u32 = 2_048;

/// Target byte size of a resource heap page. The actual page size is rounded up to a
/// multiple of `lcm(image_descriptor_size, buffer_descriptor_size, max_alignment)` so
/// every type that lives in a page packs an integer number of slots.
const TARGET_PAGE_SIZE_BYTES: u64 = 4_096;
#[derive(Debug)]
struct ResourceSubHeap {
    buffer: vk::Buffer,
    allocation: Allocation,
    device_address: vk::DeviceAddress,
    mapped: NonNull<u8>,
    /// Total heap byte size (app area + driver-reserved tail).
    byte_size: u64,
    /// Bytes the application owns; descriptors live in `[0, app_byte_size)`. The driver
    /// reserved range sits in `[app_byte_size, byte_size)` and must not be touched.
    app_byte_size: u64,

    image_descriptor_size: u64,
    buffer_descriptor_size: u64,
    page_size_bytes: u64,
    image_per_page: u32,
    buffer_per_page: u32,

    alloc: PagedSlotAllocator,
}
#[derive(Debug)]
struct SamplerSubHeap {
    buffer: vk::Buffer,
    allocation: Allocation,
    device_address: vk::DeviceAddress,
    ///non nullable pointer to a memory region in bytes for better pointer arithmetics returned by mapping cpu mem to gpu
    mapped: NonNull<u8>,
    /// Total heap byte size (app area + driver-reserved tail).
    byte_size: u64,
    /// Bytes the application owns; samplers live in `[0, app_byte_size)`.
    app_byte_size: u64,
    descriptor_size: u64,
    stride: u64,
    alloc: SlotAllocator,
}

unsafe impl Send for ResourceSubHeap {}
unsafe impl Send for SamplerSubHeap {}

pub struct DescriptorHeap {
    resource: ResourceSubHeap,
    sampler: SamplerSubHeap,
    /// Device-reported minimum reserved range that must be passed in `BindHeapInfoEXT`.
    /// We currently set the reserved range = the full heap (offset 0, size = heap size),
    /// so these are kept around mainly for the assertion at bind time.
    min_resource_reserved: u64,
    min_sampler_reserved: u64,
    ext: ext::descriptor_heap::Device,
    device: ash::Device,
}

impl DescriptorHeap { //TODO find out if there is a reserved space for the driver at the start
    pub fn new(
        device: &ash::Device,
        ext: &ext::descriptor_heap::Device,
        allocator: &mut Allocator,
        props: &vk::PhysicalDeviceDescriptorHeapPropertiesEXT,
        resource_pages: u32,
        sampler_capacity: u32,
    ) -> SrResult<Self> {
        let image_size = props.image_descriptor_size.max(1);
        let buffer_size = props.buffer_descriptor_size.max(1);
        let sampler_size = props.sampler_descriptor_size.max(1);

        let max_alignment = props
            .image_descriptor_alignment
            .max(props.buffer_descriptor_alignment)
            .max(1);

        // A page must hold an integer number of descriptors for *both* image and buffer
        // types — otherwise per-type shader indexing (`byte_offset / type_size`) would
        // land mid-descriptor across a page boundary. lcm of the two type sizes (also
        // honoring max alignment) gives the smallest page granule that works.
        let page_unit = lcm(lcm(image_size, buffer_size), max_alignment);
        let page_size_bytes = align_up(TARGET_PAGE_SIZE_BYTES.max(page_unit), page_unit);

        let image_per_page = (page_size_bytes / image_size) as u32;
        let buffer_per_page = (page_size_bytes / buffer_size) as u32;


        // VK_EXT_descriptor_heap: `BindHeapInfoEXT::reservedRange*` marks a sub-range
        // the *implementation* uses for its own internal descriptors (embedded samplers,
        // fixed ops, etc.); the application must NOT write descriptors there. We park
        // that reserved range at the tail of each heap so the app area stays at
        // `[0, app_byte_size)` and shader byte-offset addressing (`index * descriptor_size`)
        // lines up with our `alloc_resource_slot` indices unchanged. The tail still has
        // to be backed by real memory, so we size the buffer = app area + reserved range.
        let min_resource_reserved = props.min_resource_heap_reserved_range.max(1);
        let min_sampler_reserved = props.min_sampler_heap_reserved_range.max(1);

        let app_resource_byte_size = (resource_pages as u64) * page_size_bytes;
        let resource_byte_size = app_resource_byte_size + min_resource_reserved;
        let sampler_stride = align_up(sampler_size, props.sampler_descriptor_alignment.max(1));
        let app_sampler_byte_size = (sampler_capacity as u64) * sampler_stride;
        let sampler_byte_size = app_sampler_byte_size + min_sampler_reserved;

        let resource = ResourceSubHeap::new(
            device,
            allocator,
            resource_byte_size,
            app_resource_byte_size,
            image_size,
            buffer_size,
            page_size_bytes,
            image_per_page,
            buffer_per_page,
            resource_pages,
            "sunray_resource_descriptor_heap",
        )?;
        let sampler = SamplerSubHeap::new(
            device,
            allocator,
            sampler_byte_size,
            app_sampler_byte_size,
            sampler_size,
            sampler_stride,
            sampler_capacity,
            "sunray_sampler_descriptor_heap",
        )?;
        info!("sampler heap : {sampler:?}, resource heap :  {resource:?}" );
        Ok(Self {
            resource,
            sampler,
            min_resource_reserved,
            min_sampler_reserved,
            ext: ext.clone(),
            device: device.clone(),
        })
    }

    /// Free the heap's GPU allocations. Must be called before the gpu-allocator is dropped,
    /// otherwise the allocations leak (and gpu-allocator will warn). After this call the
    /// heap is in an invalid state and must not be used.
    pub fn shutdown(&mut self, allocator: &mut Allocator) {
        unsafe {
            if self.resource.buffer != vk::Buffer::null() {
                self.device.destroy_buffer(self.resource.buffer, None);
                self.resource.buffer = vk::Buffer::null();
            }
            if self.sampler.buffer != vk::Buffer::null() {
                self.device.destroy_buffer(self.sampler.buffer, None);
                self.sampler.buffer = vk::Buffer::null();
            }
        }
        let res_alloc = std::mem::take(&mut self.resource.allocation);
        let samp_alloc = std::mem::take(&mut self.sampler.allocation);
        let _ = allocator.free(res_alloc);
        let _ = allocator.free(samp_alloc);
    }

    pub fn alloc_resource_slot(&mut self, kind: ResourceDescriptorKind) -> DescriptorSlot {
        let class = kind.page_class();
        let (page_idx, slot_in_page) = self
            .resource
            .alloc
            .alloc(class)
            .expect("descriptor resource heap exhausted");
        let per_page = self.resource.alloc.per_page(class);
        // shader_index = byte_offset / type_size = page_idx * per_page + slot_in_page
        let index = (page_idx) * (per_page) + (slot_in_page);
        DescriptorSlot {
            kind: HeapKind::Resource,
            index,
            class,
        }
    }

    pub fn alloc_sampler_slot(&mut self) -> DescriptorSlot {
        let index = self
            .sampler
            .alloc
            .alloc()
            .expect("descriptor sampler heap exhausted");
        DescriptorSlot {
            kind: HeapKind::Sampler,
            index,
            class: PageClass::Buffer, // unused for samplers
        }
    }

    pub fn free(&mut self, slot: DescriptorSlot) {
        match slot.kind {
            HeapKind::Resource => {
                let per_page = self.resource.alloc.per_page(slot.class);
                let page_idx = slot.index / per_page;
                let slot_in_page = slot.index % per_page;
                self.resource.alloc.free(page_idx, slot_in_page);
            }
            HeapKind::Sampler => self.sampler.alloc.free(slot.index),
        }
    }

    pub fn resource_device_address(&self) -> vk::DeviceAddress {
        self.resource.device_address
    }

    pub fn sampler_device_address(&self) -> vk::DeviceAddress {
        self.sampler.device_address
    }

    pub fn resource_size(&self) -> u64 {
        self.resource.byte_size
    }

    pub fn sampler_size(&self) -> u64 {
        self.sampler.byte_size
    }

    /// Write an image descriptor into the resource heap at `slot`. The view-create-info
    /// is used by the driver to construct the on-heap descriptor; it does not need to
    /// outlive this call.
    pub fn write_image(
        &mut self,
        slot: DescriptorSlot,
        view_info: &vk::ImageViewCreateInfo<'_>,
        layout: vk::ImageLayout,
        kind: ResourceDescriptorKind,
    ) -> SrResult<()> {
        debug_assert_eq!(slot.kind, HeapKind::Resource);
        debug_assert_eq!(slot.class, PageClass::Image);
        let image_info = vk::ImageDescriptorInfoEXT::default()
            .view(view_info)
            .layout(layout);
        let resource_info = vk::ResourceDescriptorInfoEXT::default()
            .ty(kind.descriptor_type())
            .data(vk::ResourceDescriptorDataEXT { p_image: &image_info });
        self.write_resource(slot, resource_info, self.resource.image_descriptor_size)
    }

    pub fn write_buffer(
        &mut self,
        slot: DescriptorSlot,
        address: vk::DeviceAddress,
        size: vk::DeviceSize,
        kind: ResourceDescriptorKind,
    ) -> SrResult<()> {
        debug_assert_eq!(slot.kind, HeapKind::Resource);
        debug_assert_eq!(slot.class, PageClass::Buffer);
        let range = vk::DeviceAddressRangeEXT { address, size };
        let resource_info = vk::ResourceDescriptorInfoEXT::default()
            .ty(kind.descriptor_type())
            .data(vk::ResourceDescriptorDataEXT { p_address_range: &range });
        self.write_resource(slot, resource_info, self.resource.buffer_descriptor_size)
    }

    pub fn write_acceleration_structure(
        &mut self,
        slot: DescriptorSlot,
        address: vk::DeviceAddress,
    ) -> SrResult<()> {
        debug_assert_eq!(slot.kind, HeapKind::Resource);
        debug_assert_eq!(slot.class, PageClass::Buffer);
        let range = vk::DeviceAddressRangeEXT {
            address,
            size: 0, // ignored for AS descriptors
        };
        let resource_info = vk::ResourceDescriptorInfoEXT::default()
            .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .data(vk::ResourceDescriptorDataEXT { p_address_range: &range });
        // AS descriptors are sized like buffer descriptors on this extension.
        self.write_resource(slot, resource_info, self.resource.buffer_descriptor_size)
    }

    pub fn write_sampler(&mut self, slot: DescriptorSlot, info: &vk::SamplerCreateInfo<'_>) -> SrResult<()> {
        debug_assert_eq!(slot.kind, HeapKind::Sampler);
        let dst = self.sampler_dst(slot.index);
        let dst_range = [vk::HostAddressRangeEXT {
            address: dst.as_ptr() as *mut c_void,
            size: self.sampler.descriptor_size as usize,
            _marker: std::marker::PhantomData,
        }];
        unsafe { self.ext.write_sampler_descriptors(std::slice::from_ref(info), &dst_range) }?;
        Ok(())
    }

    /// Bind both heaps to a command buffer. Must be called before any draw/dispatch
    /// that references descriptors via the heap.
    pub fn cmd_bind(&self, cmd_buf: vk::CommandBuffer) {
        // The driver-reserved range sits at the tail of each heap (see `new` for why).
        // App descriptors live in `[0, app_byte_size)`; the driver owns
        // `[app_byte_size, byte_size)`.
        let resource_bind = vk::BindHeapInfoEXT::default()
            .heap_range(vk::DeviceAddressRangeEXT {
                address: self.resource.device_address,
                size: self.resource.byte_size,
            })
            .reserved_range_offset(self.resource.app_byte_size)
            .reserved_range_size(self.min_resource_reserved);
        let sampler_bind = vk::BindHeapInfoEXT::default()
            .heap_range(vk::DeviceAddressRangeEXT {
                address: self.sampler.device_address,
                size: self.sampler.byte_size,
            })
            .reserved_range_offset(self.sampler.app_byte_size)
            .reserved_range_size(self.min_sampler_reserved);
        unsafe {
            self.ext.cmd_bind_resource_heap(cmd_buf, &resource_bind);
            self.ext.cmd_bind_sampler_heap(cmd_buf, &sampler_bind);
        }
    }

    fn write_resource(
        &mut self,
        slot: DescriptorSlot,
        info: vk::ResourceDescriptorInfoEXT<'_>,
        descriptor_size: u64,
    ) -> SrResult<()> {
        let dst = self.resource_dst(slot.index, descriptor_size);
        let dst_range = [vk::HostAddressRangeEXT {
            address: dst.as_ptr() as *mut c_void,
            size: descriptor_size as usize,
            _marker: std::marker::PhantomData,
        }];
        unsafe { self.ext.write_resource_descriptors(std::slice::from_ref(&info), &dst_range) }?;
        Ok(())
    }

    fn resource_dst(&self, shader_index: u32, descriptor_size: u64) -> NonNull<u8> {
        let offset = (shader_index as u64) * descriptor_size;
        debug_assert!(
            offset + descriptor_size <= self.resource.app_byte_size,
            "descriptor write past the app range; would land in driver-reserved memory"
        );
        unsafe { NonNull::new_unchecked(self.resource.mapped.as_ptr().add(offset as usize)) }
    }

    fn sampler_dst(&self, index: u32) -> NonNull<u8> {
        let offset = (index as u64) * self.sampler.stride;
        debug_assert!(
            offset + self.sampler.stride <= self.sampler.app_byte_size,
            "sampler write past the app range; would land in driver-reserved memory"
        );
        unsafe { NonNull::new_unchecked(self.sampler.mapped.as_ptr().add(offset as usize)) }
    }
}

impl Drop for DescriptorHeap {
    fn drop(&mut self) {
        // Resources outlive the heap in practice (Core owns them in field-declaration order),
        // so explicit shutdown is via Core::Drop calling shutdown(). This is a backstop only.
        unsafe {
            if self.resource.buffer != vk::Buffer::null() {
                self.device.destroy_buffer(self.resource.buffer, None);
            }
            if self.sampler.buffer != vk::Buffer::null() {
                self.device.destroy_buffer(self.sampler.buffer, None);
            }
        }
    }
}

impl ResourceSubHeap {
    fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        byte_size: u64,
        app_byte_size: u64,
        image_descriptor_size: u64,
        buffer_descriptor_size: u64,
        page_size_bytes: u64,
        image_per_page: u32,
        buffer_per_page: u32,
        num_pages: u32,
        name: &'static str,
    ) -> SrResult<Self> {
        let (buffer, allocation, device_address, mapped) = create_heap_buffer(device, allocator, byte_size, name)?;
        Ok(Self {
            buffer,
            allocation,
            device_address,
            mapped,
            byte_size,
            app_byte_size,
            image_descriptor_size,
            buffer_descriptor_size,
            page_size_bytes,
            image_per_page,
            buffer_per_page,
            alloc: PagedSlotAllocator::new(num_pages, image_per_page, buffer_per_page),
        })
    }
}

impl SamplerSubHeap {
    fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        byte_size: u64,
        app_byte_size: u64,
        descriptor_size: u64,
        stride: u64,
        capacity: u32,
        name: &'static str,
    ) -> SrResult<Self> {
        let (buffer, allocation, device_address, mapped) = create_heap_buffer(device, allocator, byte_size, name)?;
        Ok(Self {
            buffer,
            allocation,
            device_address,
            mapped,
            byte_size,
            app_byte_size,
            descriptor_size,
            stride,
            alloc: SlotAllocator::new(capacity),
        })
    }
}

fn create_heap_buffer(
    device: &ash::Device,
    allocator: &mut Allocator,
    byte_size: u64,
    name: &'static str,
) -> SrResult<(vk::Buffer, Allocation, vk::DeviceAddress, NonNull<u8>)> {
    let mut usage2 = vk::BufferUsageFlags2CreateInfo::default()
        .usage(vk::BufferUsageFlags2::DESCRIPTOR_HEAP_EXT | vk::BufferUsageFlags2::SHADER_DEVICE_ADDRESS);
    let buf_info = vk::BufferCreateInfo::default()
        .size(byte_size)
        .usage(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push(&mut usage2);

    let buffer = unsafe { device.create_buffer(&buf_info, None) }?;
    let mem_reqs = unsafe { device.get_buffer_memory_requirements(buffer) };

    let mut allocation = allocator.allocate(&AllocationCreateDesc {
        name,
        requirements: mem_reqs,
        location: gpu_allocator::MemoryLocation::CpuToGpu,
        linear: true,
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    })?;

    unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())? };

    let device_address = unsafe {
        device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
    };

    let mapped_slice = allocation
        .mapped_slice_mut()
        .expect("descriptor heap buffer must be host-visible and mapped");
    let mapped = NonNull::new(mapped_slice.as_mut_ptr()).expect("non-null mapped pointer");

    Ok((buffer, allocation, device_address, mapped))
}
///aligns v up to the nearest multiple of a without floating point for faster calc with the use of integer rounding
fn align_up(v: u64, a: u64) -> u64 {
    (v + a - 1) / a * a
}



