use std::ffi::c_void;
use std::ptr::NonNull;

use ash::vk::TaggedStructure;
use ash::{ext, vk};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

use crate::error::SrResult;
use crate::vulkan_abstraction::descriptor_heap::slot::{
    DescriptorSlot, HeapKind, ResourceDescriptorKind, SlotAllocator,
};

/// Default capacities. Resource heap is large because every Image/Buffer slot lives here;
/// sampler heap is small because most engines have few sampler permutations.
/// Remember internally the SubHeap
pub const DEFAULT_RESOURCE_CAPACITY: u32 = 65_536;
pub const DEFAULT_SAMPLER_CAPACITY: u32 = 1_024;


struct SubHeap {
    buffer: vk::Buffer,
    allocation: Allocation,
    device_address: vk::DeviceAddress,
    /// Persistently-mapped host pointer to the heap memory.
    mapped: NonNull<u8>,
    byte_size: u64,
    /// Bytes between consecutive descriptor slots.
    stride: u64,
    alloc: SlotAllocator,
}

unsafe impl Send for SubHeap {}

pub struct DescriptorHeap {
    resource: SubHeap,
    sampler: SubHeap,
    /// Per-type descriptor sizes (queried from the device). Used as the `size` field of the
    /// HostAddressRangeEXT for each write — the spec requires the range size to be at least
    /// the size of the descriptor type being written, and writing exactly that size keeps us
    /// within the slot the driver promised us.
    image_descriptor_size: usize,
    buffer_descriptor_size: usize,
    sampler_descriptor_size: usize,
    ext: ext::descriptor_heap::Device,
    device: ash::Device,
}

impl DescriptorHeap {
    pub fn new(
        device: &ash::Device,
        ext: &ext::descriptor_heap::Device,
        allocator: &mut Allocator,
        props: &vk::PhysicalDeviceDescriptorHeapPropertiesEXT,
        resource_capacity: u32,
        sampler_capacity: u32,
    ) -> SrResult<Self> {
        let resource_stride = align_up(
            props
                .image_descriptor_size
                .max(props.buffer_descriptor_size)
                .max(1),
            props
                .image_descriptor_alignment
                .max(props.buffer_descriptor_alignment)
                .max(1),
        );
        let sampler_stride = align_up(
            props.sampler_descriptor_size.max(1),
            props.sampler_descriptor_alignment.max(1),
        );

        let resource = SubHeap::new(
            device,
            allocator,
            resource_capacity as u64 * resource_stride,
            resource_stride,
            resource_capacity,
            props.resource_heap_alignment.max(1),
            "sunray_resource_descriptor_heap",
        )?;
        let sampler = SubHeap::new(
            device,
            allocator,
            sampler_capacity as u64 * sampler_stride,
            sampler_stride,
            sampler_capacity,
            props.sampler_heap_alignment.max(1),
            "sunray_sampler_descriptor_heap",
        )?;

        Ok(Self {
            resource,
            sampler,
            image_descriptor_size: props.image_descriptor_size.max(1) as usize,
            buffer_descriptor_size: props.buffer_descriptor_size.max(1) as usize,
            sampler_descriptor_size: props.sampler_descriptor_size.max(1) as usize,
            ext: ext.clone(),
            device: device.clone(),
        })
    }

    /// Free the heap's GPU allocations. Must be called before the gpu-allocator
    /// is dropped, otherwise the allocations leak (and gpu-allocator will warn).
    /// After this call the heap is in an invalid state and must not be used.
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

    pub fn alloc_resource_slot(&mut self, _kind: ResourceDescriptorKind) -> DescriptorSlot {
        let index = self
            .resource
            .alloc
            .alloc()
            .expect("descriptor resource heap exhausted");
        DescriptorSlot {
            kind: HeapKind::Resource,
            index,
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
        }
    }

    pub fn free(&mut self, slot: DescriptorSlot) {
        match slot.kind {
            HeapKind::Resource => self.resource.alloc.free(slot.index),
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
        let image_info = vk::ImageDescriptorInfoEXT::default()
            .view(view_info)
            .layout(layout);
        let resource_info = vk::ResourceDescriptorInfoEXT::default()
            .ty(kind.descriptor_type())
            .data(vk::ResourceDescriptorDataEXT {
                p_image: &image_info,
            });
        self.write_resource(slot, resource_info, self.image_descriptor_size)
    }

    pub fn write_buffer(
        &mut self,
        slot: DescriptorSlot,
        address: vk::DeviceAddress,
        size: vk::DeviceSize,
        kind: ResourceDescriptorKind,
    ) -> SrResult<()> {
        debug_assert_eq!(slot.kind, HeapKind::Resource);
        let range = vk::DeviceAddressRangeEXT { address, size };
        let resource_info = vk::ResourceDescriptorInfoEXT::default()
            .ty(kind.descriptor_type())
            .data(vk::ResourceDescriptorDataEXT {
                p_address_range: &range,
            });
        self.write_resource(slot, resource_info, self.buffer_descriptor_size)
    }

    pub fn write_acceleration_structure(
        &mut self,
        slot: DescriptorSlot,
        address: vk::DeviceAddress,
    ) -> SrResult<()> {
        debug_assert_eq!(slot.kind, HeapKind::Resource);
        // ACCELERATION_STRUCTURE descriptors take a device address pointing to the AS.
        let range = vk::DeviceAddressRangeEXT {
            address,
            size: 0, // size is ignored for AS descriptors
        };
        let resource_info = vk::ResourceDescriptorInfoEXT::default()
            .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .data(vk::ResourceDescriptorDataEXT {
                p_address_range: &range,
            });
        // AS descriptor size is reported as the buffer descriptor size on this extension.
        self.write_resource(slot, resource_info, self.buffer_descriptor_size)
    }

    pub fn write_sampler(&mut self, slot: DescriptorSlot, info: &vk::SamplerCreateInfo<'_>) -> SrResult<()> {
        debug_assert_eq!(slot.kind, HeapKind::Sampler);
        let dst = self.sampler_dst(slot.index);
        let dst_range = [vk::HostAddressRangeEXT {
            address: dst.as_ptr() as *mut c_void,
            size: self.sampler_descriptor_size,
            _marker: std::marker::PhantomData,
        }];
        unsafe { self.ext.write_sampler_descriptors(std::slice::from_ref(info), &dst_range) }?;
        Ok(())
    }

    /// Bind both heaps to a command buffer. Must be called before any draw/dispatch
    /// that references descriptors via the heap.
    pub fn cmd_bind(&self, cmd_buf: vk::CommandBuffer) {
        let resource_bind = vk::BindHeapInfoEXT::default().heap_range(vk::DeviceAddressRangeEXT {
            address: self.resource.device_address,
            size: self.resource.byte_size,
        });
        let sampler_bind = vk::BindHeapInfoEXT::default().heap_range(vk::DeviceAddressRangeEXT {
            address: self.sampler.device_address,
            size: self.sampler.byte_size,
        });
        unsafe {
            self.ext.cmd_bind_resource_heap(cmd_buf, &resource_bind);
            self.ext.cmd_bind_sampler_heap(cmd_buf, &sampler_bind);
        }
    }

    fn write_resource(
        &mut self,
        slot: DescriptorSlot,
        info: vk::ResourceDescriptorInfoEXT<'_>,
        descriptor_size: usize,
    ) -> SrResult<()> {
        let dst = self.resource_dst(slot.index);
        let dst_range = [vk::HostAddressRangeEXT {
            address: dst.as_ptr() as *mut c_void,
            size: descriptor_size,
            _marker: std::marker::PhantomData,
        }];
        unsafe { self.ext.write_resource_descriptors(std::slice::from_ref(&info), &dst_range) }?;
        Ok(())
    }

    fn resource_dst(&self, index: u32) -> NonNull<u8> {
        let offset = index as u64 * self.resource.stride;
        debug_assert!(offset + self.resource.stride <= self.resource.byte_size);
        unsafe { NonNull::new_unchecked(self.resource.mapped.as_ptr().add(offset as usize)) }
    }

    fn sampler_dst(&self, index: u32) -> NonNull<u8> {
        let offset = index as u64 * self.sampler.stride;
        debug_assert!(offset + self.sampler.stride <= self.sampler.byte_size);
        unsafe { NonNull::new_unchecked(self.sampler.mapped.as_ptr().add(offset as usize)) }
    }
}

impl Drop for DescriptorHeap {
    fn drop(&mut self) {
        // Buffers are destroyed via destroy(); if Drop runs without it the buffers leak.
        // Resources outlive the heap in practice (Core owns them in field-declaration order),
        // so explicit shutdown is via Core::Drop calling destroy().
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

impl SubHeap {
    fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        byte_size: u64,
        stride: u64,
        capacity: u32,
        _heap_alignment: u64,
        name: &'static str,
    ) -> SrResult<Self> {
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

        Ok(Self {
            buffer,
            allocation,
            device_address,
            mapped,
            byte_size,
            stride,
            alloc: SlotAllocator::new(capacity),
        })
    }
}

fn align_up(v: u64, a: u64) -> u64 {
    debug_assert!(a.is_power_of_two() || a == 1);
    (v + a - 1) & !(a - 1)
}
