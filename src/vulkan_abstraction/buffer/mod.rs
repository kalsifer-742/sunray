pub mod index_buffer;
mod mapped_memory;
pub mod vertex_buffer;

//why use and not just mod?
pub use index_buffer::*;
use mapped_memory::*;
pub use vertex_buffer::*;

use crate::vulkan_abstraction::queue::Queue;

use crate::error::*;
use crate::vulkan_abstraction::cmd_buffer;
use ash::vk::{
    BufferUsageFlags, CommandPool, DeviceMemory, DeviceSize, MemoryAllocateFlags,
    MemoryAllocateFlagsInfo, MemoryPropertyFlags, PhysicalDeviceMemoryProperties,
    MemoryRequirements
};
use ash::{Device, vk};
use std::ops::Deref;

pub fn get_memory_type_index(mem_prop_flags: MemoryPropertyFlags, mem_requirements: &MemoryRequirements, mem_props: &PhysicalDeviceMemoryProperties) -> SrResult<u32> {
    type BitsType = u32;
    let bits: BitsType = mem_requirements.memory_type_bits;
    assert_ne!(bits, 0);

    let mem_types = mem_props.memory_types;
    let mut idx = -1;
    for i in 0..(8 * size_of::<BitsType>()) {
        let mem_type_is_supported = bits & (1 << i) != 0;
        if mem_type_is_supported {
            if mem_types[i].property_flags & mem_prop_flags == mem_prop_flags
            {
                idx = i as isize;
                break;
            }
        }
    }
    if idx < 0 {
        return Err(SrError::new(
            None,
            String::from("Vertex Buffer Memory Type not supported!"),
        ));
    }

    Ok(idx as u32)
}

//I think Buffer should be a trait with some default implementations
pub struct Buffer {
    usable_byte_size: DeviceSize,
    device: Device,

    buffer: vk::Buffer,
    memory: DeviceMemory,
    mapped_memory: Option<RawMappedMemory>,
}
impl Buffer {
    pub fn new_staging<V>(
        device: Device,
        size: usize,
        mem_props: &PhysicalDeviceMemoryProperties,
    ) -> SrResult<Self> {
        let memory_property_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let memory_allocate_flags = MemoryAllocateFlags::empty();
        let buffer_usage_flags = vk::BufferUsageFlags::TRANSFER_SRC;

        Self::new::<V>(
            device,
            size,
            memory_property_flags,
            memory_allocate_flags,
            buffer_usage_flags,
            mem_props,
        )
    }

    #[allow(dead_code)]
    pub fn new_uniform<T>(
        device: Device,
        mem_props: &PhysicalDeviceMemoryProperties,
    ) -> SrResult<Self> {
        let memory_property_flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let memory_allocate_flags = vk::MemoryAllocateFlags::empty();
        let buffer_usage_flags = vk::BufferUsageFlags::UNIFORM_BUFFER;

        Self::new::<T>(
            device,
            1,
            memory_property_flags,
            memory_allocate_flags,
            buffer_usage_flags,
            mem_props,
        )
    }

    pub fn new_staging_from_data<T: Copy>(
        device: Device,
        data: &[T],
        mem_props: &PhysicalDeviceMemoryProperties,
    ) -> SrResult<Self> {
        //create staging buffer
        let mut staging_buffer = Self::new_staging::<T>(device, data.len(), mem_props)?;

        //copy data to staging buffer
        let mapped_memory = staging_buffer.map::<T>()?;
        mapped_memory[0..data.len()].copy_from_slice(data);
        staging_buffer.unmap();

        Ok(staging_buffer)
    }

    /// # Create a new Buffer
    ///
    /// ## Arguments:
    /// - `len`: the number of items, not the amount of memory. the functions take care of that calculation internally
    pub fn new<V>(
        device: Device,
        len: usize,
        memory_property_flags: MemoryPropertyFlags,
        alloc_flags: MemoryAllocateFlags,
        buffer_usage_flags: BufferUsageFlags,
        mem_props: &PhysicalDeviceMemoryProperties,
    ) -> SrResult<Self> {
        let usable_byte_size = (len * size_of::<V>()) as vk::DeviceSize;
        let buffer = {
            let buf_info = vk::BufferCreateInfo::default()
                .size(usable_byte_size)
                .usage(buffer_usage_flags)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            unsafe { device.create_buffer(&buf_info, None) }?
        };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let allocation_byte_size = mem_requirements.size;
        let mem_type_idx = get_memory_type_index(memory_property_flags, &mem_requirements, &mem_props)?;

        let mut memory_allocate_flags_info = MemoryAllocateFlagsInfo::default().flags(alloc_flags);

        let mem_alloc_info = {
            let mem_alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(allocation_byte_size)
                .memory_type_index(mem_type_idx);

            if alloc_flags == MemoryAllocateFlags::empty() {
                mem_alloc_info
            } else {
                mem_alloc_info.push_next(&mut memory_allocate_flags_info)
            }
        };
        let memory =
            unsafe { device.allocate_memory(&mem_alloc_info, None) }?;

        unsafe { device.bind_buffer_memory(buffer, memory, 0) }?;

        Ok(Self {
            usable_byte_size,
            device,
            buffer,
            memory,
            mapped_memory: None,
        })
    }

    pub fn byte_size(&self) -> DeviceSize {
        self.usable_byte_size
    }
    pub fn map<V>(&mut self) -> SrResult<&mut [V]> {
        let flags = vk::MemoryMapFlags::empty();
        let p = unsafe {
            self.device
                .map_memory(self.memory, 0, self.usable_byte_size, flags)
        }?;
        let raw_mem = unsafe { RawMappedMemory::new(p, self.usable_byte_size as usize) };
        self.mapped_memory = Some(raw_mem);
        let ret = self.mapped_memory.as_mut().unwrap().borrow();

        Ok(ret)
    }

    // correctness of unmap is checked by the borrow checker: it only works if the previous
    // mut borrow of self was already dropped. drop() calls unmap() if necessary
    pub fn unmap(&mut self) {
        self.mapped_memory = None;

        unsafe { self.device.unmap_memory(self.memory) };
    }

    // mainly useful to copy from a staging buffer to a device buffer
    pub fn clone_buffer(
        device: &Device,
        queue: &Queue,
        cmd_pool: &CommandPool,
        src: &Buffer,
        dst: &Buffer,
    ) -> SrResult<()> {
        let bufcpy_cmd_buf = cmd_buffer::new(cmd_pool, device)?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { device.begin_command_buffer(bufcpy_cmd_buf, &begin_info) }?;

        debug_assert!(src.byte_size() <= dst.byte_size());

        //documentation please :) ~kalsifer
        let regions = [vk::BufferCopy::default()
            .size(src.byte_size())
            .src_offset(0)
            .dst_offset(0)];

        unsafe { device.cmd_copy_buffer(bufcpy_cmd_buf, **src, **dst, &regions) };

        unsafe { device.end_command_buffer(bufcpy_cmd_buf) }?;

        queue.submit_sync(bufcpy_cmd_buf)?;
        unsafe { device.device_wait_idle() }?;

        unsafe { device.free_command_buffers(*cmd_pool, &[bufcpy_cmd_buf]) };
        // queue.wait_idle()?;

        Ok(())
    }

    pub fn get_device_address(&self) -> vk::DeviceAddress {
        let buffer_device_address_info = vk::BufferDeviceAddressInfo::default().buffer(self.buffer);
        unsafe {
            self.device
                .get_buffer_device_address(&buffer_device_address_info)
        }
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        //unmap() must be called before the buffer is dropped
        if self.mapped_memory.is_some() {
            self.unmap();
        }

        unsafe {
            self.device.destroy_buffer(self.buffer, None);
        }
        unsafe {
            self.device.free_memory(self.memory, None);
        }
    }
}

impl Deref for Buffer {
    type Target = vk::Buffer;
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}
