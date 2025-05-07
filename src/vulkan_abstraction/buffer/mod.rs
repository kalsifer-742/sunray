#![allow(dead_code)]
mod mapped_memory;

use mapped_memory::*;
use crate::vulkan_abstraction::queue::Queue;

use std::ops::Deref;
use ash::{vk, Device};
use ash::vk::{BufferUsageFlags, DeviceMemory, DeviceSize, PhysicalDeviceMemoryProperties, CommandPool};
use crate::error::*;
use crate::vulkan_abstraction::cmd_buffer;

pub struct Buffer {
    usable_byte_size: DeviceSize,
    device: Device,

    buffer: vk::Buffer,
    memory: DeviceMemory,
    mapped_memory: Option<RawMappedMemory>
}
impl Buffer {
    pub fn new_staging<V>(device: Device, size: usize, mem_props : PhysicalDeviceMemoryProperties) -> SrResult<Self> {
        let mem_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let usage_flags = vk::BufferUsageFlags::TRANSFER_SRC;

        Self::new::<V>(device, size, mem_flags, usage_flags, mem_props)
    }
    pub fn new_vertex<V>(device: Device, size: usize, mem_props : PhysicalDeviceMemoryProperties) -> SrResult<Self> {
        let mem_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let usage_flags = BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER;

        Self::new::<V>(device, size, mem_flags, usage_flags, mem_props)
    }
    pub fn new_uniform<V>(device: Device, mem_props : PhysicalDeviceMemoryProperties) -> SrResult<Self> {
        let mem_flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let usage_flags = vk::BufferUsageFlags::UNIFORM_BUFFER;

        Self::new::<V>(device, 1, mem_flags, usage_flags, mem_props)
    }

    pub fn new_staging_from_data<V: Copy>(device: Device, data: &[V], mem_props : PhysicalDeviceMemoryProperties) -> SrResult<Self> {
        //create staging buffer
        let mut staging_buffer = Self::new_staging::<V>(device, data.len(), mem_props)?;

        //copy data to staging buffer
        let mapped_memory = staging_buffer.map::<V>()?;
        mapped_memory[0..data.len()].copy_from_slice(data);
        staging_buffer.unmap::<V>();

        Ok(staging_buffer)
    }
    pub fn new<V>(device: Device, size: usize, mem_flags: vk::MemoryPropertyFlags, usage_flags: vk::BufferUsageFlags, mem_props : PhysicalDeviceMemoryProperties) -> SrResult<Self> {
        let usable_byte_size = (size * size_of::<V>()) as vk::DeviceSize;
        let buffer = {
            let buf_info = vk::BufferCreateInfo::default()
                .size(usable_byte_size)
                .usage(usage_flags)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            unsafe { device.create_buffer(&buf_info, None) }.unwrap()
        };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let allocation_byte_size = mem_requirements.size;
        let mem_type_idx = {
            type BitsType = u32;
            let bits : BitsType = mem_requirements.memory_type_bits;
            assert_ne!(bits, 0);

            let mem_types = mem_props.memory_types;
            let mut idx = -1;
            for i in 0..(8*size_of::<BitsType>()) {
                let mem_type_is_supported = bits & (1 << i) != 0;
                if mem_type_is_supported {
                    if mem_types[i].property_flags & mem_flags == mem_flags {
                        idx = i as isize;
                        break;
                    }
                }
            }
            if idx < 0 {
                panic!("Vertex Buffer Memory Type not supported!");
                // return Err(Box::<dyn Error>::from("Vertex Buffer Memory Type not supported!"));
            }

            idx as u32
        };

        let mem_alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(allocation_byte_size)
            .memory_type_index(mem_type_idx);
        let memory = unsafe_vk! { device.allocate_memory(&mem_alloc_info, None) }?;

        unsafe_vk! { device.bind_buffer_memory(buffer, memory, 0) }?;

        Ok(Self { usable_byte_size, device, buffer, memory, mapped_memory: None })
    }

    pub fn byte_size(&self) -> DeviceSize { self.usable_byte_size }
    pub fn map<V>(&mut self) -> SrResult<&mut [V]> {
        let flags = vk::MemoryMapFlags::empty();
        let p = unsafe_vk!{ self.device.map_memory(self.memory, 0, self.usable_byte_size, flags) }?;
        let raw_mem = unsafe { RawMappedMemory::new(p, self.usable_byte_size as usize) };
        self.mapped_memory = Some(raw_mem);
        let ret = self.mapped_memory.as_mut().unwrap().borrow();


        Ok(ret)
    }

    //correctness of unmap is checked by the borrow checker: it only works if the previous mut borrow of self was already dropped
    pub fn unmap<V>(&mut self) {
        self.mapped_memory = None;

        unsafe { self.device.unmap_memory(self.memory) };
    }

    pub fn clone_buffer(device: &Device, queue: &Queue, cmd_pool: &CommandPool, src: &Buffer, dst: &Buffer) -> SrResult<()>{
        let bufcpy_cmd_buf = cmd_buffer::new(cmd_pool, device)?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe_vk!{ device.begin_command_buffer(bufcpy_cmd_buf, &begin_info) };

        debug_assert!(src.byte_size() <= dst.byte_size());

        let regions = [
            vk::BufferCopy::default()
                .size(src.byte_size())
                .src_offset(0)
                .dst_offset(0)
        ];

        unsafe { device.cmd_copy_buffer(bufcpy_cmd_buf, **src, **dst, &regions) };

        unsafe_vk! { device.end_command_buffer(bufcpy_cmd_buf) }?;

        queue.submit_sync(bufcpy_cmd_buf)?;
        unsafe_vk!{ device.device_wait_idle() }?;

        unsafe { device.free_command_buffers(*cmd_pool, &[bufcpy_cmd_buf]) };
        // vk_core.get_queue().wait_idle()?;

        Ok(())
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        assert!(self.mapped_memory.is_none(), "Buffer::unmap() must be called before the buffer is dropped");
        unsafe { self.device.destroy_buffer(self.buffer, None); }
        unsafe { self.device.free_memory(self.memory, None); }
    }
}

impl Deref for Buffer {
    type Target = vk::Buffer;
    fn deref(&self) -> &Self::Target { &self.buffer }
}