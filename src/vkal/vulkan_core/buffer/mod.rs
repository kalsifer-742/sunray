mod mapped_memory;

use mapped_memory::*;

use std::ops::Deref;
use std::rc::Rc;
use ash::vk;
use crate::vkal;

pub struct Buffer {
    usable_byte_size: vk::DeviceSize,
    device: Rc<vkal::Device>,

    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped_memory: Option<RawMappedMemory>
}
impl Buffer {
    pub fn new_staging<V>(device: Rc<vkal::Device>, size: usize) -> vkal::Result<Self> {
        let mem_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let usage_flags = vk::BufferUsageFlags::TRANSFER_SRC;

        Self::new::<V>(device, size, mem_flags, usage_flags)
    }
    pub fn new_vertex<V>(device: Rc<vkal::Device>, size: usize) -> vkal::Result<Self> {
        let mem_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let usage_flags =
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER;

        Self::new::<V>(device, size, mem_flags, usage_flags)
    }
    pub fn new_uniform<V>(device: Rc<vkal::Device>) -> vkal::Result<Self> {
        let mem_flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let usage_flags = vk::BufferUsageFlags::UNIFORM_BUFFER;

        Self::new::<V>(device, 1, mem_flags, usage_flags)
    }

    pub fn new_staging_from_data<V: Copy>(device: Rc<vkal::Device>, data: &[V]) -> vkal::Result<Self> {
        //create staging buffer
        let mut staging_buffer = Self::new_staging::<V>(Rc::clone(&device), data.len())?;

        //copy data to staging buffer
        let mapped_memory = staging_buffer.map::<V>()?;
        mapped_memory[0..data.len()].copy_from_slice(data);
        staging_buffer.unmap::<V>();

        Ok(staging_buffer)
    }
    pub fn new<V>(device: Rc<vkal::Device>, size: usize, mem_flags: vk::MemoryPropertyFlags, usage_flags: vk::BufferUsageFlags) -> vkal::Result<Self> {
        let usable_byte_size = (size * size_of::<V>()) as vk::DeviceSize;
        let buffer = {
            let buf_info = vk::BufferCreateInfo::default()
                .size(usable_byte_size)
                .usage(usage_flags)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            unsafe { device.create_buffer(&buf_info, vkal::NO_ALLOCATOR) }.unwrap()
        };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let allocation_byte_size = mem_requirements.size;
        let mem_type_idx = {
            type BitsType = u32;
            let bits : BitsType = mem_requirements.memory_type_bits;
            assert_ne!(bits, 0);

            let mem_types = &device.get_physical_device_info().memory_props.memory_types;
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
        let memory = unsafe { device.allocate_memory(&mem_alloc_info, None) }?;

        unsafe { device.bind_buffer_memory(buffer, memory, 0) }?;

        Ok(Self { usable_byte_size, device, buffer, memory, mapped_memory: None })
    }

    pub fn byte_size(&self) -> vk::DeviceSize { self.usable_byte_size }
    pub fn map<V>(&mut self) -> vkal::Result<&mut [V]> {
        let flags = vk::MemoryMapFlags::empty();
        let p = unsafe { self.device.map_memory(self.memory, 0, self.usable_byte_size, flags) }?;
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