pub mod index_buffer;
pub mod vertex_buffer;

//why use and not just mod?
pub use index_buffer::*;
pub use vertex_buffer::*;

use crate::{error::*, vulkan_abstraction};
use ash::vk;
use std::rc::Rc;

pub fn get_memory_type_index(
    core: &vulkan_abstraction::Core,
    mem_prop_flags: vk::MemoryPropertyFlags,
    mem_requirements: &vk::MemoryRequirements,
) -> SrResult<u32> {
    type BitsType = u32;
    let bits: BitsType = mem_requirements.memory_type_bits;
    assert_ne!(bits, 0);

    let mem_types = core.device().memory_properties().memory_types;
    let mut idx = -1;
    for i in 0..(8 * size_of::<BitsType>()) {
        let mem_type_is_supported = bits & (1 << i) != 0;
        if mem_type_is_supported {
            if mem_types[i].property_flags & mem_prop_flags == mem_prop_flags {
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
    core: Rc<vulkan_abstraction::Core>,

    buffer: vk::Buffer,
    allocation: gpu_allocator::vulkan::Allocation,

    byte_size: u64,
}

impl Buffer {
    pub fn new_staging<V>(core: Rc<vulkan_abstraction::Core>, size: usize) -> SrResult<Self> {
        Self::new::<V>(
            core,
            size,
            gpu_allocator::MemoryLocation::CpuToGpu,
            vk::BufferUsageFlags::TRANSFER_SRC,
            "staging buffer"
        )
    }

    #[allow(dead_code)]
    pub fn new_uniform<T>(core: Rc<vulkan_abstraction::Core>) -> SrResult<Self> {
        Self::new::<T>(
            core,
            1,
            gpu_allocator::MemoryLocation::CpuToGpu,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "uniform buffer",
        )
    }

    pub fn new_staging_from_data<T: Copy>(
        core: Rc<vulkan_abstraction::Core>,
        data: &[T],
    ) -> SrResult<Self> {
        if data.len() == 0 {
            return Ok(Self::new_null(core));
        }
        //create staging buffer
        let mut staging_buffer = Self::new_staging::<T>(core, data.len())?;

        //copy data to staging buffer
        let mapped_memory = staging_buffer.map::<T>()?;
        mapped_memory[0..data.len()].copy_from_slice(data);

        Ok(staging_buffer)
    }

    pub fn new_from_data<T: Copy>(
        core: Rc<vulkan_abstraction::Core>,
        data: &[T],
        mem_location: gpu_allocator::MemoryLocation,
        buffer_usage_flags: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        if data.len() == 0 {
            return Ok(Self::new_null(core));
        }

        let staging_buffer = Self::new_staging_from_data(Rc::clone(&core), data)?;
        let buffer = Self::new::<T>(
            Rc::clone(&core),
            data.len(),
            mem_location,
            buffer_usage_flags,
            name,
        )?;
        Self::clone_buffer(&core, &staging_buffer, &buffer)?;

        Ok(buffer)
    }

    pub fn new_null(core: Rc<vulkan_abstraction::Core>) -> Self {
        Self {
            core,
            buffer: vk::Buffer::null(),
            allocation: gpu_allocator::vulkan::Allocation::default(),
            byte_size: 0,
        }
    }

    /// # Create a new Buffer
    ///
    /// ## Arguments:
    /// - `len`: the number of items, not the amount of memory. the functions take care of that calculation internally
    pub fn new<V>(
        core: Rc<vulkan_abstraction::Core>,
        len: usize,
        memory_location: gpu_allocator::MemoryLocation,
        buffer_usage_flags: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        Self::new_aligned::<V>(core, len, 1, memory_location, buffer_usage_flags, name)
    }

    pub fn new_aligned<V>(
        core: Rc<vulkan_abstraction::Core>,
        len: usize,
        alignment: u64,
        memory_location: gpu_allocator::MemoryLocation,
        buffer_usage_flags: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        if len == 0 {
            return Ok(Self::new_null(core))
        }

        let device = core.device().inner();
        let usable_byte_size = (len * size_of::<V>()) as vk::DeviceSize;
        let buffer = {
            let buf_info = vk::BufferCreateInfo::default()
                .size(usable_byte_size)
                .usage(buffer_usage_flags)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            unsafe { device.create_buffer(&buf_info, None) }?
        };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        // fix alignment
        let mem_requirements = mem_requirements.alignment(mem_requirements.alignment.max(alignment));

        let allocation = core.allocator_mut().allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name,
            requirements: mem_requirements,
            location: memory_location,
            linear: true, // Buffers are always linear
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        })?;


        unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }?;

        Ok(Self {
            core,
            buffer,
            allocation,
            byte_size: usable_byte_size,
        })
    }

    pub fn byte_size(&self) -> vk::DeviceSize {
        self.byte_size
    }
    pub fn map<V: Sized>(&mut self) -> SrResult<&mut [V]> {
        if !self.is_null() {
            let slice = self.allocation.mapped_slice_mut().unwrap();

            let ret = unsafe { std::slice::from_raw_parts_mut(slice.as_ptr() as *mut V, slice.len() / std::mem::size_of::<V>()) };

            Ok(ret)
        } else {
            Ok(&mut [])
        }
    }


    // mainly useful to copy from a staging buffer to a device buffer
    pub fn clone_buffer(
        core: &vulkan_abstraction::Core,
        src: &Buffer,
        dst: &Buffer,
    ) -> SrResult<()> {
        if src.is_null() {
            return Ok(());
        }
        if dst.is_null() {
            return Err(SrError::new(None, String::from("attempted to clone from a non-null buffer to a null buffer")));
        }

        let device = core.device().inner();
        let bufcpy_cmd_buf = vulkan_abstraction::cmd_buffer::new_command_buffer(core.cmd_pool(), core.device().inner())?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { device.begin_command_buffer(bufcpy_cmd_buf, &begin_info) }?;

        debug_assert!(src.byte_size() <= dst.byte_size());

        //copy src.byte_size() bytes, from position 0 in src buffer to position 0 in dst buffer
        let regions = [vk::BufferCopy::default()
            .size(src.byte_size())
            .src_offset(0)
            .dst_offset(0)];

        unsafe { device.cmd_copy_buffer(bufcpy_cmd_buf, src.inner(), dst.inner(), &regions) };

        unsafe { device.end_command_buffer(bufcpy_cmd_buf) }?;

        core.queue().submit_sync(bufcpy_cmd_buf)?;

        unsafe { device.free_command_buffers(**core.cmd_pool(), &[bufcpy_cmd_buf]) };

        Ok(())
    }
    pub fn is_null(&self) -> bool { self.buffer == vk::Buffer::null() }

    pub fn get_device_address(&self) -> vk::DeviceAddress {
        if self.is_null() {
            return 0 as vk::DeviceAddress;
        }

        let buffer_device_address_info = vk::BufferDeviceAddressInfo::default().buffer(self.buffer);
        unsafe {
            self.core
                .device()
                .inner()
                .get_buffer_device_address(&buffer_device_address_info)
        }
    }

    pub fn inner(&self) -> vk::Buffer {
        self.buffer
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let device = self.core.device().inner();
        unsafe {
            device.destroy_buffer(self.buffer, None);
        }

        //need to take ownership to pass to free
        let allocation = std::mem::replace(&mut self.allocation, gpu_allocator::vulkan::Allocation::default());
        match self.core.allocator_mut().free(allocation) {
            Ok(()) => {}
            Err(e) => log::error!("gpu_allocator::vulkan::Allocator::free returned {e} in sunray::vulkan_abstraction::Buffer::drop"),
        }
    }
}
