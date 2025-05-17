use ash::{
    Device,
    vk::{
        BufferUsageFlags, MemoryAllocateFlags, MemoryPropertyFlags, PhysicalDeviceMemoryProperties,
    },
};

use crate::error::SrResult;

use super::Buffer;

//TO-DO: error handling
pub struct InstanceBuffer {
    buffer: Buffer,
}
impl InstanceBuffer {
    pub fn new<T>(
        device: Device,
        len: usize,
        mem_props: &PhysicalDeviceMemoryProperties,
    ) -> SrResult<Self> {
        let mem_flags = MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT;
        let alloc_flags = MemoryAllocateFlags::DEVICE_ADDRESS; //no idea what should be putted here
        let usage_flags = BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;

        let buffer = Buffer::new::<T>(device, len, mem_flags, alloc_flags, usage_flags, mem_props)?;

        Ok(Self { buffer })
    }
}
