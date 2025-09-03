use crate::{error::{SrResult, ToSrResult}, vulkan_abstraction};
use ash::{khr, vk::{BufferUsageFlags, MemoryAllocateFlags, MemoryPropertyFlags, PhysicalDeviceMemoryProperties, PhysicalDeviceRayTracingPipelinePropertiesKHR, StridedDeviceAddressRegionKHR}, Device};

fn aligned_size(value : u32, alignment : u32) -> u32 {
    (value + alignment - 1) & !(alignment - 1)
}

pub struct ShaderBindingTable {
    sbt_buffer: vulkan_abstraction::Buffer,
}

impl ShaderBindingTable {
    pub fn new(
        device: &Device,
        rt_pipeline_device: &khr::ray_tracing_pipeline::Device,
        rt_pipeline: &vulkan_abstraction::RayTracingPipeline,
        physical_device_rt_pipeline_properties: &PhysicalDeviceRayTracingPipelinePropertiesKHR,
        physical_device_memory_properties: &PhysicalDeviceMemoryProperties,
    ) -> SrResult<Self> {
        const RAYGEN_COUNT: u32 = 1; //There is always one and only one raygen
        let miss_count = 1;
        let hit_count = 1;
        let handle_count = RAYGEN_COUNT + miss_count + hit_count;

        let handle_size: usize = physical_device_rt_pipeline_properties.shader_group_handle_size as usize;
        let handle_alignment = physical_device_rt_pipeline_properties.shader_group_handle_alignment;
        let base_alignment = physical_device_rt_pipeline_properties.shader_group_base_alignment;
        let handle_size_aligned = aligned_size(handle_size as u32, handle_alignment);

        let raygen_stride = aligned_size(handle_size_aligned, base_alignment);
        let mut raygen_region = StridedDeviceAddressRegionKHR::default()
            .stride(raygen_stride as u64)
            .size(raygen_stride as u64);

        let mut miss_region = StridedDeviceAddressRegionKHR::default()
            .stride(handle_size_aligned as u64)
            .size(aligned_size(miss_count * handle_size_aligned, base_alignment) as u64);

        let mut hit_region = StridedDeviceAddressRegionKHR::default()
            .stride(handle_size_aligned as u64)
            .size(aligned_size(hit_count * handle_size_aligned, base_alignment) as u64);

        let call_region = StridedDeviceAddressRegionKHR::default();

        let data_size = handle_count as usize * handle_size;

        // get the shader handles
        let handles = unsafe {
            rt_pipeline_device.get_ray_tracing_shader_group_handles(rt_pipeline.get_handle(), 0, handle_count, data_size)
        }.to_sr_result()?;

        // Allocate a buffer for storing the SBT.
        let sbt_buffer_size = (raygen_region.size + miss_region.size + hit_region.size + call_region.size) as usize;
        let mut sbt_buffer = vulkan_abstraction::Buffer::new::<u8>(
            device.clone(),
            sbt_buffer_size,
            MemoryPropertyFlags::HOST_VISIBLE,
            MemoryAllocateFlags::DEVICE_ADDRESS,
            BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::SHADER_DEVICE_ADDRESS | BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
            physical_device_memory_properties
        )?;
        let sbt_buffer_data = sbt_buffer.map()?;
        let mut buffer_index = 0;
        let mut handles_index = 0;

        //copying raygen handle in the sbt_buffer
        sbt_buffer_data[buffer_index..buffer_index+handle_size].copy_from_slice(&handles[handles_index..handles_index+handle_size]);
        buffer_index += raygen_region.size as usize;
        handles_index += handle_size;

        //copying miss handles in the sbt_buffer
        for _ in 0..miss_count {
        sbt_buffer_data[buffer_index..buffer_index+handle_size].copy_from_slice(&handles[handles_index..handles_index+handle_size]);
            buffer_index += miss_region.stride as usize;
            handles_index += handle_size;
        }
        //align to next shader group start
        buffer_index = (raygen_region.size + miss_region.size) as usize;

        for _ in 0..hit_count {
        sbt_buffer_data[buffer_index..buffer_index+handle_size].copy_from_slice(&handles[handles_index..handles_index+handle_size]);
            buffer_index += hit_region.stride as usize;
            handles_index += handle_size;
        }

        sbt_buffer.unmap();

        // Find the SBT addresses of each group
        let sbt_buffer_device_address = sbt_buffer.get_device_address();
        raygen_region.device_address = sbt_buffer_device_address;
        miss_region.device_address = sbt_buffer_device_address + raygen_region.size;
        hit_region.device_address = sbt_buffer_device_address + raygen_region.size + miss_region.size;

        Ok(Self {
            sbt_buffer
        })
    }
}
