use ash::{
    Device,
    khr::acceleration_structure,
    vk::{
        AccelerationStructureBuildGeometryInfoKHR, AccelerationStructureBuildRangeInfoKHR,
        AccelerationStructureBuildSizesInfoKHR, AccelerationStructureCreateInfoKHR,
        AccelerationStructureDeviceAddressInfoKHR, AccelerationStructureGeometryDataKHR,
        AccelerationStructureGeometryInstancesDataKHR, AccelerationStructureGeometryKHR,
        AccelerationStructureInstanceKHR, AccelerationStructureKHR,
        AccelerationStructureReferenceKHR, AccelerationStructureTypeKHR, Buffer, BufferCreateInfo,
        BufferDeviceAddressInfo, BufferUsageFlags, BuildAccelerationStructureFlagsKHR,
        CommandBufferBeginInfo, CommandBufferUsageFlags, DeviceOrHostAddressConstKHR,
        GeometryFlagsKHR, GeometryInstanceFlagsKHR, GeometryTypeKHR, MemoryAllocateFlags,
        MemoryPropertyFlags, PFN_vkCreateAccelerationStructureKHR, Packed24_8,
        PhysicalDeviceAccelerationStructurePropertiesKHR, PhysicalDeviceMemoryProperties,
        TaggedStructure, TransformMatrixKHR,
    },
};

use super::BLAS;
use crate::{
    error::{SrError, SrResult},
    vulkan_abstraction::{self, instance_buffer::InstanceBuffer},
};

// Resources:
// - https://github.com/adrien-ben/vulkan-examples-rs
// - https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/
// - https://github.com/SaschaWillems/Vulkan

pub struct TLAS {
    tlas: AccelerationStructureKHR,
    buffer: 
}

/*Reading other people code I understood that a commond way of handling the tlas is to inherit everyhting from the blas:
    - device address, ecc...

  TODO:
  cleanup code
    - error handling
    - divide stuff in separate functions if needed
  comments
  make it work for multilples blas
*/

// add and check PhysicalDeviceAccelerationStructurePropertiesKHR::max_instance_count(self, max_instance_count)
impl TLAS {
    pub fn new(
        device: &Device,
        acceleration_structure_device: acceleration_structure::Device,
        device_memory_props: &PhysicalDeviceMemoryProperties,
        cmd_pool: &vulkan_abstraction::CmdPool,
        blas: Vec<BLAS>,
    ) -> SrResult<Self> {
        // This is the transformation for positioning individual BLASes
        // For now it's an Identity Matrix
        let transform_matrix = TransformMatrixKHR {
            matrix: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        };

        let blas_instances = blas.iter().map(|blas| {
            AccelerationStructureInstanceKHR {
                transform: transform_matrix,
                instance_custom_index_and_mask: Packed24_8::new(0, 0), // gl_InstanceCustomIndex = 0, mask = 0 (don't know what actually does)
                instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(
                    0, // hit_group_offset = 0, same hit group for the whole scene
                    GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8, // disable face culling for semplicity
                ),
                acceleration_structure_reference: AccelerationStructureReferenceKHR {
                    device_handle: unsafe {
                        blas.acceleration_structure_device()
                            .get_acceleration_structure_device_address(
                                &AccelerationStructureDeviceAddressInfoKHR::default()
                                    .acceleration_structure(blas.blas()), // maybe we should discuss a change of name, proposal: inner
                            )
                    },
                },
            }
        });

        let instances_buffer = InstanceBuffer::new::<u8>(device.clone(), blas_instances.len(), device_memory_props)?; //u8 as a placeholder copying from the blas, is it correct?

        let buffer_device_address_info =
            BufferDeviceAddressInfo::default().buffer(instance_buffer.buffer); //buffer is private
        let instance_buffer_device_address =
            unsafe { device.get_buffer_device_address(&buffer_device_address_info) };

        // the s_type usually never appears as a method to modify it but only as a public property.
        // why it is not required to assign it? default?
        // should we assign it?
        //
        // in general nv tutorial specify less things than this
        let acceleration_structure_geometry = AccelerationStructureGeometryKHR::default()
            .geometry_type(GeometryTypeKHR::INSTANCES)
            .flags(GeometryFlagsKHR::OPAQUE)
            .geometry(AccelerationStructureGeometryDataKHR {
                instances: AccelerationStructureGeometryInstancesDataKHR::default()
                    .array_of_pointers(false)
                    .data(DeviceOrHostAddressConstKHR {
                        device_address: instance_buffer_device_address,
                    }),
            });

        let acceleration_structure_build_sizes_info =
            AccelerationStructureBuildSizesInfoKHR::default();

        let acceleration_structure_create_info = AccelerationStructureCreateInfoKHR::default()
            .ty(AccelerationStructureTypeKHR::TOP_LEVEL)
            .size(acceleration_structure_build_sizes_info.acceleration_structure_size)
            .buffer(buffer); //figure out what to put in here

        let tlas = unsafe {
            acceleration_structure_device
                .create_acceleration_structure(&acceleration_structure_create_info, None)
        };

        // inserted less stuff than nv tutorial
        let acceleration_structure_build_geometry_info =
            AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(AccelerationStructureTypeKHR::TOP_LEVEL)
                .flags(BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .geometries(acceleration_structure_geometry) //pointers stuff
                .dst_acceleration_structure(tlas);

        let acceleration_structure_build_range_info =
            AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(1)
                .primitive_offset(0)
                .first_vertex(0)
                .transform_offset(0);

        let acceleration_structure_create_info = AccelerationStructureCreateInfoKHR::default()
            .ty(AccelerationStructureTypeKHR::TOP_LEVEL)
            .size(acceleration_structure_build_sizes_info.acceleration_structure_size);

        unsafe {
            acceleration_structure_device
                .create_acceleration_structure(&acceleration_structure_create_info, None)
        };

        let scratch_buffer = vulkan_abstraction::Buffer::new::<u8>(
            device.clone(),
            acceleration_structure_build_sizes_info.build_scratch_size, //need to be converted to usize
            MemoryPropertyFlags::DEVICE_LOCAL,
            MemoryAllocateFlags::DEVICE_ADDRESS,
            BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            device_memory_props,
        )?;

        // one-shot command buffer which we will:
        // - fill with the commands to build the BLAS
        // - pass to the queue to be executed (thus building the BLAS)
        // - free
        let command_buffer = vulkan_abstraction::cmd_buffer::new(cmd_pool, device)?;

        //record build_command_buffer with the commands to build the BLAS
        unsafe {
            device
                .begin_command_buffer(
                    command_buffer,
                    &CommandBufferBeginInfo::default()
                        .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(SrError::from)?;

            acceleration_structure_device.cmd_build_acceleration_structures(
                command_buffer,
                &[acceleration_structure_build_geometry_info],
                &[&[acceleration_structure_build_range_info]],
            );

            device
                .end_command_buffer(command_buffer)
                .map_err(SrError::from)?
        }

        //figure out about the queue
        queue.submit_sync(command_buffer)?;

        // build_command_buffer must not be in a pending state when
        // free_command_buffers is called on it
        queue.wait_idle()?;

        unsafe {
            device.free_command_buffers(**cmd_pool, &[command_buffer]);
        }

        Self { tlas }
    }
}
