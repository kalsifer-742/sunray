use std::{rc::Rc};

use ash::vk;

use super::BLAS;
use crate::{
    error::*,
    vulkan_abstraction,
};

// Resources:
// - https://github.com/adrien-ben/vulkan-examples-rs
// - https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/
// - https://github.com/SaschaWillems/Vulkan

// TODO: implement drop
pub struct TLAS {
    core: Rc<vulkan_abstraction::Core>,
    tlas: vk::AccelerationStructureKHR,
    #[allow(unused)]
    tlas_buffer: vulkan_abstraction::Buffer,
}

impl TLAS {
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        blas: &[&BLAS],
    ) -> SrResult<Self> {
        let device = core.device().inner();
        // this is the transformation for positioning individual BLASes
        // for now it's an Identity Matrix
        #[rustfmt::skip]
        let transform_matrix = vk::TransformMatrixKHR {
            matrix: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0
            ],
        };

        let blas_instances: Vec<vk::AccelerationStructureInstanceKHR> = blas
            .iter()
            .map(|blas| {
                vk::AccelerationStructureInstanceKHR {
                    transform: transform_matrix,
                    instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xFF), // gl_InstanceCustomIndex = 0, mask = 0 (don't know what actually does, NV tutorial writes "Only be hit if rayMask & instance.mask != 0")
                    instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                        0, // hit_group_offset = 0, same hit group for the whole scene
                        vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8, // disable face culling for semplicity
                    ),
                    acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                        device_handle: unsafe {
                            core.acceleration_structure_device().get_acceleration_structure_device_address(
                                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                                    .acceleration_structure(blas.inner()),
                            )
                        },
                    },
                }
            })
            .collect();
        let blas_instances_n = blas_instances.len();

        // HOST buffer to hold the instances
        let staging_instances_buffer = vulkan_abstraction::Buffer::new_staging_from_data(
            Rc::clone(&core),
            &blas_instances,
        )?;

        // GPU buffer to hold the instances
        let instances_buffer = vulkan_abstraction::Buffer::new::<vk::AccelerationStructureInstanceKHR>(
            Rc::clone(&core),
            blas_instances_n,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryAllocateFlags::DEVICE_ADDRESS,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::TRANSFER_DST,
        )?;

        vulkan_abstraction::Buffer::clone_buffer(&core, &staging_instances_buffer, &instances_buffer)?;

        let acceleration_structure_geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                    .array_of_pointers(false)
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: instances_buffer.get_device_address(),
                    }),
            });

        let binding = [acceleration_structure_geometry]; //look further into thix fix to the error "temporary value dropped while borrowed"
        let mut acceleration_structure_build_geometry_info =
            vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE) // prefer performance over size
                .geometries(&binding)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                .src_acceleration_structure(vk::AccelerationStructureKHR::null());

        let mut acceleration_structure_build_sizes_info =
            vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            core.acceleration_structure_device().get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &acceleration_structure_build_geometry_info,
                &[blas_instances_n as u32],
                &mut acceleration_structure_build_sizes_info,
            )
        };

        let tlas_buffer = vulkan_abstraction::Buffer::new::<u8>(
            Rc::clone(&core),
            acceleration_structure_build_sizes_info.acceleration_structure_size as usize,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryAllocateFlags::DEVICE_ADDRESS,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;

        let acceleration_structure_create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .size(acceleration_structure_build_sizes_info.acceleration_structure_size)
            .buffer(tlas_buffer.inner());

        let tlas = unsafe {
            core.acceleration_structure_device()
                .create_acceleration_structure(&acceleration_structure_create_info, None)
        }?;

        let scratch_buffer = vulkan_abstraction::Buffer::new::<u8>(
            Rc::clone(&core),
            acceleration_structure_build_sizes_info.build_scratch_size as usize,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryAllocateFlags::DEVICE_ADDRESS,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )?;

        // updating acceleration_structure_build_geometry_info with new information
        acceleration_structure_build_geometry_info.dst_acceleration_structure = tlas;
        acceleration_structure_build_geometry_info
            .scratch_data
            .device_address = scratch_buffer.get_device_address();

        let acceleration_structure_build_range_info =
            vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(blas_instances_n as u32)
                .primitive_offset(0)
                .first_vertex(0)
                .transform_offset(0);

        let command_buffer = vulkan_abstraction::cmd_buffer::new(core.cmd_pool(), core.device())?;

        // we can finally build the tlas
        unsafe {
            device
                .begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;

            core.acceleration_structure_device().cmd_build_acceleration_structures(
                command_buffer,
                &[acceleration_structure_build_geometry_info],
                &[&[acceleration_structure_build_range_info]],
            );

            device.end_command_buffer(command_buffer)?
        }

        core.queue().submit_sync(command_buffer)?;

        // build_command_buffer must not be in a pending state when
        // free_command_buffers is called on it
        core.queue().wait_idle()?;

        unsafe {
            device.free_command_buffers(core.cmd_pool().inner(), &[command_buffer]);
        }

        Ok(Self { core, tlas, tlas_buffer })
    }

    pub fn inner(&self) -> vk::AccelerationStructureKHR { self.tlas }
}

impl Drop for TLAS {
    fn drop(&mut self) {
        unsafe {
            self.core.acceleration_structure_device()
                .destroy_acceleration_structure(self.tlas, None)
        };
    }
}
