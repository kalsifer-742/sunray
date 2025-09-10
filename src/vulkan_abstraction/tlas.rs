use std::ops::Deref;

use ash::{
    khr::acceleration_structure, vk::{
        AccelerationStructureBuildGeometryInfoKHR, AccelerationStructureBuildRangeInfoKHR, AccelerationStructureBuildSizesInfoKHR, AccelerationStructureBuildTypeKHR, AccelerationStructureCreateInfoKHR, AccelerationStructureDeviceAddressInfoKHR, AccelerationStructureGeometryDataKHR, AccelerationStructureGeometryInstancesDataKHR, AccelerationStructureGeometryKHR, AccelerationStructureInstanceKHR, AccelerationStructureKHR, AccelerationStructureReferenceKHR, AccelerationStructureTypeKHR, BufferUsageFlags, BuildAccelerationStructureFlagsKHR, BuildAccelerationStructureModeKHR, CommandBufferBeginInfo, CommandBufferUsageFlags, DeviceOrHostAddressConstKHR, GeometryInstanceFlagsKHR, GeometryTypeKHR, MemoryAllocateFlags, MemoryPropertyFlags, Packed24_8, PhysicalDeviceMemoryProperties, TransformMatrixKHR
    }, Device
};

use super::BLAS;
use crate::{
    error::*,
    vulkan_abstraction::{self},
};

// Resources:
// - https://github.com/adrien-ben/vulkan-examples-rs
// - https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/
// - https://github.com/SaschaWillems/Vulkan

// in general i assigned only the parameters assigned in the NV tutorial
// the other examples assign a lot more stuff
// if something doesn't work look at the parameters in the other examples

// TODO: implement drop
pub struct TLAS {
    acceleration_structure_device: acceleration_structure::Device,
    tlas: AccelerationStructureKHR,
}

impl TLAS {
    pub fn new(
        device: &Device,
        acceleration_structure_device: acceleration_structure::Device,
        device_memory_props: &PhysicalDeviceMemoryProperties,
        cmd_pool: &vulkan_abstraction::CmdPool,
        queue: &vulkan_abstraction::Queue,
        blas: &[BLAS],
    ) -> SrResult<Self> {
        // this is the transformation for positioning individual BLASes
        // for now it's an Identity Matrix
        let transform_matrix = TransformMatrixKHR {
            matrix: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ],
        };

        let blas_instances: Vec<AccelerationStructureInstanceKHR> = blas
            .iter()
            .map(|blas| {
                AccelerationStructureInstanceKHR {
                    transform: transform_matrix,
                    instance_custom_index_and_mask: Packed24_8::new(0, 0), // gl_InstanceCustomIndex = 0, mask = 0 (don't know what actually does, NV tutorial writes "Only be hit if rayMask & instance.mask != 0")
                    instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(
                        0, // hit_group_offset = 0, same hit group for the whole scene
                        GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8, // disable face culling for semplicity
                    ),
                    acceleration_structure_reference: AccelerationStructureReferenceKHR {
                        device_handle: unsafe {
                            acceleration_structure_device.get_acceleration_structure_device_address(
                                &AccelerationStructureDeviceAddressInfoKHR::default()
                                    .acceleration_structure(*blas.deref()), // maybe we should discuss a change of name, proposal: inner
                            )
                        },
                    },
                }
            })
            .collect();
        let blas_instances_n = blas_instances.len();

        // HOST buffer to hold the instances
        let staging_instances_buffer = vulkan_abstraction::Buffer::new_staging_from_data(
            device.clone(),
            &blas_instances,
            device_memory_props,
        )?;

        // GPU buffer to hold the instances
        let instances_buffer = vulkan_abstraction::Buffer::new::<AccelerationStructureInstanceKHR>(
            device.clone(),
            blas_instances_n,
            MemoryPropertyFlags::DEVICE_LOCAL,
            MemoryAllocateFlags::DEVICE_ADDRESS,
            BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | BufferUsageFlags::TRANSFER_DST,
            device_memory_props,
        )?;

        vulkan_abstraction::Buffer::clone_buffer(
            device,
            queue,
            cmd_pool,
            &staging_instances_buffer,
            &instances_buffer,
        )?;

        let acceleration_structure_geometry = AccelerationStructureGeometryKHR::default()
            .geometry_type(GeometryTypeKHR::INSTANCES)
            .geometry(AccelerationStructureGeometryDataKHR {
                instances: AccelerationStructureGeometryInstancesDataKHR::default().data(
                    DeviceOrHostAddressConstKHR {
                        device_address: instances_buffer.get_device_address(),
                    },
                ),
            });

        let binding = [acceleration_structure_geometry]; //look further into thix fix to the error "temporary value dropped while borrowed"
        let mut acceleration_structure_build_geometry_info =
            AccelerationStructureBuildGeometryInfoKHR::default()
                .flags(BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE) // prefer performance over size
                .geometries(&binding)
                .mode(BuildAccelerationStructureModeKHR::BUILD)
                .ty(AccelerationStructureTypeKHR::TOP_LEVEL)
                .src_acceleration_structure(AccelerationStructureKHR::null());

        let mut acceleration_structure_build_sizes_info =
            AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            acceleration_structure_device.get_acceleration_structure_build_sizes(
                AccelerationStructureBuildTypeKHR::DEVICE,
                &acceleration_structure_build_geometry_info,
                &[blas_instances_n as u32],
                &mut acceleration_structure_build_sizes_info,
            )
        };

        let tlas_buffer = vulkan_abstraction::Buffer::new::<u8>(
            device.clone(),
            acceleration_structure_build_sizes_info.acceleration_structure_size as usize,
            MemoryPropertyFlags::DEVICE_LOCAL,
            MemoryAllocateFlags::DEVICE_ADDRESS,
            BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            device_memory_props,
        )?;

        let acceleration_structure_create_info = AccelerationStructureCreateInfoKHR::default()
            .ty(AccelerationStructureTypeKHR::TOP_LEVEL)
            .size(acceleration_structure_build_sizes_info.acceleration_structure_size)
            .buffer(*tlas_buffer.deref());

        let tlas = unsafe {
            acceleration_structure_device
                .create_acceleration_structure(&acceleration_structure_create_info, None)
        }?;

        let scratch_buffer = vulkan_abstraction::Buffer::new::<u8>(
            device.clone(),
            acceleration_structure_build_sizes_info.build_scratch_size as usize,
            MemoryPropertyFlags::DEVICE_LOCAL,
            MemoryAllocateFlags::DEVICE_ADDRESS,
            BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            device_memory_props,
        )?;

        // updating acceleration_structure_build_geometry_info with new information
        acceleration_structure_build_geometry_info.dst_acceleration_structure = tlas;
        acceleration_structure_build_geometry_info
            .scratch_data
            .device_address = scratch_buffer.get_device_address();

        let acceleration_structure_build_range_info =
            AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(blas_instances_n as u32)
                .primitive_offset(0)
                .first_vertex(0)
                .transform_offset(0);

        let command_buffer = vulkan_abstraction::cmd_buffer::new(cmd_pool, device)?;

        // we can finally build the tlas
        unsafe {
            device
                .begin_command_buffer(
                    command_buffer,
                    &CommandBufferBeginInfo::default()
                        .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;

            acceleration_structure_device.cmd_build_acceleration_structures(
                command_buffer,
                &[acceleration_structure_build_geometry_info],
                &[&[acceleration_structure_build_range_info]],
            );

            device.end_command_buffer(command_buffer)?
        }

        queue.submit_sync(command_buffer)?;

        // build_command_buffer must not be in a pending state when
        // free_command_buffers is called on it
        queue.wait_idle()?;

        unsafe {
            device.free_command_buffers(**cmd_pool, &[command_buffer]);
        }

        Ok(Self {
            acceleration_structure_device,
            tlas,
        })
    }
}

impl Deref for TLAS {
    type Target = AccelerationStructureKHR;

    fn deref(&self) -> &Self::Target {
        &self.tlas
    }
}

impl Drop for TLAS {
    fn drop(&mut self) {
        unsafe {
            self.acceleration_structure_device
                .destroy_acceleration_structure(self.tlas, None)
        };
    }
}
