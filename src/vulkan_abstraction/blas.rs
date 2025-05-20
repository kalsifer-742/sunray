use std::ops::Deref;

use crate::error::*;
use crate::vulkan_abstraction;
use ash::{
    Device,
    khr::acceleration_structure,
    vk::{
        AccelerationStructureBuildGeometryInfoKHR, AccelerationStructureBuildRangeInfoKHR,
        AccelerationStructureBuildSizesInfoKHR, AccelerationStructureBuildTypeKHR,
        AccelerationStructureCreateInfoKHR, AccelerationStructureGeometryDataKHR,
        AccelerationStructureGeometryKHR, AccelerationStructureGeometryTrianglesDataKHR,
        AccelerationStructureKHR, AccelerationStructureTypeKHR, BufferUsageFlags,
        BuildAccelerationStructureFlagsKHR, BuildAccelerationStructureModeKHR,
        CommandBufferBeginInfo, CommandBufferUsageFlags, DeviceOrHostAddressConstKHR,
        DeviceOrHostAddressKHR, Format, GeometryFlagsKHR, GeometryTypeKHR, MemoryAllocateFlags,
        MemoryPropertyFlags, PhysicalDeviceMemoryProperties,
    },
};

// Bottom-Level Acceleration Structure
pub struct BLAS {
    blas: AccelerationStructureKHR,
    blas_buffer: vulkan_abstraction::Buffer,
    acceleration_structure_device: ash::khr::acceleration_structure::Device,
}

impl BLAS {
    pub fn new(
        device: &Device,
        acceleration_structure_device: acceleration_structure::Device,
        device_memory_props: &PhysicalDeviceMemoryProperties,
        cmd_pool: &vulkan_abstraction::CmdPool,
        queue: &vulkan_abstraction::Queue,
        vertex_buffer: &vulkan_abstraction::VertexBuffer,
        index_buffer: &vulkan_abstraction::IndexBuffer,
    ) -> SrResult<Self> {
        /*
         * Building the BLAS is mostly a 3 step process (with some complications):
         * 1.  Allocate a GPU Buffer on which it will live (blas_buffer)
         *     and a scratch buffer used only for step 3
         * 2.  Create a BLAS handle (blas) pointing to this allocation
         * 3.  Build the actual BLAS data structure
         */

        // specify what the BLAS's geometry (vbo, ibo) is
        let geometry = {
            let geometry_data = AccelerationStructureGeometryDataKHR {
                triangles: AccelerationStructureGeometryTrianglesDataKHR::default()
                    .vertex_data(DeviceOrHostAddressConstKHR {
                        device_address: vertex_buffer.get_device_address(),
                    })
                    .max_vertex(vertex_buffer.len() as u32 - 1)
                    .vertex_stride(vertex_buffer.stride() as u64)
                    .vertex_format(Format::R32G32B32_SFLOAT)
                    .index_data(DeviceOrHostAddressConstKHR {
                        device_address: index_buffer.get_device_address(),
                    })
                    .index_type(index_buffer.index_type())
                    // no transform data
                    .transform_data(DeviceOrHostAddressConstKHR::default()),
            };

            AccelerationStructureGeometryKHR::default()
                .geometry_type(GeometryTypeKHR::TRIANGLES)
                .geometry(geometry_data)
                .flags(GeometryFlagsKHR::OPAQUE)
        };
        let geometries = [geometry];

        // specify the range of values to read from the ibo, vbo and transform data of a geometry.
        // there must be one build_range_info for each geometry
        let build_range_info = {
            AccelerationStructureBuildRangeInfoKHR::default()
                // the number of triangles to read (3 * the number of indices to read)
                .primitive_count(index_buffer.len() as u32 / 3)
                // an offset (in bytes) into geometry.geometry_data.index_data from which to start reading
                .primitive_offset(0)
                // the value of first_vertex is added to index values before fetching verts
                .first_vertex(0)
                // transform_offset is an offset (in bytes) into geometry.geometry_data.transform_data
                .transform_offset(0)
        };

        // parameters on how to build the BLAS.
        // this temporary version is used to calculate how much memory to allocate for it,
        // and the final version which is used to really build the blas will be based on it,
        // with some additional args based on the allocations that were performed.
        let incomplete_build_info = AccelerationStructureBuildGeometryInfoKHR::default()
            .geometries(&geometries)
            // PREFER_FAST_TRACE -> prioritize trace performance over build time
            .flags(BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            // BUILD as opposed to UPDATE
            .mode(BuildAccelerationStructureModeKHR::BUILD)
            .ty(AccelerationStructureTypeKHR::BOTTOM_LEVEL);

        // based on incomplete_build_info get the sizes of the blas buffer to allocate and
        // of the scratch buffer that will be used for building the BLAS (and can then be discarded)
        let acceleration_structure_size_info = unsafe {
            let mut size_info = AccelerationStructureBuildSizesInfoKHR::default();

            acceleration_structure_device.get_acceleration_structure_build_sizes(
                AccelerationStructureBuildTypeKHR::DEVICE,
                &incomplete_build_info,
                &[index_buffer.len() as u32 / 3],
                &mut size_info,
            );

            size_info
        };

        // the vulkan buffer on which the BLAS will live
        let blas_buffer = vulkan_abstraction::Buffer::new::<u8>(
            device.clone(),
            acceleration_structure_size_info.acceleration_structure_size as usize,
            MemoryPropertyFlags::DEVICE_LOCAL,
            MemoryAllocateFlags::DEVICE_ADDRESS,
            BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | BufferUsageFlags::STORAGE_BUFFER,
            device_memory_props,
        )?;

        // information as to how to instantiate (but not "build") the BLAS in blas_buffer.
        let blas_create_info = AccelerationStructureCreateInfoKHR::default()
            .ty(incomplete_build_info.ty)
            .size(acceleration_structure_size_info.acceleration_structure_size)
            .buffer(*blas_buffer)
            .offset(0);

        // the actual BLAS object which lives on the blas_buffer, but has not been "built" yet
        let blas = unsafe {
            acceleration_structure_device.create_acceleration_structure(&blas_create_info, None)
        }
        .to_sr_result()?;

        // the scratch buffer that will be used for building the BLAS (and can be dropped afterwards)
        let scratch_buffer = vulkan_abstraction::Buffer::new::<u8>(
            device.clone(),
            acceleration_structure_size_info.build_scratch_size as usize,
            MemoryPropertyFlags::DEVICE_LOCAL,
            MemoryAllocateFlags::DEVICE_ADDRESS,
            BufferUsageFlags::SHADER_DEVICE_ADDRESS | BufferUsageFlags::STORAGE_BUFFER,
            device_memory_props,
        )?;

        // info for building the BLAS
        let build_info = incomplete_build_info
            .dst_acceleration_structure(blas)
            .scratch_data(DeviceOrHostAddressKHR {
                device_address: scratch_buffer.get_device_address(),
            });

        // one-shot command buffer which we will:
        // - fill with the commands to build the BLAS
        // - pass to the queue to be executed (thus building the BLAS)
        // - free
        let build_command_buffer = vulkan_abstraction::cmd_buffer::new(cmd_pool, device)?;

        //record build_command_buffer with the commands to build the BLAS
        unsafe {
            device
                .begin_command_buffer(
                    build_command_buffer,
                    &CommandBufferBeginInfo::default()
                        .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .to_sr_result()?;

            acceleration_structure_device.cmd_build_acceleration_structures(
                build_command_buffer,
                &[build_info],
                &[&[build_range_info]],
            );

            device
                .end_command_buffer(build_command_buffer)
                .to_sr_result()?
        }

        queue.submit_sync(build_command_buffer)?;

        // build_command_buffer must not be in a pending state when
        // free_command_buffers is called on it
        queue.wait_idle()?;

        unsafe {
            device.free_command_buffers(**cmd_pool, &[build_command_buffer]);
        }

        Ok(Self {
            blas,
            blas_buffer,
            acceleration_structure_device,
        })
    }

    #[allow(dead_code)]
    pub fn buffer(&self) -> &vulkan_abstraction::Buffer {
        &self.blas_buffer
    }
}
impl Drop for BLAS {
    fn drop(&mut self) {
        unsafe {
            self.acceleration_structure_device
                .destroy_acceleration_structure(self.blas, None);
        }
    }
}
impl Deref for BLAS {
    type Target = AccelerationStructureKHR;

    fn deref(&self) -> &Self::Target {
        &self.blas
    }
}
