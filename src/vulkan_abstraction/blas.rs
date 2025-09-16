use std::ops::Deref;
use std::rc::Rc;

use crate::error::*;
use crate::vulkan_abstraction;
use ash::vk;
use ash::{
    vk::{
        AccelerationStructureBuildGeometryInfoKHR, AccelerationStructureBuildRangeInfoKHR,
        AccelerationStructureBuildSizesInfoKHR, AccelerationStructureBuildTypeKHR,
        AccelerationStructureCreateInfoKHR, AccelerationStructureGeometryDataKHR,
        AccelerationStructureGeometryKHR, AccelerationStructureGeometryTrianglesDataKHR,
        AccelerationStructureKHR, AccelerationStructureTypeKHR, BufferUsageFlags,
        BuildAccelerationStructureFlagsKHR, BuildAccelerationStructureModeKHR,
        CommandBufferBeginInfo, CommandBufferUsageFlags, DeviceOrHostAddressConstKHR,
        DeviceOrHostAddressKHR, Format, GeometryFlagsKHR, GeometryTypeKHR, MemoryAllocateFlags,
        MemoryPropertyFlags,
    },
};

// Bottom-Level Acceleration Structure
pub struct BLAS {
    core: Rc<vulkan_abstraction::Core>,
    blas: AccelerationStructureKHR,
    #[allow(unused)]
    blas_buffer: vulkan_abstraction::Buffer,
    #[allow(unused)]
    transform_matrix_buffer: vulkan_abstraction::Buffer,
}

impl BLAS {
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
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

        #[rustfmt::skip]
        let transform_matrix = vk::TransformMatrixKHR {
            matrix: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0
            ],
        };

        let transform_matrix_buffer = vulkan_abstraction::Buffer::new_from_data(
            Rc::clone(&core),
            &[transform_matrix],
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryAllocateFlags::DEVICE_ADDRESS,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::TRANSFER_DST
        )?;

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
                    .transform_data(DeviceOrHostAddressConstKHR {
                        device_address: transform_matrix_buffer.get_device_address(),
                    }),
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

            core.acceleration_structure_device().get_acceleration_structure_build_sizes(
                AccelerationStructureBuildTypeKHR::DEVICE,
                &incomplete_build_info,
                &[index_buffer.len() as u32 / 3],
                &mut size_info,
            );

            size_info
        };

        // the vulkan buffer on which the BLAS will live
        let blas_buffer = vulkan_abstraction::Buffer::new::<u8>(
            Rc::clone(&core),
            acceleration_structure_size_info.acceleration_structure_size as usize,
            MemoryPropertyFlags::DEVICE_LOCAL,
            MemoryAllocateFlags::DEVICE_ADDRESS,
            BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | BufferUsageFlags::STORAGE_BUFFER,
        )?;

        // information as to how to instantiate (but not "build") the BLAS in blas_buffer.
        let blas_create_info = AccelerationStructureCreateInfoKHR::default()
            .ty(incomplete_build_info.ty)
            .size(acceleration_structure_size_info.acceleration_structure_size)
            .buffer(blas_buffer.inner())
            .offset(0)
            .create_flags(vk::AccelerationStructureCreateFlagsKHR::empty());

        // the actual BLAS object which lives on the blas_buffer, but has not been "built" yet
        let blas = unsafe {
            core.acceleration_structure_device().create_acceleration_structure(&blas_create_info, None)
        }?;

        // the scratch buffer that will be used for building the BLAS (and can be dropped afterwards)
        let scratch_buffer = vulkan_abstraction::Buffer::new::<u8>(
            Rc::clone(&core),
            acceleration_structure_size_info.build_scratch_size as usize,
            MemoryPropertyFlags::DEVICE_LOCAL,
            MemoryAllocateFlags::DEVICE_ADDRESS,
            BufferUsageFlags::SHADER_DEVICE_ADDRESS | BufferUsageFlags::STORAGE_BUFFER,
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
        let build_command_buffer = vulkan_abstraction::cmd_buffer::new(core.cmd_pool(), core.device())?;

        //record build_command_buffer with the commands to build the BLAS
        unsafe {
            core.device().inner().begin_command_buffer(
                build_command_buffer,
                &CommandBufferBeginInfo::default()
                    .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            core.acceleration_structure_device().cmd_build_acceleration_structures(
                build_command_buffer,
                &[build_info],
                &[&[build_range_info]],
            );

            core.device().inner().end_command_buffer(build_command_buffer)?
        }

        core.queue().submit_sync(build_command_buffer)?;

        // build_command_buffer must not be in a pending state when
        // free_command_buffers is called on it
        core.queue().wait_idle()?;

        unsafe {
            core.device().inner().free_command_buffers(**core.cmd_pool(), &[build_command_buffer]);
        }

        Ok(Self {
            core,
            blas,
            blas_buffer,
            transform_matrix_buffer,
        })
    }

    pub fn inner(&self) -> AccelerationStructureKHR { self.blas }
}
impl Drop for BLAS {
    fn drop(&mut self) {
        unsafe {
            self.core.acceleration_structure_device()
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
