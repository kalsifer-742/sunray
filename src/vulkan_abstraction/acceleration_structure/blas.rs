use std::rc::Rc;

use crate::error::*;
use crate::vulkan_abstraction;
use ash::vk;

// Bottom-Level Acceleration Structure
pub struct BLAS {
    blas: vulkan_abstraction::AccelerationStructure,
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

        let transform_matrix = vulkan_abstraction::IDENTITY_MATRIX;

        //it is, as far as I can tell, ok to drop the transform buffer after constructing the BLAS
        let transform_matrix_buffer = vulkan_abstraction::Buffer::new_from_data(
            Rc::clone(&core),
            &[transform_matrix],
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryAllocateFlags::DEVICE_ADDRESS,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::TRANSFER_DST
        )?;

        // specify what the BLAS's geometry (vbo, ibo) is
        let geometry = {
            let geometry_data = vk::AccelerationStructureGeometryDataKHR {
                triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                    .vertex_data(vk::DeviceOrHostAddressConstKHR {
                        device_address: vertex_buffer.get_device_address(),
                    })
                    .max_vertex(vertex_buffer.len() as u32 - 1)
                    .vertex_stride(vertex_buffer.stride() as u64)
                    .vertex_format(vk::Format::R32G32B32_SFLOAT)
                    .index_data(vk::DeviceOrHostAddressConstKHR {
                        device_address: index_buffer.get_device_address(),
                    })
                    .index_type(index_buffer.index_type())
                    .transform_data(vk::DeviceOrHostAddressConstKHR {
                        device_address: transform_matrix_buffer.get_device_address(),
                    }),
            };

            vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .geometry(geometry_data)
                .flags(vk::GeometryFlagsKHR::OPAQUE)
        };
        let geometries = [geometry];

        // specify the range of values to read from the ibo, vbo and transform data of a geometry.
        // there must be one build_range_info for each geometry
        let build_range_info =
            vk::AccelerationStructureBuildRangeInfoKHR::default()
                // the number of triangles to read (3 * the number of indices to read)
                .primitive_count(index_buffer.len() as u32 / 3)
                // an offset (in bytes) into geometry.geometry_data.index_data from which to start reading
                .primitive_offset(0)
                // the value of first_vertex is added to index values before fetching verts
                .first_vertex(0)
                // transform_offset is an offset (in bytes) into geometry.geometry_data.transform_data
                .transform_offset(0);
        let build_range_infos = [build_range_info];

        let blas = vulkan_abstraction::AccelerationStructure::new(core, vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL, &build_range_infos, &geometries, false)?;

        Ok(Self {
            blas,
        })
    }

    pub fn inner(&self) -> vk::AccelerationStructureKHR { self.blas.inner() }
}
