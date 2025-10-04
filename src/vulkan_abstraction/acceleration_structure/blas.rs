use std::rc::Rc;

use crate::error::*;
use crate::vulkan_abstraction;
use ash::vk;

pub struct BlasInstance<'a> {
    pub blas: &'a vulkan_abstraction::BLAS,
    pub transform: vk::TransformMatrixKHR,
    pub blas_instance_index: u32, // blas custom index, contains the index of the instance, NOT of the blas, so we can fetch instance-specific information in the shader
}

// Bottom-Level Acceleration Structure
pub struct BLAS {
    blas: vulkan_abstraction::AccelerationStructure,
    #[allow(unused)]
    vertex_buffer: vulkan_abstraction::VertexBuffer,
    #[allow(unused)]
    index_buffer: vulkan_abstraction::IndexBuffer,
}

impl BLAS {
    /// the vertex_buffer is assumed to have a vec3 position attribute as its first (not necessarily the only) attribute in memory
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        vertex_buffer: vulkan_abstraction::VertexBuffer,
        index_buffer: vulkan_abstraction::IndexBuffer,
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
                    .index_type(index_buffer.index_type()),
            };
            // .transform_data(vk::DeviceOrHostAddressConstKHR { device_address: transform_buffer.get_device_address() })

            vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .geometry(geometry_data)
                .flags(vk::GeometryFlagsKHR::OPAQUE | vk::GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION)
        };

        // specify the range of values to read from the ibo, vbo and transform data of a geometry.
        // there must be one build_range_info for each geometry
        let build_range_info = vk::AccelerationStructureBuildRangeInfoKHR::default()
            // the value of first_vertex is added to index values before fetching verts
            .first_vertex(0 as u32)
            // the number of triangles to read (3 * the number of indices to read)
            .primitive_count((index_buffer.len() / 3) as u32)
            // an offset (in bytes) into geometry.geometry_data.index_data from which to start reading
            .primitive_offset(0 as u32)
            // transform_offset is an offset (in bytes) into geometry.geometry_data.transform_data
            .transform_offset(0);

        let blas = vulkan_abstraction::AccelerationStructure::new(
            core,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            &[build_range_info],
            &[geometry],
            false,
        )?;

        Ok(Self {
            blas,
            vertex_buffer,
            index_buffer,
        })
    }

    pub fn inner(&self) -> vk::AccelerationStructureKHR {
        self.blas.inner()
    }

    pub fn vertex_buffer(&self) -> &vulkan_abstraction::VertexBuffer {
        &self.vertex_buffer
    }

    pub fn index_buffer(&self) -> &vulkan_abstraction::IndexBuffer {
        &self.index_buffer
    }
}
