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
        model: &vulkan_abstraction::Model,
    ) -> SrResult<Self> {
        /*
         * Building the BLAS is mostly a 3 step process (with some complications):
         * 1.  Allocate a GPU Buffer on which it will live (blas_buffer)
         *     and a scratch buffer used only for step 3
         * 2.  Create a BLAS handle (blas) pointing to this allocation
         * 3.  Build the actual BLAS data structure
         */

        //it is, as far as I can tell, ok to drop the transform buffer after constructing the BLAS
        let transform_matrices_buffer = vulkan_abstraction::Buffer::new_from_data(
            Rc::clone(&core),
            model.transforms(),
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryAllocateFlags::DEVICE_ADDRESS,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
        )?;

        let geometry_data = vk::AccelerationStructureGeometryDataKHR {
            triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                .vertex_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: model.vertex_buffer().get_device_address(),
                })
                .max_vertex(model.vertex_buffer().len() as u32 - 1)
                .vertex_stride(model.vertex_buffer().stride() as u64)
                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                .index_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: model.index_buffer().get_device_address(),
                })
                .index_type(model.index_buffer().index_type())
                .transform_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: transform_matrices_buffer.get_device_address(),
                }),
        };

        let mut geometries = Vec::new();
        let mut build_range_infos = Vec::new();
        for (i, mesh) in model.meshes().iter().enumerate() {
            // specify what the BLAS's geometry (vbo, ibo) is
            let geometry = {
                vk::AccelerationStructureGeometryKHR::default()
                    .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                    .geometry(geometry_data)
                    .flags(vk::GeometryFlagsKHR::OPAQUE)
            };
            geometries.push(geometry);

            // specify the range of values to read from the ibo, vbo and transform data of a geometry.
            // there must be one build_range_info for each geometry
            let build_range_info = vk::AccelerationStructureBuildRangeInfoKHR::default()
                // the value of first_vertex is added to index values before fetching verts
                .first_vertex(mesh.vertex_offset as u32)
                // the number of triangles to read (3 * the number of indices to read)
                .primitive_count((mesh.index_count / 3) as u32)
                // an offset (in bytes) into geometry.geometry_data.index_data from which to start reading
                .primitive_offset((mesh.index_offset * std::mem::size_of::<u32>()) as u32)
                // transform_offset is an offset (in bytes) into geometry.geometry_data.transform_data
                .transform_offset((i * size_of::<vk::TransformMatrixKHR>()) as u32); //TODO: calculate transform offset
            build_range_infos.push(build_range_info);
        }

        let blas = vulkan_abstraction::AccelerationStructure::new(
            core,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            &build_range_infos,
            &geometries,
            false,
        )?;

        Ok(Self { blas })
    }

    pub fn inner(&self) -> vk::AccelerationStructureKHR {
        self.blas.inner()
    }
}
