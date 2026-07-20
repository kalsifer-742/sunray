use std::rc::Rc;

use crate::error::*;
use crate::vulkan_abstraction;
use crate::vulkan_abstraction::{Buffer, IndexBuffer, VertexBuffer};
use ash::vk;
use ash::vk::AccelerationStructureBuildRangeInfoKHR;

pub struct BlasInstance<'a> {
    pub blas: &'a vulkan_abstraction::BLAS,
    pub transform: vk::TransformMatrixKHR,
    pub blas_instance_index: u32, // contains the index of the instance, NOT of the blas, so we can fetch instance-specific information in the shader, by passing it as gl_InstanceCustomIndexEXT
}

//TODO this is cause I can't self reference the blas
pub struct BlasMetaData {
    pub transform: vk::TransformMatrixKHR,
    pub blas_instance_index: u32,
}

pub enum BlasState {
    Optimal,
    Changing(Dynamic),
}

pub struct Dynamic {
    // when it changes it goes into a fast rebuild or update and after 30 frames unchanged it goes into a slow rebuild
    frame_since_last_update_or_fast_rebuild: u32,
    number_of_updates_and_fast_rebuilds: u32,
}

// Bottom-Level Acceleration Structure
pub struct BLAS {
    blas: vulkan_abstraction::AccelerationStructure,
    #[allow(unused)]
    vertex_buffer: vulkan_abstraction::VertexBuffer,
    #[allow(unused)]
    index_buffer: vulkan_abstraction::IndexBuffer,

    /// Whether the geometry is flagged `VK_GEOMETRY_OPAQUE_BIT_KHR`. False only
    /// for alpha-cutout (glTF MASK) geometry, which must invoke the any-hit
    /// shader; opaque geometry stays on the traversal fast path. Retained so
    /// `rebuild` reproduces the same flag.
    opaque: bool,

    state: BlasState,
}

impl BLAS {
    /// the vertex_buffer is assumed to have a vec3 position attribute as its first (not necessarily the only) attribute in memory.
    /// Emissive triangles are no longer tracked here — the `ResourceManager` owns
    /// the per-BLAS emissive triangle slots.
    /// `opaque` maps to `VK_GEOMETRY_OPAQUE_BIT_KHR`: pass `false` only for
    /// alpha-cutout (glTF MASK) meshes so the any-hit shader runs on them; `true`
    /// for everything else keeps them on the fixed-function traversal fast path.
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        vertex_buffer: vulkan_abstraction::VertexBuffer,
        index_buffer: vulkan_abstraction::IndexBuffer,
        opaque: bool,
        fast_build: bool,
    ) -> SrResult<Self> {
        /*
         * Building the BLAS is mostly a 3 step process (with some complications):
         * 1.  Allocate a GPU Buffer on which it will live (blas_buffer)
         *     and a scratch buffer used only for step 3
         * 2.  Create a BLAS handle (blas) pointing to this allocation
         * 3.  Build the actual BLAS data structure
         */

        // specify what the BLAS's geometry (vbo, ibo) is
        let geometry = Self::make_geometry(&vertex_buffer, &index_buffer, opaque);

        // specify the range of values to read from the ibo, vbo and transform data of a geometry.
        // there must be one build_range_info for each geometry
        let build_range_info = Self::make_build_range_info(&index_buffer);

        let blas = vulkan_abstraction::AccelerationStructure::new(
            core,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            &[build_range_info],
            &[geometry],
            false,
            fast_build,
        )?;

        Ok(Self {
            blas,
            vertex_buffer,
            index_buffer,
            opaque,
            state: BlasState::Optimal,
        })
    }

    pub fn state(&self) -> &BlasState {
        &self.state
    }

    fn make_geometry<'a>(
        vertex_buffer: &VertexBuffer,
        index_buffer: &IndexBuffer,
        opaque: bool,
    ) -> vk::AccelerationStructureGeometryKHR<'a> {
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

        // Opaque geometry is committed by fixed-function traversal (no any-hit).
        // Cutout geometry drops OPAQUE so the any-hit alpha test runs, and keeps
        // NO_DUPLICATE_ANY_HIT_INVOCATION so a given triangle is tested at most
        // once per ray (the test is a pure function of the hit, so dedup is safe).
        let flags = if opaque {
            vk::GeometryFlagsKHR::OPAQUE
        } else {
            vk::GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION
        };

        vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(geometry_data)
            .flags(flags)
    }
    #[allow(unused)]
    pub fn rebuild(
        &mut self,
        vertex_buffer: vulkan_abstraction::VertexBuffer,
        index_buffer: vulkan_abstraction::IndexBuffer,
        fast_build: bool,
    ) -> SrResult<()> {
        let geometry = Self::make_geometry(&vertex_buffer, &index_buffer, self.opaque);

        let build_range_info = Self::make_build_range_info(&index_buffer);

        self.blas.rebuild(&[build_range_info], &[geometry], fast_build)?;

        Ok(())
    }

    fn make_build_range_info(index_buffer: &IndexBuffer) -> AccelerationStructureBuildRangeInfoKHR {
        vk::AccelerationStructureBuildRangeInfoKHR::default()
            // the value of first_vertex is added to index values before fetching verts
            .first_vertex(0u32)
            // the number of triangles to read (3 * the number of indices to read)
            .primitive_count((index_buffer.len() / 3) as u32)
            // an offset (in bytes) into geometry.geometry_data.index_data from which to start reading
            .primitive_offset(0u32)
            // transform_offset is an offset (in bytes) into geometry.geometry_data.transform_data
            .transform_offset(0)
    }

    #[allow(unused)]
    pub fn update(
        &mut self,
        vertex_buffer: vulkan_abstraction::VertexBuffer,
        index_buffer: vulkan_abstraction::IndexBuffer,
    ) -> SrResult<()> {
        if !self.blas.allow_update {
            return SrResult::Err(SrError::new_custom("The structure is not updatable".to_string()));
        }

        let geometry = Self::make_geometry(&vertex_buffer, &index_buffer, self.opaque);

        let build_range_info = Self::make_build_range_info(&index_buffer);

        self.blas.update(&[build_range_info], &[geometry])?;

        Ok(())
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
