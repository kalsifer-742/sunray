use std::hash::Hash;
use std::rc::Rc;

use crate::error::*;
use crate::vulkan_abstraction;
use crate::vulkan_abstraction::{
    AccelerationStructure, AsBuildInputs, Buffer, CompactionQueryPool, IndexBuffer, ResourceManager, VertexBuffer,
};
use ash::vk;

// ─── Build description (plain data — no handles, no lifetimes) ────────────────

/// A single triangle geometry, described purely by device addresses + formats +
/// counts. Holds no buffer handles, so it can be stored and re-realized freely.
#[derive(Clone, Debug)]
pub struct TriangleGeometryDesc {
    pub vertex_address: vk::DeviceAddress,
    pub vertex_stride: u64,
    pub vertex_format: vk::Format,
    pub max_vertex: u32,
    pub index_address: vk::DeviceAddress,
    pub index_type: vk::IndexType,
    pub primitive_count: u32,
    pub flags: vk::GeometryFlagsKHR,
}

/// Where a BLAS geometry's data comes from. Triangles today; clustered geometry
/// (NVIDIA CLAS) later, fed as a list of CLAS device addresses.
#[derive(Clone, Debug)]
pub enum GeometrySource {
    Triangles(TriangleGeometryDesc),
    // Clusters { clas_refs: Vec<vk::DeviceAddress> }, // future CLAS path (VK_NV_cluster_acceleration_structure)
}

/// Plain-data description of a BLAS build: its geometries and build flags. No
/// handles and no lifetimes — realized into the transient `vk::*` structs only
/// at build time via [`BlasDesc::realize`].
#[derive(Clone, Debug)]
pub struct BlasDesc {
    pub geometries: Vec<GeometrySource>,
    pub flags: vk::BuildAccelerationStructureFlagsKHR,
}

impl BlasDesc {
    /// Realize this description into the transient geometry + range arrays an
    /// [`AsBuildInputs`] needs. The arrays own no borrowed data (every field is
    /// a device address / scalar), so they are `'static`. //TODO this is unsafe since it loses the lifetime of the original data
    unsafe fn realize(
        &self,
    ) -> (
        Vec<vk::AccelerationStructureGeometryKHR<'static>>,
        Vec<vk::AccelerationStructureBuildRangeInfoKHR>,
    ) {
        let mut geometries = Vec::with_capacity(self.geometries.len());
        let mut ranges = Vec::with_capacity(self.geometries.len());
        
      
        
        for source in &self.geometries {
            match source {
                GeometrySource::Triangles(tri) => {
                    let geometry_data = vk::AccelerationStructureGeometryDataKHR {
                        triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                                device_address: tri.vertex_address,
                            })
                            .max_vertex(tri.max_vertex)
                            .vertex_stride(tri.vertex_stride)
                            .vertex_format(tri.vertex_format)
                            .index_data(vk::DeviceOrHostAddressConstKHR {
                                device_address: tri.index_address,
                            })
                            .index_type(tri.index_type),
                    };

                    geometries.push(
                        vk::AccelerationStructureGeometryKHR::default()
                            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                            .geometry(geometry_data)
                            .flags(tri.flags),
                    );

                    ranges.push(
                        vk::AccelerationStructureBuildRangeInfoKHR::default()
                            // the value of first_vertex is added to index values before fetching verts
                            .first_vertex(0u32)
                            // number of triangles (indices / 3)
                            .primitive_count(tri.primitive_count)
                            // byte offset into the index data
                            .primitive_offset(0u32)
                            // byte offset into the (unused) transform data
                            .transform_offset(0),
                    );
                } // GeometrySource::Clusters { clas_refs } => { /* realize CLAS geometry from the leaf addresses */ }
            }
        }

        (geometries, ranges)
    }
}

// ─── BLAS rebuild policy (dormant scaffolding — kept as-is) ───────────────────

pub enum BlasState {
    Optimal,
    Changing(Dynamic),
}

pub struct Dynamic {
    // when it changes it goes into a fast rebuild or update and after 30 frames unchanged it goes into a slow rebuild
    #[allow(dead_code)]
    frame_since_last_update_or_fast_rebuild: u32,
    #[allow(dead_code)]
    number_of_updates_and_fast_rebuilds: u32,
}

// ─── BLAS geometry ownership ──────────────────────────────────────────────────

/// The live GPU buffers a BLAS is built from. They must outlive the BLAS (a
/// rebuild / update reads them), so the BLAS owns them here.
pub struct BlasGeometry {
    vertex_buffer: VertexBuffer,
    index_buffer: IndexBuffer,
}

// ─── BLAS ─────────────────────────────────────────────────────────────────────

/// Bottom-Level Acceleration Structure: the resource ([`AccelerationStructure`]),
/// the geometry buffers it was built from, its plain-data build description, and
/// its rebuild-policy state.
pub struct Blas {
    accel: AccelerationStructure,
    geometry: BlasGeometry,
    desc: BlasDesc,
    #[allow(dead_code)]
    state: BlasState,
}

impl Blas {
    /// the vertex_buffer is assumed to have a vec3 position attribute as its first (not necessarily the only) attribute in memory.
    /// Emissive triangles are no longer tracked here — the `ResourceManager` owns
    /// the per-BLAS emissive triangle slots.
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        vertex_buffer: VertexBuffer,
        index_buffer: IndexBuffer,
        fast_build: bool,
    ) -> SrResult<Self> {
        // PREFER_FAST_BUILD -> prioritize build time; PREFER_FAST_TRACE ->
        // prioritize trace performance. Matches the pre-rework default flags.
        let flags = if fast_build {
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD
        } else {
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
        };
        Self::new_with_build_flags(core, vertex_buffer, index_buffer, flags)
    }

    /// Build a BLAS with an explicit set of build flags. Use this to opt into
    /// `ALLOW_COMPACTION` (required before [`Self::record_compaction`] /
    /// [`Self::compact_sync`]) or `ALLOW_UPDATE` (required before [`Self::update`]).
    pub fn new_with_build_flags(
        core: Rc<vulkan_abstraction::Core>,
        vertex_buffer: VertexBuffer,
        index_buffer: IndexBuffer,
        flags: vk::BuildAccelerationStructureFlagsKHR,
    ) -> SrResult<Self> {
        let desc = BlasDesc {
            geometries: vec![GeometrySource::Triangles(Self::triangle_desc(&vertex_buffer, &index_buffer))],
            flags,
        };

        let (geometries, ranges) = unsafe { desc.realize() };
        let accel = AccelerationStructure::build_sync(
            core,
            &AsBuildInputs {
                ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                flags: desc.flags,
                geometries: &geometries,
                ranges: &ranges,
            },
        )?;

        Ok(Self {
            accel,
            geometry: BlasGeometry {
                vertex_buffer,
                index_buffer,
            },
            desc,
            state: BlasState::Optimal,
        })
    }

    /// Build a [`TriangleGeometryDesc`] from a vertex + index buffer, using the
    /// fixed geometry flags the renderer's triangle meshes use.
    fn triangle_desc(vertex_buffer: &VertexBuffer, index_buffer: &IndexBuffer) -> TriangleGeometryDesc {
        TriangleGeometryDesc {
            vertex_address: vertex_buffer.get_device_address(),
            vertex_stride: vertex_buffer.stride() as u64,
            vertex_format: vk::Format::R32G32B32_SFLOAT,
            max_vertex: vertex_buffer.len() as u32 - 1,
            index_address: index_buffer.get_device_address(),
            index_type: index_buffer.index_type(),
            primitive_count: (index_buffer.len() / 3) as u32,
            //TODO why always opaque?
            flags: vk::GeometryFlagsKHR::OPAQUE | vk::GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION,
        }
    }

    pub fn state(&self) -> &BlasState {
        &self.state
    }

    #[allow(unused)]
    pub fn rebuild(&mut self, vertex_buffer: VertexBuffer, index_buffer: IndexBuffer, fast_build: bool) -> SrResult<()> {
        *self = Self::new(Rc::clone(self.accel.core()), vertex_buffer, index_buffer, fast_build)?;
        log::debug!("BLAS rebuilt");
        Ok(())
    }

    #[allow(unused)]
    pub fn update(&mut self, vertex_buffer: VertexBuffer, index_buffer: IndexBuffer) -> SrResult<()> {
        if !self.desc.flags.contains(vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE) {
            return Err(SrError::new_custom("The structure is not updatable".to_string()));
        }

        // Same geometry count / layout, new buffer contents.
        let desc = BlasDesc {
            geometries: vec![GeometrySource::Triangles(Self::triangle_desc(&vertex_buffer, &index_buffer))],
            flags: self.desc.flags,
        };
        let (geometries, ranges) = unsafe {desc.realize()} ;
        self.accel.update_sync(&AsBuildInputs {
            ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            flags: desc.flags,
            geometries: &geometries,
            ranges: &ranges,
        })?;

        self.geometry = BlasGeometry {
            vertex_buffer,
            index_buffer,
        };
        self.desc = desc;
        Ok(())
    }

    // ─── Compaction ──────────────────────────────────────────────────────────
    //
    // Granular path (caller picks timing / queue):
    //   1. build with `ALLOW_COMPACTION` (`new_with_build_flags`)
    //   2. `pool.cmd_reset_and_query(cmd_buf, &[blas.accel()])`; submit; `pool.read_size(i)`
    //   3. `let old = blas.record_compaction(cmd_buf, size)?`; submit; drop `old`
    //      once that submission completes.

    /// The underlying resource — pass `blas.accel()` to
    /// [`CompactionQueryPool::cmd_reset_and_query`].
    pub fn accel(&self) -> &AccelerationStructure {
        &self.accel
    }

    /// Record a COMPACT copy into a minimum-sized buffer and swap it in,
    /// returning the **pre-compaction** structure. The returned structure backs
    /// the recorded copy, so the caller must keep it alive until `cmd_buf`'s
    /// submission completes, then drop it. Requires the BLAS was built with
    /// `ALLOW_COMPACTION`.
    pub fn record_compaction(
        &mut self,
        cmd_buf: vk::CommandBuffer,
        compacted_size: vk::DeviceSize,
    ) -> SrResult<AccelerationStructure> {
        if !self
            .desc
            .flags
            .contains(vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION)
        {
            return Err(SrError::new_custom(
                "record_compaction requires the BLAS was built with ALLOW_COMPACTION".to_string(),
            ));
        }

        let compacted =
            self.accel
                .record_compact_copy(cmd_buf, vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL, compacted_size)?;
        Ok(std::mem::replace(&mut self.accel, compacted))
    }

    /// Convenience: run the whole query → copy compaction round-trip
    /// synchronously on the graphics queue. The pre-compaction structure is
    /// safe to drop as soon as the (waited-on) copy submit returns, so this
    /// returns nothing. Requires the BLAS was built with `ALLOW_COMPACTION`.
    #[allow(unused)]
    pub fn compact_sync(&mut self) -> SrResult<()> {
        if !self
            .desc
            .flags
            .contains(vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION)
        {
            return Err(SrError::new_custom(
                "compact_sync requires the BLAS was built with ALLOW_COMPACTION".to_string(),
            ));
        }

        let core = Rc::clone(self.accel.core());
        let pool = CompactionQueryPool::new(Rc::clone(&core), 1)?;

        // The BLAS build already completed synchronously (queue idle), so the
        // size query can read it without an extra barrier.
        let query_cmd = vulkan_abstraction::cmd_buffer::new_command_buffer(core.graphics_cmd_pool(), core.device().inner())?;
        unsafe {
            core.device().inner().begin_command_buffer(
                query_cmd,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        pool.cmd_reset_and_query(query_cmd, &[&self.accel]);
        unsafe { core.device().inner().end_command_buffer(query_cmd)? }
        core.graphics_queue().submit_sync(query_cmd)?;
        unsafe {
            core.device()
                .inner()
                .free_command_buffers(core.graphics_cmd_pool().inner(), &[query_cmd]);
        }

        let compacted_size = pool.read_size(0)?;

        let copy_cmd = vulkan_abstraction::cmd_buffer::new_command_buffer(core.graphics_cmd_pool(), core.device().inner())?;
        unsafe {
            core.device().inner().begin_command_buffer(
                copy_cmd,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        let old = self.record_compaction(copy_cmd, compacted_size)?;
        unsafe { core.device().inner().end_command_buffer(copy_cmd)? }
        core.graphics_queue().submit_sync(copy_cmd)?;
        unsafe {
            core.device()
                .inner()
                .free_command_buffers(core.graphics_cmd_pool().inner(), &[copy_cmd]);
        }

        // The copy submit was waited on, so the source is no longer in use.
        drop(old);
        Ok(())
    }

    // ─── Accessors ───────────────────────────────────────────────────────────

    pub fn inner(&self) -> vk::AccelerationStructureKHR {
        self.accel.inner()
    }

    /// `vkGetAccelerationStructureDeviceAddressKHR` of this BLAS (cached).
    pub fn device_address(&self) -> vk::DeviceAddress {
        self.accel.device_address()
    }

    pub fn vertex_buffer(&self) -> &VertexBuffer {
        &self.geometry.vertex_buffer
    }

    pub fn index_buffer(&self) -> &IndexBuffer {
        &self.geometry.index_buffer
    }
}

// ─── Instances ────────────────────────────────────────────────────────────────

/// Plain `Copy` description of one TLAS instance. References its BLAS by the
/// existing resource-manager key `K` — never a borrow, never a new id, and the
/// BLAS device address is resolved **late**, at [`InstanceDesc::lower`] time.
#[derive(Copy, Clone)]
pub struct InstanceDesc<K> {
    /// The resource-manager key of the referenced BLAS.
    pub blas: K,
    pub transform: vk::TransformMatrixKHR,
    /// → `gl_InstanceCustomIndexEXT` (was `blas_instance_index`).
    pub custom_index: u32,
    pub mask: u8,
    pub sbt_offset: u32,
    pub flags: vk::GeometryInstanceFlagsKHR,
    // pub partition: PartitionId, // reserved for PTLAS (VK_NV_partitioned_acceleration_structure);
    //                             // PartitionId would be a plain-data index type (e.g. `pub type PartitionId = u32;`).
}

impl<K: Hash + Eq + Copy> InstanceDesc<K> {
    /// Lower to the packed `vk::AccelerationStructureInstanceKHR`. Resolves the
    /// BLAS device address **here**, through the resource manager — so a slow
    /// rebuild that reallocated the BLAS's backing buffer is picked up with zero
    /// patching. The address is never stored on the instance.
    ///
    /// Resolution goes key → device address (not key → handle), so a future
    /// handle-less cluster BLAS fits this path unchanged.
    pub(crate) fn lower(&self, res: &ResourceManager<K>) -> vk::AccelerationStructureInstanceKHR {
        let address = res
            .blas_device_address(self.blas)
            .expect("InstanceDesc::lower: instance references a BLAS key that was never loaded");

        vk::AccelerationStructureInstanceKHR {
            transform: self.transform,
            // mask = 0xFF: "Only be hit if rayMask & instance.mask != 0".
            instance_custom_index_and_mask: vk::Packed24_8::new(self.custom_index, self.mask),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                self.sbt_offset,
                self.flags.as_raw() as u8,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR { device_handle: address },
        }
    }
}
