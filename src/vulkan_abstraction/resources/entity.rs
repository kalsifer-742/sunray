use crate::vulkan_abstraction::Material;
use ash::vk;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EntityId(pub u64);

/// CPU-side entity metadata. Not uploaded to GPU — used for TLAS rebuilds,
/// emissive indirection, and reconstructing EntityGpuData on transform updates.Contains all the info needed to build a blas instance
#[derive(Copy, Clone)]
pub struct Entity {
    pub id: EntityId,
    /// Index into Renderer's blases vec (shared geometry).
    pub blas_index: u64,
    /// Index into Renderer's instances buffer.
    pub blas_instance_index: u64,

    /// GPU-ready material (kept CPU-side to reconstruct EntityGpuData on updates).
    pub material: Material,

    pub transform: vk::TransformMatrixKHR,
}

/// Per-entity data uploaded to GPU and read by shaders.
/// Stored in the entities arena buffer (indexed by arena slot =
/// `gl_InstanceCustomIndexEXT`).
#[derive(Clone, Copy)]
#[repr(C, packed)]
pub(crate) struct EntityGpuData {
    pub(crate) vertex_buffer: vk::DeviceAddress,
    pub(crate) index_buffer: vk::DeviceAddress,
    pub(crate) material: Material,
}
