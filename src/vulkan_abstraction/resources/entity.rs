use crate::vulkan_abstraction::Material;
use ash::vk;

/// Per-BLAS data uploaded to GPU and read by shaders. Stored in the meshes-info
/// arena buffer; the slot index is what every instance of that BLAS passes as
/// `gl_InstanceCustomIndexEXT`, so instances sharing a BLAS share one entry.
#[derive(Clone, Copy)]
#[repr(C, packed)]
pub(crate) struct EntityGpuData {
    pub(crate) vertex_buffer: vk::DeviceAddress,
    pub(crate) index_buffer: vk::DeviceAddress,
    pub(crate) material: Material,
}
