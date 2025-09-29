use std::rc::Rc;

use ash::vk;

use crate::{
    error::SrResult,
    vulkan_abstraction::{self},
};

#[rustfmt::skip]
pub const IDENTITY_MATRIX : vk::TransformMatrixKHR = vk::TransformMatrixKHR {
    matrix: [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0
    ],
};

pub struct Node {
    transform: vk::TransformMatrixKHR,
    mesh: Option<vulkan_abstraction::Mesh>,
    children: Option<Vec<vulkan_abstraction::Node>>,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            transform: vulkan_abstraction::IDENTITY_MATRIX,
            mesh: None,
            children: None,
        }
    }
}

impl Node {
    pub fn new(
        transform: vk::TransformMatrixKHR,
        mesh: Option<vulkan_abstraction::Mesh>,
        children: Option<Vec<Node>>,
    ) -> SrResult<Self> {
        Ok(Self {
            transform,
            mesh,
            children,
        })
    }

    pub fn transform(&self) -> &vk::TransformMatrixKHR {
        &self.transform
    }

    pub fn mesh(&self) -> &Option<vulkan_abstraction::Mesh> {
        &self.mesh
    }

    pub fn children(&self) -> &Option<Vec<vulkan_abstraction::Node>> {
        &self.children
    }

    pub fn load_into_gpu_memory(
        &self,
        core: &Rc<vulkan_abstraction::Core>,
    ) -> SrResult<(
        vulkan_abstraction::Buffer,
        Option<vulkan_abstraction::VertexBuffer>,
        Option<vulkan_abstraction::IndexBuffer>,
    )> {
        let transform_buffer = vulkan_abstraction::Buffer::new_from_data(
            Rc::clone(&core),
            &[self.transform],
            gpu_allocator::MemoryLocation::GpuOnly,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
            "node BLAS transform buffer"
        )?;
        if let Some(mesh) = &self.mesh {
            let vertex_buffer = vulkan_abstraction::VertexBuffer::new_for_blas_from_data(Rc::clone(&core), mesh.vertices())?;
            let index_buffer = vulkan_abstraction::IndexBuffer::new_for_blas_from_data::<u32>(Rc::clone(&core), mesh.indices())?;

            return Ok((transform_buffer, Some(vertex_buffer), Some(index_buffer)));
        }

        Ok((transform_buffer, None, None))
    }
}
