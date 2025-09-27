use std::rc::Rc;

use ash::vk;

use crate::{error::SrResult, vulkan_abstraction};

#[derive(Clone, Copy)]
pub struct Vertex {
    #[allow(unused)]
    pub position: [f32; 3],
}

pub struct Mesh {
    pub vertex_offset: usize,
    pub index_offset: usize,
    pub index_count: usize,
}

pub struct Model {
    vertex_buffer: vulkan_abstraction::VertexBuffer,
    index_buffer: vulkan_abstraction::IndexBuffer,
    meshes: Vec<vulkan_abstraction::Mesh>,
    transforms: Vec<vk::TransformMatrixKHR>,
}

impl Model {
    pub fn new(
        core: &Rc<vulkan_abstraction::Core>,
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
        meshes: Vec<vulkan_abstraction::Mesh>,
        transforms: Vec<vk::TransformMatrixKHR>,
    ) -> SrResult<Self> {
        let vertex_buffer = {
            let staging_buffer = vulkan_abstraction::Buffer::new_staging_from_data::<Vertex>(
                Rc::clone(&core),
                &vertices,
            )?;

            let vertex_buffer = vulkan_abstraction::VertexBuffer::new_for_blas::<Vertex>(
                Rc::clone(&core),
                vertices.len(),
            )?;
            vulkan_abstraction::Buffer::clone_buffer(&core, &staging_buffer, &vertex_buffer)?;

            vertex_buffer
        };

        let index_buffer = {
            let staging_buffer = vulkan_abstraction::Buffer::new_staging_from_data::<u32>(
                Rc::clone(&core),
                &indices,
            )?;
            let index_buffer = vulkan_abstraction::IndexBuffer::new_for_blas::<u32>(
                Rc::clone(&core),
                indices.len(),
            )?;
            vulkan_abstraction::Buffer::clone_buffer(&core, &staging_buffer, &index_buffer)?;

            index_buffer
        };

        Ok(Self {
            vertex_buffer,
            index_buffer,
            meshes,
            transforms,
        })
    }

    pub fn vertex_buffer(&self) -> &vulkan_abstraction::VertexBuffer {
        &self.vertex_buffer
    }

    pub fn index_buffer(&self) -> &vulkan_abstraction::IndexBuffer {
        &self.index_buffer
    }

    pub fn meshes(&self) -> &[vulkan_abstraction::Mesh] {
        &self.meshes
    }

    pub fn transforms(&self) -> &[vk::TransformMatrixKHR] {
        &self.transforms
    }
}
