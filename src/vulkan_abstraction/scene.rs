use std::rc::Rc;

use crate::{error::SrResult, vulkan_abstraction};

#[derive(Clone, Copy)]
pub struct Vertex {
    #[allow(unused)]
    pos: [f32; 3],
}

pub struct Scene {
    vertex_buffer: vulkan_abstraction::VertexBuffer,
    index_buffer: vulkan_abstraction::IndexBuffer,
}

impl Scene {
    /// This function exists for testing purposes during the refactor
    pub fn new_default(core: &Rc<vulkan_abstraction::Core>) -> SrResult<Self> {
        let verts = [
            Vertex {
                pos: [-1.0, -0.5, 0.0],
            },
            Vertex {
                pos: [1.0, -0.5, 0.0],
            },
            Vertex {
                pos: [0.0, 1.0, 0.0],
            },
        ];
        let indices: [u32; 3] = [0, 1, 2];

        Self::new(core, &verts, &indices)
    }

    pub fn new(
        core: &Rc<vulkan_abstraction::Core>,
        verts: &[Vertex],
        indices: &[u32],
    ) -> SrResult<Self> {
        let vertex_buffer = {
            let staging_buffer = vulkan_abstraction::Buffer::new_staging_from_data::<Vertex>(
                Rc::clone(&core),
                &verts,
            )?;

            let vertex_buffer = vulkan_abstraction::VertexBuffer::new_for_blas::<Vertex>(
                Rc::clone(&core),
                verts.len(),
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
        })
    }

    pub fn vertex_buffer(&self) -> &vulkan_abstraction::VertexBuffer {
        &self.vertex_buffer
    }

    pub fn index_buffer(&self) -> &vulkan_abstraction::IndexBuffer {
        &self.index_buffer
    }
}
