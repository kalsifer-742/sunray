use crate::{error::SrResult, vulkan_abstraction};

pub struct Primitive {
    vertex_buffer: vulkan_abstraction::VertexBuffer,
    index_buffer: vulkan_abstraction::IndexBuffer,
}

impl Primitive {
    pub fn new(
        vertex_buffer: vulkan_abstraction::VertexBuffer,
        index_buffer: vulkan_abstraction::IndexBuffer,
    ) -> SrResult<Self> {
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
