use std::rc::Rc;

use ash::vk;

use crate::{error::SrResult, vulkan_abstraction};

#[derive(Clone, Copy)]
pub struct Vertex {
    #[allow(unused)]
    pub position: [f32; 3],
}

pub struct Mesh {
    vertex_buffer: vulkan_abstraction::VertexBuffer,
    index_buffer: vulkan_abstraction::IndexBuffer,
    transform_buffer: vulkan_abstraction::Buffer,
}

impl Mesh {
    pub fn new(
        core: &Rc<vulkan_abstraction::Core>,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> SrResult<Self> {
        let vertex_buffer = {
            let staging_buffer = vulkan_abstraction::Buffer::new_staging_from_data::<Vertex>(
                Rc::clone(&core),
                vertices,
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
                indices,
            )?;
            let index_buffer = vulkan_abstraction::IndexBuffer::new_for_blas::<u32>(
                Rc::clone(&core),
                indices.len(),
            )?;
            vulkan_abstraction::Buffer::clone_buffer(&core, &staging_buffer, &index_buffer)?;

            index_buffer
        };

        //it is, as far as I can tell, ok to drop the transform buffer after constructing the BLAS
        let transform_buffer = vulkan_abstraction::Buffer::new_from_data(
            Rc::clone(&core),
            &[vulkan_abstraction::IDENTITY_MATRIX],
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryAllocateFlags::DEVICE_ADDRESS,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
        )?;

        Ok(Self {
            vertex_buffer,
            index_buffer,
            transform_buffer,
        })
    }

    pub fn vertex_buffer(&self) -> &vulkan_abstraction::VertexBuffer {
        &self.vertex_buffer
    }

    pub fn index_buffer(&self) -> &vulkan_abstraction::IndexBuffer {
        &self.index_buffer
    }

    pub fn transform_buffer(&self) -> &vulkan_abstraction::Buffer {
        &self.transform_buffer
    }
}
