use std::{rc::Rc, sync::Arc};

use ash::vk;
use nalgebra as na;

use crate::{
    error::SrResult,
    vulkan_abstraction::{self},
};

pub struct Node {
    transform: na::Matrix4<f32>,
    mesh: Option<vulkan_abstraction::Mesh>,
    children: Option<Vec<vulkan_abstraction::Node>>,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            transform: na::Matrix4::identity(),
            mesh: None,
            children: None,
        }
    }
}

impl Node {
    pub fn new(
        transform: na::Matrix4<f32>,
        mesh: Option<vulkan_abstraction::Mesh>,
        children: Option<Vec<Node>>,
    ) -> SrResult<Self> {
        Ok(Self {
            transform,
            mesh,
            children,
        })
    }

    pub fn transform(&self) -> &na::Matrix4<f32> {
        &self.transform
    }

    pub fn mesh(&self) -> &Option<vulkan_abstraction::Mesh> {
        &self.mesh
    }

    pub fn children(&self) -> &Option<Vec<vulkan_abstraction::Node>> {
        &self.children
    }

    pub fn load_mesh_into_gpu_memory(
        &self,
        core: &Rc<vulkan_abstraction::Core>,
    ) -> SrResult<(
        vulkan_abstraction::VertexBuffer,
        vulkan_abstraction::IndexBuffer,
    )> {
        let mesh = self.mesh.as_ref().unwrap();

        let vertex_buffer = {
            let staging_buffer = vulkan_abstraction::Buffer::new_staging_from_data::<
                vulkan_abstraction::Vertex,
            >(Rc::clone(&core), mesh.vertices())?;

            let vertex_buffer = vulkan_abstraction::VertexBuffer::new_for_blas::<
                vulkan_abstraction::Vertex,
            >(Rc::clone(&core), mesh.vertices().len())?;
            vulkan_abstraction::Buffer::clone_buffer(&core, &staging_buffer, &vertex_buffer)?;

            vertex_buffer
        };

        let index_buffer = {
            let staging_buffer = vulkan_abstraction::Buffer::new_staging_from_data::<u32>(
                Rc::clone(&core),
                mesh.indices(),
            )?;
            let index_buffer = vulkan_abstraction::IndexBuffer::new_for_blas::<u32>(
                Rc::clone(&core),
                mesh.indices().len(),
            )?;
            vulkan_abstraction::Buffer::clone_buffer(&core, &staging_buffer, &index_buffer)?;

            index_buffer
        };

        Ok((vertex_buffer, index_buffer))
    }
}
