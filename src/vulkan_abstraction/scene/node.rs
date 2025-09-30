use std::rc::Rc;

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
    ) -> SrResult<
        Vec<(
            vulkan_abstraction::VertexBuffer,
            vulkan_abstraction::IndexBuffer,
        )>
    > {
        let mesh = self.mesh.as_ref().unwrap();

        mesh.primitives().iter().map(|primitive| {
            let vertex_buffer = vulkan_abstraction::VertexBuffer::new_for_blas_from_data(
                Rc::clone(&core),
                &primitive.vertices,
            )?;

            let index_buffer = vulkan_abstraction::IndexBuffer::new_for_blas_from_data::<u32>(
                Rc::clone(&core),
                &primitive.indices,
            )?;

            Ok((vertex_buffer, index_buffer))
        }).collect()
    }
}
