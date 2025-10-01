use nalgebra as na;

use crate::{
    error::SrResult,
    vulkan_abstraction::{self},
};

pub struct Node {
    transform: na::Matrix4<f32>,
    mesh: Option<vulkan_abstraction::gltf::Mesh>,
    children: Option<Vec<vulkan_abstraction::gltf::Node>>,
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
        mesh: Option<vulkan_abstraction::gltf::Mesh>,
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

    pub fn mesh(&self) -> &Option<vulkan_abstraction::gltf::Mesh> {
        &self.mesh
    }

    pub fn children(&self) -> &Option<Vec<vulkan_abstraction::gltf::Node>> {
        &self.children
    }
}
