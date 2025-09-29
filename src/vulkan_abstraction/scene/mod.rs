pub mod mesh;
pub mod node;

pub use mesh::*;
pub use node::*;

use crate::{
    error::SrResult,
    vulkan_abstraction::{self},
};

pub struct Scene {
    nodes: Vec<Node>,
}

impl Default for Scene {
    fn default() -> Self {
        let transform = vulkan_abstraction::IDENTITY_MATRIX;
        let vertices = vec![
            vulkan_abstraction::Vertex {
                position: [-1.0, -0.5, 0.0],
            },
            vulkan_abstraction::Vertex {
                position: [1.0, -0.5, 0.0],
            },
            vulkan_abstraction::Vertex {
                position: [0.0, 1.0, 0.0],
            },
        ];
        let indices = vec![0, 1, 2];
        let mesh = vulkan_abstraction::Mesh::new(vertices, indices).unwrap();
        let nodes = vec![vulkan_abstraction::Node::new(transform, Some(mesh), None).unwrap()];

        Self { nodes }
    }
}

impl Scene {
    pub fn new(nodes: Vec<Node>) -> SrResult<Self> {
        Ok(Self { nodes })
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
}
