pub mod model;
use std::rc::Rc;

use ash::vk;
pub use model::*;

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

pub struct Scene {
    pub meshes: Vec<vulkan_abstraction::Mesh>,
}

impl Scene {
    pub fn new_testing(core: &Rc<vulkan_abstraction::Core>) -> SrResult<Self> {
        let vertices = [
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
        let indices: [u32; 3] = [0, 1, 2];

        let _transforms = vec![vulkan_abstraction::IDENTITY_MATRIX];

        let meshes = vec![vulkan_abstraction::Mesh::new(core, &vertices, &indices)?];

        Ok(Self { meshes })
    }
}
