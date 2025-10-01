use crate::{error::SrResult, vulkan_abstraction};

pub struct Mesh {
    primitives: Vec<vulkan_abstraction::gltf::Primitive>,
}

impl Mesh {
    pub fn new(primitives: Vec<vulkan_abstraction::gltf::Primitive>) -> SrResult<Self> {
        Ok(Self { primitives })
    }

    pub fn primitives(&self) -> &Vec<vulkan_abstraction::gltf::Primitive> {
        &self.primitives
    }
}
