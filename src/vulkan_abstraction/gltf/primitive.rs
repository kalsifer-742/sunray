use crate::vulkan_abstraction;

pub type PrimitiveUniqueKey = (usize, usize);

pub struct PrimitiveData {
    pub vertex_buffer: vulkan_abstraction::VertexBuffer,
    pub index_buffer: vulkan_abstraction::IndexBuffer,
    pub material: vulkan_abstraction::gltf::Material,
}

pub struct Primitive {
    pub unique_key: PrimitiveUniqueKey,
}
