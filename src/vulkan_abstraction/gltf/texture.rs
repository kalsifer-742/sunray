use crate::vulkan_abstraction;

pub enum MagFilter {
    Linear,
    Nearest,
}

pub enum MinFilter {
    Nearest,
    Linear,
    NearestMipmapNearest,
    LinearMipmapNearest,
    NearestMipmapLinear,
    LinearMipmapLinear,
}

pub enum WrappingMode {
    ClampToEdge,
    MirroredRepeat,
    Repeat,
}

pub struct Sampler {
    pub mag_filter: Option<vulkan_abstraction::gltf::MagFilter>,
    pub min_filter: Option<vulkan_abstraction::gltf::MinFilter>,
    pub wrap_s_u: WrappingMode,
    pub wrap_t_v: WrappingMode,
}

pub struct Texture {
    pub sampler: Option<usize>,
    pub source: usize, // this is technically not required by the spec
}
