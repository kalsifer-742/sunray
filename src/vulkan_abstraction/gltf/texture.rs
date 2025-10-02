pub struct Sampler {
    pub mag_filter: Option<gltf::texture::MagFilter>,
    pub min_filter: Option<gltf::texture::MinFilter>,
    pub wrap_s_u: gltf::texture::WrappingMode,
    pub wrap_t_v: gltf::texture::WrappingMode,
}

pub struct Texture {
    pub sampler: Option<usize>,
    pub source: usize, // this is technically not required by the spec
}
