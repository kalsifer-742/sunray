// type Color = [f32; 4];

// type PbrVvalues = (Color, f32, f32);

// pub enum PbrMetallicRoughnessProperties {
//     Values(PbrVvalues),
//     Textures(usize, usize),
// }

pub struct PbrMetallicRoughnessProperties {
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub base_color_texture_index: Option<usize>,
    pub metallic_roughness_texture_index: Option<usize>,
}

pub struct Material {
    pub pbr_mettalic_roughness_properties: PbrMetallicRoughnessProperties,
    pub normal_texture_index: Option<usize>,
    pub occlusion_texture_index: Option<usize>,
    pub emissive_factor: [f32; 3],
    pub emissive_texture_index: Option<usize>,
    pub alpha_mode: gltf::material::AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
}
