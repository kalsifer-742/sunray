#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct Vertex {
    #[allow(unused)]
    pub position: [f32; 3],
    pub base_color_tex_coord: [f32; 2],
    pub metallic_roughness_tex_coord: [f32; 2],
    pub normal_tex_coord: [f32; 2],
    pub occlusion_tex: [f32; 2],
    pub emissive_tex: [f32; 2],
}
