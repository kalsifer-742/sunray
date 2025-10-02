#[derive(Debug, Clone, Copy, Default)]
#[repr(C, packed)]
pub struct Vertex {
    // NOTE: don't move position or place any attributes before it:
    // the BLAS assumes that the vertex_buffer has a vec3 position attribute as its first (not necessarily the only) attribute in memory
    pub position: [f32; 3],
    pub _padding0: [f32; 1],
    pub base_color_tex_coord: [f32; 2],
    pub metallic_roughness_tex_coord: [f32; 2],
    pub normal_tex_coord: [f32; 2],
    pub occlusion_tex: [f32; 2],
    pub emissive_tex: [f32; 2],
    pub _padding2: [f32; 2],
}
