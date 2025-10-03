#[derive(Debug, Clone, Copy, Default)]
#[repr(C, packed)]
pub struct Vertex {
    /*
    NOTE: don't move Self::position or place any attributes before it: the BLAS assumes
    that the vertex_buffer has a vec3 position attribute as its first (not necessarily
    the only) attribute in memory.

    repr(C) is used to avoid reordering of this data, since it will be sent to the gpu,
    and repr(packed) gives us full control of the alignment and we can force it
    (using _padding* attributes) to follow GLSL's rules (specifically std430).

    The main thing to look out for is that vectors cannot straddle over the 4-word
    (1 word = 1 float) boundary: for example a vec4 is always aligned to 4w, and a float
    is aligned to 1w, but a vec3 is either aligned to 4w or to the next word after a 4w
    boundary, and can only be packed together with a single float, whereas a vec2 is
    either aligned to 2w or to the next word after a 4w boundary.

    GLSL (float) arrays do not follow these rules, but it was chosen to use vec(n)
    instead because alignment is a big deal in SIMD-level parallelism (which the GPU
    does massively) and if the GLSL spec specifies a preferred alignment we shouldn't
    ignore it just for slight convenience.
    */
    pub position: [f32; 3],
    pub _padding0: [f32; 1],
    pub base_color_tex_coord: [f32; 2],
    pub metallic_roughness_tex_coord: [f32; 2],
    pub normal_tex_coord: [f32; 2],
    pub occlusion_tex: [f32; 2],
    pub emissive_tex: [f32; 2],
    pub _padding1: [f32; 2],
}
