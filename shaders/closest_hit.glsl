#version 460
#extension GL_EXT_ray_tracing : require

#extension GL_EXT_buffer_reference2 : require
//uint32_t
#extension GL_EXT_shader_explicit_arithmetic_types : require

hitAttributeEXT vec2 attribs;

struct vertex_t {
    vec3 position;
    vec2 base_color_tex_coord;
    vec2 metallic_roughness_tex_coord;
    vec2 normal_tex_coord;
    vec2 occlusion_tex;
    vec2 emissive_tex;
};

layout(std430, buffer_reference, buffer_reference_align = 8) buffer vertex_buffer_reference_t {
    vertex_t v[]; //use .length()?
};


layout(std430, buffer_reference, buffer_reference_align = 8) buffer index_buffer_reference_t {
    uint32_t i[];
};

struct material_t {
    vec4 base_color_value;
    uint32_t base_color_texture_index;

    float metallic_factor;
    float roughness_factor;
    uint32_t metallic_roughness_texture_index;

    uint32_t normal_texture_index;
    uint32_t occlusion_texture_index;

    vec3 emissive_factor;
    uint32_t emissive_texture_index;
};

struct mesh_info_t {
    vertex_buffer_reference_t vertices;
    index_buffer_reference_t indices;

    material_t material;
};

layout(set = 0, binding = 2) uniform matrices_uniform_buffer_t {
    mat4 view_inverse, proj_inverse;

} matrices_uniform_buffer;

layout(set = 0, binding = 3) buffer meshes_info_storage_buffer_t {
    mesh_info_t m[];
} meshes_info_uniform_buffer;

layout(set = 0, binding = 4) uniform sampler2D texture_samplers[1024];

layout(location = 0) rayPayloadInEXT ray_payload_t {
    vec3 color;
} prd;



void main() {
    // Get barycentric coordinates
    vec3 barycentrics = vec3(1 - attribs.x - attribs.y, attribs.x, attribs.y);

    uint blas_instance_id = gl_InstanceCustomIndexEXT;
    mesh_info_t mesh_info = meshes_info_uniform_buffer.m[blas_instance_id];
    material_t material = mesh_info.material;
    uint texture_index = material.base_color_texture_index;


    uint index_buffer_offset = gl_PrimitiveID * 3;

    uint i0 = mesh_info.indices.i[index_buffer_offset+0];
    uint i1 = mesh_info.indices.i[index_buffer_offset+1];
    uint i2 = mesh_info.indices.i[index_buffer_offset+2];

    vertex_t v0 = mesh_info.vertices.v[i0];
    vertex_t v1 = mesh_info.vertices.v[i1];
    vertex_t v2 = mesh_info.vertices.v[i2];

    vec2 base_color_tex_coords =
          vec2(v0.base_color_tex_coord[0], v0.base_color_tex_coord[1]) * barycentrics.x
        + vec2(v1.base_color_tex_coord[0], v1.base_color_tex_coord[1]) * barycentrics.y
        + vec2(v2.base_color_tex_coord[0], v2.base_color_tex_coord[1]) * barycentrics.z;

    // texture_samplers[texture_index] is our texture
    prd.color = texture(texture_samplers[texture_index], base_color_tex_coords).xyz;
}
