#version 460
#include <shaders/common.glsl>
#include <shaders/utils.glsl>

layout(location = 0) rayPayloadInEXT ray_payload_t prd;

hitAttributeEXT vec2 attribs;

void main() {
    // Get barycentric coordinates
    vec3 barycentrics = vec3(1 - attribs.x - attribs.y, attribs.x, attribs.y);

    uint blas_instance_id = gl_InstanceCustomIndexEXT;
    mesh_info_t mesh_info = meshes_info_uniform_buffer.m[blas_instance_id];
    material_t material = mesh_info.material;

    uint index_buffer_offset = gl_PrimitiveID * 3;

    uint i0 = mesh_info.indices.i[index_buffer_offset+0];
    uint i1 = mesh_info.indices.i[index_buffer_offset+1];
    uint i2 = mesh_info.indices.i[index_buffer_offset+2];

    vertex_t v0 = mesh_info.vertices.v[i0];
    vertex_t v1 = mesh_info.vertices.v[i1];
    vertex_t v2 = mesh_info.vertices.v[i2];

    vec2 base_color_tex_coords = INTERPOLATE_TEX_COORDS(base_color_tex_coord, v0, v1, v2, barycentrics);
    vec2 emissive_tex_coords = INTERPOLATE_TEX_COORDS(emissive_tex_coord, v0, v1, v2, barycentrics);

    vec4 base_color = sample_texture(material.base_color_texture_index, base_color_tex_coords, material.base_color_value);
    vec4 emissive_color = sample_texture(material.emissive_texture_index, emissive_tex_coords, vec4(material.emissive_factor, 0.0));

    prd.color = base_color.xyz + emissive_color.xyz;
}
