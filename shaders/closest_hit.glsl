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

    vertex_t triangle[3] = {
        mesh_info.vertices.v[i0],
        mesh_info.vertices.v[i1],
        mesh_info.vertices.v[i2],
    };

    vec3 pos = INTERPOLATE_VERTEX_ATTRIBUTE(position, triangle, barycentrics);
    vec3 vertex_normal = INTERPOLATE_VERTEX_ATTRIBUTE(normal, triangle, barycentrics);
    vec2 base_color_tex_coords = INTERPOLATE_VERTEX_ATTRIBUTE(base_color_tex_coord, triangle, barycentrics);
    vec2 emissive_tex_coords = INTERPOLATE_VERTEX_ATTRIBUTE(emissive_tex_coord, triangle, barycentrics);
    vec2 normal_tex_coords = INTERPOLATE_VERTEX_ATTRIBUTE(normal_tex_coord, triangle, barycentrics);

    // Computing the coordinates of the hit position
    const vec3 world_pos = vec3(gl_ObjectToWorldEXT * vec4(pos, 1.0)); // Transforming the position to world space

    vec4 base_color = sample_texture(material.base_color_texture_index, base_color_tex_coords, material.base_color_value);
    vec3 emissive_color = sample_texture(material.emissive_texture_index, emissive_tex_coords, vec4(material.emissive_factor, 0.0)).xyz;
    vec3 texture_normal = sample_texture(material.normal_texture_index, normal_tex_coords, vec4(0.0, 0.0, 1.0, 0.0)).xyz;

    // Computing the normal at hit position
    vec3 world_normal = normalize(vec3(vertex_normal * gl_WorldToObjectEXT)); // Transforming the normal to world space

    float light_intensity = 1.0;
    vec3 light_direction = normalize(vec3(-1.0, -1.0, -0.5));

    float lighting = max(dot(-light_direction, world_normal), 0.2);

    prd.color = lighting * base_color.xyz + emissive_color.xyz;
}
