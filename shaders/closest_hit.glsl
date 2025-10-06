#version 460
#include <shaders/common.glsl>
#include <shaders/utils.glsl>

layout(location = 0) rayPayloadInEXT ray_payload_t ray_payload_in;
layout(location = 1) rayPayloadEXT ray_payload_t ray_payload_out;

hitAttributeEXT vec2 attribs;

void main() {
    // Get barycentric coordinates
    vec3 barycentrics = vec3(1 - attribs.x - attribs.y, attribs.x, attribs.y);

    uint blas_instance_id = gl_InstanceCustomIndexEXT;
    mesh_info_t mesh_info = meshes_info_uniform_buffer.m[blas_instance_id];
    material_t material = mesh_info.material;

    uint index_buffer_offset = gl_PrimitiveID * 3;
    uint triangle_indices[3] = {
        mesh_info.indices.i[index_buffer_offset+0],
        mesh_info.indices.i[index_buffer_offset+1],
        mesh_info.indices.i[index_buffer_offset+2],
    };
    vertex_attributes_t triangle[3] = {
        mesh_info.vertices.v[triangle_indices[0]],
        mesh_info.vertices.v[triangle_indices[1]],
        mesh_info.vertices.v[triangle_indices[2]],
    };

    vertex_attributes_t vertex_attribs = interpolate_vertex_attributes(triangle, barycentrics);

    // Computing the coordinates of the hit position
    const vec3 world_pos = vec3(gl_ObjectToWorldEXT * vec4(vertex_attribs.position, 1.0)); // Transforming the position to world space

    vec4 base_color = sample_texture(material.base_color_texture_index, vertex_attribs.base_color_tex_coord, material.base_color_value);
    vec3 emissive_color = sample_texture(material.emissive_texture_index, vertex_attribs.emissive_tex_coord, vec4(material.emissive_factor, 0.0)).xyz;
    vec3 texture_normal = sample_texture(material.normal_texture_index, vertex_attribs.normal_tex_coord, vec4(0.0, 0.0, 1.0, 0.0)).xyz;

    // Computing the normal at hit position
    vec3 world_normal = normalize(vec3(vertex_attribs.normal * gl_WorldToObjectEXT)); // Transforming the normal to world space


    vec3 light_pos = vec3(5.0, 5.0, -5.0);
    vec3 light_dir = normalize(world_pos - light_pos);
    float light_dist = distance(world_pos, light_pos);
    float light_intensity = 2.0;

    float light = light_intensity * max(dot(-light_dir, world_normal), 0.2);

    // SHADOW
    float shadow = 1.0;
    ray_payload_out.shadow_ray_miss = false;

    uint ray_flags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT;
    vec3 shadow_ray_origin = world_pos; // + world_normal * 0.001; // why add world_normal * 0.001? we already have tMin=0.001
    vec3 shadow_ray_direction = -light_dir;
    float tMin = 0.001;
    float tMax = light_dist; // anything behind the light should not cause a shadow
    traceRayEXT(
        tlas,
        ray_flags,              
        0xFF,                   
        0,
        0,                    
        0,                   
        shadow_ray_origin,          
        tMin,                  
        shadow_ray_direction,          
        tMax,
        1 // payload has location=1
    );
    if(ray_payload_out.shadow_ray_miss) {
        shadow = 0.2;
    }

    ray_payload_in.color = light * (1.0 - shadow) * base_color.xyz + emissive_color.xyz;
}
