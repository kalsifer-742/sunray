#ifndef SHADERS_COMMON_GLSL
#define SHADERS_COMMON_GLSL

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types : require // uint32_t

// the actual payload is not defined here because raygen would define it as RayPayloadEXT, whereas hit/miss would define it as RayPayloadInEXT
struct ray_payload_t {
    vec3 color;
};

struct vertex_t {
    vec3 position;
    vec3 normal;
    vec2 base_color_tex_coord;
    vec2 metallic_roughness_tex_coord;
    vec2 normal_tex_coord;
    vec2 occlusion_tex_coord;
    vec2 emissive_tex_coord;
};

layout(std430, buffer_reference, buffer_reference_align = 8) buffer vertex_buffer_reference_t {
    vertex_t v[];
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

layout(push_constant) uniform push_constant_t {
    bool use_srgb;
};
layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 1, rgba8) uniform image2D image;
layout(set = 0, binding = 2) uniform matrices_uniform_buffer_t {
    mat4 view_inverse, proj_inverse;
} matrices_uniform_buffer;
layout(set = 0, binding = 3) buffer meshes_info_storage_buffer_t {
    mesh_info_t m[];
} meshes_info_uniform_buffer;

layout(set = 0, binding = 4) uniform sampler2D texture_samplers[1024];

#endif
