#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 1, rgba8) uniform image2D image;
layout(set = 0, binding = 2) uniform uniform_buffer_t {
    mat4 view_inverse, proj_inverse;
} uniform_buffer;

struct ray_payload_t {
    vec3 color;
};
layout(location = 0) rayPayloadEXT ray_payload_t prd;


void main() {
    const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5); //the coordinates are of the corner, +0.5 gets the pixel center
    const vec2 inUV = pixelCenter / vec2(gl_LaunchSizeEXT.xy); //normalize value in [0, 1]
    vec2 d = inUV * 2.0 - 1.0; //map [0, 1] to [-1, 1]

    uint  ray_flags = gl_RayFlagsOpaqueEXT;
    float tMin     = 0.001;
    float tMax     = 10000.0;

    vec4 origin    = uniform_buffer.view_inverse * vec4(0, 0, 0, 1);
    vec4 target    = uniform_buffer.proj_inverse * vec4(d.x, d.y, 1, 1);
    vec4 direction = uniform_buffer.view_inverse * vec4(normalize(target.xyz), 0);

    prd.color = vec3(.3, 0, 0);

    traceRayEXT(
        tlas,                   // acceleration structure
        ray_flags,              // rayFlags
        0xFF,                   // cullMask
        0,                      // sbtRecordOffset
        0,                      // sbtRecordStride
        0,                      // missIndex
        origin.xyz,             // ray origin
        tMin,                   // ray min range
        direction.xyz,          // ray direction
        tMax,                   // ray max range
        0                       // payload (location = 0)
    );

    imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(prd.color, 1.0));
}
