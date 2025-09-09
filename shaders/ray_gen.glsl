#version 460
#extension GL_EXT_ray_tracing : require

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 1, set = 0, rgba8) uniform image2D image;

struct ray_payload_t {
    vec3 color;
};
layout(location = 0) rayPayloadEXT ray_payload_t prd;

void main() {
    const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
    const vec2 inUV = pixelCenter/vec2(gl_LaunchSizeEXT.xy);
    vec2 d = inUV * 2.0 - 1.0;

    uint  rayFlags = gl_RayFlagsOpaqueEXT;
    float tMin     = 0.001;
    float tMax     = 10000.0;

    prd.color = vec3(.3, 0, 0);

    traceRayEXT(
        topLevelAS,             // acceleration structure
        rayFlags,               // rayFlags
        0xFF,                   // cullMask
        0,                      // sbtRecordOffset
        0,                      // sbtRecordStride
        0,                      // missIndex
        vec3(0,0,1),            // ray origin
        tMin,                   // ray min range
        normalize(vec3(d,-1)),  // ray direction
        tMax,                   // ray max range
        0                       // payload (location = 0)
    );

    imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(prd.color.bgr, 1.0));
}
