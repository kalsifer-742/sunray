#version 460
#include <shaders/common.glsl>

layout(location = 0) rayPayloadInEXT ray_payload_t prd;

void main() {
    prd.color = vec3(0, 0, 0.2);
}
