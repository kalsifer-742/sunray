#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable //found in NV tutorial, don't know what it does

hitAttributeEXT vec3 attribs;

struct ray_payload_t {
    vec3 color;
};
layout(location = 0) rayPayloadInEXT ray_payload_t prd;

void main() {
  prd.color = vec3(0, 0.5, 0);
}
