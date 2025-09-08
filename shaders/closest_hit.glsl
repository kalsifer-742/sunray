#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable


struct hitPayload
{
  vec3 hitValue;
};

hitAttributeEXT vec3 attribs;
layout(location = 0) rayPayloadEXT hitPayload prd;

void main()
{
  prd.hitValue = vec3(0, 1, 0);
}
