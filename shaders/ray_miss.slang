// Miss shader. Negative `dist` is the sentinel the ray-gen shaders check
// to detect "hit the sky"; we leave emission at zero so the sky contributes
// no direct light. Mirrors shaders/ray_miss.glsl.

import rt_types;

[vk::push_constant] ConstantBuffer<RaytracingPC> pc;

[shader("miss")]
void ray_miss(inout RayPayload payload) {
    payload.dist = -1.0;
    payload.emission = float3(0.0);
}
