// Postprocess: tonemap + gamma. First Slang shader in the project, used to
// validate the VK_EXT_descriptor_heap path end-to-end before porting the rest.

struct PostprocessPC {
    DescriptorHandle<RWTexture2D<float4>> input;   // r11g11b10f HDR (read as float4)
    DescriptorHandle<RWTexture2D<float4>> output;  // rgba8 LDR
    float exposure;
}

[vk::push_constant]
ConstantBuffer<PostprocessPC> pc;

// ACES Fitted (Narkowicz approximation).
float3 aces_film(float3 x) {
    x = clamp(x, 0.0, 100.0);
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    RWTexture2D<float4> src = pc.input;
    RWTexture2D<float4> dst = pc.output;

    uint2 size;
    dst.GetDimensions(size.x, size.y);
    if (tid.x >= size.x || tid.y >= size.y) {
        return;
    }

    float3 color = src[tid.xy].rgb;
    if (any(isnan(color)) || any(isinf(color))) {
        color = float3(0.0);
    }

    color *= pc.exposure;
    float3 mapped = aces_film(color);
    float3 final_color = pow(mapped, float3(1.0 / 2.2));

    dst[tid.xy] = float4(final_color, 1.0);
}
