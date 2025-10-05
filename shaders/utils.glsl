#ifndef SHADERS_UTILS_GLSL
#define SHADERS_UTILS_GLSL
// necessary since this file uses the texture_samplers uniform in sample_texture
#include <shaders/common.glsl>

// a texture index of ~0 == u32(-1) == 0xffffffff may be passed to indicate that no texture should be used, and the provided value should be used as replacement for all texels
const uint null_texture = ~0;

// sample from the provided texture index or, if it is null, return the fallback color
// texture_samplers is not currently passed as a
vec4 sample_texture(in uint texture_index, in vec2 tex_coords, in vec4 fallback_color) {
    if(texture_index == null_texture) {
        return fallback_color;
    } else {
        return texture(texture_samplers[texture_index], tex_coords);
    }
}



//given 3 vertices, a texture coordinate attribute and barycentric coordinates interpolate the texture coordinate attribute
#define INTERPOLATE_TEX_COORDS(tex_coord_attribute, v0, v1, v2, barycentrics) \
          vec2(v0.tex_coord_attribute[0], v0.tex_coord_attribute[1]) * barycentrics.x \
        + vec2(v1.tex_coord_attribute[0], v1.tex_coord_attribute[1]) * barycentrics.y \
        + vec2(v2.tex_coord_attribute[0], v2.tex_coord_attribute[1]) * barycentrics.z

// take a value that should be interpreted as linear and return the equivalent that should be interpreted as sRGB.
// this is useful to write to an sRGB image from a compute or raytracing shader.
// source: https://github.com/Microsoft/DirectX-Graphics-Samples/blob/master/MiniEngine/Core/Shaders/ColorSpaceUtility.hlsli
// note: if this is ever a bottleneck (shouldn't be) consider using the fast version, from the same source
float remove_srgb_curve(float x) {
    // Approximately pow(x, 2.2)
    return x < 0.04045 ?  x / 12.92 : pow((x + 0.055) / 1.055, 2.4);
}

#endif
