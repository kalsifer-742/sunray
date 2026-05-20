#version 460
#extension GL_EXT_descriptor_heap : require
#extension GL_EXT_nonuniform_qualifier : require

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint input_idx;
    uint output_idx;
    float exposure;
} pc;

layout(descriptor_heap, r11f_g11f_b10f) uniform readonly image2D input_image[];
layout(descriptor_heap, rgba8) uniform writeonly image2D output_image[];


// ACES Fitted (Narkowicz approximation)
vec3 ACESFilm(vec3 x) {
    // Clamp to prevent Inf/Inf division if x is extremely high
    x = clamp(x, 0.0, 100.0);
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    ivec2 size = imageSize(input_image[pc.input_idx]);
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

    if (pixel_coords.x >= size.x || pixel_coords.y >= size.y) {
        return;
    }

    vec4 hdr_data = imageLoad(input_image[pc.input_idx], pixel_coords);
    vec3 color = hdr_data.rgb;

    if (any(isnan(color)) || any(isinf(color))) {
        color = vec3(0.0);
    }

    color *= pc.exposure;
    vec3 mapped = ACESFilm(color);
    vec3 final_color = pow(mapped, vec3(1.0 / 2.2));

    imageStore(output_image[pc.output_idx], pixel_coords, vec4(final_color, 1.0));
}