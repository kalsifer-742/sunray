#version 460

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint frame_count;
    int step_width;
    uint width;
    uint height;
} pc;

layout(set = 0, binding = 0, r11f_g11f_b10f) uniform readonly image2D temporal_result; // Input
layout(set = 0, binding = 1) uniform sampler2D depth_image;
layout(set = 0, binding = 2) uniform sampler2D normal_image;
layout(set = 0, binding = 3) uniform sampler2D diffuse_image;
layout(set = 0, binding = 4, r11f_g11f_b10f) uniform writeonly image2D spatial_output;

float get_luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722)); // [cite: 62]
}

// Fixed B-Spline weights for 5x5 A-Trous
const float kernel[5] = { 1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0 };

void main() {
    // `imageSize(spatial_output)` returns bogus dimensions on this driver, see
    // matching workaround in temporal_accumulation.glsl. Use push-constant size.
    ivec2 size = ivec2(pc.width, pc.height);
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    if (pixel_coords.x >= size.x || pixel_coords.y >= size.y) return;

    vec3 center_color = imageLoad(temporal_result, pixel_coords).rgb;
    float center_depth = texelFetch(depth_image, pixel_coords, 0).r;

    //Early bypass for sky

    if (center_depth >= 10000.0) {
        imageStore(spatial_output, pixel_coords, vec4(center_color, 1.0));
        return;
    }

    vec4 normal_data = texelFetch(normal_image, pixel_coords, 0);
    vec3 center_normal = normal_data.rgb;
    float center_roughness = normal_data.a;
    vec3 center_diffuse = texelFetch(diffuse_image, pixel_coords, 0).rgb;

    if(center_roughness < 0.1){
        imageStore(spatial_output, pixel_coords, vec4(center_color, 1.0)); // Removed multiplication
        return;
    }


    //imageStore(spatial_output, pixel_coords, vec4(center_color, 1.0));
    //return;

    float center_weight = kernel[2] * kernel[2];
    vec3 sum_color = center_color * center_weight;
    float sum_weight = center_weight;


    float DEPTH_SENSITIVITY = 8.0;
    float NORMAL_SENSITIVITY = 80.0;
    float DIFFUSE_SENSITIVITY = 50.0;

    float center_luma = get_luminance(center_color);

    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            ivec2 sample_offset = ivec2(x, y) * pc.step_width;

            ivec2 sample_coord = pixel_coords + sample_offset;

            if (sample_coord.x < 0 || sample_coord.y < 0 ||
            sample_coord.x >= size.x || sample_coord.y >= size.y) {
                continue;
            }

            vec3 sample_color = imageLoad(temporal_result, sample_coord).rgb;
            float sample_depth = texelFetch(depth_image, sample_coord, 0).r;
            vec3 sample_normal = texelFetch(normal_image, sample_coord, 0).rgb;

            vec3 sample_diffuse = texelFetch(diffuse_image, sample_coord, 0).rgb;

            float sample_luma = get_luminance(sample_color);

            float diffuse_diff = distance(center_diffuse, sample_diffuse);
            float luma_diff = abs(center_luma - sample_luma);
            float luma_sigma = max(center_luma, sample_luma) * 0.4 + 0.01;
            float luma_ratio = luma_diff / luma_sigma;

            // Squared luma penalty
            float combined_power =
            -abs(center_depth - sample_depth) * DEPTH_SENSITIVITY
            + (dot(center_normal, sample_normal) - 1.0) * NORMAL_SENSITIVITY
            - diffuse_diff * DIFFUSE_SENSITIVITY
            - luma_ratio * luma_ratio;

            float weight = exp(combined_power) * kernel[x + 2] * kernel[y + 2];

            sum_color += sample_color * weight;
            sum_weight += weight;
        }
    }

    vec3 spatially_denoised_color = (sum_color / max(sum_weight, 0.0001)) * center_diffuse;

    imageStore(spatial_output, pixel_coords, vec4(spatially_denoised_color, 1.0));
}