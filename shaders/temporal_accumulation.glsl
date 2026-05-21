#version 460

#define TILE_SIZE 16
#define TILE_BORDER 1
#define TILE_FULL (TILE_SIZE + 2 * TILE_BORDER) // 18

layout(local_size_x = TILE_SIZE, local_size_y = TILE_SIZE, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint frame_count;
    uint width;
    uint height;
} pc;

layout(set = 0, binding = 0) uniform sampler2D raw_rt_color;
layout(set = 0, binding = 1, rg16f)   uniform readonly image2D motion_vector_image;
layout(set = 0, binding = 2, r11f_g11f_b10f) uniform image2D accumulation_images[2];
layout(set = 0, binding = 3)          uniform sampler2D history_samplers[2];

const float ACCUMULATION_FACTOR = 0.14;

// Shared tile for 3x3 neighborhood clamp. 256 threads cooperatively load 324 texels (~1.27/thread).
// Reduces 9 texelFetch/pixel to ~1.27/pixel plus avoids L1 cache contention.
shared vec3 tile_color[TILE_FULL * TILE_FULL];

float get_luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

void main() {
    // `imageSize(accumulation_images[0])` returns bogus dimensions on this driver
    // (single-pixel writes observed), so pass the dispatch's pixel extent via push
    // constant to keep the early-exit guard correct.
    ivec2 size = ivec2(pc.width, pc.height);
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 local_coords = ivec2(gl_LocalInvocationID.xy);
    ivec2 tile_base = ivec2(gl_WorkGroupID.xy) * TILE_SIZE - TILE_BORDER;

    // Cooperative tile load: all 256 threads must participate (do NOT early-exit before barrier).
    uint thread_idx = gl_LocalInvocationIndex;
    for (uint i = thread_idx; i < TILE_FULL * TILE_FULL; i += TILE_SIZE * TILE_SIZE) {
        ivec2 local_pos = ivec2(i % TILE_FULL, i / TILE_FULL);
        ivec2 sample_pos = clamp(tile_base + local_pos, ivec2(0), size - 1);
        tile_color[i] = texelFetch(raw_rt_color, sample_pos, 0).rgb;
    }
    barrier();

    if (pixel_coords.x >= size.x || pixel_coords.y >= size.y) return;

    // Read center from tile.
    ivec2 tile_center = local_coords + ivec2(TILE_BORDER);
    vec3 current_color = tile_color[tile_center.y * TILE_FULL + tile_center.x];

    vec3 min_color = current_color;
    vec3 max_color = current_color;
    float center_luma = get_luminance(current_color);

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            if (x == 0 && y == 0) continue;
            ivec2 tile_pos = tile_center + ivec2(x, y);
            vec3 neighbor_color = tile_color[tile_pos.y * TILE_FULL + tile_pos.x];

            float neighbor_luma = get_luminance(neighbor_color);
            float luma_threshold = max(center_luma * 5.0, 0.08);
            if (abs(neighbor_luma - center_luma) < luma_threshold) {
                min_color = min(min_color, neighbor_color);
                max_color = max(max_color, neighbor_color);
            }
        }
    }

    vec2 uv = (vec2(pixel_coords) + 0.5) / vec2(size);
    vec2 motion_vector = imageLoad(motion_vector_image, pixel_coords).rg;
    vec2 prev_uv = uv - motion_vector;

    uint history_idx = pc.frame_count % 2;
    uint accum_idx   = (pc.frame_count + 1) % 2;

    vec3 accumulated_color = current_color;

    bool is_off_screen = any(lessThan(prev_uv, vec2(0.0))) || any(greaterThan(prev_uv, vec2(1.0)));

    if (!is_off_screen && pc.frame_count > 2) {
        vec3 history_color = texture(history_samplers[history_idx], prev_uv).rgb;
        vec3 clamped_history = clamp(history_color, min_color, max_color);
        accumulated_color = mix(clamped_history, current_color, ACCUMULATION_FACTOR);
    }

    imageStore(accumulation_images[accum_idx], pixel_coords, vec4(accumulated_color, 1.0));
}