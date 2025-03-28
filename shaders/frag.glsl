#version 460
layout(location=0) in vec3 incolor;
layout(location=0) out vec4 outcolor;
void main() {
    float levels = 10.0; // quantization levels
    vec3 quantized_color = floor(incolor * levels) / (levels - 1.0);
    outcolor = vec4(quantized_color, 1.0);
}
