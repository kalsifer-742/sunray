#version 460

struct Vertex {
    vec3 pos;
    vec3 color;
};
layout(binding = 0, std430) readonly buffer Vertices {
    Vertex data[];
} in_verts;

layout(binding = 1) readonly uniform UniformBuffer {
    mat4 transform;
} ubo;

layout(location = 0) out vec3 color;

void main() {
    Vertex vert = in_verts.data[gl_VertexIndex];

    gl_Position = ubo.transform * vec4(vert.pos, 1.0);
    color = vert.color;
}
