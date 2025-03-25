#version 460

struct Vertex {
    //float pos[3];
    //float color[3];
    vec3 pos;
    vec3 color;
};
layout(binding = 0, std430) readonly buffer Vertices {
    Vertex data[];
} in_verts;

layout(binding = 1) readonly uniform UniformBuffer {
    mat4 transform;
} ubo;

layout(location = 1) out vec3 color;

void main() {
    Vertex vert = in_verts.data[gl_VertexIndex];
    vec4 pos = vec4(vert.pos[0], vert.pos[1], vert.pos[2], 1.0);

    // gl_Position = pos;
    gl_Position = ubo.transform * pos;
    color = vec3(vert.color[0], vert.color[1], vert.color[2]);
}
