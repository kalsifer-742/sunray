#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

layout(location = 0) out vec3 out_color;

void main() {
    gl_Position = vec4(position, 1.0);
    out_color = color; //i really don't understand how this works, why the colors gradient automaticaly without doing anything?
}