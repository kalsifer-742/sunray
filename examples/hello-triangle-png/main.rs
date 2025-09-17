use std::path::Path;

use ash::vk;
use image::ImageFormat;
use sunray::Renderer;

fn render_to_file(
    image: vk::Image,
    image_extent: (u32, u32),
    path: impl AsRef<Path>,
    format: ImageFormat,
) {
    let buf = image; //TODO: conversion
    let (width, height) = image_extent;

    image::save_buffer_with_format(
        path,
        buf,
        width,
        height,
        image::ExtendedColorType::Rgba8,
        format,
    );
}

fn main() {
    let verts = [
        Vertex {
            pos: [-1.0, -0.5, 0.0],
        },
        Vertex {
            pos: [1.0, -0.5, 0.0],
        },
        Vertex {
            pos: [0.0, 1.0, 0.0],
        },
    ];
    let indices: [u32; 3] = [0, 1, 2];

    let image_extent = (800, 600);
    let renderer = Renderer::new(image_extent).unwrap();

    renderer.load_file(verts, indices);
    renderer.set_camera();

    let image = renderer.render().unwrap();
    render_to_file(image, image_extent, "hello_triangle", ImageFormat::Png);
}
