use std::path::Path;

use ash::vk;
use image::{ExtendedColorType, ImageFormat};
use sunray::{Camera, Renderer};

fn get_vk_format(image_format: ExtendedColorType) -> vk::Format {
    match image_format {
        ExtendedColorType::Rgba8 => vk::Format::R8G8B8A8_UNORM,
        _ => vk::Format::UNDEFINED,
    }
}

fn render_to_file(
    image: vk::Image,
    image_extent: (u32, u32),
    path: impl AsRef<Path>,
    format: ImageFormat,
) {
    let buf = image; //TODO: conversion
    let (width, height) = image_extent;

    image::save_buffer_with_format(path, buf, width, height, ExtendedColorType::Rgba8, format);
}

fn main() {
    let image_extent = (800, 600);
    let image_format = get_vk_format(ExtendedColorType::Rgba8);
    let mut renderer = Renderer::new(image_extent, image_format).unwrap();

    renderer.load_file();

    let camera = Camera::new();

    renderer.set_camera(camera);

    let image = renderer.render().unwrap();
    render_to_file(image, image_extent, "hello_triangle", ImageFormat::Png);
}
