use crate::vulkan_abstraction;

#[derive(Clone, Copy)]
pub struct Texture<'a>(pub &'a vulkan_abstraction::Image, pub &'a vulkan_abstraction::Sampler);
