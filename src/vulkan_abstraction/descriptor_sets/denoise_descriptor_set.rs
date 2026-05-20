use crate::error::SrResult;
use crate::vulkan_abstraction;
use ash::vk;
use std::rc::Rc;

pub struct DenoiseDescriptorSetLayout {
    descriptor_set_layout: vk::DescriptorSetLayout,
    core: Rc<vulkan_abstraction::Core>,
}

impl DenoiseDescriptorSetLayout {
    pub const INPUT_BINDING: u32 = 0; // Input from Temporal Pass
    pub const DEPTH_BINDING: u32 = 1; // Edge preservation
    pub const NORMAL_BINDING: u32 = 2; // Edge preservation
    pub const DIFFUSE_BINDING: u32 = 3; //edge preservation
    pub const FINAL_OUTPUT_BINDING: u32 = 4; // Final image to the screen

    pub const NUMBER_OF_BINDINGS: usize = 5;

    pub fn new(core: Rc<vulkan_abstraction::Core>) -> SrResult<Self> {
        let device = core.device().inner();

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::INPUT_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::DEPTH_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::NORMAL_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::DIFFUSE_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::FINAL_OUTPUT_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&create_info, None)? };

        Ok(Self {
            descriptor_set_layout,
            core,
        })
    }

    pub fn inner(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }
}

impl Drop for DenoiseDescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.core
                .device()
                .inner()
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

/// Wrapper struct for the denoise pass descriptor set
pub struct DenoiseDescriptorSets {
    descriptor_sets: Vec<vk::DescriptorSet>,
    descriptor_pool: vk::DescriptorPool,
    core: Rc<vulkan_abstraction::Core>,
}

impl DenoiseDescriptorSets {
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        layout: &DenoiseDescriptorSetLayout,
        temporal_results: &[vulkan_abstraction::Image; 2],
        depth_image: &vulkan_abstraction::Image,
        normal_image: &vulkan_abstraction::Image,
        diffuse_image: &vulkan_abstraction::Image,
        denoise_ping_pong_images: &[vulkan_abstraction::Image; 2],
        sampler: vk::Sampler,
    ) -> SrResult<Self> {
        let device = core.device().inner();

        // 1. Pool Sizes: 2 Storage Images per set * 3 Sets = 6 total
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(6),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(9),
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default().pool_sizes(&pool_sizes).max_sets(3);

        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        // 2. Allocate 3 Descriptor Sets
        let set_layouts = [layout.inner(), layout.inner(), layout.inner()];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);

        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };

        let create_info = |img: &vulkan_abstraction::Image| {
            vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(img.image_view())
        };

        let create_sampled_info = |img: &vulkan_abstraction::Image| {
            vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(img.image_view())
                .sampler(sampler)
        };

        // Common Textures
        let depth_info = create_sampled_info(depth_image);
        let normal_info = create_sampled_info(normal_image);
        let diffuse_info = create_sampled_info(diffuse_image);

        // Ping-Pong specific infos
        let temp_0 = create_info(&temporal_results[0]);
        let temp_1 = create_info(&temporal_results[1]);
        let denoise_0 = create_info(&denoise_ping_pong_images[0]);
        let denoise_1 = create_info(&denoise_ping_pong_images[1]);

        let mut writes = Vec::new();

        // --- SET 0: Initial Pass (Temporal 0 -> Denoise 0) ---
        // Note: You can swap temp_0 for temp_1 based on frame_count in the caller,
        // or just update this specific binding every frame.
        writes.push(self::create_write(
            descriptor_sets[0],
            DenoiseDescriptorSetLayout::INPUT_BINDING,
            &temp_0,
            vk::DescriptorType::STORAGE_IMAGE,
        ));
        writes.push(self::create_write(
            descriptor_sets[0],
            DenoiseDescriptorSetLayout::FINAL_OUTPUT_BINDING,
            &denoise_0,
            vk::DescriptorType::STORAGE_IMAGE,
        ));

        // --- SET 1: Ping-Pong A (Denoise 0 -> Denoise 1) ---
        writes.push(self::create_write(
            descriptor_sets[1],
            DenoiseDescriptorSetLayout::INPUT_BINDING,
            &denoise_0,
            vk::DescriptorType::STORAGE_IMAGE,
        ));
        writes.push(self::create_write(
            descriptor_sets[1],
            DenoiseDescriptorSetLayout::FINAL_OUTPUT_BINDING,
            &denoise_1,
            vk::DescriptorType::STORAGE_IMAGE,
        ));

        // --- SET 2: Ping-Pong B (Denoise 1 -> Denoise 0) ---
        writes.push(self::create_write(
            descriptor_sets[2],
            DenoiseDescriptorSetLayout::INPUT_BINDING,
            &denoise_1,
            vk::DescriptorType::STORAGE_IMAGE,
        ));
        writes.push(self::create_write(
            descriptor_sets[2],
            DenoiseDescriptorSetLayout::FINAL_OUTPUT_BINDING,
            &denoise_0,
            vk::DescriptorType::STORAGE_IMAGE,
        ));

        // Add Depth and Normal to all sets
        for &set in &descriptor_sets {
            writes.push(self::create_write(
                set,
                DenoiseDescriptorSetLayout::DEPTH_BINDING,
                &depth_info,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            ));
            writes.push(self::create_write(
                set,
                DenoiseDescriptorSetLayout::NORMAL_BINDING,
                &normal_info,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            ));
            writes.push(create_write(
                set,
                DenoiseDescriptorSetLayout::DIFFUSE_BINDING,
                &diffuse_info,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            ))
        }

        unsafe { device.update_descriptor_sets(&writes, &[]) };

        Ok(Self {
            core,
            descriptor_sets,
            descriptor_pool,
        })
    }

    pub fn inner(&self) -> &[vk::DescriptorSet] {
        &self.descriptor_sets
    }
}

// Helper function to keep the code clean
fn create_write(
    set: vk::DescriptorSet,
    binding: u32,
    info: &'_ vk::DescriptorImageInfo,
    d_type: vk::DescriptorType,
) -> vk::WriteDescriptorSet<'_> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(d_type)
        .image_info(std::slice::from_ref(info))
}

impl Drop for DenoiseDescriptorSets {
    fn drop(&mut self) {
        unsafe {
            self.core.device().inner().destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}
