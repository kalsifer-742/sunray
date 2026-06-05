use crate::error::SrResult;
use crate::vulkan_abstraction;
use ash::vk;
use std::rc::Rc;

pub struct TemporalAccumulationDescriptorSetLayout {
    descriptor_set_layout: vk::DescriptorSetLayout,
    core: Rc<vulkan_abstraction::Core>,
}

impl TemporalAccumulationDescriptorSetLayout {
    pub const RAW_COLOR_BINDING: u32 = 0; // Current Noisy Frame
    pub const MOTION_VECTOR_BINDING: u32 = 1; // Input from Raytrace pass
    pub const OUTPUT_IMAGES_BINDING: u32 = 2; // Array of 2 (Ping-Pong Storage)
    pub const HISTORY_SAMPLERS_BINDING: u32 = 3; // Array of 2 (Ping-Pong Sampler)

    pub const NUMBER_OF_BINDINGS: usize = 4;

    pub fn new(core: Rc<vulkan_abstraction::Core>) -> SrResult<Self> {
        let device = core.device().inner();

        let bindings = [
            // 0: Current Noisy Frame (Storage)
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::RAW_COLOR_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // 1: Motion Vectors (Storage or Sampler, usually Storage is fine)
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::MOTION_VECTOR_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // 2: Accumulation Images [Ping, Pong]
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::OUTPUT_IMAGES_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(2)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // 3: History Samplers [Ping, Pong]
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::HISTORY_SAMPLERS_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(2)
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

impl Drop for TemporalAccumulationDescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.core
                .device()
                .inner()
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

pub struct TemporalAccumulationDescriptorSets {
    descriptor_sets: Vec<vk::DescriptorSet>,
    descriptor_pool: vk::DescriptorPool,
    core: Rc<vulkan_abstraction::Core>,
}

impl TemporalAccumulationDescriptorSets {
    pub fn new(
        core: &Rc<vulkan_abstraction::Core>,
        layout: &TemporalAccumulationDescriptorSetLayout,
        input_image: &vulkan_abstraction::Image,         // Binding 0: Noisy Raytrace result
        motion_vector_image: &vulkan_abstraction::Image, // Binding 1: Motion Vectors
        accumulation_images: [&vulkan_abstraction::Image; 2], // Binding 2: Storage Images (Ping-Pong Output)
        history_images: [&vulkan_abstraction::Image; 2], // Binding 3: Samplers (Ping-Pong History)
        history_sampler: vk::Sampler,
    ) -> SrResult<Self> {
        let device = core.device().inner();

        // 1. Define Pool Sizes (4 Storage, 2 Samplers)
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(3), // 1 Motion + 2 Accumulation
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(3), // 1 Input + 2 History
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default().pool_sizes(&pool_sizes).max_sets(1);

        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        // 2. Allocate Descriptor Set
        let set_layouts = [layout.inner()];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);

        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };
        let set = descriptor_sets[0];

        // Helper closure
        let create_info = |img: &vulkan_abstraction::Image| {
            vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(img.image_view())
        };

        // 3. Create Image Infos
        let input_info = {
            vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(input_image.image_view())
                .sampler(history_sampler)
        };
        let mv_info = create_info(motion_vector_image);

        let accumulation_infos = [create_info(accumulation_images[0]), create_info(accumulation_images[1])];

        let history_infos = [
            vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(history_images[0].image_view())
                .sampler(history_sampler),
            vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(history_images[1].image_view())
                .sampler(history_sampler),
        ];

        // 4. Write to the Descriptor Set
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(TemporalAccumulationDescriptorSetLayout::RAW_COLOR_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&input_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(TemporalAccumulationDescriptorSetLayout::MOTION_VECTOR_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&mv_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(TemporalAccumulationDescriptorSetLayout::OUTPUT_IMAGES_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&accumulation_infos),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(TemporalAccumulationDescriptorSetLayout::HISTORY_SAMPLERS_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&history_infos),
        ];

        unsafe { device.update_descriptor_sets(&writes, &[]) };

        Ok(Self {
            core: Rc::clone(core),
            descriptor_sets,
            descriptor_pool,
        })
    }

    pub fn update_temporal_descriptors(&self, images: &[vulkan_abstraction::image::Image; 2], sampler: vk::Sampler) {
        let device = self.core.device().inner();

        // 1. Prepare for Binding 2 (Storage Images - Writing)
        let accum_infos = images
            .iter()
            .map(|img| {
                vk::DescriptorImageInfo::default()
                    .image_view(img.image_view())
                    .image_layout(vk::ImageLayout::GENERAL)
            })
            .collect::<Vec<_>>();

        // 2. Prepare for Binding 3 (Combined Image Samplers - Reading History)
        let history_infos = images
            .iter()
            .map(|img| {
                vk::DescriptorImageInfo::default()
                    .sampler(sampler)
                    .image_view(img.image_view())
                    .image_layout(vk::ImageLayout::GENERAL)
            })
            .collect::<Vec<_>>();

        let mut writes = Vec::new();

        // We only have one descriptor set for the Temporal pass
        let set = self.inner()[0];

        // Write Binding 2: accumulation_images[2]
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(2) // Matches our new GLSL/Layout
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .dst_array_element(0)
                .image_info(&accum_infos),
        );

        // Write Binding 3: history_samplers[2]
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(3) // Matches our new GLSL/Layout
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .dst_array_element(0)
                .image_info(&history_infos),
        );

        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
    }

    pub fn inner(&self) -> &[vk::DescriptorSet] {
        &self.descriptor_sets
    }
}

impl Drop for TemporalAccumulationDescriptorSets {
    fn drop(&mut self) {
        unsafe {
            self.core.device().inner().destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}
