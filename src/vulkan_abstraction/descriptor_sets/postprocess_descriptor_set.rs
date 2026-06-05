use crate::error::SrResult;
use crate::vulkan_abstraction;
use ash::vk;
use std::rc::Rc;

pub struct PostprocessDescriptorSetLayout {
    descriptor_set_layout: vk::DescriptorSetLayout,
    core: Rc<vulkan_abstraction::Core>,
}

impl PostprocessDescriptorSetLayout {
    pub const INPUT_IMAGE: u32 = 0;
    pub const OUTPUT_IMAGE: u32 = 1;
    pub const NUMBER_OF_BINDINGS: usize = 2;

    pub fn new(core: Rc<vulkan_abstraction::Core>) -> SrResult<Self> {
        let device = core.device().inner();

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::INPUT_IMAGE)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::OUTPUT_IMAGE)
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

impl Drop for PostprocessDescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.core
                .device()
                .inner()
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

pub struct PostProcessDescriptorSets {
    descriptor_sets: Vec<vk::DescriptorSet>,
    descriptor_pool: vk::DescriptorPool,

    core: Rc<vulkan_abstraction::Core>,
}

impl PostProcessDescriptorSets {
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        layout: &PostprocessDescriptorSetLayout,
        denoised_input_images: [&vulkan_abstraction::Image; 2],
        output_image: &vulkan_abstraction::Image, // The final output image
    ) -> SrResult<Self> {
        let device = core.device().inner();

        // 1. We now need 4 storage image descriptors total (2 bindings * 2 sets)
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(4)];

        let pool_info = vk::DescriptorPoolCreateInfo::default().pool_sizes(&pool_sizes).max_sets(2); // Allocate 2 sets

        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        // 2. Allocate 2 Descriptor Sets
        let set_layouts = [layout.inner(), layout.inner()];

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);

        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };

        // Helper
        let create_info = |img: &vulkan_abstraction::Image| {
            vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(img.image_view())
        };

        // 3. Image Infos
        let input_info_0 = create_info(denoised_input_images[0]);
        let input_info_1 = create_info(denoised_input_images[1]);
        let output_info = create_info(output_image);

        let writes = [
            // --- SET 0 (Reads from Denoise Ping) ---
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_sets[0])
                .dst_binding(PostprocessDescriptorSetLayout::INPUT_IMAGE)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&input_info_0)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_sets[0])
                .dst_binding(PostprocessDescriptorSetLayout::OUTPUT_IMAGE)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&output_info)),
            // --- SET 1 (Reads from Denoise Pong) ---
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_sets[1])
                .dst_binding(PostprocessDescriptorSetLayout::INPUT_IMAGE)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&input_info_1)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_sets[1])
                .dst_binding(PostprocessDescriptorSetLayout::OUTPUT_IMAGE)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&output_info)),
        ];

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
impl Drop for PostProcessDescriptorSets {
    fn drop(&mut self) {
        unsafe {
            self.core.device().inner().destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}
