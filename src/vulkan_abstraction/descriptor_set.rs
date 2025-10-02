use std::rc::Rc;

use ash::vk;

use crate::{error::*, vulkan_abstraction};

use vulkan_abstraction::TLAS;

pub struct DescriptorSetLayout {
    descriptor_set_layout: vk::DescriptorSetLayout,

    core: Rc<vulkan_abstraction::Core>,
}


impl DescriptorSetLayout {
    pub const TLAS_BINDING: u32 = 0;
    pub const OUTPUT_IMAGE_BINDING: u32 = 1;
    pub const MATRICES_UNIFORM_BUFFER_BINDING: u32 = 2;
    pub const MESHES_INFO_STORAGE_BUFFER_BINDING: u32 = 3;
    pub const MATERIALS_STORAGE_BUFFER_BINDING: u32 = 4;
    pub const SAMPLERS_BINDING: u32 = 5;
    pub const NUMBER_OF_BINDINGS: usize = 6;

    pub const NUMBER_OF_SAMPLERS : u32 = vulkan_abstraction::ShaderDataBuffers::NUMBER_OF_SAMPLERS as u32;

    pub fn new(core: Rc<vulkan_abstraction::Core>) -> SrResult<Self> {
        let device = core.device().inner();

        // TODO: check the stage_flags for each binding
        let descriptor_set_layout_bindings = [
            // TLAS layout binding
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::TLAS_BINDING)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            // output image layout binding
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::OUTPUT_IMAGE_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            // matrices uniform buffer layout binding
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::MATRICES_UNIFORM_BUFFER_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL),
            // meshes info uniform buffer layout binding
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::MESHES_INFO_STORAGE_BUFFER_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL),
            // materials storage buffer layout binding
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::MATERIALS_STORAGE_BUFFER_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL),
            // samplers buffer layout binding
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::SAMPLERS_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(Self::NUMBER_OF_SAMPLERS)
                .stage_flags(vk::ShaderStageFlags::ALL),
        ];

        let descriptor_set_layout_create_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_set_layout_bindings);

        let descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
        }?;

        Ok(Self {
            descriptor_set_layout,
            core,
        })
    }

    pub fn inner(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }
}
impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.core
                .device()
                .inner()
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None)
        };
    }
}

pub(crate) struct DescriptorSets {
    descriptor_sets: Vec<vk::DescriptorSet>,
    descriptor_pool: vk::DescriptorPool,

    core: Rc<vulkan_abstraction::Core>,
}

impl DescriptorSets {
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        descriptor_set_layout: &vulkan_abstraction::DescriptorSetLayout,
        tlas: &TLAS,
        output_image_view: vk::ImageView,
        shader_data: &vulkan_abstraction::ShaderDataBuffers,
    ) -> SrResult<Self> {
        let device = core.device().inner();
        let descriptor_pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(2),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(DescriptorSetLayout::NUMBER_OF_SAMPLERS),
        ];

        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&descriptor_pool_sizes)
            .max_sets(1);

        let descriptor_pool = unsafe { device.create_descriptor_pool(&descriptor_pool_create_info, None) }?;

        let descriptor_set_layouts = [descriptor_set_layout.inner()];

        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&descriptor_set_layouts);

        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info) }?;

        let mut descriptor_writes = Vec::new();

        // write TLAS to descriptor set
        let tlases = [tlas.inner()];
        let mut write_descriptor_set_acceleration_structure =
            vk::WriteDescriptorSetAccelerationStructureKHR::default()
                .acceleration_structures(&tlases);
        descriptor_writes.push(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .push_next(&mut write_descriptor_set_acceleration_structure)
                .descriptor_count(1)
                .dst_set(descriptor_sets[0])
                .dst_binding(DescriptorSetLayout::TLAS_BINDING)
        );

        // write image to descriptor set
        let descriptor_image_infos = [
            vk::DescriptorImageInfo::default()
                .image_view(output_image_view)
                .image_layout(vk::ImageLayout::GENERAL)
        ];
        descriptor_writes.push(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&descriptor_image_infos)
                .dst_set(descriptor_sets[0])
                .dst_binding(DescriptorSetLayout::OUTPUT_IMAGE_BINDING)
        );

        // write matrices uniform buffer to descriptor set
        let descriptor_buffer_infos = [
            vk::DescriptorBufferInfo::default()
                .buffer(shader_data.get_matrices_uniform_buffer())
                .range(vk::WHOLE_SIZE)
        ];
        descriptor_writes.push(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&descriptor_buffer_infos)
                .dst_set(descriptor_sets[0])
                .dst_binding(DescriptorSetLayout::MATRICES_UNIFORM_BUFFER_BINDING)
        );

        // write meshes info uniform buffer to descriptor set
        let descriptor_buffer_infos = [
            vk::DescriptorBufferInfo::default()
                .buffer(shader_data.get_meshes_info_storage_buffer())
                .range(vk::WHOLE_SIZE)
        ];
        descriptor_writes.push(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&descriptor_buffer_infos)
                .dst_set(descriptor_sets[0])
                .dst_binding(DescriptorSetLayout::MESHES_INFO_STORAGE_BUFFER_BINDING)
        );

        // write materials storage buffer to descriptor set
        let descriptor_buffer_infos = [
            vk::DescriptorBufferInfo::default()
                .buffer(shader_data.get_materials_storage_buffer())
                .range(vk::WHOLE_SIZE)
        ];
        descriptor_writes.push(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&descriptor_buffer_infos)
                .dst_set(descriptor_sets[0])
                .dst_binding(DescriptorSetLayout::MATERIALS_STORAGE_BUFFER_BINDING)
        );

        // write samplers to descriptor set
        assert_eq!(shader_data.get_textures().len(), DescriptorSetLayout::NUMBER_OF_SAMPLERS as usize);

        let descriptor_sampler_infos =
            shader_data.get_textures().iter()
            .map(|(sampler, image_view)|
                vk::DescriptorImageInfo::default()
                    .sampler(*sampler)
                    .image_view(*image_view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            )
            .collect::<Vec<_>>();

        descriptor_writes.push(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&descriptor_sampler_infos)
                .dst_set(descriptor_sets[0])
                .dst_binding(DescriptorSetLayout::SAMPLERS_BINDING)
        );

        unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) };

        Ok(Self {
            core,
            descriptor_sets,
            descriptor_pool,
        })
    }

    pub fn get_handles(&self) -> &[vk::DescriptorSet] {
        &self.descriptor_sets
    }
}

impl Drop for DescriptorSets {
    fn drop(&mut self) {
        //only do this if you set VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT
        //unsafe { self.core.device().free_descriptor_sets(self.descriptor_pool, &self.descriptor_sets) }.unwrap();

        unsafe {
            self.core
                .device()
                .inner()
                .destroy_descriptor_pool(self.descriptor_pool, None)
        };
    }
}
