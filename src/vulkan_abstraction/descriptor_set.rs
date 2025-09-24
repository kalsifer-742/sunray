use std::{rc::Rc};

use ash::vk;

use crate::{error::*, vulkan_abstraction};

use vulkan_abstraction::TLAS;

pub struct DescriptorSets {
    core: Rc<vulkan_abstraction::Core>,
    descriptor_sets: Vec<vk::DescriptorSet>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
}

impl DescriptorSets {
    const TLAS_BINDING: u32 = 0;
    const IMAGE_BINDING: u32 = 1;
    const UNIFORM_BUFFER_BINDING: u32 = 2;

    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        tlas: &TLAS,
        output_image_view: &vk::ImageView,
        uniform_buffer: &vulkan_abstraction::Buffer,
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
        ];

        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&descriptor_pool_sizes)
            .max_sets(1);

        let descriptor_pool =
            unsafe { device.create_descriptor_pool(&descriptor_pool_create_info, None) }?;

        let descriptor_set_layout_bindings = [
            // TLAS layout binding
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::TLAS_BINDING)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            // output image layout binding
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::IMAGE_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            // uniform buffer layout binding
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::UNIFORM_BUFFER_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL),
        ];

        let descriptor_set_layout_create_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_set_layout_bindings);

        let descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
        }?;

        let descriptor_set_layouts = vec![descriptor_set_layout];

        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&descriptor_set_layouts);

        let descriptor_sets =
            unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info) }?;

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
                .dst_set(descriptor_sets[0])
                .dst_binding(Self::TLAS_BINDING)
                .descriptor_count(1),
        );

        // write image to descriptor set
        let descriptor_image_infos = [
            vk::DescriptorImageInfo::default()
                .image_view(*output_image_view)
                .image_layout(vk::ImageLayout::GENERAL)
        ];
        descriptor_writes.push(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&descriptor_image_infos)
                .dst_set(descriptor_sets[0])
                .dst_binding(Self::IMAGE_BINDING)
                .descriptor_count(1),
        );

        // write uniform buffer to descriptor set
        let descriptor_buffer_infos = [
            vk::DescriptorBufferInfo::default()
                .buffer(uniform_buffer.inner())
                .range(vk::WHOLE_SIZE)
        ];
        descriptor_writes.push(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&descriptor_buffer_infos)
                .dst_set(descriptor_sets[0])
                .dst_binding(Self::UNIFORM_BUFFER_BINDING)
                .descriptor_count(1)
        );

        assert_eq!(descriptor_writes.len(), 3);


        unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) };

        Ok(Self {
            core,
            descriptor_sets,
            descriptor_pool,
            descriptor_set_layouts,
        })
    }

    pub fn get_layouts(&self) -> &[vk::DescriptorSetLayout] {
        &self.descriptor_set_layouts
    }
    pub fn get_handles(&self) -> &[vk::DescriptorSet] {
        &self.descriptor_sets
    }
}

impl Drop for DescriptorSets {
    fn drop(&mut self) {
        //only do this if you set VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT
        //unsafe { self.core.device().free_descriptor_sets(self.descriptor_pool, &self.descriptor_sets) }.unwrap();

        unsafe { self.core.device().inner().destroy_descriptor_pool(self.descriptor_pool, None) };

        for layout in self.descriptor_set_layouts.iter() {
            unsafe { self.core.device().inner().destroy_descriptor_set_layout(*layout, None) };
        }
    }
}
