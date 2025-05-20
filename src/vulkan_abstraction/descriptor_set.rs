use std::ops::Deref;

use ash::{
    Device,
    vk::{
        DescriptorImageInfo, DescriptorPoolCreateInfo, DescriptorPoolSize,
        DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding,
        DescriptorSetLayoutCreateInfo, DescriptorType, ImageLayout, ImageView, ShaderStageFlags,
        WriteDescriptorSet, WriteDescriptorSetAccelerationStructureKHR,
    },
};

use crate::error::*;

use super::TLAS;

struct DescriptorSets {}
impl DescriptorSets {
    pub fn new(device: Device, tlas: &TLAS, output_image_view: &ImageView) -> SrResult<Self> {
        let descriptor_pool_sizes = [
            DescriptorPoolSize::default()
                .ty(DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1),
            DescriptorPoolSize::default()
                .ty(DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1),
        ];

        let descriptor_pool_create_info = DescriptorPoolCreateInfo::default()
            .pool_sizes(&descriptor_pool_sizes)
            .max_sets(1);

        let descriptor_pool =
            unsafe { device.create_descriptor_pool(&descriptor_pool_create_info, None) }
                .to_sr_result()?;

        let descriptor_set_layout_bindings = [
            // TLAS layout binding
            DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .stage_flags(ShaderStageFlags::RAYGEN_KHR),
            // output image layout binding
            DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(ShaderStageFlags::RAYGEN_KHR),
        ];

        let descriptor_set_layout_create_info =
            DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_set_layout_bindings);

        let descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
        }
        .to_sr_result()?;

        let descriptor_set_layouts = [descriptor_set_layout];

        let descriptor_set_allocate_info = DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&descriptor_set_layouts);

        let descriptor_sets =
            unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info) }
                .to_sr_result()?;

        let mut descriptor_writes = Vec::new();

        let tlases = [*tlas.deref()];

        let mut write_descriptor_set_acceleration_structure =
            WriteDescriptorSetAccelerationStructureKHR::default().acceleration_structures(&tlases);

        descriptor_writes.push(
            WriteDescriptorSet::default()
                .descriptor_type(DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .push_next(&mut write_descriptor_set_acceleration_structure)
                .dst_set(descriptor_sets[0])
                .dst_binding(0)
                .descriptor_count(1),
        );

        let descriptor_image_info = DescriptorImageInfo::default()
            .image_view(*output_image_view)
            .image_layout(ImageLayout::GENERAL);
        let descriptor_image_infos = [descriptor_image_info];

        descriptor_writes.push(
            WriteDescriptorSet::default()
                .descriptor_type(DescriptorType::STORAGE_IMAGE)
                .image_info(&descriptor_image_infos)
                .dst_set(descriptor_sets[0])
                .dst_binding(1) // TODO: make const
                .descriptor_count(1),
        );

        unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) };

        Ok(todo!())
    }
}
