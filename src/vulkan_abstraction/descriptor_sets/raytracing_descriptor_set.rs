use crate::error::SrResult;
use crate::vulkan_abstraction;
use crate::vulkan_abstraction::{Buffer, TLAS};
use ash::vk;
use std::rc::Rc;
use ash::vk::TaggedStructure;

pub struct RaytracingDescriptorSetLayout {
    descriptor_set_layout: vk::DescriptorSetLayout,

    core: Rc<vulkan_abstraction::Core>,
}

impl RaytracingDescriptorSetLayout {
    pub const TLAS_BINDING: u32 = 0;
    pub const OUTPUT_IMAGE_BINDING: u32 = 1;
    pub const MATRICES_UNIFORM_BUFFER_BINDING: u32 = 2;
    pub const MESHES_INFO_STORAGE_BUFFER_BINDING: u32 = 3;
    pub const SAMPLERS_BINDING: u32 = 4;
    pub const DEPTH_BINDING: u32 = 5;
    pub const NORMAL_BINDING: u32 = 6;
    pub const DIFFUSE_BINDING: u32 = 7;
    pub const MOTION_VECTOR_BINDING: u32 = 8;
    pub const EMISSIVE_TRIANGLES_BINDING: u32 = 9;
    pub const BLUE_NOISE_BINDING: u32 = 10;
    pub const EMISSIVE_INDIRECTION_BINDING: u32 = 11;
    pub const ENTITY_TRANSFORMS_BINDING: u32 = 12;
    pub const RESERVOIR_BUFFERS_BINDING: u32 = 13;
    pub const RESERVOIR_GI_BUFFERS_BINDING: u32 = 14;
    pub const NUMBER_OF_BINDINGS: usize = 15;

    pub const NUMBER_OF_SAMPLERS: u32 = vulkan_abstraction::ResourceManager::NUMBER_OF_SAMPLERS as u32;

    pub fn new(core: Rc<vulkan_abstraction::Core>) -> SrResult<Self> {
        let device = core.device().inner();

        // TODO: check the stage_flags for each binding
        let descriptor_set_layout_bindings = [
            // TLAS layout binding
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::TLAS_BINDING)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR),
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
            // samplers buffer layout binding
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::SAMPLERS_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(Self::NUMBER_OF_SAMPLERS)
                .stage_flags(vk::ShaderStageFlags::ALL),
            // Depth layout binding
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::DEPTH_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            // normal layout binding
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::NORMAL_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::DIFFUSE_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            // motion vectors binding
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::MOTION_VECTOR_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::EMISSIVE_TRIANGLES_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::BLUE_NOISE_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            // Emissive indirection buffer (dense NEE sampling entries)
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::EMISSIVE_INDIRECTION_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            // Entity transforms buffer (indexed by arena slot)
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::ENTITY_TRANSFORMS_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            //ping pong buffers for ReSTIR, accessed as reservoirs[frame_parity] on the shader side
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::RESERVOIR_BUFFERS_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(2)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            //ping pong buffers for ReSTIR GI, accessed as reservoirs_gi[frame_parity] on the shader side
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::RESERVOIR_GI_BUFFERS_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(2)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
        ];

        let descriptor_set_layout_create_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_set_layout_bindings);

        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&descriptor_set_layout_create_info, None) }?;

        Ok(Self {
            descriptor_set_layout,
            core,
        })
    }

    pub fn inner(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }
}
impl Drop for RaytracingDescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.core
                .device()
                .inner()
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None)
        };
    }
}

pub struct RaytracingDescriptorSets {
    descriptor_sets: Vec<vk::DescriptorSet>,
    descriptor_pool: vk::DescriptorPool,

    core: Rc<vulkan_abstraction::Core>,
}

impl RaytracingDescriptorSets {
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        descriptor_set_layout: &RaytracingDescriptorSetLayout,
        tlas: &TLAS,
        output_image: &vulkan_abstraction::Image,
        depth_image: &vulkan_abstraction::Image,
        normal_image: &vulkan_abstraction::Image,
        diffuse_image: &vulkan_abstraction::Image,
        motion_vector_image: &vulkan_abstraction::Image,
        blue_noise_image: &vulkan_abstraction::Image,
        blue_noise_sampler: vk::Sampler,
        reservoir_buffers: &[vulkan_abstraction::GpuOnlyBuffer; 2],
        reservoir_gi_buffers: &[vulkan_abstraction::GpuOnlyBuffer; 2],
        shader_data: &vulkan_abstraction::ResourceManager,
    ) -> SrResult<Self> {
        let device = core.device().inner();
        let descriptor_pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(5), // 1 Output + 1 Depth + 1 Normal + 1 Motion Vector + 1 Diffuse
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER) //Meshes info + Emissive triangles+ Indirection + Entity transforms + 2 ping pong Reservoir buffers  + 2 ping pong GI Reservoir buffers
                .descriptor_count(8),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(RaytracingDescriptorSetLayout::NUMBER_OF_SAMPLERS + 1), //The +1 is for the blue noise texture
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
            vk::WriteDescriptorSetAccelerationStructureKHR::default().acceleration_structures(&tlases);

        // Using push() for writes, just like your original code
        let mut push_write = |write| descriptor_writes.push(write);

        push_write(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .push(&mut write_descriptor_set_acceleration_structure)
                .dst_set(descriptor_sets[0])
                .dst_binding(RaytracingDescriptorSetLayout::TLAS_BINDING)
                .descriptor_count(1), // Added descriptor_count
        );

        // --- NEW: Helper for images to keep code clean ---
        let create_info = |img: &vulkan_abstraction::Image| {
            [vk::DescriptorImageInfo::default()
                .image_view(img.image_view())
                .image_layout(vk::ImageLayout::GENERAL)]
        };

        let output_info = create_info(output_image);
        let depth_info = create_info(depth_image);
        let normal_info = create_info(normal_image);
        let diffuse_info = create_info(diffuse_image);
        let mv_info = create_info(motion_vector_image);

        // Write Output Image
        push_write(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&output_info)
                .dst_set(descriptor_sets[0])
                .dst_binding(RaytracingDescriptorSetLayout::OUTPUT_IMAGE_BINDING),
        );

        // Write Depth Image
        push_write(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&depth_info)
                .dst_set(descriptor_sets[0])
                .dst_binding(RaytracingDescriptorSetLayout::DEPTH_BINDING),
        );

        // Write Normal Image
        push_write(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&normal_info)
                .dst_set(descriptor_sets[0])
                .dst_binding(RaytracingDescriptorSetLayout::NORMAL_BINDING),
        );

        push_write(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&diffuse_info)
                .dst_set(descriptor_sets[0])
                .dst_binding(RaytracingDescriptorSetLayout::DIFFUSE_BINDING),
        );

        // Write Motion Vector Image
        push_write(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&mv_info)
                .dst_set(descriptor_sets[0])
                .dst_binding(RaytracingDescriptorSetLayout::MOTION_VECTOR_BINDING),
        );

        // write matrices uniform buffer to descriptor set
        let descriptor_buffer_infos = [vk::DescriptorBufferInfo::default()
            .buffer(shader_data.get_matrices_uniform_buffer())
            .range(vk::WHOLE_SIZE)];
        push_write(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&descriptor_buffer_infos)
                .dst_set(descriptor_sets[0])
                .dst_binding(RaytracingDescriptorSetLayout::MATRICES_UNIFORM_BUFFER_BINDING),
        );

        // write meshes info uniform buffer to descriptor set
        let descriptor_buffer_infos = [vk::DescriptorBufferInfo::default()
            .buffer(shader_data.get_meshes_info_storage_buffer())
            .range(vk::WHOLE_SIZE)];
        push_write(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&descriptor_buffer_infos)
                .dst_set(descriptor_sets[0])
                .dst_binding(RaytracingDescriptorSetLayout::MESHES_INFO_STORAGE_BUFFER_BINDING),
        );

        let emissive_buffer_infos = [vk::DescriptorBufferInfo::default()
            .buffer(shader_data.get_emissive_triangles_storage_buffer())
            .range(vk::WHOLE_SIZE)];
        push_write(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&emissive_buffer_infos)
                .dst_set(descriptor_sets[0])
                .dst_binding(RaytracingDescriptorSetLayout::EMISSIVE_TRIANGLES_BINDING),
        );

        // write emissive indirection buffer
        let indirection_buffer_infos = [vk::DescriptorBufferInfo::default()
            .buffer(shader_data.get_emissive_indirection_buffer())
            .range(vk::WHOLE_SIZE)];
        push_write(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&indirection_buffer_infos)
                .dst_set(descriptor_sets[0])
                .dst_binding(RaytracingDescriptorSetLayout::EMISSIVE_INDIRECTION_BINDING),
        );

        // write entity transforms buffer
        let entity_transforms_infos = [vk::DescriptorBufferInfo::default()
            .buffer(shader_data.get_entity_transforms_buffer())
            .range(vk::WHOLE_SIZE)];
        push_write(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&entity_transforms_infos)
                .dst_set(descriptor_sets[0])
                .dst_binding(RaytracingDescriptorSetLayout::ENTITY_TRANSFORMS_BINDING),
        );

        // write samplers to descriptor set
        assert_eq!(
            shader_data.get_textures().len(),
            RaytracingDescriptorSetLayout::NUMBER_OF_SAMPLERS as usize
        );

        let descriptor_sampler_infos = shader_data
            .get_textures()
            .iter()
            .map(|(sampler, image_view)| {
                vk::DescriptorImageInfo::default()
                    .sampler(*sampler)
                    .image_view(*image_view)
                    .image_layout(vk::ImageLayout::GENERAL)
            })
            .collect::<Vec<_>>();

        push_write(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&descriptor_sampler_infos)
                .dst_set(descriptor_sets[0])
                .dst_binding(RaytracingDescriptorSetLayout::SAMPLERS_BINDING),
        );

        let blue_noise_info = [vk::DescriptorImageInfo::default()
            .image_view(blue_noise_image.image_view())
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .sampler(blue_noise_sampler)];

        push_write(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&blue_noise_info)
                .dst_set(descriptor_sets[0])
                .dst_binding(RaytracingDescriptorSetLayout::BLUE_NOISE_BINDING),
        );

        let reservoir_infos = [
            vk::DescriptorBufferInfo::default()
                .buffer(reservoir_buffers[0].inner())
                .range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default()
                .buffer(reservoir_buffers[1].inner())
                .range(vk::WHOLE_SIZE),
        ];
        push_write(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&reservoir_infos)
                .dst_set(descriptor_sets[0])
                .dst_binding(RaytracingDescriptorSetLayout::RESERVOIR_BUFFERS_BINDING),
        );

        let reservoir_gi_infos = [
            vk::DescriptorBufferInfo::default()
                .buffer(reservoir_gi_buffers[0].inner())
                .range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default()
                .buffer(reservoir_gi_buffers[1].inner())
                .range(vk::WHOLE_SIZE),
        ];
        push_write(
            vk::WriteDescriptorSet::default()
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&reservoir_gi_infos)
                .dst_set(descriptor_sets[0])
                .dst_binding(RaytracingDescriptorSetLayout::RESERVOIR_GI_BUFFERS_BINDING),
        );

        unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) };

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

impl Drop for RaytracingDescriptorSets {
    fn drop(&mut self) {
        //only do this if you set VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT
        //unsafe { self.core.device().free_descriptor_sets(self.descriptor_pool, &self.descriptor_sets) }.unwrap();

        unsafe { self.core.device().inner().destroy_descriptor_pool(self.descriptor_pool, None) };
    }
}
