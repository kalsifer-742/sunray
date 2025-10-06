use std::{ffi::CStr, rc::Rc};

use crate::error::SrResult;
use crate::vulkan_abstraction;

use ash::vk;

// should match the one defined in build.rs
const SHADER_ENTRY_POINT: &CStr = c"main";

#[allow(dead_code)] // read by the gpu
#[repr(C, packed)]
pub struct PushConstant {
    pub use_srgb: bool,
    pub _padding: [u8; 3], //push constant size must be a multiple of 4
}

pub struct RayTracingPipeline {
    core: Rc<vulkan_abstraction::Core>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
}

impl RayTracingPipeline {
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        descriptor_set_layout: &vulkan_abstraction::DescriptorSetLayout,
        generate_debug_info: bool,
    ) -> SrResult<Self> {
        if generate_debug_info {
            log::info!("Building shaders with debug symbols");
        }
        let device = core.device().inner();

        let make_shader_stage_create_info =
            |stage: vk::ShaderStageFlags, spirv: &[u8]| -> SrResult<vk::PipelineShaderStageCreateInfo> {

                let spirv_u32 = bytemuck::cast_slice(spirv);

                let module_create_info = vk::ShaderModuleCreateInfo::default()
                    .flags(vk::ShaderModuleCreateFlags::empty())
                    .code(spirv_u32);


                let module = unsafe { device.create_shader_module(&module_create_info, None) }?;

                let stage_create_info = vk::PipelineShaderStageCreateInfo::default()
                    .name(SHADER_ENTRY_POINT)
                    .module(module)
                    .stage(stage);

                Ok(stage_create_info)
            };

        let ray_gen_stage_create_info = make_shader_stage_create_info(
            vk::ShaderStageFlags::RAYGEN_KHR,
            include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/ray_gen.spirv")),
        )?;

        let ray_miss_stage_create_info = make_shader_stage_create_info(
            vk::ShaderStageFlags::MISS_KHR,
            include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/ray_miss.spirv")),
        )?;

        let closest_hit_stage_create_info = make_shader_stage_create_info(
            vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/closest_hit.spirv")),
        )?;

        let mut stages = Vec::new();
        let ray_gen_stage_index = stages.len();
        stages.push(ray_gen_stage_create_info);
        let ray_miss_stage_index = stages.len();
        stages.push(ray_miss_stage_create_info);
        let closest_hit_stage_index = stages.len();
        stages.push(closest_hit_stage_create_info);

        let mut shader_groups = Vec::new();
        assert_eq!(ray_gen_stage_index, 0);
        assert_eq!(ray_miss_stage_index, 1);
        assert_eq!(closest_hit_stage_index, 2);

        let ray_gen_shader_group_create_info = vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR)
            .general_shader(ray_gen_stage_index as u32);

        shader_groups.push(ray_gen_shader_group_create_info);

        let ray_miss_shader_group_create_info = vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR)
            .general_shader(ray_miss_stage_index as u32);

        shader_groups.push(ray_miss_shader_group_create_info);

        let closest_hit_shader_group_create_info = vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
            .intersection_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .closest_hit_shader(closest_hit_stage_index as u32)
            .general_shader(vk::SHADER_UNUSED_KHR);

        shader_groups.push(closest_hit_shader_group_create_info);

        let push_constants = [vk::PushConstantRange::default()
            .stage_flags(
                vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR | vk::ShaderStageFlags::MISS_KHR,
            )
            .offset(0)
            .size(std::mem::size_of::<PushConstant>() as u32)];

        let set_layouts = [descriptor_set_layout.inner()];

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(&push_constants)
            .set_layouts(&set_layouts);

        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None) }?;

        let pipeline_create_info = vk::RayTracingPipelineCreateInfoKHR::default()
            .stages(&stages)
            .groups(&shader_groups)
            .max_pipeline_ray_recursion_depth(2)
            .layout(pipeline_layout);

        let pipelines = unsafe {
            core.rt_pipeline_device().create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                &[pipeline_create_info],
                None,
            )
        }
        .map_err(|(_, e)| e)?;

        let pipeline = pipelines[0];

        stages.iter().for_each(|stage| unsafe {
            device.destroy_shader_module(stage.module, None);
        });

        Ok(Self {
            core,
            pipeline,
            pipeline_layout,
        })
    }

    pub fn inner(&self) -> vk::Pipeline {
        self.pipeline
    }
    pub fn layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }
}



impl Drop for RayTracingPipeline {
    fn drop(&mut self) {
        unsafe {
            self.core.device().inner().destroy_pipeline(self.pipeline, None);
            self.core.device().inner().destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
