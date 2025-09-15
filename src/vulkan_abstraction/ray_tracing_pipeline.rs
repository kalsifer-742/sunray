use std::{ffi::CStr, rc::Rc};

use crate::error::SrResult;
use crate::vulkan_abstraction;

use ash::{vk};

const SHADER_ENTRY_POINT: &CStr = c"main";

fn compile_shader_internal(
    src: &str,
    file_name: &str,
    shader_type: shaderc::ShaderKind,
) -> shaderc::CompilationArtifact {
    //TODO: unwrap
    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_4 as u32);


    let binary_result = compiler
        .compile_into_spirv(
            src,
            shader_type,
            file_name,
            SHADER_ENTRY_POINT.to_str().unwrap(),
            Some(&options),
        )
        .unwrap();

    binary_result
}

macro_rules! compile_shader {
    ($file_name : expr, $shader_type : expr) => {
        compile_shader_internal(
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/", $file_name)),
            $file_name,
            $shader_type,
        )
    };
}

#[allow(dead_code)] // read by the gpu
// for now push constants are never used, but it is enough to specify a value in this struct, to instantiate it in lib.rs and to add in the shader a uniform block with "layout(push_constant)" for it to work
pub struct PushConstant {
    pub clear_color: [f32; 4],
}

pub struct RayTracingPipeline {
    core: Rc<vulkan_abstraction::Core>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
}

impl RayTracingPipeline {
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        descriptor_sets: &vulkan_abstraction::DescriptorSets,
    ) -> SrResult<Self> {
        let mut stages = Vec::new();
        let device = core.device().inner();

        let ray_gen_module = {
            let spirv = compile_shader!("shaders/ray_gen.glsl", shaderc::ShaderKind::RayGeneration);

            let create_info = vk::ShaderModuleCreateInfo::default()
                .code(spirv.as_binary())
                .flags(vk::ShaderModuleCreateFlags::empty());

            unsafe { device.create_shader_module(&create_info, None) }?
        };
        let ray_gen_create_info = vk::PipelineShaderStageCreateInfo::default()
            .name(SHADER_ENTRY_POINT)
            .module(ray_gen_module)
            .stage(vk::ShaderStageFlags::RAYGEN_KHR);


        let ray_miss_module = {
            let spirv =
                compile_shader!("shaders/ray_miss.glsl", shaderc::ShaderKind::Miss);

            let create_info = vk::ShaderModuleCreateInfo::default()
                .code(spirv.as_binary())
                .flags(vk::ShaderModuleCreateFlags::empty());

            unsafe { device.create_shader_module(&create_info, None) }?
        };
        let ray_miss_create_info = vk::PipelineShaderStageCreateInfo::default()
            .name(SHADER_ENTRY_POINT)
            .module(ray_miss_module)
            .stage(vk::ShaderStageFlags::MISS_KHR);


        let closest_hit_module = {
            let spirv =
                compile_shader!("shaders/closest_hit.glsl", shaderc::ShaderKind::ClosestHit);

            let create_info = vk::ShaderModuleCreateInfo::default()
                .code(spirv.as_binary())
                .flags(vk::ShaderModuleCreateFlags::empty());

            unsafe { device.create_shader_module(&create_info, None) }?
        };
        let closest_hit_create_info = vk::PipelineShaderStageCreateInfo::default()
            .name(SHADER_ENTRY_POINT)
            .module(closest_hit_module)
            .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR);

        let ray_gen_stage_index = stages.len();
        stages.push(ray_gen_create_info);
        let ray_miss_stage_index = stages.len();
        stages.push(ray_miss_create_info);
        let closest_hit_stage_index = stages.len();
        stages.push(closest_hit_create_info);

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
            .closest_hit_shader(closest_hit_stage_index as u32);

        shader_groups.push(closest_hit_shader_group_create_info);

        let push_constants = [vk::PushConstantRange::default()
            .stage_flags(
                vk::ShaderStageFlags::RAYGEN_KHR
                | vk::ShaderStageFlags::CLOSEST_HIT_KHR
                | vk::ShaderStageFlags::MISS_KHR,
            )
            .offset(0)
            .size(std::mem::size_of::<PushConstant>() as u32)];

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(&push_constants)
            .set_layouts(descriptor_sets.get_layouts());

        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None) }?;

        let pipeline_create_info = vk::RayTracingPipelineCreateInfoKHR::default()
            .stages(&stages)
            .groups(&shader_groups)
            .max_pipeline_ray_recursion_depth(1)
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

    pub fn get_handle(&self) -> vk::Pipeline { self.pipeline }
    pub fn get_layout(&self) -> vk::PipelineLayout { self.pipeline_layout }
}

impl Drop for RayTracingPipeline {
    fn drop(&mut self) {
        unsafe {
            self.core.device().inner().destroy_pipeline(self.pipeline, None);
            self.core.device().inner().destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
