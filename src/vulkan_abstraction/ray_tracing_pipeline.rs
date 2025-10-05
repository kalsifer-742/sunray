use std::collections::HashMap;
use std::{ffi::CStr, rc::Rc};

use crate::error::SrResult;
use crate::vulkan_abstraction;
use shaderc::ShaderKind;

use ash::vk;

const SHADER_ENTRY_POINT: &CStr = c"main";

fn compile_shader_internal(
    src: &str,
    file_name: &str,
    shader_type: shaderc::ShaderKind,
    generate_debug_info: bool,
    header_hashmap: &HashMap<&str, &str>,
) -> shaderc::Result<shaderc::CompilationArtifact> {
    //TODO: unwrap
    let compiler = shaderc::Compiler::new().unwrap();

    let mut options = shaderc::CompileOptions::new().unwrap();
    if generate_debug_info {
        options.set_generate_debug_info();
    }
    options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_4 as u32);
    options.set_include_callback(|included_file_name, _included_type, _including_file_name, _include_depth| {
        if _included_type == shaderc::IncludeType::Relative {
            return Err("Found relative include; only standard include is allowed".to_string());
        }
        match header_hashmap.get(&included_file_name) {
            Some(source) => Ok(shaderc::ResolvedInclude { resolved_name: included_file_name.to_string(), content: source.to_string() }),
            None => Err(format!("no such shader header found; remember that it must first be included in the header hashmap to be found by the preprocessor!")),
        }
    });

    let preprocessed = compiler
        .preprocess(src, file_name, SHADER_ENTRY_POINT.to_str().unwrap(), Some(&options))?
        .as_text();

    let binary_result = compiler.compile_into_spirv(
        &preprocessed,
        shader_type,
        file_name,
        SHADER_ENTRY_POINT.to_str().unwrap(),
        Some(&options),
    )?;

    Ok(binary_result)
}

macro_rules! compile_shader {
    ($file_name : expr, $shader_type : expr, $generate_debug_info: expr, $header_hashmap: expr) => {
        compile_shader_internal(
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/", $file_name)),
            $file_name,
            $shader_type,
            $generate_debug_info,
            $header_hashmap,
        )
    };
}

macro_rules! add_shader_header_to_hashmap {
    ($file_name : expr, $hashmap: ident) => {
        $hashmap.insert($file_name, include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/", $file_name)));
    };
}

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

        let mut shader_header_hashmap = HashMap::<&str, &str>::new();
        add_shader_header_to_hashmap!("shaders/common.glsl", shader_header_hashmap);
        add_shader_header_to_hashmap!("shaders/utils.glsl", shader_header_hashmap);

        let make_shader_stage_create_info =
            |stage: vk::ShaderStageFlags, spirv: shaderc::CompilationArtifact| -> SrResult<vk::PipelineShaderStageCreateInfo> {
                let module_create_info = vk::ShaderModuleCreateInfo::default()
                    .code(spirv.as_binary())
                    .flags(vk::ShaderModuleCreateFlags::empty());

                let module = unsafe { device.create_shader_module(&module_create_info, None) }?;

                let stage_create_info = vk::PipelineShaderStageCreateInfo::default()
                    .name(SHADER_ENTRY_POINT)
                    .module(module)
                    .stage(stage);

                Ok(stage_create_info)
            };

        let ray_gen_stage_create_info = make_shader_stage_create_info(
            vk::ShaderStageFlags::RAYGEN_KHR,
            compile_shader!(
                "shaders/ray_gen.glsl",
                ShaderKind::RayGeneration,
                generate_debug_info,
                &shader_header_hashmap
            )?,
        )?;

        let ray_miss_stage_create_info = make_shader_stage_create_info(
            vk::ShaderStageFlags::MISS_KHR,
            compile_shader!(
                "shaders/ray_miss.glsl",
                ShaderKind::Miss,
                generate_debug_info,
                &shader_header_hashmap
            )?,
        )?;

        let closest_hit_stage_create_info = make_shader_stage_create_info(
            vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            compile_shader!(
                "shaders/closest_hit.glsl",
                ShaderKind::ClosestHit,
                generate_debug_info,
                &shader_header_hashmap
            )?,
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
