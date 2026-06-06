use std::{ffi::CStr, rc::Rc};

use crate::error::SrResult;
use crate::vulkan_abstraction;

use ash::vk;
use ash::vk::TaggedStructure;

// should match the one defined in build.rs
const SHADER_ENTRY_POINT: &CStr = c"main";

#[allow(dead_code)] // read by the gpu
#[repr(C, packed)]
#[derive(Debug)]
pub struct RaytracingPushConstant {
    pub frame_count: u32,
    pub use_srgb: bool,
    pub _padding: [u8; 3], //push constant size must be a multiple of 4
}

/// Push-constant layout for the heap-mode (Slang) raytracing pipeline. Every
/// `DescriptorHandle<T>` field in `shaders/rt_types.slang::RaytracingPC`
/// lowers to a `uint2`, so each is mirrored here as `[u32; 2]` (low word =
/// heap shader index, high word = 0). Total size: 152 bytes — well within
/// the 256-byte minimum push-constant range required by Vulkan.
#[allow(dead_code)] // read by the gpu
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct RaytracingHeapPushConstant {
    /// AS device address (uint64) instead of a heap-handle pair — workaround for
    /// Slang issue #10671: `DescriptorHandle<RaytracingAccelerationStructure>` +
    /// `spvDescriptorHeapEXT` omits `OpConvertUToAccelerationStructureKHR`, so
    /// `TraceRayKHR` faults at runtime. The shader does the convert via inline
    /// SPIR-V (`shaders/rt_utils.slang::tlas_from_address`). Switch back to a
    /// `[u32; 2]` heap handle once the upstream Slang fix lands.
    pub tlas: u64,
    pub raw_color: [u32; 2],
    pub depth_img: [u32; 2],
    pub normal_img: [u32; 2],
    pub diffuse_img: [u32; 2],
    pub motion_vec_img: [u32; 2],
    /// Buffer-device-address of the matrices buffer (not a heap handle — see
    /// `shaders/rt_types.slang::RaytracingPC.matrices`). Still 8 bytes, so the
    /// rest of the struct layout is unchanged.
    pub matrices: u64,
    pub meshes_info: [u32; 2],
    pub emissive_triangles: [u32; 2],
    pub emissive_indirection: [u32; 2],
    pub entity_transforms: [u32; 2],
    pub blue_noise_tex: [u32; 2],
    pub blue_noise_sampler: [u32; 2],
    /// Buffer-device-addresses for the ping-pong reservoir buffers (see
    /// `shaders/rt_types.slang::RaytracingPC.reservoirs`). 16 bytes total,
    /// matching the previous `[[u32; 2]; 2]` heap-handle layout.
    pub reservoirs: [u64; 2],
    pub reservoirs_gi: [u64; 2],
    pub textures_lookup: [u32; 2],
    pub frame_count: u32,
    pub use_srgb: u32,
}

pub struct RayTracingPipeline {
    core: Rc<vulkan_abstraction::Core>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
}

impl RayTracingPipeline {
    #[deprecated(note = "This method is legacy, new_heap should be used ")]
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        descriptor_set_layout: &vulkan_abstraction::RaytracingDescriptorSetLayout,
        generate_debug_info: bool,
        ray_gen_spirv: &[u8],
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

        let ray_gen_stage_create_info = make_shader_stage_create_info(vk::ShaderStageFlags::RAYGEN_KHR, ray_gen_spirv)?;

        let ray_miss_stage_create_info = make_shader_stage_create_info(
            vk::ShaderStageFlags::MISS_KHR,
            include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/ray_miss.spirv")),
        )?;

        let any_hit_stage_create_info = make_shader_stage_create_info(
            vk::ShaderStageFlags::ANY_HIT_KHR,
            include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/any_hit.spirv")),
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
        let any_hit_stage_index = stages.len();
        stages.push(any_hit_stage_create_info);

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
            .any_hit_shader(any_hit_stage_index as u32)
            .closest_hit_shader(closest_hit_stage_index as u32)
            .general_shader(vk::SHADER_UNUSED_KHR);

        shader_groups.push(closest_hit_shader_group_create_info);

        let push_constants = [vk::PushConstantRange::default()
            .stage_flags(
                vk::ShaderStageFlags::RAYGEN_KHR
                    | vk::ShaderStageFlags::CLOSEST_HIT_KHR
                    | vk::ShaderStageFlags::MISS_KHR
                    | vk::ShaderStageFlags::ANY_HIT_KHR,
            )
            .offset(0)
            .size(std::mem::size_of::<RaytracingPushConstant>() as u32)];

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

    /// Heap-mode constructor: pipeline layout is `VK_NULL_HANDLE` and the
    /// pipeline is flagged `DESCRIPTOR_HEAP_EXT`. All descriptors and the
    /// push-constant block come from the Slang shaders' SPIR-V interface,
    /// driven at command time by `cmd_bind_resource/sampler_heap` and
    /// `cmd_push_data`. Caller supplies the four SPIR-V byte slices for
    /// ray-gen, miss, closest-hit, and any-hit.
    pub fn new_heap(
        core: Rc<vulkan_abstraction::Core>,
        ray_gen_spirv: &[u8],
        miss_spirv: &[u8],
        closest_hit_spirv: &[u8],
        any_hit_spirv: &[u8],
    ) -> SrResult<Self> {
        let device = core.device().inner();

        let make_stage = |stage: vk::ShaderStageFlags, spirv: &[u8]| -> SrResult<vk::PipelineShaderStageCreateInfo> {
            let spirv_u32 = bytemuck::cast_slice(spirv);
            let module_info = vk::ShaderModuleCreateInfo::default().code(spirv_u32);
            let module = unsafe { device.create_shader_module(&module_info, None) }?;
            Ok(vk::PipelineShaderStageCreateInfo::default()
                .name(SHADER_ENTRY_POINT)
                .module(module)
                .stage(stage))
        };

        let stages = [
            make_stage(vk::ShaderStageFlags::RAYGEN_KHR, ray_gen_spirv)?,
            make_stage(vk::ShaderStageFlags::MISS_KHR, miss_spirv)?,
            make_stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR, closest_hit_spirv)?,
            make_stage(vk::ShaderStageFlags::ANY_HIT_KHR, any_hit_spirv)?,
        ];

        let shader_groups = [
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(0)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(1)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(2)
                .any_hit_shader(3)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
        ];

        // Heap-mode requires `layout = VK_NULL_HANDLE` plus the
        // `DESCRIPTOR_HEAP_EXT` flag; the push-constant block lives in the
        // shader interface and is fed by `vkCmdPushDataEXT`.
        let mut flags2 = vk::PipelineCreateFlags2CreateInfo::default().flags(vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT);

        let pipeline_info = vk::RayTracingPipelineCreateInfoKHR::default()
            .stages(&stages)
            .groups(&shader_groups)
            .max_pipeline_ray_recursion_depth(2)
            .layout(vk::PipelineLayout::null())
            .push(&mut flags2);

        let pipelines = unsafe {
            core.rt_pipeline_device().create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            )
        }
        .map_err(|(_, e)| e)?;
        let pipeline = pipelines[0];

        for stage in &stages {
            unsafe { device.destroy_shader_module(stage.module, None) };
        }

        Ok(Self {
            core,
            pipeline,
            pipeline_layout: vk::PipelineLayout::null(),
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
        let device = self.core.device().inner();
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            // Heap-mode pipelines own no `VkPipelineLayout`; only the legacy
            // descriptor-set constructor creates one.
            if self.pipeline_layout != vk::PipelineLayout::null() {
                device.destroy_pipeline_layout(self.pipeline_layout, None);
            }
        }
    }
}
