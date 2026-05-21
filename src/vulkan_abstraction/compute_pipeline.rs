use crate::error::SrResult;
use crate::vulkan_abstraction::TemporalAccumulationDescriptorSetLayout;
use crate::vulkan_abstraction::{self, Core, DenoiseDescriptorSetLayout, PostProcessDescriptorSetLayout};
use ash::vk;
use ash::vk::TaggedStructure;
use std::marker::PhantomData;
use std::{ffi::CStr, rc::Rc};

const SHADER_ENTRY_POINT: &CStr = c"main";

pub trait ComputeTypeDef {
    type PushConstant;
    ///This serves two purposes, descriptor sets layout or descriptor heap layout depending on impl
    type DescriptorsLayout;
    fn spirv_bytes() -> &'static [u8];
}

pub struct DenoisePass;
pub struct TemporalPass;
pub struct PostprocessPass;

impl ComputeTypeDef for DenoisePass {
    type PushConstant = DenoisePushConstant;
    type DescriptorsLayout = DenoiseDescriptorSetLayout;
    fn spirv_bytes() -> &'static [u8] {
        include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/denoise.spirv"))
    }
}

impl ComputeTypeDef for TemporalPass {
    type PushConstant = TemporalAccumulationPushConstant;
    type DescriptorsLayout = TemporalAccumulationDescriptorSetLayout;
    fn spirv_bytes() -> &'static [u8] {
        include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/temporal_accumulation.spirv"))
    }
}

impl ComputeTypeDef for PostprocessPass {
    type PushConstant = PostprocessPushConstant;
    type DescriptorsLayout = PostProcessDescriptorSetLayout;

    fn spirv_bytes() -> &'static [u8] {
        include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/postprocess.spirv"))
    }
}





///Push Constant for the denoiser pass.
/// Frame count is self explicative.
/// Step width references the distance between each pixel used as a sample during the a-trous filtering.
#[allow(dead_code)] // read by the gpu
#[repr(C, packed)]
#[derive(Debug)]
pub struct DenoisePushConstant {
    pub frame_count: u32,
    pub step_width: u32,
    pub width: u32,
    pub height: u32,
}

#[allow(dead_code)] // read by the gpu
#[repr(C, packed)]
#[derive(Debug)]
pub struct TemporalAccumulationPushConstant {
    pub frame_count: u32,
    pub width: u32,
    pub height: u32,
}

#[allow(dead_code)] // read by the gpu
#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
pub struct PostprocessPushConstant {
    // Slang's `DescriptorHandle<T>` lowers to `uint2` (8 bytes); the `_pad` fields
    // keep `output_idx` at offset 8 and `exposure` at offset 16 to match the shader.
    pub input_idx: u32,
    pub _input_pad: u32,
    pub output_idx: u32,
    pub _output_pad: u32,
    pub exposure: f32,
}

pub struct ComputePipeline<T: ComputeTypeDef> {
    core: Rc<Core>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    _marker: PhantomData<T>,
}

impl<T: ComputeTypeDef> ComputePipeline<T> {
    /// Heap-mode constructor: pipeline layout has no descriptor sets, only push constants;
    /// the pipeline itself is flagged `DESCRIPTOR_HEAP_EXT`. Caller supplies the SPIR-V
    /// directly (e.g. from the Slang `ShaderCompiler`) since heap-mode shaders are not
    /// the build-time-baked GLSL ones referenced by `T::spirv_bytes`.
    pub fn new_heap(core: Rc<Core>, spirv_bytes: &[u8]) -> SrResult<Self> {
        let device = core.device().inner();
        let spirv_u32 = bytemuck::cast_slice(spirv_bytes);

        let module_create_info = vk::ShaderModuleCreateInfo::default().code(spirv_u32);
        let shader_module = unsafe { device.create_shader_module(&module_create_info, None) }?;

        let shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
            .name(SHADER_ENTRY_POINT)
            .module(shader_module)
            .stage(vk::ShaderStageFlags::COMPUTE);

        // VK_EXT_descriptor_heap requires `layout = VK_NULL_HANDLE` when
        // `DESCRIPTOR_HEAP_BIT_EXT` is set; the push-constant interface lives in the
        // shader's SPIR-V interface block instead, and is fed via `vkCmdPushDataEXT`.
        let mut flags2 = vk::PipelineCreateFlags2CreateInfo::default()
            .flags(vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(shader_stage_create_info)
            .layout(vk::PipelineLayout::null())
            .push(&mut flags2);

        let pipelines = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, err)| {
                    device.destroy_shader_module(shader_module, None);
                    err
                })?
        };
        let pipeline = pipelines[0];

        unsafe { device.destroy_shader_module(shader_module, None) };

        Ok(Self {
            core,
            pipeline,
            pipeline_layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            _marker: PhantomData,
        })
    }

    pub fn new(core: Rc<Core>, descriptor_set_layout: vk::DescriptorSetLayout) -> SrResult<Self> {
        let device = core.device().inner();

        // 1. Get the SPIR-V bytes from the trait implementation
        let spirv_bytes = T::spirv_bytes();
        let spirv_u32 = bytemuck::cast_slice(spirv_bytes);

        // 2. Create the Shader Module
        let module_create_info = vk::ShaderModuleCreateInfo::default().code(spirv_u32);

        let shader_module = unsafe { device.create_shader_module(&module_create_info, None) }?;

        // 3. Set up the stage info
        let shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
            .name(SHADER_ENTRY_POINT)
            .module(shader_module)
            .stage(vk::ShaderStageFlags::COMPUTE);

        // 4. Use the generic PushConstant type for size
        let size = std::mem::size_of::<T::PushConstant>() as u32;

        // 1. Only create the range if the size is actually greater than 0
        let push_constant_ranges = if size > 0 {
            vec![
                vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .offset(0)
                    .size(size),
            ]
        } else {
            // If it's a ZST, we provide an empty Vec
            Vec::new()
        };

        let set_layouts = [descriptor_set_layout];

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_constant_ranges);

        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

        // 5. Create the Pipeline
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(shader_stage_create_info)
            .layout(pipeline_layout);

        let pipelines = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, err)| {
                    // Clean up layout and module if creation fails
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_shader_module(shader_module, None);
                    err
                })?
        };
        let pipeline = pipelines[0];

        // 6. Cleanup Shader Module (it is no longer needed once the pipeline is created)
        unsafe {
            device.destroy_shader_module(shader_module, None);
        }

        Ok(Self {
            core,
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            _marker: PhantomData,
        })
    }

    // Getters for usage in the command buffer
    pub fn inner(&self) -> vk::Pipeline {
        self.pipeline
    }

    pub fn layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    pub fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }
}

impl<T: ComputeTypeDef> Drop for ComputePipeline<T> {
    fn drop(&mut self) {
        let device = self.core.device().inner();
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            // Heap-mode pipelines are constructed with a null layout; only legacy
            // descriptor-set pipelines own one.
            if self.pipeline_layout != vk::PipelineLayout::null() {
                device.destroy_pipeline_layout(self.pipeline_layout, None);
            }
        }
    }
}
