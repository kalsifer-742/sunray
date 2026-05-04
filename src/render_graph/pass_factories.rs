use crate::render_graph::graph::{Handle, RawResourceHandle, ResourceDesc, TransientResources};
use std::collections::HashMap;
use std::path::PathBuf;
use ash::vk;
use ash::vk::{CommandBuffer, DescriptorSet};
use derive_builder::Builder;
use rspirv_reflect::DescriptorInfo;
use crate::error::SrResult;
use crate::vulkan_abstraction::RaytracingDescriptorSets;

/// The maximum number of descriptor sets bound simultaneously.
///
/// Vulkan guarantees a minimum hardware support of 4 bound descriptor sets
/// (`maxBoundDescriptorSets`). To maximize binding efficiency, resources are
/// grouped into sets based on their frequency of update:
///
/// * **Set 0 (Global/Frame):** Data that changes once per frame.
///   *(e.g., Camera View/Projection matrices, global time, directional lights)*
/// * **Set 1 (Pass/Scene):** Data that changes per render pass.
///   *(e.g., Environment maps, shadow maps, subpass inputs)*
/// * **Set 2 (Material):** Data that changes when switching materials.
///   *(e.g., Albedo/Normal textures, roughness/metallic factors)*
/// * **Set 3 (Object/Draw):** Data that changes per individual draw call.
///   *(e.g., Model transform matrices, animation bone data)*
pub const MAX_DESCRIPTOR_SETS: usize = 4;


//TODO unire il punto comune magari con trait
pub struct RayTracingShaderDesc{
    pub descriptor_set_opts: [Option<(u32, DescriptorSetLayoutOpts)>; MAX_DESCRIPTOR_SETS],
    pub(crate) shader: ShaderSource,
    pub(crate) pipeline_stage : RasterPipelineStage,
}

pub struct RasterShaderDesc{
    pub descriptor_set_opts: [Option<(u32, DescriptorSetLayoutOpts)>; MAX_DESCRIPTOR_SETS],
    pub(crate) shader: ShaderSource,
    pub(crate) pipeline_stage : RasterPipelineStage,
}

pub struct ComputeShaderDesc{
    pub descriptor_set_opts: [Option<(u32, DescriptorSetLayoutOpts)>; MAX_DESCRIPTOR_SETS],
    pub(crate) shader: ShaderSource,
}



type DescriptorSetLayout = HashMap<u32, rspirv_reflect::DescriptorInfo>;
type StageDescriptorSetLayouts = HashMap<u32, DescriptorSetLayout>;

#[derive(Builder, Default, Debug, Clone)]
#[builder(pattern = "owned", derive(Clone))]
pub struct DescriptorSetLayoutOpts {
    #[builder(setter(strip_option), default)]
    pub flags: Option<vk::DescriptorSetLayoutCreateFlags>,
    #[builder(setter(strip_option), default)]
    pub replace: Option<DescriptorSetLayout>,
}

struct RgComputePipeline {
    //TODO
}

struct RgRasterPipeline {
    //TODO
}
struct RgRaytracingPipeline {
    raytracing_descriptor_sets: DescriptorInfo,
    shaders : Vec<RayTracingShaderDesc> ,

}


#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug)]
pub enum RayTracingPipelineStage{
    RayGen,
    RayMiss,
    RayClosestHit,
}

#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug)]
pub enum RasterPipelineStage { //TODO check for missing since I don't raster yet
    Vertex,
    Pixel,
}

pub trait ShaderDesc{

}

pub enum ShaderSource {
    //TODO supported shaders, for now glsl
    Glsl(PathBuf)
}
pub trait RenderPassFactory<Desc : ShaderDesc>{


    fn render_fn(self) -> ( Box<DynRenderFn>, CommonPipelineData , Vec<Desc> );
}
pub(crate) type DynRenderFn = dyn FnOnce(&mut CommandBuffer, &mut TransientResources) -> SrResult<()>; //TODO TransientResources here is intended to be a way to dereference the resources,but this implies it handles also external ones

struct CommonPipelineData {
    //TODO dont think this is correct
    descriptor_set:  HashMap<u32,rspirv_reflect::DescriptorInfo >,

}

