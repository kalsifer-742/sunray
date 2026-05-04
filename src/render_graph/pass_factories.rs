use crate::render_graph::graph::{ResourceDesc, TransientResources};
use std::collections::HashMap;
use std::path::PathBuf;
use ash::vk::CommandBuffer;
use rspirv_reflect::DescriptorInfo;
use crate::error::SrResult;
use crate::vulkan_abstraction::RaytracingDescriptorSets;

pub struct RayTracingShaderDesc{
    pub(crate) shader: ShaderSource,
    pub(crate) pipeline_stage : RasterPipelineStage,
}

pub struct RasterShaderDesc{
    pub(crate) shader: ShaderSource,
    pub(crate) pipeline_stage : RasterPipelineStage,
}

pub struct ComputeShaderDesc{
    pub(crate) shader: ShaderSource,
}

pub struct DescriptorSetOps{
        
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

