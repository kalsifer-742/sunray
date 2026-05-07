use crate::error::{SrError, SrResult};
use crate::render_graph::graph::{AnyRenderPass, Handle, PassResourceAccessSyncType, PassResourceAccessType, RawResourceHandle, RenderGraph, Resource, ResourceRef, Setup, TransientResources};
use ash::vk;
use ash::vk::CommandBuffer;
use derive_builder::Builder;
use std::collections::HashMap;
use std::path::PathBuf;
use crate::render_graph::graph_error::GraphError;

pub enum BindingElement {
    //TODO maybe compile time check the value corresponds to the inserted one
    RgResource {
        resource: RawResourceHandle,
    },

    /// Buffer Device Address: Directly pass a 64-bit GPU pointer. TODO this is unsafe and suggested by gemini
    /// Highly recommended for SSBOs in a modern bindless engine.
    DeviceAddress {
        resource: vk::DeviceMemory,
    },
}

pub enum BindingIntent {
    Single { name: &'static str },
    ArrayElement { name: &'static str, array_index: u32 },
}

type DescriptorsLayout = HashMap<String, rspirv_reflect::DescriptorInfo>;

type DescriptorOps = HashMap<BindingIntent, BindingElement>;
pub struct RayTracingShaderDesc {
    pub descriptor_operations: DescriptorOps,
    pub(crate) shader: ShaderSource,
}

pub struct RasterShaderDesc {
    //TODO
    pub descriptor_operations: DescriptorOps,
    pub(crate) shader: ShaderSource,
    pub(crate) pipeline_stage: RasterPipelineStage,
}

pub struct ComputeShaderDesc {
    pub descriptor_operations: DescriptorOps,
    pub(crate) shader: ShaderSource,
}

pub(crate) struct PassCommonData {
    pub(crate) read: Vec<ResourceRef>,
    pub(crate) write: Vec<ResourceRef>,

    pub(crate) name: String,
    id: u32,
}

pub struct PassCommonDataBuilder {
    pass_common_data: PassCommonData,
}
impl PassCommonDataBuilder {
    pub fn new(rg: & mut RenderGraph<Setup>, name: impl Into<String>) -> Self {
        Self {
            pass_common_data: PassCommonData {
                read: vec![],
                write: vec![],
                name: name.into(),
                id: rg.next_pass_id(),
            },
        }
    }
    pub fn read<Res: Resource>(&mut self, resource: &Handle<Res>, access_type: vk_sync_fork::AccessType) -> SrResult<()> {
        if !access_type.is_write_access() {
            self.pass_common_data.read.push(ResourceRef {
                raw: resource.raw,
                access: PassResourceAccessType {
                    access_type,
                    sync_type: PassResourceAccessSyncType::NeverSync,
                },
            });
            Ok(())
        } else {
            Err( SrError::new(GraphError::IncorrectRenderAccessFlags.into() , format!( "asked to read with such access: {access_type:?}" ) ))
        }
    }

    pub fn write<Res: Resource>(&mut self, resource: &Handle<Res>, access_type: vk_sync_fork::AccessType) -> SrResult<()> {
        //TODO this needs to change the resource version
        //TODO more complex not always sync write+write and read+write and render graph state id lookup

        if access_type.is_write_access() {
            self.pass_common_data.write.push(ResourceRef {
                raw: resource.raw,
                access: PassResourceAccessType {
                    access_type,
                    sync_type: PassResourceAccessSyncType::AlwaysSync,
                },
            });
            Ok(())
        } else {
            Err( SrError::new(GraphError::IncorrectRenderAccessFlags.into() , format!( "asked to write with such access: {access_type:?}" ) ))
        }
    }

}


impl Into<AnyRenderPass>  for RaytracingRenderPass {
    fn into(self) -> AnyRenderPass {
        AnyRenderPass::Rt(self)
    }
}

impl Into<AnyRenderPass>  for RasterRenderPass {
    fn into(self) -> AnyRenderPass {
        AnyRenderPass::Raster(self)
    }
}

impl Into<AnyRenderPass>  for ComputeRenderPass {
    fn into(self) -> AnyRenderPass {
        AnyRenderPass::Compute(self)
    }
}



#[derive(Builder)]
#[builder(pattern = "owned")]
pub(crate) struct RaytracingRenderPass {
    pub(super) common: PassCommonData,
    pub(super) ray_gen: RayTracingShaderDesc,
    #[builder(setter(each = "add_closest_hit"))]
    pub(super) closest_hit: Vec<RayTracingShaderDesc>,
    #[builder(setter(each = "add_miss"))]
    pub(super) miss: Vec<RayTracingShaderDesc>,
    pub(super) trace_extent: [u32; 3],
}


pub(crate) struct RasterRenderPass {
    pub(super) common: PassCommonData,
    //TODO
}
#[derive(Builder)]
#[builder(pattern = "owned")]
pub(crate) struct ComputeRenderPass {
    pub(super) common: PassCommonData,
    #[builder(setter(each = "add_shader"))]
    pub(super) shaders: Vec<ShaderSource>,
    pub(super) entry_point: String,
}

#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug)]
pub enum RayTracingPipelineStage {
    RayGen,
    RayMiss,
    RayClosestHit,
}

#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug)]
pub enum RasterPipelineStage {
    //TODO check for missing since I don't raster yet like task, mesh, tessellation , geometry
    Vertex,
    Pixel,
}

pub trait ShaderDesc {}

#[derive(Clone, Debug)]
pub enum ShaderSource {
    //TODO supported shaders, for now glsl
    Glsl(PathBuf),
}

pub(crate) type DynRenderFn =  dyn FnOnce(&mut CommandBuffer, &mut TransientResources) -> SrResult<()>; //TODO TransientResources here is intended to be a way to dereference the resources,but this implies it handles also external ones
