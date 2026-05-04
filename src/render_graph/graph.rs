use crate::error::SrResult;
use crate::vulkan_abstraction::{AccelerationStructure, Buffer, CmdBuffer, Core, Image, RawBuffer, RaytracingDescriptorSets};
use ash::vk;
use ash::vk::{CommandBuffer, DescriptorPool, DescriptorSet};
use derive_builder::Builder;
use enum_as_inner::EnumAsInner;
use std::any::Any;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use vk_sync_fork as vk_sync;
use crate::render_graph::pass_builder::RenderPassBuilder;

pub trait Resource {
    type Desc: ResourceDesc;
    fn borrow_resource(res: &AnyRenderResource) -> &Self;
}
pub trait ResourceDesc: Clone + std::fmt::Debug + Into<GraphResourceDesc> {
    type Resource: Resource;
}

#[derive(Clone, Copy)]
pub(crate) struct RawResourceHandle {
    pub(crate) id: u32,
    pub(crate) version: u32,
    render_state_index: u32 ,
}

pub struct Handle<ResourceType: Resource> {
    pub(crate) raw: RawResourceHandle,
    pub(crate) desc: <ResourceType as Resource>::Desc,
    pub(crate) marker: PhantomData<ResourceType>,
}

pub(crate) struct ResourceRef {
    pub(crate) raw: RawResourceHandle,
    pub(crate) usage: PassResourceAccessType,
}

pub enum AnyRenderResource {
    OwnedImage(Image),
    ImportedImage(Arc<Image>),
    OwnedBuffer(RawBuffer),
    ImportedBuffer(Arc<dyn Buffer>),
    ImportedRayTracingAcceleration(Arc<AccelerationStructure>),
}


type DynRenderFn = dyn FnOnce(&mut CommandBuffer, &mut TransientResources) -> SrResult<()>; //TODO TransientResources here is intended to be a way to dereference the resources,but this implies it handles also external ones


pub(crate) struct RenderPass {
    pub(crate)  read: Vec<ResourceRef>,
    pub(crate)  write: Vec<ResourceRef>,
    pub(crate)  render_fn: Option<Box<DynRenderFn>>,
    pub(crate)  name: String,
    pub(crate)  idx: usize,
}



#[allow(dead_code)]
fn global_barrier(core: &Core, cb: &CmdBuffer, previous_accesses: &[vk_sync::AccessType], next_accesses: &[vk_sync::AccessType]) {
    vk_sync::cmd::pipeline_barrier(
        core.device().inner(),
        cb.inner(),
        Some(vk_sync::GlobalBarrier {
            previous_accesses,
            next_accesses,
        }),
        &[],
        &[],
    );
}


pub struct TransientResources {
    //TODO this struct needs to be emptied after the next frame creation so that resources can be reused
}

#[derive(Clone)]
pub enum GraphResourceImportInfo {
    Image {
        resource: Arc<Image>,
        access_type: vk_sync::AccessType,
    },
    Buffer {
        resource: Arc<RawBuffer>,
        access_type: vk_sync::AccessType,
    },
    RayTracingAcceleration {
        resource: Arc<AccelerationStructure>,
        access_type: vk_sync::AccessType,
    },
    SwapchainImage,
}

pub struct ImageDesc {}

pub struct BufferDesc {}
pub struct RaytracingASDesc {}

pub enum GraphResourceDesc {
    Image(ImageDesc),
    Buffer(BufferDesc),
    RaytracingAS(RaytracingASDesc),
}
#[derive(EnumAsInner)]
pub enum GraphResourceInfo {
    //this is description of what I need to allocate to satisfy the request pof the render pass
    Created(GraphResourceDesc),
    Imported(GraphResourceImportInfo),
}

struct PipelineCache {}

pub trait RenderGraphState {}
#[derive(Default)]
pub(crate) struct Setup {}
impl RenderGraphState for Setup {}

struct RgComputePipeline {
    //TODO
}

struct RgRasterPipeline {
    //TODO
}


pub(crate) struct PassResourceRef {
    pub handle: RawResourceHandle,
    pub access: PassResourceAccessType,
}

#[derive(Copy, Clone)]
pub enum PassResourceAccessSyncType {
    AlwaysSync,
    SkipSyncIfSameAccessType,
}

#[derive(Copy, Clone)]
pub struct PassResourceAccessType {
    pub(crate) access_type: vk_sync::AccessType,
    pub(crate) sync_type: PassResourceAccessSyncType,
}

struct ShaderDesc{
    shader: Shader,
}

pub(crate ) enum Shader{
    //TODO supported shaders, for now glsl
    Glsl(PathBuf)
}
pub trait RenderPassFactory{

    fn render_fn(self) -> ( Box<DynRenderFn>, HashMap<u32,rspirv_reflect::DescriptorInfo > , Vec<ShaderDesc> );



}

struct CommonPipelineData {
    //TODO temp theorically the binding number
    descriptor_set:  HashMap<u32,rspirv_reflect::DescriptorInfo >,

}

struct RgRaytracingPipeline {
    raytracing_descriptor_sets: RaytracingDescriptorSets,
    shader : Shader
}


pub struct RenderGraph<State: RenderGraphState> {
    state_index:  u32,

    //TODO debug hooks and tools
    pub(crate) passes: Vec<RenderPass>,
    resources: Vec<GraphResourceInfo>,

    pub(crate) compute_pipelines: Vec<RgComputePipeline>,
    pub(crate) raster_pipelines: Vec<RgRasterPipeline>,
    pub(crate) rt_pipelines: Vec<RgRaytracingPipeline>,
    pub(crate) passes_data: HashMap<u32,CommonPipelineData >,
    // transient_resources: TransientResources,
    frame_descriptor_set: vk::DescriptorSet, //
    state_data: State,
}



impl RenderGraph<Setup> {
    pub fn new() -> SrResult<Self> {
        Ok(RenderGraph {
            passes: vec![],
            resources: vec![],
            //transient_resources: TransientResources {},
            //frame_descriptor_set: Default::default(),
            compute_pipelines: vec![],
            raster_pipelines: vec![],
            rt_pipelines: vec![],
            frame_descriptor_set: Default::default(),
            state_data: Setup::default(),
        })
    }
    pub fn create<Desc: ResourceDesc>(&mut self, desc: Desc) -> Handle<<Desc as ResourceDesc>::Resource>
    where
        Desc: TypeEquals<Other = <<Desc as ResourceDesc>::Resource as Resource>::Desc>,
    {
        self.create_raw_resource(desc.clone().into());
        Handle {
            raw: RawResourceHandle { id: 0, version: 0, render_state_index: self.state_index },
            desc: TypeEquals::same(desc),
            marker: Default::default(),
        }
    }

    pub fn create_raw_resource(&mut self, resource_desc: GraphResourceDesc) {
        self.resources.push(GraphResourceInfo::Created(resource_desc));
    }

    pub fn add_render_pass(&mut self, render_pass_builder: RenderPassBuilder) {
        let render_pass = render_pass_builder.submit(self);
        todo!()
    }

    pub fn compile(mut self) -> RenderGraph<Built> {

    }

}


pub(crate) struct Render {}


pub(crate) struct Built {

}
impl RenderGraphState for Built{

}

pub struct BuiltRenderGraph {
    cmd_buffer: CmdBuffer
    //ready to execute
}

pub trait TypeEquals {
    type Other;
    fn same(value: Self) -> Self::Other;
}

impl<T: Sized> TypeEquals for T {
    type Other = Self;
    fn same(value: Self) -> Self::Other {
        value
    }
}


