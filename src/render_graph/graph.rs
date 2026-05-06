use crate::error::SrResult;
use crate::render_graph::pass_builder::{ComputeRenderPass, RasterRenderPass, RaytracingRenderPass, RenderPassBuilder};
use crate::vulkan_abstraction::{
    AccelerationStructure, Buffer, CmdBuffer, Core, Image, RawBuffer,
};
use enum_as_inner::EnumAsInner;
use std::marker::PhantomData;
use std::sync::Arc;
use ash::vk::RenderPass;
use vk_sync_fork as vk_sync;

pub trait Resource {
    type Desc: ResourceDesc;
    fn borrow_resource(res: &AnyRenderResource) -> &Self;
}
pub trait ResourceDesc: Clone + std::fmt::Debug + Into<GraphResourceDesc> {
    type Resource: Resource;
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RawResourceHandle {
    pub(crate) id: u32,
    pub(crate) version: u32,
    pub(super) render_state_index: u32,
}

#[derive(Clone, Debug)]
pub struct Handle<ResourceType: Resource> {
    pub(crate) raw: RawResourceHandle,
    pub(crate) desc: <ResourceType as Resource>::Desc,
    pub(crate) marker: PhantomData<ResourceType>,
}

#[derive(Clone, Debug)]
pub(crate) struct ResourceRef {
    pub(crate) raw: RawResourceHandle,
    pub(crate) access: PassResourceAccessType,
}

pub enum AnyRenderResource {
    OwnedImage(Image),
    ImportedImage(Arc<Image>),
    OwnedBuffer(RawBuffer),
    ImportedBuffer(Arc<dyn Buffer>),
    ImportedRayTracingAcceleration(Arc<AccelerationStructure>),
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

pub trait RenderGraphState {}
#[derive(Default)]
pub(crate) struct Setup {}
impl RenderGraphState for Setup {}

#[derive(Copy, Clone, Debug)]
pub enum PassResourceAccessSyncType {
    AlwaysSync,
    SkipSyncIfSameAccessType,
    NeverSync,
}

#[derive(Copy, Clone, Debug)]
pub struct PassResourceAccessType {
    pub(crate) access_type: vk_sync::AccessType,
    pub(crate) sync_type: PassResourceAccessSyncType,
}

pub enum AnyRenderPass {
    Rt(RaytracingRenderPass),
    Raster(RasterRenderPass),
    Computer(ComputeRenderPass)
}

pub struct RenderGraph<State: RenderGraphState> {
    state_index: u32,
    next_pass_id: usize,
    //TODO debug hooks and tools
    resources: Vec<GraphResourceInfo>,
    passes : Vec<AnyRenderPass>,
     transient_resources: TransientResources,
    state_data: State,
}

impl RenderGraph<Setup> {
    
    pub fn new() -> SrResult<Self> {
        Ok(RenderGraph {
            state_index: 0,
            next_pass_id: 0,
            passes: vec![],
            resources: vec![],
            transient_resources: TransientResources {},
            state_data: Setup::default(),
        })
    }

    pub(super) fn next_pass_id(&mut self) -> usize {
        let id = self.next_pass_id;
        self.next_pass_id += 1;
        id
    }
    pub fn create<Desc: ResourceDesc>(&mut self, desc: Desc) -> Handle<<Desc as ResourceDesc>::Resource>
    where
        Desc: TypeEquals<Other = <<Desc as ResourceDesc>::Resource as Resource>::Desc>,
    {
        self.create_raw_resource(desc.clone().into());
        Handle {
            raw: RawResourceHandle {
                id: 0,
                version: 0,
                render_state_index: self.state_index,
            },
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

pub(crate) struct Built {}
impl RenderGraphState for Built {}

pub struct BuiltRenderGraph {
    cmd_buffer: CmdBuffer, //ready to execute
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
