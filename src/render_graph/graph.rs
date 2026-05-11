use crate::error::SrResult;
use crate::render_graph::pass_builder::{ComputeRenderPass, DynRenderFn, RasterRenderPass, RaytracingRenderPass};
use crate::vulkan_abstraction::{
    AccelerationStructure, Buffer, CmdBuffer, Core, Image, RawBuffer,
};
use enum_as_inner::EnumAsInner;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
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


impl Into<GraphResourceInfo> for GraphResourceImportInfo {
    fn into(self) -> GraphResourceInfo {
        GraphResourceInfo::Imported(self)
    }
}
#[derive(Clone, Debug)]
pub struct ImageDesc {

}

impl Into<GraphResourceDesc> for ImageDesc {
    fn into(self) -> GraphResourceDesc {
        GraphResourceDesc::Image(self)
    }
}

impl ResourceDesc for ImageDesc {
    type Resource = Image;
}

pub struct BufferDesc {

}
pub struct RaytracingASDesc {

}

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
    Compute(ComputeRenderPass),
}


pub struct RenderGraph<State: RenderGraphState> {
    state_index: u32,
    next_pass_id: u32,
    next_resource_id: u32,
    //TODO debug hooks and tools
    virtual_resources: Vec<GraphResourceInfo>,
    passes: Vec<AnyRenderPass>,
    transient_resources: TransientResources,
    state_data: State,
}

impl RenderGraph<Setup> {
    pub fn new() -> Self {
        RenderGraph {
            state_index: 0,
            next_pass_id: 0,
            next_resource_id: 0,
            passes: vec![],
            virtual_resources: vec![],
            transient_resources: TransientResources::default(),
            state_data: Setup::default(),
        }
    }

    pub(super) fn next_pass_id(&mut self) -> u32 {
        let id = self.next_pass_id;
        self.next_pass_id += 1;
        id
    }
    pub(super) fn next_resource_id(&mut self) -> u32 {
        let id = self.next_resource_id;
        self.next_resource_id += 1;
        id
    }
    pub fn create<Desc: ResourceDesc>(&mut self, desc: Desc) -> Handle<<Desc as ResourceDesc>::Resource>
    where
        Desc: TypeEquals<Other=<<Desc as ResourceDesc>::Resource as Resource>::Desc>,
    {
        self.create_raw_resource(desc.clone().into());
        Handle {
            raw: RawResourceHandle {
                id: self.next_resource_id(),
                version: 0,
                render_state_index: self.state_index,
            },
            desc: TypeEquals::same(desc),
            marker: Default::default(),
        }
    }

    pub fn create_raw_resource(&mut self, resource_desc: GraphResourceDesc) {
        self.virtual_resources.push(GraphResourceInfo::Created(resource_desc));
    }

    pub fn import<Desc: ResourceDesc>(&mut self, res: impl RgImportable<Desc> + Into<GraphResourceImportInfo> ) -> Handle<<Desc as ResourceDesc>::Resource>
    where
        Desc: TypeEquals<Other=<<Desc as ResourceDesc>::Resource as Resource>::Desc>,
    {
        let desc = res.import();
        self.virtual_resources.push(GraphResourceInfo::Imported(res.into()));
        Handle {
            raw: RawResourceHandle {
                id: self.next_resource_id(),
                version: 0,
                render_state_index: self.state_index,
            },
            desc: TypeEquals::same(desc),
            marker: Default::default(),
        }
    }

    pub fn add_render_pass(&mut self, render_pass: AnyRenderPass) {
        self.passes.push(render_pass)
    }


    pub fn compile(mut self) -> RenderGraph<Built> { //TODO there are some complex optimizations as shown here https://www.youtube.com/watch?v=v9LaTFLhP38 and this is the site where it will be published the paper https://dl.acm.org/profile/99661091135
        // let mut actual_resources = ResourceDeref

        for pass in self.passes.iter_mut() {
            let common =match pass {
                AnyRenderPass::Rt(rt) => {
                    &mut rt.common
                }
                AnyRenderPass::Raster(raster) => {
                    &mut raster.common
                }
                AnyRenderPass::Compute(compute) => {
                    &mut compute.common
                }
            };

            // for read in common.read.iter_mut() {
            //     read.access.
            // }
        }

        todo!()
        // let mut graph = ;
    }
}

pub(super) struct CompiledPass {
    render: Box<DynRenderFn>,
    pub(crate) name: String,
    id: u32,
}

#[derive(Default)]
pub struct TransientResources {
    external_images: HashMap<u32, Arc<Image>>,
    transient_images: HashMap<u32, Image>,
    external_buffers: HashMap<u32, Arc<dyn Buffer>>,
    transient_buffers: HashMap<u32, Box<dyn Buffer>>,
    external_raytracing_ac: HashMap<u32, Arc<AccelerationStructure>>,
    transient_raytracing_ac: HashMap<u32,AccelerationStructure>,
    //TODO this struct needs to be emptied after the next frame creation so that resources can be reused
}
impl TransientResources {
    pub fn populate(&mut self , virtual_resources: &[GraphResourceInfo]) {

    }
}

pub trait RgImportable<ResDesc: ResourceDesc> { //TODO do I want to take ownership of the data?
    fn import(&self) -> ResDesc;
    
}


pub(crate) struct Render {}

pub(crate) struct Built {

}
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
