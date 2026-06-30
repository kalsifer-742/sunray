use crate::render_graph::graph::{AnyRenderPass, PassResourceAccessType};
use crate::vulkan_abstraction::acceleration_structure::ASDesc;
use crate::vulkan_abstraction::buffer::BufferDesc;
use crate::vulkan_abstraction::image::ImageDesc;
use crate::vulkan_abstraction::image::sampler::SamplerDesc;
use crate::vulkan_abstraction::{AccelerationStructure, Buffer, Image, RawBuffer, Sampler};
use enum_as_inner::EnumAsInner;
use std::marker::PhantomData;
use std::sync::Arc;
use vk_sync_fork as vk_sync;

pub trait Resource {
    type Desc: ResourceDesc;
    fn borrow_resource(res: &AnyRenderResource) -> &Self; //TODO this is useless basically
}

pub trait ResourceDesc: Clone + std::fmt::Debug + Into<GraphResourceDesc> {
    type Resource: Resource;
}

#[derive(Debug)]
pub struct Handle<ResourceType: Resource> {
    pub(crate) id: u32,
    pub(crate) desc: <ResourceType as Resource>::Desc,
    pub(crate) marker: PhantomData<ResourceType>,
}

// Manual `Clone` so a `Handle` is cloneable regardless of whether the resource
// type itself is `Clone` (it never needs to be — only the `Desc` is stored).
// `#[derive(Clone)]` would add a spurious `ResourceType: Clone` bound.
impl<ResourceType: Resource> Clone for Handle<ResourceType> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            desc: self.desc.clone(),
            marker: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ResourceRef {
    pub(crate) id: u32,
    pub(crate) access: PassResourceAccessType,
}

pub enum AnyRenderResource {
    OwnedImage(Image),
    ImportedImage(Arc<Image>),
    OwnedBuffer(RawBuffer),
    ImportedBuffer(Arc<dyn Buffer>),
    OwnedSampler(Sampler),
    ImportedSampler(Arc<Sampler>),
    ImportedRayTracingAcceleration(Arc<AccelerationStructure>),
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
    Sampler {
        resource: Arc<Sampler>,
    },
    RayTracingAcceleration {
        resource: Arc<AccelerationStructure>,
        access_type: vk_sync::AccessType,
    },
    /// Swapchain target image. Only one swapchain resource may exist per graph.
    /// The Arc is the current frame's acquired image; replace by calling
    /// `RenderGraph::import_swapchain` again on the next frame.
    SwapchainImage {
        resource: Arc<Image>,
    },
}

impl From<GraphResourceImportInfo> for GraphResourceInfo {
    fn from(val: GraphResourceImportInfo) -> Self {
        GraphResourceInfo::Imported(val)
    }
}

impl From<ImageDesc> for GraphResourceDesc {
    fn from(val: ImageDesc) -> Self {
        GraphResourceDesc::Image(val)
    }
}

impl From<BufferDesc> for GraphResourceDesc {
    fn from(val: BufferDesc) -> Self {
        GraphResourceDesc::Buffer(val)
    }
}

impl From<SamplerDesc> for GraphResourceDesc {
    fn from(val: SamplerDesc) -> Self {
        GraphResourceDesc::Sampler(val)
    }
}

impl From<ASDesc> for GraphResourceDesc {
    fn from(val: ASDesc) -> Self {
        GraphResourceDesc::RaytracingAS(val)
    }
}

pub enum GraphResourceDesc {
    Image(ImageDesc),
    Buffer(BufferDesc),
    Sampler(SamplerDesc),
    RaytracingAS(ASDesc),
}

#[derive(EnumAsInner)]
pub enum GraphResourceInfo {
    //TODO imported res with ownership taking option for internal aliasing later
    Created(GraphResourceDesc),
    Imported(GraphResourceImportInfo),
}

pub trait RgImportable<ResDesc: ResourceDesc> {
    //TODO do I want to take ownership of the data?
    fn import(&self) -> ResDesc;
}
