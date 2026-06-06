use crate::error::{SrError, SrResult};
use crate::render_graph::error::GraphError;
use crate::render_graph::graph::{
    AnyRenderPass, Handle, PassResourceAccessSyncType, PassResourceAccessType, RawResourceHandle, RenderGraph, Resource,
    ResourceRef, TransientResources,
};
use crate::vulkan_abstraction::{Core, RayTracingPipeline, ShaderBindingTable};
use ash::vk;
use ash::vk::CommandBuffer;
use derive_builder::Builder;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

pub(crate) enum BindingElement {
    //TODO maybe compile time check the value corresponds to the inserted one
    RgResource {
        resource: RawResourceHandle,
    },

    /// Buffer Device Address: Directly pass a 64-bit GPU pointer. TODO this is unsafe and suggested by gemini, this is a bda basically
    /// Highly recommended for SSBOs in a modern bindless engine.
    DeviceAddress {
        resource: vk::DeviceMemory,
    },
}

pub enum BindingIntent {
    Single { name: &'static str },
    ArrayElement { name: &'static str, array_index: u32 },
}

type DescriptorsLayout = HashMap<String, rspirv_reflect::DescriptorInfo>; //TODO rspirv_reflect does not support descriptor_heap

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
    #[allow(dead_code)]
    id: u32,
    /// The user-supplied recording function for this pass. Invoked by
    /// `RenderGraph::compile` once per pass in topological order, after the
    /// barriers required by that pass's incoming edges have already been issued
    /// into the command buffer. `None` means "nothing to record" (e.g.
    /// pass kept only to anchor an external resource transition).
    pub(crate) render: Option<Box<DynRenderFn>>,
}

pub struct PassCommonDataBuilder {
    pass_common_data: PassCommonData,
}
impl PassCommonDataBuilder {
    pub fn new(rg: &mut RenderGraph, name: impl Into<String>) -> Self {
        Self {
            pass_common_data: PassCommonData {
                read: vec![],
                write: vec![],
                name: name.into(),
                id: rg.next_pass_id(),
                render: None,
            },
        }
    }

    /// Attach the recording closure to this pass. Replaces any previous one.
    pub fn render<F>(&mut self, f: F) -> &mut Self
    where
        F: FnMut(&mut CommandBuffer, &TransientResources) -> SrResult<()> + 'static,
    {
        self.pass_common_data.render = Some(Box::new(f));
        self
    }

    /// Finalize the builder and consume it into the `PassCommonData` that the
    /// concrete pass builders embed.
    pub fn build(self) -> PassCommonData {
        self.pass_common_data
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
            Err(SrError::new(
                GraphError::IncorrectRenderAccessFlags.into(),
                format!("asked to read with such access: {access_type:?}"),
            ))
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
            Err(SrError::new(
                GraphError::IncorrectRenderAccessFlags.into(),
                format!("asked to write with such access: {access_type:?}"),
            ))
        }
    }
}

impl From<RaytracingRenderPass> for AnyRenderPass {
    fn from(val: RaytracingRenderPass) -> Self {
        AnyRenderPass::Rt(val)
    }
}

impl From<RasterRenderPass> for AnyRenderPass {
    fn from(val: RasterRenderPass) -> Self {
        AnyRenderPass::Raster(val)
    }
}

impl From<ComputeRenderPass> for AnyRenderPass {
    fn from(val: ComputeRenderPass) -> Self {
        AnyRenderPass::Compute(val)
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
    #[builder(default = "Vec::new()", setter(each = "add_any_hit"))]
    pub(super) any_hit: Vec<RayTracingShaderDesc>,
    pub(super) trace_extent: [u32; 3],
}

impl RaytracingRenderPassBuilder {
    /// Eagerly build the heap-mode raytracing pipeline + shader binding table
    /// for this pass and install a render closure on `common` that, when
    /// invoked by the graph at record time, binds the descriptor heap, binds
    /// the pipeline, and issues `cmd_trace_rays` with the configured
    /// `trace_extent`.
    ///
    /// Requires `common`, `ray_gen`, exactly one `miss`, exactly one
    /// `closest_hit`, exactly one `any_hit`, and `trace_extent` to already be
    /// set on the builder. Every shader stage's `ShaderSource` must be
    /// `ShaderSource::Spirv` — heap-mode shaders are not compiled by this
    /// helper; callers compile them externally (e.g. via the Slang
    /// `ShaderCompiler`) and hand the bytes in.
    ///
    /// Push constants are not pushed by the generated closure: per-frame
    /// scalar data is pass-specific and lives outside the builder's knowledge.
    /// Wire it up by composing an extra render step on top, or by extending
    /// this helper with a typed push-constant callback.
    pub fn generate_render(mut self, core: Rc<Core>) -> SrResult<Self> {
        fn extract_spirv(src: &ShaderSource) -> SrResult<&[u8]> {
            match src {
                ShaderSource::Spirv(bytes) => Ok(bytes.as_slice()),
                ShaderSource::Glsl(path) => Err(SrError::new_custom(format!(
                    "generate_render only accepts ShaderSource::Spirv; got Glsl({path:?})"
                ))),
            }
        }

        let ray_gen = self
            .ray_gen
            .as_ref()
            .ok_or_else(|| SrError::new_custom("generate_render: ray_gen not set".into()))?;
        let miss = self
            .miss
            .as_ref()
            .and_then(|v| v.first())
            .ok_or_else(|| SrError::new_custom("generate_render: at least one miss shader required".into()))?;
        let closest_hit = self
            .closest_hit
            .as_ref()
            .and_then(|v| v.first())
            .ok_or_else(|| SrError::new_custom("generate_render: at least one closest_hit shader required".into()))?;
        let any_hit = self
            .any_hit
            .as_ref()
            .and_then(|v| v.first())
            .ok_or_else(|| SrError::new_custom("generate_render: at least one any_hit shader required".into()))?;
        let trace_extent = self
            .trace_extent
            .ok_or_else(|| SrError::new_custom("generate_render: trace_extent not set".into()))?;

        // SBT currently hardcodes 1 raygen + 1 miss + 1 hit group; extra
        // shader stages set on the builder are ignored by the dispatch.
        let ray_gen_spirv = extract_spirv(&ray_gen.shader)?.to_vec();
        let miss_spirv = extract_spirv(&miss.shader)?.to_vec();
        let closest_hit_spirv = extract_spirv(&closest_hit.shader)?.to_vec();
        let any_hit_spirv = extract_spirv(&any_hit.shader)?.to_vec();

        let pipeline = Rc::new(RayTracingPipeline::new_heap(
            Rc::clone(&core),
            &ray_gen_spirv,
            &miss_spirv,
            &closest_hit_spirv,
            &any_hit_spirv,
        )?);
        let sbt = Rc::new(ShaderBindingTable::new(&core, &pipeline)?);

        let mut common = self
            .common
            .take()
            .ok_or_else(|| SrError::new_custom("generate_render: common not set".into()))?;

        let pipeline_c = Rc::clone(&pipeline);
        let sbt_c = Rc::clone(&sbt);
        let core_c = Rc::clone(&core);
        common.render = Some(Box::new(move |cb, _tr| {
            let device = core_c.device().inner();
            unsafe {
                core_c.descriptor_heap().cmd_bind(*cb);
                device.cmd_bind_pipeline(*cb, vk::PipelineBindPoint::RAY_TRACING_KHR, pipeline_c.inner());
                core_c.rt_pipeline_device().cmd_trace_rays(
                    *cb,
                    sbt_c.raygen_region(),
                    sbt_c.miss_region(),
                    sbt_c.hit_region(),
                    sbt_c.callable_region(),
                    trace_extent[0],
                    trace_extent[1],
                    trace_extent[2],
                );
            }
            Ok(())
        }));
        self.common = Some(common);

        // Keep the built pipeline + SBT alive by stashing the Rc clones in the
        // closure; the builder is consumed by `.build()` afterwards so nothing
        // else needs to hold them.
        let _ = (pipeline, sbt);
        Ok(self)
    }
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
    //TODO supported shaders, for now glsl + pre-compiled SPIR-V
    Glsl(PathBuf),
    /// Pre-compiled SPIR-V bytes — produced upstream (e.g. by the Slang
    /// `ShaderCompiler`) and consumed verbatim by heap-mode pipeline helpers
    /// like `RaytracingRenderPassBuilder::generate_render`.
    Spirv(Vec<u8>),
}

pub(crate) type DynRenderFn = dyn FnMut(&mut CommandBuffer, &TransientResources) -> SrResult<()>; //TODO TransientResources here is intended to be a way to dereference the resources,but this implies it handles also external ones
