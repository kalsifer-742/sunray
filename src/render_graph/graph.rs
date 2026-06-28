use crate::error::{SrError, SrResult};
use crate::render_graph::error::GraphError;
use crate::render_graph::pass_builder::{ComputeRenderPass, RasterRenderPass, RaytracingRenderPass};
pub(crate) use crate::render_graph::resource::{
    GraphResourceDesc, GraphResourceImportInfo, GraphResourceInfo, Handle, Resource, ResourceDesc, RgImportable,
};
use crate::render_graph::transient_resources::TransientResources;
use crate::vulkan_abstraction::image::ImageDesc;
use crate::vulkan_abstraction::{
    Buffer, CmdBuffer, ComputePipeline, Core, Fence, GraphicsPipeline, GraphicsPipelineShaders, HeapComputePass, Image, Pipeline,
    RayTracingPipeline, RayTracingPipelineShaders, ShaderBindingTable,
};
use ash::vk;
use petgraph::Graph;
use petgraph::visit::{EdgeRef, IntoNeighbors};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use petgraph::graph::EdgeIndex;
use vk_sync_fork as vk_sync;
use crate::MAX_FRAMES_IN_FLIGHT;

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



pub(super) enum AnyRenderPass {
    Rt(RaytracingRenderPass),
    Raster(RasterRenderPass),
    Compute(ComputeRenderPass),
}


/// Lightweight reference to a pipeline interned in the graph's [`PipelineCache`].
/// Render closures resolve it to the concrete pipeline at record time via
/// `TransientResources::{compute,raytracing,graphics}_pipeline`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PipelineHandle(u32);

/// A heap-mode pipeline owned by the cache. RT additionally owns its shader
/// binding table (built alongside the pipeline).
pub(super) enum CachedPipeline {
    Compute(Rc<ComputePipeline<HeapComputePass>>),
    RayTracing(Rc<RayTracingPipeline>, Rc<ShaderBindingTable>),
    Graphics(Rc<GraphicsPipeline>),
}

/// Content-addressed cache of heap-mode pipelines, owned by the graph and kept
/// alive across the per-frame `reset()` / rebuild cycle. Passes describe their
/// shaders every frame, but the underlying `vk::Pipeline` (an expensive object)
/// is built exactly once per distinct shader set and reused — no per-frame
/// pipeline churn, and identical shaders shared by several passes are never
/// duplicated on the GPU.
///
/// Entries are addressed by [`PipelineHandle`] (an index into `entries`); the
/// `by_key` map dedups by a hash of the shader bytes. The cache is never cleared
/// by `free_internal_state` (that only frees per-frame transient resources), so
/// it lives for the whole graph; `core` is held so cached pipelines outlive the
/// device-owning `Core` no matter the surrounding drop order.
///
/// TODO: there is no eviction yet — an interned pipeline is kept until the graph
/// is dropped. Fine while the renderer uses a fixed, small shader set (every
/// entry is "still needed" every frame); add refcount/GC eviction once shaders
/// can come and go at runtime.
#[derive(Default)]
pub(super) struct PipelineCache {
    entries: Vec<CachedPipeline>,
    by_key: HashMap<u64, PipelineHandle>,
    core: Option<Rc<Core>>,
}

impl PipelineCache {
    pub(super) fn get(&self, handle: PipelineHandle) -> Option<&CachedPipeline> {
        self.entries.get(handle.0 as usize)
    }

    /// Return the handle for `key` if already interned, otherwise build the
    /// pipeline via `build`, store it, and return its fresh handle.
    pub(super) fn intern(
        &mut self,
        key: u64,
        core: &Rc<Core>,
        build: impl FnOnce() -> SrResult<CachedPipeline>,
    ) -> SrResult<PipelineHandle> {
        if let Some(handle) = self.by_key.get(&key) {
            return Ok(*handle);
        }
        let entry = build()?;
        if self.core.is_none() {
            self.core = Some(Rc::clone(core));
        }
        let handle = PipelineHandle(self.entries.len() as u32);
        self.entries.push(entry);
        self.by_key.insert(key, handle);
        Ok(handle)
    }
}

/// Hash a set of byte slices (plus a `kind` discriminant so a compute shader and
/// a same-bytes graphics shader never collide) into the cache key.
fn pipeline_cache_key(kind: u8, parts: &[&[u8]]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    kind.hash(&mut hasher);
    for part in parts {
        part.hash(&mut hasher);
    }
    hasher.finish()
}

/// Where a graph resource ends up once the graph's submission completes: the
/// last pass that touched it and the access it is left in.
///
/// TODO(temp impl): this is the seed of the cross-submission sync contract.
/// The goal is that whatever runs *after* the graph (the external blit, the
/// present transition, the next frame's graph) reads the end state of the
/// resources it shares with the graph and emits a precise pipeline barrier
/// from `end_access` instead of `vkDeviceWaitIdle`, and that `compile` itself
/// consumes the previous submission's end states as the initial access of
/// imported resources — so consecutive frames chain render-pass-to-render-pass
/// with no idle. Right now the states are only collected and exposed; nothing
/// consumes them yet.
#[derive(Clone, Debug)]
pub struct ResourceEndState {
    /// Pass id of the last pass that touched the resource, `None` if the
    /// resource was registered but never used by any pass.
    pub last_use_pass: Option<usize>,
    /// Access type the resource is left in when the submission completes
    /// (`Nothing` when never used).
    pub end_access: vk_sync::AccessType,
    /// Graph-created (transient) resource: its backing memory is recycled on
    /// `reset`, so its end state only matters for in-graph aliasing, never to
    /// the caller.
    pub internal: bool,
}

/// A single transition required before a destination pass can run, derived from
/// a read/write hazard on `resource_id` against an earlier producer or reader.
#[derive(Clone, Debug)]
pub(crate) struct ResourceBarrier {
    pub(crate) resource_id: u32,
    pub(crate) prev_access: vk_sync::AccessType,
    pub(crate) next_access: vk_sync::AccessType,
}




/// Edge weight on the pass dependency graph: all barriers that must be issued
/// before the destination pass runs because of the source pass.
#[derive(Clone, Debug, Default)]
pub(crate) struct PassDependency {
    pub(crate) barriers: Vec<ResourceBarrier>,
}

/// Per-resource lifetime + ordered list of (pass, access) touches. Lifetime is
/// inclusive: the resource must be live from `first_pass` through `last_pass`.
#[derive(Debug)]
pub(crate) struct ResourceLifetimeUsage {
    pub(crate) first_pass: usize,
    pub(crate) last_pass: usize,
    pub(crate) usages: Vec<(usize, PassResourceAccessType)>,
}

/// Hazard-tracking state for a single resource while scanning passes in order.
#[derive(Debug, Default)]
struct ResourceHazardState {
    last_writer: Option<(usize, vk_sync::AccessType)>,
    readers_since_write: Vec<(usize, vk_sync::AccessType)>,
}

/// A weakly-connected component of the dependency graph: a set of passes that
/// transitively share resources, plus the resource ids those passes touch.
/// Transient memory aliasing is computed independently per component.
#[derive(Debug)]
pub(crate) struct PassComponent {
    pub(crate) passes: Vec<usize>,
    pub(crate) resources: Vec<u32>,
}

fn record_usage(usages: &mut BTreeMap<u32, ResourceLifetimeUsage>, res_id: u32, pass_id: usize, access: PassResourceAccessType) {
    usages
        .entry(res_id)
        .and_modify(|u| {
            u.last_pass = pass_id;
            u.usages.push((pass_id, access));
        })
        .or_insert_with(|| ResourceLifetimeUsage {
            first_pass: pass_id,
            last_pass: pass_id,
            usages: vec![(pass_id, access)],
        });
}

fn add_dep_edge(
    graph: &mut petgraph::graph::DiGraph<usize, PassDependency>,
    nodes: &[petgraph::graph::NodeIndex],
    src: usize,
    dst: usize,
    barrier: ResourceBarrier,
) {
    // A pass that reads-then-writes its own resource produces a self-edge; the hazard
    // is already serialized by the pass itself, so skip it.
    if src == dst {
        return;
    }
    let s = nodes[src];
    let d = nodes[dst];
    if let Some(e) = graph.find_edge(s, d) {
        graph
            .edge_weight_mut(e)
            .expect("edge just found must have a weight")
            .barriers
            .push(barrier);
    } else {
        graph.add_edge(s, d, PassDependency { barriers: vec![barrier] });
    }
}

struct TemporalResource{
    frame_of_creation : usize ,
    ids : [u32 ; MAX_FRAMES_IN_FLIGHT],
}

pub struct RenderGraph {
    next_pass_id: u32,
    next_resource_id: u32,
    //TODO debug hooks and tools
    virtual_resources: Vec<GraphResourceInfo>,
    temporal_resources: Vec<TemporalResource>,
    passes: Vec<AnyRenderPass>,
    transient_resources: TransientResources,
    /// At most one swapchain target per graph; remembered so the compile step can
    /// flag the present-target layout transition without scanning every resource.
    swapchain_resource_id: Option<u32>,
    /// Per-resource end state (last use + final access) collected by `compile`,
    /// keyed by resource id. See [`ResourceEndState`] — temp impl, exposed so a
    /// later stage can sync against the graph with a barrier instead of a
    /// device-wait-idle; nothing consumes it yet.
    resource_end_states: HashMap<u32, ResourceEndState>,
    /// Lives across `compile` / `run` cycles so the same primary command buffer
    /// is re-recorded each frame rather than reallocated.
    cmd_buffer: CmdBuffer,
    /// Cached so `run` can submit and `compile` can record without the caller
    /// having to re-thread `Core` through every call.
    core: Rc<Core>,
}
//TODO per frame global data uploaded each frame like transforms and the camera, these can then live in the descriptor heap based on kajiya DYNAMIC_CONSTANTS_BUFFER
//TODO Reintroduce the typestate of the render graph,as it is intended to work like this, the setup phase is where you can add stuff and so on, when you want to run it you compile it once done than you can return to the setup phase, this should empty out reset the cmdbuffer and allow to add again resources, this should make sure the resources in use are
//   not overwritten though while still allowing new resources to be added,also while on a built state it should be able to handle n frames in flight with internal sync to minimize the wait idle time and allow multiple frame to be run concurrently, this
impl RenderGraph {
    pub fn new(core: Rc<Core>) -> SrResult<Self> {
        let cmd_buffer = CmdBuffer::new(Rc::clone(&core))?;
        Ok(RenderGraph {
            next_pass_id: 0,
            next_resource_id: 0,
            passes: vec![],
            virtual_resources: vec![],
            temporal_resources: vec![],
            transient_resources: TransientResources::default(),
            swapchain_resource_id: None,
            resource_end_states: HashMap::new(),
            cmd_buffer,
            core,
        })
    }

    /// Clear all per-frame state (passes, virtual resources, transient bindings,
    /// swapchain import, recorded barrier trace) so the graph can be rebuilt
    /// from scratch on the next frame. The persistent `CmdBuffer` and cached
    /// `Core` survive: this is the entry point for "graph is an attribute of
    /// the renderer, rebuilt each frame, but the underlying primary command
    /// buffer is reused".
    pub fn reset(&mut self) {
        self.next_pass_id = 0;
        self.next_resource_id = 0;
        self.passes.clear();
        self.virtual_resources.clear();
        self.swapchain_resource_id = None;
        self.resource_end_states.clear();
        self.transient_resources.free_internal_state();
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
    pub fn create_resource<Desc: ResourceDesc>(&mut self, desc: Desc) -> Handle<<Desc as ResourceDesc>::Resource>
    where
        Desc: TypeEquals<Other = <<Desc as ResourceDesc>::Resource as Resource>::Desc>,
    {
        self.create_raw_resource(desc.clone().into());
        Handle {
            id: self.next_resource_id(),
            desc: TypeEquals::same(desc),
            marker: Default::default(),
        }
    }

    pub fn create_temporal_resource<Desc : ResourceDesc>(&mut self, desc: Desc) -> [Handle<<Desc as ResourceDesc>::Resource>; MAX_FRAMES_IN_FLIGHT]
    where
        Desc: TypeEquals<Other = <<Desc as ResourceDesc>::Resource as Resource>::Desc>,
    {
        let mut ids = [0; MAX_FRAMES_IN_FLIGHT];

        let handles = std::array::from_fn(|i| {
            self.create_raw_resource(desc.clone().into());
            let id = self.next_resource_id();

            ids[i] = id;

            Handle {
                id,
                desc: TypeEquals::same(desc.clone()),
                marker: Default::default(),
            }
        });

        let temporal_resource = TemporalResource {
            frame_of_creation: *self.core.absolute_frame_count.borrow(),
            ids,
        };

        self.temporal_resources.push(temporal_resource);

        handles
    }


    fn create_raw_resource(&mut self, resource_desc: GraphResourceDesc) {
        self.virtual_resources.push(GraphResourceInfo::Created(resource_desc));
    }

    pub fn import<Desc: ResourceDesc>(
        &mut self,
        res: impl RgImportable<Desc> + Into<GraphResourceImportInfo>,
    ) -> Handle<<Desc as ResourceDesc>::Resource>
    where
        Desc: TypeEquals<Other = <<Desc as ResourceDesc>::Resource as Resource>::Desc>,
    {
        let desc = res.import();
        self.virtual_resources.push(GraphResourceInfo::Imported(res.into()));
        Handle {
            id: self.next_resource_id(),
            desc: TypeEquals::same(desc),
            marker: Default::default(),
        }
    }

    pub fn add_render_pass(&mut self, render_pass: impl Into<AnyRenderPass>) {
        self.passes.push(render_pass.into())
    }

    /// The graph's cached `Core`. Pass builders use this so callers no longer
    /// have to thread `Rc<Core>` into `generate_render` separately.
    pub(crate) fn core(&self) -> Rc<Core> {
        Rc::clone(&self.core)
    }

    /// Intern a heap-mode compute pipeline for `spirv`, returning a handle the
    /// render closure resolves at record time. Built once per distinct shader and
    /// reused across frame rebuilds (see [`PipelineCache`]).
    pub(crate) fn cache_compute_pipeline(&mut self, spirv: &[u8]) -> SrResult<PipelineHandle> {
        let key = pipeline_cache_key(0, &[spirv]);
        let core = Rc::clone(&self.core);
        self.transient_resources.pipeline_cache.intern(key, &core, || {
            let pipeline = ComputePipeline::<HeapComputePass>::new(core.clone_device(), spirv)?;
            Ok(CachedPipeline::Compute(Rc::new(pipeline)))
        })
    }

    /// Intern a heap-mode ray-tracing pipeline + its shader binding table.
    pub(crate) fn cache_raytracing_pipeline(&mut self, shaders: &RayTracingPipelineShaders) -> SrResult<PipelineHandle> {
        let key = pipeline_cache_key(1, &[&shaders.ray_gen, &shaders.miss, &shaders.closest_hit, &shaders.any_hit]);
        let core = Rc::clone(&self.core);
        self.transient_resources.pipeline_cache.intern(key, &core, || {
            let pipeline = Rc::new(RayTracingPipeline::new(Rc::clone(&core), shaders)?);
            let sbt = Rc::new(ShaderBindingTable::new(&core, &pipeline)?);
            Ok(CachedPipeline::RayTracing(pipeline, sbt))
        })
    }

    /// Intern a heap-mode graphics pipeline. The vertex layout is currently not
    /// part of the cache key (only vertex+fragment SPIR-V and color format are) —
    /// fine while the raster path is experimental and single-layout.
    pub(crate) fn cache_graphics_pipeline(&mut self, shaders: &GraphicsPipelineShaders) -> SrResult<PipelineHandle> {
        let key = pipeline_cache_key(
            2,
            &[
                &shaders.vertex,
                &shaders.fragment,
                &shaders.color_format.as_raw().to_ne_bytes(),
            ],
        );
        let core = Rc::clone(&self.core);
        self.transient_resources.pipeline_cache.intern(key, &core, || {
            let pipeline = GraphicsPipeline::new(Rc::clone(&core), shaders)?;
            Ok(CachedPipeline::Graphics(Rc::new(pipeline)))
        })
    }

    /// Import the current frame's swapchain image as the graph's present target.
    /// At most one swapchain may be imported per graph; subsequent calls return
    /// `GraphError::SwapchainAlreadyImported`. The returned handle can be used as
    /// any other image handle for reads/writes; compile will tag the final
    /// transition into `PRESENT_SRC_KHR` on this resource.
    pub fn import_swapchain(&mut self, image: Arc<Image>) -> SrResult<Handle<Image>> {
        //TODO this approach doesn't work I'll change it later so that the swapchain can change without actually rebuild the graph
        if self.swapchain_resource_id.is_some() {
            return Err(SrError::new(
                GraphError::SwapchainAlreadyImported.into(),
                "render graph already has a swapchain import".to_string(),
            ));
        }
        let desc = ImageDesc {
            extent: image.extent(),
            format: image.format(),
            tiling: vk::ImageTiling::OPTIMAL,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            usage: vk::ImageUsageFlags::empty(),
            name: "swapchain",
        };
        let id = self.next_resource_id();
        self.virtual_resources
            .push(GraphResourceInfo::Imported(GraphResourceImportInfo::SwapchainImage {
                resource: image,
            }));
        self.swapchain_resource_id = Some(id);
        Ok(Handle {
            id,
            desc,
            marker: Default::default(),
        })
    }


    pub fn compile(&mut self) -> SrResult<()> {
        //TODO force injection of previous temporal data on rebuild, the graph is incapable of understanding temporal dep across compilations
        //TODO mark the render pass goals as the result of the graph so anything unnecessary can be removed
        //TODO there are some complex optimizations as shown here https://www.youtube.com/watch?v=v9LaTFLhP38 and this is the site where it will be published the paper https://dl.acm.org/profile/99661091135
        //TODO respect PassResourceAccessSyncType (NeverSync / SkipSyncIfSameAccessType) when deciding whether to emit a barrier
        //TODO it currently returns a one time submit, but the cmd buffer can be reuse as long as the graph doesn't get rebuilt this requires the temporal stuff though and some rework on the sync side between each frame
        //TODO the render graph currently has no way to export the data, this is useful to synchronize across frames. The data should be released and reused basically each frame. This put a constraint, mutable data imported into the graph is hard to work with,
        // for example you could build a tlas the next frame if this is seen as an internal or created on the spot data structure, but exporting it would block the cpu on interacting with it until the previous frame has ended.
        // To further emphasise this there will need to be a dedicated way to handle multiple data based of frames in flight , transformation matrices and the camera should only live as long as a frame.

        let pass_count = self.passes.len();

        let mut resource_usages: BTreeMap<u32, ResourceLifetimeUsage> = BTreeMap::new();
        let mut hazard_states: HashMap<u32, ResourceHazardState> = HashMap::new();

        let mut dep_graph = petgraph::graph::DiGraph::<usize, PassDependency>::with_capacity(pass_count, pass_count * 2);

        //Note: acyclic graph ir with phi

        let pass_nodes: Vec<petgraph::graph::NodeIndex> = (0..pass_count).map(|i| dep_graph.add_node(i)).collect();

        for (pass_id, pass) in self.passes.iter().enumerate() {
            let common = match pass {
                AnyRenderPass::Rt(rt) => &rt.common,
                AnyRenderPass::Raster(raster) => &raster.common,
                AnyRenderPass::Compute(compute) => &compute.common,
            };

            for read in &common.read {
                let res_id = read.id;
                record_usage(&mut resource_usages, res_id, pass_id, read.access);
                let state = hazard_states.entry(res_id).or_default();
                if let Some((w_pass, w_access)) = state.last_writer {
                    add_dep_edge(
                        &mut dep_graph,
                        &pass_nodes,
                        w_pass,
                        pass_id,
                        ResourceBarrier {
                            resource_id: res_id,
                            prev_access: w_access,
                            next_access: read.access.access_type,
                        },
                    );
                }
                state.readers_since_write.push((pass_id, read.access.access_type));
            }

            for write in &common.write {
                let res_id = write.id;
                record_usage(&mut resource_usages, res_id, pass_id, write.access);
                let state = hazard_states.entry(res_id).or_default();
                if !state.readers_since_write.is_empty() {
                    for (r_pass, r_access) in &state.readers_since_write {
                        add_dep_edge(
                            &mut dep_graph,
                            &pass_nodes,
                            *r_pass,
                            pass_id,
                            ResourceBarrier {
                                resource_id: res_id,
                                prev_access: *r_access,
                                next_access: write.access.access_type,
                            },
                        );
                    }
                } else if let Some((w_pass, w_access)) = state.last_writer {
                    add_dep_edge(
                        &mut dep_graph,
                        &pass_nodes,
                        w_pass,
                        pass_id,
                        ResourceBarrier {
                            resource_id: res_id,
                            prev_access: w_access,
                            next_access: write.access.access_type,
                        },
                    );
                }
                state.last_writer = Some((pass_id, write.access.access_type));
                state.readers_since_write.clear();
            }
        }

        // Export the end state of every resource: the last pass that touches it
        // and the access it is left in when the submission completes. Usages are
        // recorded in pass-id order, so the last entry is the latest pass.
        // TODO(temp impl): nothing consumes these yet — see `ResourceEndState`
        // for the intended pass-to-pass cross-submission sync.
        self.resource_end_states.clear();
        for (res_id, info) in self.virtual_resources.iter().enumerate() {
            let res_id = res_id as u32;
            let internal = matches!(info, GraphResourceInfo::Created(_));
            let (last_use_pass, end_access) = match resource_usages.get(&res_id).and_then(|u| u.usages.last()) {
                Some((pass, access)) => (Some(*pass), access.access_type),
                None => (None, vk_sync::AccessType::Nothing),
            };
            self.resource_end_states.insert(
                res_id,
                ResourceEndState {
                    last_use_pass,
                    end_access,
                    internal,
                },
            );
        }

        // Weakly-connected components via union-find over dependency edges. Any resource
        // shared by multiple passes already produced at least one hazard edge above, so
        // passes that share a resource end up in the same component.
        let mut uf = petgraph::unionfind::UnionFind::<usize>::new(pass_count);
        for edge in dep_graph.edge_indices() {
            let (a, b) = dep_graph.edge_endpoints(edge).expect("edge from iterator must exist");
            uf.union(a.index(), b.index());
        }
        let labels = uf.into_labeling();

        let mut components_by_root: HashMap<usize, PassComponent> = HashMap::new();
        for (pass_id, root) in labels.iter().enumerate() {
            components_by_root
                .entry(*root)
                .or_insert_with(|| PassComponent {
                    passes: vec![],
                    resources: vec![],
                })
                .passes
                .push(pass_id);
        }
        for (res_id, usage) in &resource_usages {
            let root = labels[usage.first_pass];
            components_by_root
                .get_mut(&root)
                .expect("pass component must exist for any resource that was touched")
                .resources
                .push(*res_id);
        }
        let components: Vec<PassComponent> = components_by_root.into_values().collect();

        self.transient_resources
            .populate(Rc::clone(&self.core), &self.virtual_resources, &components, &resource_usages)?;

        // Topological order of passes. petgraph's toposort fails iff there is a
        // cycle, which would be a logic bug since hazards only ever produce
        // forward (lower pass_id → higher pass_id) edges by construction.
        let topo = petgraph::algo::toposort(&dep_graph, None)
            .map_err(|_| SrError::new_custom("render graph dependency graph contains a cycle".to_string()))?;

        // Pre-group all barriers by the pass that needs them issued *before* it
        // runs. Each dep edge contributes its barriers to the destination pass.
        let mut incoming: HashMap<usize, Vec<ResourceBarrier>> = HashMap::new();
        for edge in dep_graph.edge_references() {
            let dst = dep_graph[edge.target()];
            incoming
                .entry(dst)
                .or_default()
                .extend(edge.weight().barriers.iter().cloned());
        }

        let device = self.core.device().inner().clone();
        // The persistent cmd buffer was allocated in `RenderGraph::new`. Reset
        // it before re-recording — the pool was created with
        // `RESET_COMMAND_BUFFER`, so per-buffer reset is allowed. No
        // `ONE_TIME_SUBMIT` flag since the buffer is re-used across compiles.
        let raw_cb = self.cmd_buffer.inner();
        unsafe {
            device.reset_command_buffer(raw_cb, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(raw_cb, &vk::CommandBufferBeginInfo::default())?;
        }

        // Initial layout transitions for created (transient) images. Their memory
        // is freshly bound this frame, so the image is in UNDEFINED; the first
        // pass that touches one accesses it through a storage/sampled descriptor
        // that requires GENERAL / SHADER_READ_ONLY. The hazard graph only emits
        // producer->consumer barriers, so a created image that is *written first*
        // (the common case: an RT/compute pass producing it) would otherwise be
        // accessed while still UNDEFINED. Discard-transition each one up front to
        // the layout implied by its first access. Imported resources are excluded:
        // they carry a layout from outside the graph (or across frames).
        let mut init_barriers: Vec<ResourceBarrier> = Vec::new();
        for (res_id, usage) in &resource_usages {
            let is_created_image = matches!(
                self.virtual_resources.get(*res_id as usize),
                Some(GraphResourceInfo::Created(GraphResourceDesc::Image(_)))
            );
            if !is_created_image {
                continue;
            }
            if let Some((_, first_access)) = usage.usages.first() {
                init_barriers.push(ResourceBarrier {
                    resource_id: *res_id,
                    prev_access: vk_sync::AccessType::Nothing,
                    next_access: first_access.access_type,
                });
            }
        }
        if !init_barriers.is_empty() {
            self.transient_resources.emit_barriers(&device, raw_cb, &init_barriers);
        }

        // Drive each pass in topological order. We borrow `self.passes` mutably
        // (closures are FnMut) but only `self.transient_resources` immutably, so
        // the disjoint-field split borrow is fine.
        for node in &topo {
            let pass_id = dep_graph[*node];

            if let Some(barriers) = incoming.remove(&pass_id) {
                self.transient_resources.emit_barriers(&device, raw_cb, &barriers);
                self.transient_resources.recorded_barriers.push((pass_id, barriers));
            }

            let common = match &mut self.passes[pass_id] {
                AnyRenderPass::Rt(rt) => &mut rt.common,
                AnyRenderPass::Raster(raster) => &mut raster.common,
                AnyRenderPass::Compute(compute) => &mut compute.common,
            };
            if let Some(render) = common.render.as_mut() {
                let mut cb_handle = raw_cb;
                render(&mut cb_handle, &self.transient_resources)?;
            }
        }

        unsafe { device.end_command_buffer(raw_cb)? };

        Ok(())
    }

    /// End states of every resource as collected by the last [`Self::compile`],
    /// keyed by resource id (cleared on [`Self::reset`]). For *imported*
    /// resources this is the state the resource is left in when the graph's
    /// submission completes — the caller can chain further GPU work with a
    /// plain pipeline barrier from `end_access` instead of waiting the device
    /// idle. Internal (created/transient) resources are reported too but die
    /// with the next `reset`. Temp impl, see [`ResourceEndState`].
    pub fn resource_end_states(&self) -> &HashMap<u32, ResourceEndState> {
        &self.resource_end_states
    }

    /// End state of one resource by handle, if the graph compiled it.
    pub fn end_state<R: Resource>(&self, handle: &Handle<R>) -> Option<&ResourceEndState> {
        self.resource_end_states.get(&handle.id)
    }

    /// Submit the recorded command buffer to the graphics queue, signaling
    /// `signal_fence` when GPU execution completes. The caller owns the fence
    /// and is responsible for waiting on it before the next `compile`
    /// re-records into the same command buffer.
    pub fn run(
        &mut self,
        signal_fence: &mut Fence,
        wait_semaphores: &[vk::Semaphore],
        wait_stages: &[vk::PipelineStageFlags],
    ) -> SrResult<()> {
        let fence_handle = signal_fence.submit()?;
        self.core
            .graphics_queue()
            .submit_async(self.cmd_buffer.inner(), wait_semaphores, wait_stages, &[], fence_handle)?;
        Ok(())
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vulkan_abstraction::buffer::BufferDesc;
    use crate::vulkan_abstraction::image::sampler::SamplerDesc;
    use gpu_allocator::MemoryLocation;

    fn image(size: u32, name: &'static str) -> ImageDesc {
        ImageDesc {
            extent: vk::Extent3D {
                width: size,
                height: size,
                depth: 1,
            },
            format: vk::Format::R8G8B8A8_UNORM,
            tiling: vk::ImageTiling::OPTIMAL,
            location: MemoryLocation::GpuOnly,
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            name,
        }
    }

    fn buffer(bytes: u64, name: &'static str) -> BufferDesc {
        BufferDesc {
            byte_size: bytes,
            alignment: 16,
            memory_location: MemoryLocation::GpuOnly,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            name,
        }
    }

    fn lifetime(first: usize, last: usize) -> ResourceLifetimeUsage {
        ResourceLifetimeUsage {
            first_pass: first,
            last_pass: last,
            usages: vec![],
        }
    }

    /// Exercises `TransientResources::populate` on a hand-built set of virtual
    /// resources whose lifetimes deliberately allow reuse. Prints the slot
    /// assignment + aliasing groups so the allocator decisions are inspectable
    /// with `cargo test transient_aliasing -- --nocapture`.
    ///
    /// Resource layout:
    ///   res 0: 256x256 image,   lifetime [0,1]  ┐ overlap → distinct slots
    ///   res 1: 512x512 image,   lifetime [0,1]  ┘
    ///   res 2: 128x128 image,   lifetime [2,3]  ┐ overlap, both 0/1 dead → reuse
    ///   res 3: 4096-byte buffer,lifetime [2,3]  ┘
    ///   res 4: 1024-byte buffer,lifetime [4,5]  → reuses earliest free slot
    ///   res 5: sampler                          (not aliased)
    #[test]
    fn transient_aliasing_debug() {
        let core = Rc::new(Core::new(false, false, vk::Format::R8G8B8A8_UNORM).expect("Core::new failed"));

        let virtual_resources = vec![
            GraphResourceInfo::Created(GraphResourceDesc::Image(image(256, "img_256"))),
            GraphResourceInfo::Created(GraphResourceDesc::Image(image(512, "img_512"))),
            GraphResourceInfo::Created(GraphResourceDesc::Image(image(128, "img_128"))),
            GraphResourceInfo::Created(GraphResourceDesc::Buffer(buffer(4096, "buf_4k"))),
            GraphResourceInfo::Created(GraphResourceDesc::Buffer(buffer(1024, "buf_1k"))),
            GraphResourceInfo::Created(GraphResourceDesc::Sampler(SamplerDesc {
                min_filter: vk::Filter::LINEAR,
                mag_filter: vk::Filter::LINEAR,
                address_mode_u: vk::SamplerAddressMode::REPEAT,
                address_mode_v: vk::SamplerAddressMode::REPEAT,
                address_mode_w: vk::SamplerAddressMode::REPEAT,
                mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            })),
        ];

        let mut usages: BTreeMap<u32, ResourceLifetimeUsage> = BTreeMap::new();
        usages.insert(0, lifetime(0, 1));
        usages.insert(1, lifetime(0, 1));
        usages.insert(2, lifetime(2, 3));
        usages.insert(3, lifetime(2, 3));
        usages.insert(4, lifetime(4, 5));
        // sampler: lifetime doesn't matter for slot assignment, but populate uses
        // `usages` only via `pending` keys, so we still record one.
        usages.insert(5, lifetime(0, 5));

        let components = vec![PassComponent {
            passes: (0..6).collect(),
            resources: vec![0, 1, 2, 3, 4, 5],
        }];

        let mut transient = TransientResources::default();
        transient
            .populate(Rc::clone(&core), &virtual_resources, &components, &usages)
            .expect("populate failed");

        println!("{transient:?}");

        // Sanity: overlapping-lifetime resources must NOT share a slot.
        assert_ne!(
            transient.resource_slots[&0], transient.resource_slots[&1],
            "res 0 and 1 overlap; must be in different slots"
        );
        assert_ne!(
            transient.resource_slots[&2], transient.resource_slots[&3],
            "res 2 and 3 overlap; must be in different slots"
        );
        // Total slot count must be strictly fewer than the number of aliasable
        // resources — otherwise no aliasing happened at all.
        assert!(
            transient.slot_allocations.len() < 5,
            "expected aliasing to reduce 5 resources to fewer slots; got {} slots",
            transient.slot_allocations.len()
        );
        // Sampler must have been materialized but not slot-aliased.
        assert_eq!(transient.transient_samplers.len(), 1);
        assert!(!transient.resource_slots.contains_key(&5));
    }

    /// End-to-end compile test: two compute passes with a producer→consumer
    /// dependency. Verifies that (a) the topological traversal invokes the
    /// passes in dependency order, (b) the render closure receives a non-null
    /// command buffer that's been begin-recorded, and (c) compile returns a
    /// `RenderGraph<Built>` carrying a real `CmdBuffer`.
    #[test]
    fn compile_runs_passes_in_topo_order() {
        use crate::render_graph::pass_builder::{ComputeRenderPassBuilder, PassCommonDataBuilder};
        use std::cell::RefCell;

        let core = Rc::new(Core::new(false, false, vk::Format::R8G8B8A8_UNORM).expect("Core::new failed"));
        let mut rg = RenderGraph::new(Rc::clone(&core)).expect("RenderGraph::new failed");

        let img_a = rg.create_resource(image(64, "img_a"));
        let img_b = rg.create_resource(image(64, "img_b"));

        // Shared trace: each render closure pushes its name.
        let trace: Rc<RefCell<Vec<&'static str>>> = Rc::new(RefCell::new(Vec::new()));

        // Pass 0: write img_a.
        let mut common0 = PassCommonDataBuilder::new(&mut rg, "producer");
        common0
            .write(&img_a, vk_sync::AccessType::ComputeShaderWrite)
            .expect("producer write");
        {
            let trace = Rc::clone(&trace);
            common0.render(move |cb, _tr| {
                assert_ne!(*cb, vk::CommandBuffer::null(), "producer got null cmd buffer");
                trace.borrow_mut().push("producer");
                Ok(())
            });
        }
        let producer = ComputeRenderPassBuilder::default()
            .common(common0.build())
            .build()
            .expect("build producer pass");
        rg.add_render_pass(AnyRenderPass::Compute(producer));

        // Pass 1: read img_a, write img_b → depends on pass 0.
        let mut common1 = PassCommonDataBuilder::new(&mut rg, "consumer");
        common1
            .read(&img_a, vk_sync::AccessType::ComputeShaderReadOther)
            .expect("consumer read");
        common1
            .write(&img_b, vk_sync::AccessType::ComputeShaderWrite)
            .expect("consumer write");
        {
            let trace = Rc::clone(&trace);
            common1.render(move |cb, tr| {
                assert_ne!(*cb, vk::CommandBuffer::null(), "consumer got null cmd buffer");
                // img_a must be bound to a transient slot at this point.
                assert!(tr.resource_slots.contains_key(&0), "img_a not bound after populate");
                trace.borrow_mut().push("consumer");
                Ok(())
            });
        }
        let consumer = ComputeRenderPassBuilder::default()
            .common(common1.build())
            .build()
            .expect("build consumer pass");
        rg.add_render_pass(AnyRenderPass::Compute(consumer));

        rg.compile().expect("compile failed");

        // Print the transient state — the report now includes the barrier
        // trace recorded by compile.
        println!("{:?}", rg.transient_resources);

        // Both render closures must have fired, producer before consumer.
        {
            let trace = trace.borrow();
            assert_eq!(*trace, vec!["producer", "consumer"], "topo order violated");
        }
        // Persistent cmd buffer must be recorded.
        let recorded_cb = rg.cmd_buffer.inner();
        assert_ne!(recorded_cb, vk::CommandBuffer::null());
        // At least one barrier must have been recorded (producer→consumer RAW on img_a).
        assert!(
            !rg.transient_resources.recorded_barriers.is_empty(),
            "expected at least one recorded barrier between producer and consumer"
        );

        // Submit + wait. The graph stays usable for re-compile after run.
        let mut fence = Fence::new_unsignaled(Rc::clone(core.device())).expect("Fence::new");
        rg.run(&mut fence, &[], &[]).expect("run failed");
        fence.wait().expect("fence wait failed");

        // The same primary command buffer persists; not reallocated by run().
        assert_eq!(rg.cmd_buffer.inner(), recorded_cb, "cmd_buffer was reallocated across run()");
    }
}
