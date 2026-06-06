use crate::error::{SrError, SrResult};
use crate::render_graph::error::GraphError;
use crate::render_graph::pass_builder::{ComputeRenderPass, RasterRenderPass, RaytracingRenderPass};
use crate::vulkan_abstraction::{AccelerationStructure, CmdBuffer, Core, Fence, Image, RawBuffer, Sampler};
use ash::vk;
use enum_as_inner::EnumAsInner;
use petgraph::visit::EdgeRef;
use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use vk_sync_fork as vk_sync;

/// Pick the right `vk::ImageAspectFlags` for a given format. Used when building
/// image subresource ranges for layout transitions in `emit_barriers`.
fn aspect_for(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::X8_D24_UNORM_PACK32 => vk::ImageAspectFlags::DEPTH,
        vk::Format::D16_UNORM_S8_UINT | vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        _ => vk::ImageAspectFlags::COLOR,
    }
}

pub trait Resource {
    type Desc: ResourceDesc;
    fn borrow_resource(res: &AnyRenderResource) -> &Self; //TODO this is useless basically
}
pub trait ResourceDesc: Clone + std::fmt::Debug + Into<GraphResourceDesc> {
    type Resource: Resource;
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RawResourceHandle {
    pub(crate) id: u32,
    pub(crate) version: u32,
}

#[derive(Debug)]
pub struct Handle<ResourceType: Resource> {
    pub(crate) raw: RawResourceHandle,
    pub(crate) desc: <ResourceType as Resource>::Desc,
    pub(crate) marker: PhantomData<ResourceType>,
}

// Manual `Clone` so a `Handle` is cloneable regardless of whether the resource
// type itself is `Clone` (it never needs to be — only the `Desc` is stored).
// `#[derive(Clone)]` would add a spurious `ResourceType: Clone` bound.
impl<ResourceType: Resource> Clone for Handle<ResourceType> {
    fn clone(&self) -> Self {
        Self {
            raw: self.raw,
            desc: self.desc.clone(),
            marker: PhantomData,
        }
    }
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
    ImportedBuffer(Arc<RawBuffer>),
    OwnedSampler(Sampler),
    ImportedSampler(Arc<Sampler>),
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
#[derive(Clone, Debug)]
pub struct ImageDesc {
    pub extent: vk::Extent3D,
    pub format: vk::Format,
    pub tiling: vk::ImageTiling,
    pub location: gpu_allocator::MemoryLocation,
    pub usage: vk::ImageUsageFlags,
    pub name: &'static str,
}

impl From<ImageDesc> for GraphResourceDesc {
    fn from(val: ImageDesc) -> Self {
        GraphResourceDesc::Image(val)
    }
}

impl ResourceDesc for ImageDesc {
    type Resource = Image;
}

#[derive(Clone, Debug)]
pub struct BufferDesc {
    pub byte_size: vk::DeviceSize,
    pub alignment: u64,
    pub memory_location: gpu_allocator::MemoryLocation,
    pub usage: vk::BufferUsageFlags,
    pub name: &'static str,
}

impl From<BufferDesc> for GraphResourceDesc {
    fn from(val: BufferDesc) -> Self {
        GraphResourceDesc::Buffer(val)
    }
}

impl ResourceDesc for BufferDesc {
    type Resource = RawBuffer;
}

#[derive(Clone, Debug)]
pub struct SamplerDesc {
    pub min_filter: vk::Filter,
    pub mag_filter: vk::Filter,
    pub address_mode_u: vk::SamplerAddressMode,
    pub address_mode_v: vk::SamplerAddressMode,
    pub address_mode_w: vk::SamplerAddressMode,
    pub mipmap_mode: vk::SamplerMipmapMode,
}

impl From<SamplerDesc> for GraphResourceDesc {
    fn from(val: SamplerDesc) -> Self {
        GraphResourceDesc::Sampler(val)
    }
}

impl ResourceDesc for SamplerDesc {
    type Resource = Sampler;
}

#[derive(Clone, Debug)]
pub struct RaytracingASDesc {}

impl From<RaytracingASDesc> for GraphResourceDesc {
    fn from(val: RaytracingASDesc) -> Self {
        GraphResourceDesc::RaytracingAS(val)
    }
}

pub enum GraphResourceDesc {
    Image(ImageDesc),
    Buffer(BufferDesc),
    Sampler(SamplerDesc),
    RaytracingAS(RaytracingASDesc),
}
#[derive(EnumAsInner)]
pub enum GraphResourceInfo {
    //TODO imported res with ownership taking option for internal aliasing later
    //this is description of what I need to allocate to satisfy the request pof the render pass
    Created(GraphResourceDesc),
    Imported(GraphResourceImportInfo),
}

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

pub struct RenderGraph {
    next_pass_id: u32,
    next_resource_id: u32,
    //TODO debug hooks and tools
    virtual_resources: Vec<GraphResourceInfo>,
    passes: Vec<AnyRenderPass>,
    transient_resources: TransientResources,
    /// At most one swapchain target per graph; remembered so the compile step can
    /// flag the present-target layout transition without scanning every resource.
    swapchain_resource_id: Option<u32>,
    /// Lives across `compile` / `run` cycles so the same primary command buffer
    /// is re-recorded each frame rather than reallocated.
    cmd_buffer: CmdBuffer,
    /// Cached so `run` can submit and `compile` can record without the caller
    /// having to re-thread `Core` through every call.
    core: Rc<Core>,
}
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
            transient_resources: TransientResources::default(),
            swapchain_resource_id: None,
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
            raw: RawResourceHandle {
                id: self.next_resource_id(),
                version: 0,
            },
            desc: TypeEquals::same(desc),
            marker: Default::default(),
        }
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
            raw: RawResourceHandle {
                id: self.next_resource_id(),
                version: 0,
            },
            desc: TypeEquals::same(desc),
            marker: Default::default(),
        }
    }

    pub fn add_render_pass(&mut self, render_pass: AnyRenderPass) {
        self.passes.push(render_pass)
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
            raw: RawResourceHandle { id, version: 0 },
            desc,
            marker: Default::default(),
        })
    }

    pub fn compile(&mut self) -> SrResult<()> {
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
        let pass_nodes: Vec<petgraph::graph::NodeIndex> = (0..pass_count).map(|i| dep_graph.add_node(i)).collect();

        for (pass_id, pass) in self.passes.iter().enumerate() {
            let common = match pass {
                AnyRenderPass::Rt(rt) => &rt.common,
                AnyRenderPass::Raster(raster) => &raster.common,
                AnyRenderPass::Compute(compute) => &compute.common,
            };

            for read in &common.read {
                let res_id = read.raw.id;
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
                let res_id = write.raw.id;
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
        self.core.graphics_queue().submit_async(
            self.cmd_buffer.inner(),
            wait_semaphores,
            wait_stages,
            &[],
            fence_handle,
        )?;
        Ok(())
    }
}

#[derive(Default)]
pub struct TransientResources {
    external_images: HashMap<u32, Arc<Image>>,
    external_buffers: HashMap<u32, Arc<RawBuffer>>,
    external_samplers: HashMap<u32, Arc<Sampler>>,
    external_raytracing_ac: HashMap<u32, Arc<AccelerationStructure>>,
    /// One wrapper per *resource id*, even when several resources share a memory
    /// slot. Each wrapper holds its own `vk::Image` handle + view; the underlying
    /// memory is owned by `slot_allocations` (Image::owns_memory == false).
    transient_images: HashMap<u32, Image>,
    /// Same indirection as `transient_images` for buffers.
    transient_buffers: HashMap<u32, RawBuffer>,
    /// Samplers are not memory-backed in the aliasable sense; one per resource id.
    transient_samplers: HashMap<u32, Sampler>,
    /// One `gpu_allocator` allocation per memory slot. Resources sharing a slot
    /// bind to the same `Allocation` at offset 0. Indexed by slot id.
    slot_allocations: Vec<gpu_allocator::vulkan::Allocation>,
    /// Maps each aliased transient resource id (images + buffers) to its slot id
    /// in `slot_allocations`. Samplers and AS are absent (they're not aliased).
    resource_slots: HashMap<u32, u32>,
    /// Trace of the barriers that `compile` issued, in topological order, one
    /// entry per pass that needed at least one barrier. Populated by `compile`
    /// after `populate` has wired resources, cleared on `free_internal_state`.
    /// Purely informational — used by the `Debug` impl; the actual barrier
    /// commands are already recorded into the command buffer at this point.
    pub(crate) recorded_barriers: Vec<(usize, Vec<ResourceBarrier>)>,
    /// Cached for `Drop`. Set on first `populate`.
    core: Option<Rc<Core>>,
}

/// Aggregated memory requirements for a single transient alias slot — built up as
/// resources are folded in, then handed to `gpu_allocator` once per slot.
#[derive(Clone, Copy, Debug)]
struct SlotMemReqs {
    size: u64,
    alignment: u64,
    /// AND of every member's `memory_type_bits`. A slot whose intersection is
    /// empty cannot be allocated and signals a bad aliasing decision.
    memory_type_bits: u32,
    location: gpu_allocator::MemoryLocation,
}

/// Pre-built `vk::Image` / `vk::Buffer` handle with its memory requirements; held
/// during populate between the "create handles" pass and the "bind to slot memory"
/// pass.
enum PendingTransient {
    Image {
        handle: vk::Image,
        reqs: vk::MemoryRequirements,
        desc: ImageDesc,
    },
    Buffer {
        handle: vk::Buffer,
        reqs: vk::MemoryRequirements,
        desc: BufferDesc,
    },
}

impl PendingTransient {
    fn reqs(&self) -> vk::MemoryRequirements {
        match self {
            PendingTransient::Image { reqs, .. } => *reqs,
            PendingTransient::Buffer { reqs, .. } => *reqs,
        }
    }
    fn location(&self) -> gpu_allocator::MemoryLocation {
        match self {
            PendingTransient::Image { desc, .. } => desc.location,
            PendingTransient::Buffer { desc, .. } => desc.memory_location,
        }
    }
}

impl TransientResources {
    /// Allocate (or import) backing storage for every virtual resource.
    ///
    /// Slots are assigned by lifetime alone — a buffer's memory can later back an
    /// image and vice versa. Compatibility is enforced at slot creation: when a
    /// candidate slot is reused, its accumulated `memory_type_bits` must still
    /// intersect with the new member's; otherwise a fresh slot is opened. The
    /// resulting slot allocation's `requirements = (max size, max alignment, AND
    /// of memory_type_bits)` is what `gpu_allocator` actually sees.
    ///
    /// Lifetime + Drop:
    ///   - Slot allocations live on `Self`. `Drop` frees them via the cached `core`.
    ///   - Individual transient `Image` / `RawBuffer` wrappers are constructed with
    ///     `owns_memory == false` so their own `Drop` only destroys the `vk::Image`
    ///     / `vk::Buffer` handle (+ view), never `Allocator::free`.
    ///   - This lets `TransientResources` outlive a single frame: the same graph
    ///     can be replayed each frame without re-allocating.
    pub(crate) fn populate(
        &mut self,
        core: Rc<Core>,
        virtual_resources: &[GraphResourceInfo],
        components: &[PassComponent],
        usages: &BTreeMap<u32, ResourceLifetimeUsage>,
    ) -> SrResult<()> {
        // Drop previous frame's bindings + free their allocations. The graph is
        // designed to be re-populated; cross-frame reuse of the same allocations is
        // a future optimization.
        // TODO: detect that desc+lifetimes haven't changed and keep `slot_allocations`
        //       alive across populate calls so we don't churn the allocator each frame.
        self.free_internal_state();
        self.core = Some(Rc::clone(&core));

        // ---------- Phase 1: create unbound handles + collect memory requirements. ----------
        // Sampler / AS are handled separately because they don't participate in slot aliasing.
        let mut pending: HashMap<u32, PendingTransient> = HashMap::new();
        for (res_id, resource_info) in virtual_resources.iter().enumerate() {
            let res_id = res_id as u32;
            let desc = match resource_info {
                GraphResourceInfo::Created(desc) => desc,
                GraphResourceInfo::Imported(_) => continue,
            };
            match desc {
                GraphResourceDesc::Image(image_desc) => {
                    let (handle, reqs) = Image::create_unbound(
                        &core,
                        image_desc.extent,
                        image_desc.format,
                        image_desc.tiling,
                        image_desc.usage,
                    )?;
                    pending.insert(
                        res_id,
                        PendingTransient::Image {
                            handle,
                            reqs,
                            desc: image_desc.clone(),
                        },
                    );
                }
                GraphResourceDesc::Buffer(buffer_desc) => {
                    let (handle, reqs) = RawBuffer::create_unbound(&core, buffer_desc.byte_size, buffer_desc.usage)?;
                    pending.insert(
                        res_id,
                        PendingTransient::Buffer {
                            handle,
                            reqs,
                            desc: buffer_desc.clone(),
                        },
                    );
                }
                GraphResourceDesc::Sampler(sampler_desc) => {
                    // Samplers aren't aliased. Build the wrapper and (eagerly) reserve
                    // its descriptor heap slot.
                    let sampler = Sampler::new_from_desc(Rc::clone(&core), sampler_desc)?;
                    // TODO: this descriptor pre-assignment exists for the legacy single-
                    //       slot-per-resource model; the heap rework will replace it.
                    let _ = sampler.slot();
                    self.transient_samplers.insert(res_id, sampler);
                }
                GraphResourceDesc::RaytracingAS(_) => {
                    //TODO transient AS allocation is intentionally unimplemented and will
                    //     stay that way until the acceleration-structure module is
                    //     refactored — likely in tandem with introducing clustered BLAS
                    //     and partial TLAS updates, which change the lifetime/aliasing
                    //     model enough that designing transient AS now would be wasted.
                }
            }
        }

        // ---------- Phase 2: per-component slot assignment by lifetime, with mem compat. ----------
        // Slots are global (numbered across all components). Reuse condition: the
        // candidate slot's last_pass < this resource's first_pass AND its current
        // (memory_type_bits, location) is compatible with this resource's
        // (memory_type_bits, location). On reuse, the slot's accumulated requirements
        // grow to the max(size), max(alignment), AND(memory_type_bits).
        //TODO this conditions can be relaxed especially on the memory type side
        let mut next_slot: u32 = 0;
        let mut slot_reqs: HashMap<u32, SlotMemReqs> = HashMap::new();
        for component in components {
            // (last_pass_in_slot, slot_id) — kind isn't tracked here, slot_reqs drives compat.
            let mut active: Vec<(usize, u32)> = Vec::new();

            let mut transients: Vec<u32> = component
                .resources
                .iter()
                .copied()
                .filter(|res_id| pending.contains_key(res_id))
                .collect();
            transients.sort_by_key(|res_id| usages[res_id].first_pass);

            for res_id in transients {
                let lifetime = &usages[&res_id];
                let p = &pending[&res_id];
                let this_reqs = p.reqs();
                let this_loc = p.location();

                let candidate = active.iter().position(|(last_pass, slot)| {
                    if *last_pass >= lifetime.first_pass {
                        return false;
                    }
                    let s = &slot_reqs[slot];
                    s.location == this_loc && (s.memory_type_bits & this_reqs.memory_type_bits) != 0
                });

                let slot = if let Some(idx) = candidate {
                    let (_, slot) = active[idx];
                    let s = slot_reqs.get_mut(&slot).expect("slot reqs missing");
                    s.size = s.size.max(this_reqs.size);
                    s.alignment = s.alignment.max(this_reqs.alignment);
                    s.memory_type_bits &= this_reqs.memory_type_bits;
                    active[idx] = (lifetime.last_pass, slot);
                    slot
                } else {
                    let slot = next_slot;
                    next_slot += 1;
                    slot_reqs.insert(
                        slot,
                        SlotMemReqs {
                            size: this_reqs.size,
                            alignment: this_reqs.alignment,
                            memory_type_bits: this_reqs.memory_type_bits,
                            location: this_loc,
                        },
                    );
                    active.push((lifetime.last_pass, slot));
                    slot
                };
                self.resource_slots.insert(res_id, slot);
            }
        }

        // ---------- Phase 3: allocate one chunk of memory per slot. ----------
        // Iterate by slot id so `slot_allocations[i]` corresponds to slot `i`.
        let mut slot_allocations: Vec<gpu_allocator::vulkan::Allocation> = Vec::with_capacity(next_slot as usize);
        for slot in 0..next_slot {
            let s = slot_reqs[&slot];
            if s.memory_type_bits == 0 {
                // Should not be reachable: the compat check above guarantees a non-zero
                // intersection on every reuse.
                return Err(SrError::new_custom(format!(
                    "transient slot {slot}: empty memory_type_bits after aliasing"
                )));
            }
            let mem_reqs = vk::MemoryRequirements {
                size: s.size,
                alignment: s.alignment,
                memory_type_bits: s.memory_type_bits,
            };
            // linear: false — slots may host optimal-tiled images, and since members
            // never co-occupy the slot, bufferImageGranularity within the allocation
            // isn't an issue. Buffers placed in non-linear regions still work.
            let allocation = core.allocator_mut().allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "render_graph_transient_slot",
                requirements: mem_reqs,
                location: s.location,
                linear: false,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })?;
            slot_allocations.push(allocation);
        }

        // ---------- Phase 4: bind each handle into its slot's memory, wrap, pre-reserve descriptors. ----------
        let device = core.device().inner();
        for (res_id, p) in pending {
            let slot = self.resource_slots[&res_id];
            let alloc = &slot_allocations[slot as usize];
            match p {
                PendingTransient::Image { handle, reqs, desc } => {
                    unsafe { device.bind_image_memory(handle, alloc.memory(), alloc.offset()) }?;
                    let image = Image::from_aliased(Rc::clone(&core), handle, desc.extent, desc.format, reqs.size)?;
                    // TODO: this descriptor pre-assignment exists for the legacy single-
                    //       slot-per-resource model; the heap rework will replace it.
                    if desc.usage.contains(vk::ImageUsageFlags::STORAGE) {
                        let _ = image.storage_slot();
                    }
                    if desc.usage.contains(vk::ImageUsageFlags::SAMPLED) {
                        let _ = image.sampled_slot();
                    }
                    self.transient_images.insert(res_id, image);
                }
                PendingTransient::Buffer { handle, reqs: _, desc } => {
                    unsafe { device.bind_buffer_memory(handle, alloc.memory(), alloc.offset()) }?;
                    let buffer = RawBuffer::from_aliased(Rc::clone(&core), handle, desc.byte_size, desc.usage)?;
                    // TODO: same legacy-descriptor caveat as the image path above.
                    if desc.usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER) {
                        let _ = buffer.storage_slot();
                    }
                    if desc.usage.contains(vk::BufferUsageFlags::UNIFORM_BUFFER) {
                        let _ = buffer.uniform_slot();
                    }
                    self.transient_buffers.insert(res_id, buffer);
                }
            }
        }
        self.slot_allocations = slot_allocations;

        // ---------- Phase 5: wire imported handles into external_* maps. ----------
        for (res_id, resource_info) in virtual_resources.iter().enumerate() {
            let res_id = res_id as u32;
            let import = match resource_info {
                GraphResourceInfo::Created(_) => continue,
                GraphResourceInfo::Imported(import) => import,
            };
            match import {
                GraphResourceImportInfo::Image { resource, .. } => {
                    self.external_images.insert(res_id, resource.clone());
                }
                GraphResourceImportInfo::Buffer { resource, .. } => {
                    self.external_buffers.insert(res_id, resource.clone());
                }
                GraphResourceImportInfo::Sampler { resource } => {
                    self.external_samplers.insert(res_id, resource.clone());
                }
                GraphResourceImportInfo::RayTracingAcceleration { resource, .. } => {
                    self.external_raytracing_ac.insert(res_id, resource.clone());
                }
                GraphResourceImportInfo::SwapchainImage { resource } => {
                    self.external_images.insert(res_id, resource.clone());
                }
            }
        }

        Ok(())
    }

    /// Drop all transient wrappers (which destroy their vk handles but won't free
    /// memory) and free every slot allocation. Used by both `populate` (rebuild)
    /// and `Drop`.
    pub(crate) fn free_internal_state(&mut self) {
        self.external_images.clear();
        self.external_buffers.clear();
        self.external_samplers.clear();
        self.external_raytracing_ac.clear();
        // Drop the wrappers first: their Drop destroys vk handles and skips
        // Allocator::free (owns_memory == false), so the underlying allocations are
        // still valid afterwards.
        self.transient_images.clear();
        self.transient_buffers.clear();
        self.transient_samplers.clear();
        self.resource_slots.clear();
        self.recorded_barriers.clear();

        if let Some(core) = self.core.as_ref() {
            let mut allocator = core.allocator_mut();
            for allocation in self.slot_allocations.drain(..) {
                if let Err(e) = allocator.free(allocation) {
                    log::error!("Allocator::free returned {e} in TransientResources::free_internal_state");
                }
            }
        } else {
            // No core cached: there cannot be allocations to free, but defensively clear.
            self.slot_allocations.clear();
        }
    }
}

impl Drop for TransientResources {
    fn drop(&mut self) {
        self.free_internal_state();
    }
}

impl TransientResources {
    /// Resolve a graph image handle to the concrete `Image` bound for this frame,
    /// whether it's a transient (created) resource or an imported one. Render
    /// closures call this to read an image's heap descriptor slots
    /// (`storage_slot()` / `sampled_slot()`) when the image is graph-managed
    /// rather than captured directly.
    pub fn image(&self, handle: &Handle<Image>) -> SrResult<&Image> {
        let id = handle.raw.id;
        if let Some(img) = self.transient_images.get(&id) {
            return Ok(img);
        }
        if let Some(img) = self.external_images.get(&id) {
            return Ok(img.as_ref());
        }
        Err(SrError::new_custom(format!(
            "render graph: no image bound for resource id {id} (not created or imported as an image)"
        )))
    }
}

impl TransientResources {
    /// Issue a `vkCmdPipelineBarrier` covering every `ResourceBarrier` in
    /// `barriers`. Each barrier is dispatched as an image barrier, buffer
    /// barrier, or global barrier depending on what kind of resource its
    /// `resource_id` resolves to here. Resources unknown to this struct (e.g.
    /// non-aliased samplers or AS-only ids that slipped through) collapse to a
    /// `GlobalBarrier` carrying just the access transition.
    ///
    /// Layouts: we use `vk_sync::ImageLayout::Optimal` for both sides, which
    /// tells vk_sync to pick the right `VK_IMAGE_LAYOUT_*` from the access
    /// types. Subresource range covers all mips / array layers — per-mip /
    /// per-layer barriers are a future optimization once passes can express
    /// subresource access.
    ///
    /// Queue family transfer is `IGNORED` on both sides — we're single-queue.
    ///
    /// TODO this is currently doing a 1 resource barriers to 1 actual barrier, this can be reduced to 1 barrier per image layout transition and a global barrier for the other stuff, this doesn't add any involuntary sync since I have already built the dependencies graph
    pub(crate) fn emit_barriers(&self, device: &ash::Device, cmd_buffer: vk::CommandBuffer, barriers: &[ResourceBarrier]) {
        if barriers.is_empty() {
            return;
        }
        let mut image_barriers: Vec<vk_sync::ImageBarrier> = Vec::new();
        let mut buffer_barriers: Vec<vk_sync::BufferBarrier> = Vec::new();
        let mut global_prev: Vec<vk_sync::AccessType> = Vec::new();
        let mut global_next: Vec<vk_sync::AccessType> = Vec::new();

        for b in barriers {
            let prev_slice = std::slice::from_ref(&b.prev_access);
            let next_slice = std::slice::from_ref(&b.next_access);

            // Image? (transient first, then imported — same resource id can never
            // appear in both maps so the order doesn't matter for correctness).
            let image_info = self
                .transient_images
                .get(&b.resource_id)
                .map(|img| (img.inner(), img.format()))
                .or_else(|| {
                    self.external_images
                        .get(&b.resource_id)
                        .map(|img| (img.inner(), img.format()))
                });

            if let Some((handle, format)) = image_info {
                image_barriers.push(vk_sync::ImageBarrier {
                    previous_accesses: prev_slice,
                    next_accesses: next_slice,
                    previous_layout: vk_sync::ImageLayout::Optimal,
                    next_layout: vk_sync::ImageLayout::Optimal,
                    discard_contents: false,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: handle,
                    range: vk::ImageSubresourceRange {
                        aspect_mask: aspect_for(format),
                        base_mip_level: 0,
                        level_count: vk::REMAINING_MIP_LEVELS,
                        base_array_layer: 0,
                        layer_count: vk::REMAINING_ARRAY_LAYERS,
                    },
                });
                continue;
            }

            let buffer_info = self
                .transient_buffers
                .get(&b.resource_id)
                .map(|buf| (buf.inner(), buf.byte_size()))
                .or_else(|| {
                    self.external_buffers
                        .get(&b.resource_id)
                        .map(|buf| (buf.inner(), buf.byte_size()))
                });

            if let Some((handle, size)) = buffer_info {
                buffer_barriers.push(vk_sync::BufferBarrier {
                    previous_accesses: prev_slice,
                    next_accesses: next_slice,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    buffer: handle,
                    offset: 0,
                    size: size as usize,
                });
                continue;
            }

            // Fallback: AS / sampler / anything else — collapse into a global
            // barrier so the access ordering is still expressed.
            global_prev.push(b.prev_access);
            global_next.push(b.next_access);
        }

        let global = if !global_prev.is_empty() {
            Some(vk_sync::GlobalBarrier {
                previous_accesses: &global_prev,
                next_accesses: &global_next,
            })
        } else {
            None
        };

        vk_sync::cmd::pipeline_barrier(device, cmd_buffer, global, &buffer_barriers, &image_barriers);
    }
}

impl std::fmt::Debug for TransientResources {
    /// Renders the aliasing decisions in the same "report" layout used by the
    /// transient_aliasing_debug test: header, per-slot allocation table, per-resource
    /// slot assignment, and grouped aliasing sets. Lifetimes aren't
    /// stored on `self`, so they're omitted here — only what `populate` left
    /// behind is printed.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let aliasable = self.resource_slots.len();
        let imported = self.external_images.len()
            + self.external_buffers.len()
            + self.external_samplers.len()
            + self.external_raytracing_ac.len();
        let total = self.transient_images.len() + self.transient_buffers.len() + self.transient_samplers.len() + imported;

        writeln!(f)?;
        writeln!(f, "=== TransientResources aliasing report ===")?;
        writeln!(f, "resources tracked           : {total}")?;
        writeln!(f, "  transient images          : {}", self.transient_images.len())?;
        writeln!(f, "  transient buffers         : {}", self.transient_buffers.len())?;
        writeln!(f, "  transient samplers        : {}", self.transient_samplers.len())?;
        writeln!(f, "  imported                  : {imported}")?;
        writeln!(f, "aliasable resources (img+buf): {aliasable}")?;
        writeln!(f, "slot allocations            : {}", self.slot_allocations.len())?;
        writeln!(f)?;

        writeln!(f, "Per-slot allocation:")?;
        for (i, alloc) in self.slot_allocations.iter().enumerate() {
            let mem = unsafe { alloc.memory() };
            writeln!(
                f,
                "  slot {i}: size={:>8} offset={:>8} memory={:?}",
                alloc.size(),
                alloc.offset(),
                mem,
            )?;
        }
        writeln!(f)?;

        writeln!(f, "Per-resource slot assignment:")?;
        let mut assignments: Vec<(u32, u32)> = self.resource_slots.iter().map(|(r, s)| (*r, *s)).collect();
        assignments.sort();
        for (res_id, slot_id) in &assignments {
            let kind = if let Some(img) = self.transient_images.get(res_id) {
                let e = img.extent();
                format!("Image {}x{}x{}", e.width, e.height, e.depth)
            } else if let Some(buf) = self.transient_buffers.get(res_id) {
                format!("Buffer {} bytes", buf.byte_size())
            } else {
                "???".to_string()
            };
            writeln!(f, "  res {res_id} {kind:<32} -> slot {slot_id}")?;
        }
        writeln!(f)?;

        let mut by_slot: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for (r, s) in &self.resource_slots {
            by_slot.entry(*s).or_default().push(*r);
        }
        writeln!(f, "Aliasing groups (slot -> resources sharing memory):")?;
        for (slot, members) in &by_slot {
            let mut m = members.clone();
            m.sort();
            let aliased = if m.len() > 1 { " (ALIASED)" } else { "" };
            writeln!(f, "  slot {slot} <- resources {m:?}{aliased}")?;
        }

        // Non-aliased extras
        if !self.transient_samplers.is_empty() {
            writeln!(f)?;
            let mut samplers: Vec<u32> = self.transient_samplers.keys().copied().collect();
            samplers.sort();
            writeln!(f, "Transient samplers (not slot-aliased): {samplers:?}")?;
        }
        if imported > 0 {
            writeln!(f)?;
            let mut imports: Vec<(u32, &'static str)> = Vec::new();
            imports.extend(self.external_images.keys().map(|k| (*k, "Image")));
            imports.extend(self.external_buffers.keys().map(|k| (*k, "Buffer")));
            imports.extend(self.external_samplers.keys().map(|k| (*k, "Sampler")));
            imports.extend(self.external_raytracing_ac.keys().map(|k| (*k, "AccelStruct")));
            imports.sort();
            writeln!(f, "Imported resources:")?;
            for (id, kind) in imports {
                writeln!(f, "  res {id} {kind}")?;
            }
        }

        // ---- Barriers recorded during compile ----
        writeln!(f)?;
        let total_barriers: usize = self.recorded_barriers.iter().map(|(_, b)| b.len()).sum();
        if self.recorded_barriers.is_empty() {
            writeln!(f, "Barriers recorded: none (graph not compiled or single-pass)")?;
        } else {
            writeln!(
                f,
                "Barriers recorded ({} total across {} pass(es), in topo order):",
                total_barriers,
                self.recorded_barriers.len(),
            )?;
            for (pass_id, barriers) in &self.recorded_barriers {
                writeln!(f, "  before pass {pass_id}:")?;
                for b in barriers {
                    let kind = if self.transient_images.contains_key(&b.resource_id)
                        || self.external_images.contains_key(&b.resource_id)
                    {
                        "Image"
                    } else if self.transient_buffers.contains_key(&b.resource_id)
                        || self.external_buffers.contains_key(&b.resource_id)
                    {
                        "Buffer"
                    } else if self.transient_samplers.contains_key(&b.resource_id)
                        || self.external_samplers.contains_key(&b.resource_id)
                    {
                        "Sampler"
                    } else if self.external_raytracing_ac.contains_key(&b.resource_id) {
                        "AccelStruct"
                    } else {
                        "Global"
                    };
                    writeln!(
                        f,
                        "    res {:>3} ({kind:<11}) {:?} -> {:?}",
                        b.resource_id, b.prev_access, b.next_access
                    )?;
                }
            }
        }
        writeln!(f, "==========================================")
    }
}

pub trait RgImportable<ResDesc: ResourceDesc> {
    //TODO do I want to take ownership of the data?
    fn import(&self) -> ResDesc;
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
            .shaders(vec![])
            .entry_point("main".to_string())
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
            .shaders(vec![])
            .entry_point("main".to_string())
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
        assert_eq!(
            rg.cmd_buffer.inner(),
            recorded_cb,
            "cmd_buffer was reallocated across run()"
        );
    }
}
