use crate::error::{SrError, SrResult};
use crate::render_graph::error::GraphError;
use crate::render_graph::pass_builder::{ComputeRenderPass, DynRenderFn, RasterRenderPass, RaytracingRenderPass};
use crate::vulkan_abstraction::{AccelerationStructure, Buffer, CmdBuffer, Core, Image, RawBuffer, Sampler};
use ash::vk;
use enum_as_inner::EnumAsInner;
use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use vk_sync_fork as vk_sync;

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

impl Into<GraphResourceInfo> for GraphResourceImportInfo {
    fn into(self) -> GraphResourceInfo {
        GraphResourceInfo::Imported(self)
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

impl Into<GraphResourceDesc> for ImageDesc {
    fn into(self) -> GraphResourceDesc {
        GraphResourceDesc::Image(self)
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

impl Into<GraphResourceDesc> for BufferDesc {
    fn into(self) -> GraphResourceDesc {
        GraphResourceDesc::Buffer(self)
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

impl Into<GraphResourceDesc> for SamplerDesc {
    fn into(self) -> GraphResourceDesc {
        GraphResourceDesc::Sampler(self)
    }
}

impl ResourceDesc for SamplerDesc {
    type Resource = Sampler;
}

#[derive(Clone, Debug)]
pub struct RaytracingASDesc {}

impl Into<GraphResourceDesc> for RaytracingASDesc {
    fn into(self) -> GraphResourceDesc {
        GraphResourceDesc::RaytracingAS(self)
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

fn record_usage(
    usages: &mut BTreeMap<u32, ResourceLifetimeUsage>,
    res_id: u32,
    pass_id: usize,
    access: PassResourceAccessType,
) {
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
        graph.edge_weight_mut(e).expect("edge just found must have a weight").barriers.push(barrier);
    } else {
        graph.add_edge(s, d, PassDependency { barriers: vec![barrier] });
    }
}

pub struct RenderGraph<State: RenderGraphState> {
    next_pass_id: u32,
    next_resource_id: u32,
    //TODO debug hooks and tools
    virtual_resources: Vec<GraphResourceInfo>,
    passes: Vec<AnyRenderPass>,
    transient_resources: TransientResources,
    /// At most one swapchain target per graph; remembered so the compile step can
    /// flag the present-target layout transition without scanning every resource.
    swapchain_resource_id: Option<u32>,
    state_data: State,
}

impl RenderGraph<Setup> {
    pub fn new() -> Self {
        RenderGraph {
            next_pass_id: 0,
            next_resource_id: 0,
            passes: vec![],
            virtual_resources: vec![],
            transient_resources: TransientResources::default(),
            swapchain_resource_id: None,
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

    pub fn create_raw_resource(&mut self, resource_desc: GraphResourceDesc) {
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
    pub fn import_swapchain(&mut self, image: Arc<Image>) -> SrResult<Handle<Image>> { //TODO this approach doesn't work I'll change it later so that the swapchain can change without actually rebuild the graph
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

    pub fn compile(mut self, core: Rc<Core>) -> SrResult<RenderGraph<Built>> {
        //TODO mark the render pass goals as the result of the graph so anything unnecessary can be removed
        //TODO there are some complex optimizations as shown here https://www.youtube.com/watch?v=v9LaTFLhP38 and this is the site where it will be published the paper https://dl.acm.org/profile/99661091135
        //TODO respect PassResourceAccessSyncType (NeverSync / SkipSyncIfSameAccessType) when deciding whether to emit a barrier

        let pass_count = self.passes.len();

        let mut resource_usages: BTreeMap<u32, ResourceLifetimeUsage> = BTreeMap::new();
        let mut hazard_states: HashMap<u32, ResourceHazardState> = HashMap::new();

        let mut dep_graph =
            petgraph::graph::DiGraph::<usize, PassDependency>::with_capacity(pass_count, pass_count * 2);
        let pass_nodes: Vec<petgraph::graph::NodeIndex> =
            (0..pass_count).map(|i| dep_graph.add_node(i)).collect();

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
                .or_insert_with(|| PassComponent { passes: vec![], resources: vec![] })
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
            .populate(core, &self.virtual_resources, &components, &resource_usages)?;

        //TODO topological-order traversal of dep_graph, emit barriers per edge, invoke each pass's DynRenderFn
        //TODO build the final BuiltRenderGraph (cmd buffer recording) and transition into RenderGraph<Built>
        todo!("compile: command-buffer recording from dep_graph + components is not implemented yet")
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
                    let (handle, reqs) =
                        RawBuffer::create_unbound(&core, buffer_desc.byte_size, buffer_desc.usage)?;
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
                    let image =
                        Image::from_aliased(Rc::clone(&core), handle, desc.extent, desc.format, reqs.size)?;
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
                    let buffer =
                        RawBuffer::from_aliased(Rc::clone(&core), handle, desc.byte_size, desc.usage)?;
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
    fn free_internal_state(&mut self) {
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

        if let Some(core) = self.core.as_ref() {
            let mut allocator = core.allocator_mut();
            for allocation in self.slot_allocations.drain(..) {
                if let Err(e) = allocator.free(allocation) {
                    log::error!(
                        "Allocator::free returned {e} in TransientResources::free_internal_state"
                    );
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

pub trait RgImportable<ResDesc: ResourceDesc> {
    //TODO do I want to take ownership of the data?
    fn import(&self) -> ResDesc;
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
