use crate::render_graph::pass_builder::{ComputeRenderPass, DynRenderFn, RasterRenderPass, RaytracingRenderPass};
use crate::vulkan_abstraction::{AccelerationStructure, Buffer, CmdBuffer, Core, Image, RawBuffer};
use enum_as_inner::EnumAsInner;
use std::collections::{BTreeMap, HashMap};
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
pub struct ImageDesc {}

impl Into<GraphResourceDesc> for ImageDesc {
    fn into(self) -> GraphResourceDesc {
        GraphResourceDesc::Image(self)
    }
}

impl ResourceDesc for ImageDesc {
    type Resource = Image;
}

pub struct BufferDesc {}
pub struct RaytracingASDesc {}

pub enum GraphResourceDesc {
    Image(ImageDesc),
    Buffer(BufferDesc),
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

    pub fn compile(mut self) -> RenderGraph<Built> {
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
            .populate(&self.virtual_resources, &components, &resource_usages);

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
    transient_images: HashMap<u32, Image>,
    external_buffers: HashMap<u32, Arc<RawBuffer>>,
    transient_buffers: HashMap<u32, Box<dyn Buffer>>,
    external_raytracing_ac: HashMap<u32, Arc<AccelerationStructure>>,
    transient_raytracing_ac: HashMap<u32, AccelerationStructure>,
    /// Maps each *transient* (Created) resource id to the alias slot id it shares with
    /// other resources whose lifetimes do not overlap. Populated by `populate`; consumed
    /// by the (still-TODO) per-slot physical allocation step.
    resource_slots: HashMap<u32, u32>,
    //TODO swapchain image: the import variant carries no Arc; resolve at frame-begin time
    //     and stash the per-frame swapchain Image here.
}
impl TransientResources {
    /// Allocate (or import) backing storage for every virtual resource. `components`
    /// groups passes that transitively share resources; aliasing decisions are made
    /// per-component because resources in different components have disjoint pass
    /// sets and can always reuse memory. `usages` carries the per-resource lifetime
    /// the interval-graph aliaser needs.
    pub(crate) fn populate(
        &mut self,
        virtual_resources: &[GraphResourceInfo],
        components: &[PassComponent],
        usages: &BTreeMap<u32, ResourceLifetimeUsage>,
    ) {
        // Clear last frame's bindings so re-compiling is idempotent.
        // TODO: once we have a real recycler we should reuse transient allocations across
        // frames instead of dropping + reallocating every compile.
        self.external_images.clear();
        self.transient_images.clear();
        self.external_buffers.clear();
        self.transient_buffers.clear();
        self.external_raytracing_ac.clear();
        self.transient_raytracing_ac.clear();
        self.resource_slots.clear();

        // 1. Per-component greedy interval coloring -> alias slots.
        // Within a component, sort transient resources by `first_pass` then for each one
        // reuse any slot whose last active resource finished before this one starts.
        // Slots are globally unique across components so callers can key allocations on
        // them without remembering which component a slot came from.
        // TODO: also gate reuse on resource-type / size / format compatibility once the
        //       descriptor structs (currently empty placeholders) actually carry that info.
        let mut next_slot: u32 = 0;
        for component in components {
            // (last_pass_in_slot, slot_id) for slots currently holding a still-live resource.
            let mut active: Vec<(usize, u32)> = Vec::new();

            let mut transients: Vec<u32> = component
                .resources
                .iter()
                .copied()
                .filter(|res_id| {
                    matches!(
                        virtual_resources.get(*res_id as usize),
                        Some(GraphResourceInfo::Created(_))
                    )
                })
                .collect();
            transients.sort_by_key(|res_id| usages[res_id].first_pass);

            for res_id in transients {
                let lifetime = &usages[&res_id];
                let reused = active
                    .iter()
                    .position(|(last_pass, _)| *last_pass < lifetime.first_pass);
                let slot = if let Some(idx) = reused {
                    let (_, slot) = active[idx];
                    active[idx] = (lifetime.last_pass, slot);
                    slot
                } else {
                    let slot = next_slot;
                    next_slot += 1;
                    active.push((lifetime.last_pass, slot));
                    slot
                };
                self.resource_slots.insert(res_id, slot);
            }
        }

        // 2. Wire imported handles into the matching external_* maps, keyed by resource
        // id so later barrier emission and pass execution can look them up.
        // Created resources are deferred (see TODO below) because the *Desc placeholders
        // do not yet carry the format/extent/usage info needed to construct anything.
        for (res_id, resource_info) in virtual_resources.iter().enumerate() {
            let res_id = res_id as u32;
            match resource_info {
                GraphResourceInfo::Created(_desc) => {
                    //TODO allocate the physical image / buffer / AS for `self.resource_slots[&res_id]`
                    //     once ImageDesc / BufferDesc / RaytracingASDesc gain real fields.
                    //     The slot id already tells us which resources can share memory.
                }
                GraphResourceInfo::Imported(import) => match import {
                    GraphResourceImportInfo::Image { resource, .. } => {
                        self.external_images.insert(res_id, resource.clone());
                    }
                    GraphResourceImportInfo::Buffer { resource, .. } => {
                        self.external_buffers.insert(res_id, resource.clone());
                    }
                    GraphResourceImportInfo::RayTracingAcceleration { resource, .. } => {
                        self.external_raytracing_ac.insert(res_id, resource.clone());
                    }
                    GraphResourceImportInfo::SwapchainImage => {
                        //TODO resolve the swapchain image at frame-begin time and insert
                        //     into `external_images` keyed by `res_id`.
                    }
                },
            }
        }
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
