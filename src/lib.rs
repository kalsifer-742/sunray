
pub mod camera;
pub mod error;
pub mod render_graph;
pub mod scene;
pub mod shader_compiler;
pub mod utils;
pub mod vulkan_abstraction;
pub mod finello_pathtracing_pipeline;


/// Bevy 0.19 plugin that drives this renderer from inside a Bevy `App`.
///
/// Gated behind the `bevy` feature. See `docs/bevy_integration.md` for the
/// architecture and `examples/bevy_app` for usage. Declared after `utils` so the
/// `include_bytes_align_as!` macro is in textual scope.
#[cfg(feature = "bevy")]
pub mod bevy_integration;

pub use camera::*;
use error::*;
pub use scene::*;
pub use crate::vulkan_abstraction::DiagnosticTool;

use std::hash::Hash;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::{collections::HashMap, rc::Rc, sync::Arc};

use crate::render_graph::graph::{AnyRenderPass, BufferDesc, Handle, ImageDesc, RenderGraph};
use crate::render_graph::pass_builder::{
    ComputeRenderPassBuilder, ComputeShaders, PassCommonDataBuilder, RayTracingShaders, RaytracingRenderPassBuilder, ShaderSource,
};
use crate::utils::env_var_as_bool;
use crate::vulkan_abstraction::image::swapchain::{Surface, Swapchain};
use crate::vulkan_abstraction::swapchain::{SwapchainData, SwapchainFrame};
use crate::vulkan_abstraction::{Buffer, HostAccessibleBuffer, PostprocessPushConstant, Reservoir, ReservoirGI};
use ash::vk;
use vk_sync_fork as vk_sync;

//TODO finello
pub const DENOISE_PASSES: u32 = 8;
//TODO finello
pub const EXPOSURE: f32 = 1.0;


/// Key identifying a GPU asset (BLAS or image) inside the renderer's
/// `ResourceManager`. `group` ties together every asset created by one
/// `load_scene` call so a whole scene can be deallocated in bulk (see
/// [`Renderer::unload_scene`]); `index` is unique within the group.
///
/// Scene loading generates these and converts them into the renderer's actual
/// key type via `K: From<ResourceKey>` — use `ResourceKey` itself when no
/// third party (e.g. Bevy's asset system) supplies its own ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResourceKey {
    pub group: u64,
    pub index: u64,
}

/// The number of concurrent frames that are processed (both by CPU and GPU).
///
/// Apparently 2 is the most common choice. Empirically it seems like the performance doesn't really
/// get any better with a higher number, but it does get measurably worse with only 1.
///
/// TODO this feature is actually not doing what it is supposed and needs to be reworked, do not go over 2 I think it will crash
/// the render graph is incapable of starting a second frame with a current frame still ongoing
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

//TODO add a list of callbacks to call at the end of frames for cleanup or at start for setup
//TODO deferred deallocation for buffers and acceleration structures
//TODO validate max_frame_in_flight against the swapchain


/// Per-output-image data. The render graph now owns the intermediate G-buffer /
/// RT-output images as internal (transient) resources, so the only image that
/// still lives here is the post-process result, which the external blit copies
/// to the caller's target. `blit_cmd_buf` holds the pre-recorded blit.
struct ImageDependentData {
    pub blit_cmd_buf: vulkan_abstraction::CmdBuffer,
    postprocess_result_image: Arc<vulkan_abstraction::Image>,
}


pub type CreateSurfaceFn = dyn Fn(&ash::Entry, &ash::Instance) -> SrResult<vk::SurfaceKHR>;



pub struct Renderer<K: Hash + Eq + Copy + 'static = ResourceKey> {
    image_dependant_data: HashMap<vk::Image, ImageDependentData>,

    resource_manager: vulkan_abstraction::ResourceManager<K>,

    /// Swapchain + present plumbing, present when constructed with a surface
    /// (`new_with_surface`) — given at startup, owned internally.
    swapchain_data: Option<SwapchainData>,

    /// Next `ResourceKey::group`; one group per `load_scene` call.
    next_group: u64,
    /// Group → every key created for that scene load, for bulk deallocation.
    scene_groups: HashMap<u64, Vec<K>>,

    // Heap-mode ray-tracing SPIR-V (one blob per stage).
    // `RaytracingRenderPassBuilder::generate_render`, which interns the pipeline
    // and SBT in the graph's persistent cache (built once, reused across frame
    // rebuilds). RIS and final share miss/closest-hit/any-hit and differ only in
    // the ray-gen stage, so they intern as two distinct cache entries.
    /// Ray-gen for the RIS pass: finds the best candidates per pixel without tracing many rays.
    ray_gen_ris_spirv: &'static [u8],
    /// Ray-gen for the final pass: traces rays based on the reservoirs the RIS pass produced.
    ray_gen_final_spirv: &'static [u8],

    ray_miss_spirv: &'static [u8],
    closest_hit_spirv: &'static [u8],
    any_hit_spirv: &'static [u8],
    ///The first pass after raytracing merges the previous frame on the next one to reduce bias
    temporal_accumulation_spirv: &'static [u8],
    ///The denoise pass is run after the temporal accumulation to reduce noise even more (a-trous filter)
    denoise_spirv:&'static [u8],
    ///An extra pass to handle post-processing like exposure and color correction. Should be mathematically easy to calculate
    postprocess_spirv: &'static [u8],


    // this is about the frame being worked on by the cpu
    image_extent: vk::Extent3D,
    image_format: vk::Format,
    

    blue_noise_image: vulkan_abstraction::Image,
    blue_noise_sampler: vulkan_abstraction::Sampler,


    core: Rc<vulkan_abstraction::Core>,

    //TODO finello all of this params are temporal stuff to be incorporated in a future version of the graph when temporal stuff can be handled internally
    pub accumulation_images: [Arc<vulkan_abstraction::Image>; 2],
    pub denoising_images: [Arc<vulkan_abstraction::Image>; 2],
    ///this is used for temporal accumulation, there is an absolute frame counter in the core
    pub relative_frame_count: u32,

    /// Ping-pong reservoir buffers. Stored as `Arc<RawBuffer>` (rather than plain
    /// `GpuOnlyBuffer`) so the same buffer can be imported into the render graph
    /// each frame for hazard tracking — the RIS pass writes them and the final
    /// pass reads them, so the graph emits the reservoir hand-off barrier between
    /// the two RT passes automatically. The shader still addresses them by
    /// device-address (see `RaytracingHeapPushConstant::reservoirs`); the graph
    /// import only governs synchronization.
    reservoir_buffers: [Arc<vulkan_abstraction::RawBuffer>; 2],
    /// Ping-pong pair of GI reservoir buffers for ReSTIR GI (Ouyang 2021)
    /// contract as reservoir_buffers above, but storing surface samples (x2) instead of light samples.
    reservoir_gi_buffers: [Arc<vulkan_abstraction::RawBuffer>; 2],



    prev_view_proj: nalgebra::Matrix4<f32>, //used to calculate motion vectors 

    /// Persistent render graph.
    /// Re-populated each frame (passes / imports change because the ping-pong
    /// indices and per-frame descriptor sets do), but the underlying command
    /// buffer is reused across `compile` calls.
    pub render_graph: RenderGraph,
    /// Fence signaled when the render graph's submission completes.
    render_graph_fence: vulkan_abstraction::Fence,

    /// Timeline semaphore signaled with the **absolute frame count** when a
    /// frame's GPU work (including the final blit) completes. This replaces
    /// per-frame fences as the "wait for render end" primitive: waiting for
    /// frame N is `frame_timeline.wait(N)`.
    frame_timeline: vulkan_abstraction::TimelineSemaphore,
    /// Last frame value the watcher thread observed on `frame_timeline`.
    completed_frame: Arc<AtomicU64>,
    /// Tells the watcher thread to exit (set in `Drop`).
    frame_watcher_shutdown: Arc<AtomicBool>,
    /// Thread (spawned at construction) that waits the frame timeline and
    /// publishes `completed_frame`. The callbacks themselves run on the render
    /// thread (they capture `Rc`-based GPU resources, which are `!Send`) —
    /// `render` drains the ones whose frame the watcher reported complete.
    frame_watcher: Option<std::thread::JoinHandle<()>>,

    //TODO these would love #![feature(unboxed_closures)]
    //these are ordered,the u64 is the absolute frame on which to execute and the actual callback
    start_of_frame_callbacks: Vec<(u64, Box<dyn FnOnce()>)>,
    /// Persistent (FnMut) callbacks invoked on every `resize`.
    resize_callbacks: Vec<Box<dyn FnMut()>>,
    /// Run on the render thread once the tagged frame has *completed on the
    /// GPU* (per `completed_frame`). The per-frame CpuToGpu buffers `render`
    /// creates are deallocated through here.
    end_of_frame_callbacks: Vec<(u64, Box<dyn FnOnce()>)>,
}

/// Per-frame GPU inputs of the unified graph that live in frame-local buffers
/// (created on the spot in `render`, deferred-freed via the end-of-frame
/// callbacks): the camera matrices UBO address and the heap slots of the flat
/// transform / emissive indirection buffers.
struct FrameGpuData {
    matrices_address: vk::DeviceAddress,
    entity_transforms_slot: u32,
    emissive_indirection_slot: u32,
}
// `K: 'static` propagated from `ResourceManager` (its deferred frame work is
// stored as boxed callbacks).
impl<K: Hash + Eq + Copy + 'static> Renderer<K> {
    pub fn new(image_extent: (u32, u32), image_format: vk::Format) -> SrResult<Self> {
        Self::new_impl(image_extent, image_format, &[], None)
    }

    // It's necessary to pass a fn to create the surface, because it depends on instance, device depends on it (if present), and both device and
    // instance are created and owned inside Renderer (in Core) so this was deemed a good approach to allow the user to build their own surface.
    // The swapchain for that surface is created here too and kept internal — drive it with `render_to_swapchain`.
    pub fn new_with_surface(
        image_extent: (u32, u32),
        image_format: vk::Format,
        instance_exts: &'static [*const i8],
        create_surface: &CreateSurfaceFn,
    ) -> SrResult<Self> {
        Self::new_impl(image_extent, image_format, instance_exts, Some(create_surface))
    }

    fn new_impl(
        image_extent: (u32, u32),
        image_format: vk::Format,
        instance_exts: &'static [*const i8],
        create_surface: Option<&CreateSurfaceFn>,
    ) -> SrResult<Self> {
        let with_validation_layer = env_var_as_bool(ENABLE_VALIDATION_LAYER_ENV_VAR).unwrap_or(IS_DEBUG_BUILD);
        let with_gpuav = env_var_as_bool(ENABLE_GPUAV_ENV_VAR_NAME).unwrap_or(false);
        // Map the ENABLE_NVIDIA_AFTERMATH env var (legacy) onto the new
        // DiagnosticTool enum. When the user wants RenderDoc / RGP support,
        // add the corresponding env vars here and switch the match arm.
        let diagnostics = if env_var_as_bool(ENABLE_NVIDIA_AFTERMATH_VAR_NAME).unwrap_or(false) {
            DiagnosticTool::NvidiaAftermath
        } else {
            DiagnosticTool::None
        };

        let (core, surface) = vulkan_abstraction::Core::new_with_surface(
            with_validation_layer,
            with_gpuav,
            diagnostics,
            image_format,
            instance_exts,
            create_surface,
        )?;

        let core = Rc::new(core);

        let window_extent = image_extent;
        let image_extent = utils::tuple_to_extent3d(image_extent);

        //must be filled by loading a scene
        let resource_manager = vulkan_abstraction::ResourceManager::new_empty(Rc::clone(&core))?;

        let ray_gen_ris_spirv: &'static [u8] = include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/ray_gen_ris.spirv"));
        let ray_gen_final_spirv: &'static [u8] = include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/ray_gen_final.spirv"));
        let ray_miss_spirv: &'static [u8] = include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/ray_miss.spirv"));
        let closest_hit_spirv: &'static [u8] = include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/closest_hit.spirv"));
        let any_hit_spirv: &'static [u8] = include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/any_hit.spirv"));
        let denoise_spirv = include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/denoise.spirv"));
        let postprocess_spirv = include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/postprocess.spirv"));
        let temporal_accumulation_spirv = include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/temporal_accumulation.spirv"));

        let image_dependant_data = HashMap::new();

        let create_accum_image = |name: &'static str| -> SrResult<Arc<vulkan_abstraction::Image>> {
            Ok(Arc::new(vulkan_abstraction::Image::new(
                core.clone(),
                image_extent, // <--- USE THIS (it's already a vk::Extent3D)
                vk::Format::B10G11R11_UFLOAT_PACK32,
                vk::ImageTiling::OPTIMAL,
                gpu_allocator::MemoryLocation::GpuOnly,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                name,
            )?))
        };

        let accumulation_images = [
            create_accum_image("Accumulation_Ping")?,
            create_accum_image("Accumulation_Pong")?,
        ];

        let denoising_images = [create_accum_image("Denoise_Ping")?, create_accum_image("Denoise_Pong")?];

        let num_pixels = (image_extent.width * image_extent.height) as usize;
        let reservoir_buffers = Self::create_reservoir_buffers(&core, num_pixels)?;
        let reservoir_gi_buffers = Self::create_reservoir_gi_buffers(&core, num_pixels)?;

        let blue_noise_bytes = include_bytes!("finello_pathtracing_pipeline/util_files/noise.png");
        let blue_noise_img = image::load_from_memory(blue_noise_bytes).unwrap().to_rgba8();
        let (noise_width, noise_height) = blue_noise_img.dimensions();
        let blue_noise_data = blue_noise_img.into_raw();

        let blue_noise_image = vulkan_abstraction::Image::new_from_data(
            Rc::clone(&core),
            blue_noise_data,
            vk::Extent3D {
                width: noise_width,
                height: noise_height,
                depth: 1,
            },
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageTiling::OPTIMAL,
            gpu_allocator::MemoryLocation::GpuOnly,
            vk::ImageUsageFlags::SAMPLED,
            "blue noise texture",
        )?;

        let blue_noise_sampler = vulkan_abstraction::Sampler::new(
            Rc::clone(&core),
            vk::Filter::NEAREST,
            vk::Filter::NEAREST,
            vk::SamplerAddressMode::REPEAT,
            vk::SamplerAddressMode::REPEAT,
            vk::SamplerAddressMode::REPEAT,
            vk::SamplerMipmapMode::NEAREST,
        )?;

        let render_graph = RenderGraph::new(Rc::clone(&core))?;
        let render_graph_fence = vulkan_abstraction::Fence::new_unsignaled(Rc::clone(core.device()))?;

        // Frame timeline: signaled with the absolute frame count when each
        // frame's GPU work completes. Starts at 0 = "frame 0 (nothing) done".
        let frame_timeline = vulkan_abstraction::TimelineSemaphore::new(Rc::clone(&core), 0)?;
        let completed_frame = Arc::new(AtomicU64::new(0));
        let frame_watcher_shutdown = Arc::new(AtomicBool::new(false));

        // Watcher thread: waits the timeline value-by-value and publishes the
        // last completed frame. It only *observes* — the end-of-frame callbacks
        // run on the render thread because they capture `Rc`-based (!Send) GPU
        // resources. `ash::Device` is Send + Sync, so the raw waits are fine here.
        let frame_watcher = {
            let device = core.device().inner().clone();
            let timeline = frame_timeline.inner();
            let completed = Arc::clone(&completed_frame);
            let shutdown = Arc::clone(&frame_watcher_shutdown);
            std::thread::Builder::new()
                .name("sunray-frame-watcher".to_string())
                .spawn(move || {
                    let mut next_frame = 1u64;
                    while !shutdown.load(Ordering::Acquire) {
                        let semaphores = [timeline];
                        let values = [next_frame];
                        let wait_info = vk::SemaphoreWaitInfo::default().semaphores(&semaphores).values(&values);
                        // Short timeout so shutdown is honored promptly.
                        match unsafe { device.wait_semaphores(&wait_info, 100_000_000 /* 100ms */) } {
                            Ok(()) => {
                                completed.store(next_frame, Ordering::Release);
                                next_frame += 1;
                            }
                            Err(vk::Result::TIMEOUT) => continue,
                            Err(e) => {
                                log::error!("sunray frame watcher: vkWaitSemaphores failed with {e:?}; exiting");
                                break;
                            }
                        }
                    }
                })
                .map_err(|e| SrError::new_custom(format!("failed to spawn frame watcher thread: {e}")))?
        };

        // The swapchain abstraction is owned internally and built at startup
        // from the surface the caller's closure created.
        let swapchain_data = match surface {
            Some(surface_khr) => {
                let surface = Surface::new(core.entry(), core.instance(), surface_khr);
                Some(SwapchainData::new(&core, surface, window_extent)?)
            }
            None => None,
        };

        let mut renderer = Self {
                image_dependant_data,

                swapchain_data,
                next_group: 0,
                scene_groups: HashMap::new(),

                render_graph,
                render_graph_fence,

                frame_timeline,
                completed_frame,
                frame_watcher_shutdown,
                frame_watcher: Some(frame_watcher),

                reservoir_buffers,

                ray_gen_ris_spirv,
                ray_gen_final_spirv,
                ray_miss_spirv,
                closest_hit_spirv,
                any_hit_spirv,
                denoise_spirv,
                temporal_accumulation_spirv,
                postprocess_spirv,

                prev_view_proj: nalgebra::zero(),

                image_extent,
                image_format,



                accumulation_images,
                denoising_images,
                relative_frame_count: 0,

                blue_noise_image,
                blue_noise_sampler,

                resource_manager,
                reservoir_gi_buffers,


                core,

                start_of_frame_callbacks: vec![],
                resize_callbacks: vec![],
                end_of_frame_callbacks: vec![],
        };

        // The pre-recorded blit into each swapchain image must exist before the
        // first `render_to_swapchain` call.
        if let Some(sc) = &renderer.swapchain_data {
            let images = sc.swapchain.images().to_vec();
            renderer.build_image_dependent_data(&images)?;
        }

        Ok(renderer)
    }

    /// Allocate a ping-pong pair of GPU-only buffers holding `num_pixels`
    /// elements of `T`, usable as SSBOs addressed by device-address and importable
    /// into the render graph (`Arc<RawBuffer>`).
    fn create_reservoir_pair<T>(
        core: &Rc<vulkan_abstraction::Core>,
        num_pixels: usize,
        name_a: &'static str,
        name_b: &'static str,
    ) -> SrResult<[Arc<vulkan_abstraction::RawBuffer>; 2]> {
        let byte_size = (num_pixels * size_of::<T>()) as vk::DeviceSize;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_DST;
        let make = |name: &'static str| -> SrResult<Arc<vulkan_abstraction::RawBuffer>> {
            Ok(Arc::new(vulkan_abstraction::RawBuffer::new_aligned(
                Rc::clone(core),
                byte_size,
                1,
                gpu_allocator::MemoryLocation::GpuOnly,
                usage,
                name,
            )?))
        };
        Ok([make(name_a)?, make(name_b)?])
    }

    fn create_reservoir_buffers(
        core: &Rc<vulkan_abstraction::Core>,
        num_pixels: usize,
    ) -> SrResult<[Arc<vulkan_abstraction::RawBuffer>; 2]> {
        Self::create_reservoir_pair::<Reservoir>(core, num_pixels, "ReSTIR Reservoir Buffer A", "ReSTIR Reservoir Buffer B")
    }

    fn create_reservoir_gi_buffers(
        core: &Rc<vulkan_abstraction::Core>,
        num_pixels: usize,
    ) -> SrResult<[Arc<vulkan_abstraction::RawBuffer>; 2]> {
        Self::create_reservoir_pair::<ReservoirGI>(
            core,
            num_pixels,
            "ReSTIR GI Reservoir Buffer A",
            "ReSTIR GI Reservoir Buffer B",
        )
    }
    // ─── Frame / resize callbacks ───────────────────────────────────────────

    /// Schedule `callback` to run on the CPU at the start of the next frame
    /// (before any per-frame upload).
    pub fn add_start_of_frame_callback(&mut self, callback: impl FnOnce() + 'static) {
        let next_frame = *self.core.absolute_frame_count.borrow() as u64 + 1;
        self.start_of_frame_callbacks.push((next_frame, Box::new(callback)));
    }

    /// Schedule `callback` to run once the next rendered frame has *completed
    /// on the GPU* (per the frame timeline). This is the deferred-deallocation
    /// hook: dropping a GPU resource inside the callback is safe because the
    /// frame that used it is provably done.
    pub fn add_end_of_frame_callback(&mut self, callback: impl FnOnce() + 'static) {
        let next_frame = *self.core.absolute_frame_count.borrow() as u64 + 1;
        self.end_of_frame_callbacks.push((next_frame, Box::new(callback)));
    }

    /// Register a persistent callback invoked on every [`Self::resize`].
    pub fn add_resize_callback(&mut self, callback: impl FnMut() + 'static) {
        self.resize_callbacks.push(Box::new(callback));
    }

    /// Run the start-of-frame callbacks due for `upcoming_frame` (kept ordered
    /// by frame).
    fn run_start_of_frame_callbacks(&mut self, upcoming_frame: u64) {
        let mut i = 0;
        while i < self.start_of_frame_callbacks.len() {
            if self.start_of_frame_callbacks[i].0 <= upcoming_frame {
                let (_, callback) = self.start_of_frame_callbacks.remove(i);
                callback();
            } else {
                i += 1;
            }
        }
    }

    /// Run the end-of-frame callbacks whose frame the watcher thread reported
    /// complete on the GPU (deferred deallocation of per-frame resources).
    fn run_due_end_of_frame_callbacks(&mut self) {
        let completed = self.completed_frame.load(Ordering::Acquire);
        let mut i = 0;
        while i < self.end_of_frame_callbacks.len() {
            if self.end_of_frame_callbacks[i].0 <= completed {
                let (_, callback) = self.end_of_frame_callbacks.remove(i);
                callback();
            } else {
                i += 1;
            }
        }
    }

    //TODO this needs to be changes to a subscription based approach where all the necessary methods to recreate the necessary image dependant data are rebuilt
    pub fn resize(&mut self, image_extent: (u32, u32)) -> SrResult<()> {
        self.resize_internal_images(image_extent)?;
        self.resize_swapchain(image_extent)?;

        for callback in self.resize_callbacks.iter_mut() {
            callback();
        }

        Ok(())
    }

    fn resize_internal_images(&mut self, image_extent: (u32, u32)) -> SrResult<()> {
        let new_extent = utils::tuple_to_extent3d(image_extent);
        if new_extent == self.image_extent {
            return Ok(());
        }
        // Drop the in-flight references to images before we destroy them — without
        // this, a resize that arrives while the previous frame's fence hasn't signaled
        // tears down images/descriptor-heap slots the GPU is still reading.
        unsafe { self.core.device().inner().device_wait_idle() }?;
        self.clear_image_dependent_data();

        let num_pixels = (new_extent.width * new_extent.height) as usize;
        self.reservoir_buffers = Self::create_reservoir_buffers(&self.core, num_pixels)?;
        self.reservoir_gi_buffers = Self::create_reservoir_gi_buffers(&self.core, num_pixels)?;

        self.image_extent = new_extent;

        let create_accum_image = |name: &'static str| -> SrResult<Arc<vulkan_abstraction::Image>> {
            Ok(Arc::new(vulkan_abstraction::Image::new(
                self.core.clone(),
                new_extent,
                vk::Format::B10G11R11_UFLOAT_PACK32,
                vk::ImageTiling::OPTIMAL,
                gpu_allocator::MemoryLocation::GpuOnly,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                name,
            )?))
        };

        self.accumulation_images = [create_accum_image("Accumulation_1")?, create_accum_image("Accumulation_2")?];

        self.denoising_images = [create_accum_image("Denoising_1")?, create_accum_image("Denoising_2")?];

        let device = self.core.device().inner();
        let mut setup_cmd_buf = vulkan_abstraction::CmdBuffer::new(self.core.clone())?;

        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(setup_cmd_buf.inner(), &begin_info)?;

            let create_barrier = |image: vk::Image| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::empty())
                    .dst_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
            };

            let barriers = [
                create_barrier(self.accumulation_images[0].inner()),
                create_barrier(self.accumulation_images[1].inner()),
                create_barrier(self.denoising_images[0].inner()),
                create_barrier(self.denoising_images[1].inner()),
            ];

            let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
            device.cmd_pipeline_barrier2(setup_cmd_buf.inner(), &dep_info);

            device.end_command_buffer(setup_cmd_buf.inner())?;

            let fence = setup_cmd_buf.fence_mut().submit()?;
            self.core
                .graphics_queue()
                .submit_async(setup_cmd_buf.inner(), &[], &[], &[], fence)?;
            setup_cmd_buf.fence_mut().wait()?;
        }

        self.relative_frame_count = 0;

        Ok(())
    }

    /// Rebuild the internal swapchain (and everything tied to its images) when
    /// the surface extent changed. No-op without a surface.
    fn resize_swapchain(&mut self, window_extent: (u32, u32)) -> SrResult<()> {
        let Some(sc) = self.swapchain_data.as_mut() else {
            return Ok(());
        };

        self.core
            .device()
            .update_surface_support_details(sc.surface.inner(), sc.surface.instance());
        let new_extent = Swapchain::get_extent(window_extent, &self.core.device().surface_support_details());
        if sc.swapchain.extent() == new_extent {
            return Ok(());
        }

        unsafe { self.core.device().inner().device_wait_idle() }?;

        let surface_khr = sc.surface.inner();
        sc.swapchain.rebuild(surface_khr, window_extent)?;
        let (present_barrier_cmd_bufs, ready_to_present_sems) = SwapchainData::build_per_image_objects(&self.core, &sc.swapchain)?;
        sc.present_barrier_cmd_bufs = present_barrier_cmd_bufs;
        sc.ready_to_present_sems = ready_to_present_sems;
        let images = sc.swapchain.images().to_vec();

        // The blit command buffers (and their fences, which the in-flight slots
        // hold) reference the old images.
        self.clear_image_dependent_data();
        self.build_image_dependent_data(&images)?;

        Ok(())
    }

    pub fn clear_image_dependent_data(&mut self) {
        // No fence bookkeeping needed: the in-flight slots hold frame-timeline
        // *values*, which stay valid forever (unlike the destroyed blit fences
        // this used to have to null out).
        self.image_dependant_data.clear();
    }

    pub fn build_image_dependent_data(&mut self, images: &[vk::Image]) -> SrResult<()> {
        for post_blit_image in images {
            // The post-process result is the only intermediate image the renderer
            // still owns. It must persist (the pre-recorded blit captures its
            // handle) and it's consumed by the external blit, which runs outside
            // the render graph. Every other intermediate (RT raw color, depth,
            // normal, diffuse, motion vectors, denoise ping-pong) is now a
            // graph-internal (transient) resource.
            let postprocess_result_image = Arc::new(vulkan_abstraction::Image::new(
                Rc::clone(&self.core),
                self.image_extent,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageTiling::OPTIMAL,
                gpu_allocator::MemoryLocation::GpuOnly,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
                "sunray (internal, pre-blit) postprocess result image",
            )?);

            // Discard-init the post-process image to GENERAL. The graph's
            // postprocess pass writes it through a storage descriptor (GENERAL),
            // but it's an *imported* resource, so the graph's own
            // created-resource init transition doesn't cover it.
            {
                let device = self.core.device().inner();
                let mut setup_cmd_buf = vulkan_abstraction::CmdBuffer::new(Rc::clone(&self.core))?;
                unsafe {
                    let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                    device.begin_command_buffer(setup_cmd_buf.inner(), &begin_info)?;
                    let barrier = vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .src_access_mask(vk::AccessFlags2::empty())
                        .dst_access_mask(vk::AccessFlags2::SHADER_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(postprocess_result_image.inner())
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        });
                    let dep_info = vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
                    device.cmd_pipeline_barrier2(setup_cmd_buf.inner(), &dep_info);
                    device.end_command_buffer(setup_cmd_buf.inner())?;
                    let fence = setup_cmd_buf.fence_mut().submit()?;
                    self.core
                        .graphics_queue()
                        .submit_async(setup_cmd_buf.inner(), &[], &[], &[], fence)?;
                    setup_cmd_buf.fence_mut().wait()?;
                }
            }

            let blit_cmd_buf = vulkan_abstraction::CmdBuffer::new(Rc::clone(&self.core))?;

            //record blit
            {
                let cmd_buf_begin_info =
                    vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);
                unsafe {
                    self.core
                        .device()
                        .inner()
                        .begin_command_buffer(blit_cmd_buf.inner(), &cmd_buf_begin_info)
                }?;

                Self::cmd_blit_image(
                    &self.core,
                    blit_cmd_buf.inner(),
                    postprocess_result_image.inner(),
                    postprocess_result_image.extent(),
                    *post_blit_image,
                    postprocess_result_image.image_subresource_range(),
                )?;

                unsafe { self.core.device().inner().end_command_buffer(blit_cmd_buf.inner()) }?;
            }

            self.image_dependant_data.insert(
                *post_blit_image,
                ImageDependentData {
                    blit_cmd_buf,
                    postprocess_result_image,
                },
            );
        }

        Ok(())
    }

    /// Load a glTF file's default scene. See [`Self::load_scene`] for the
    /// return contract.
    pub fn load_gltf(&mut self, path: &str) -> SrResult<(u64, Vec<(K, Vec<vk::TransformMatrixKHR>)>)>
    where
        K: From<ResourceKey>,
    {
        let gltf = vulkan_abstraction::gltf::Gltf::new(Rc::clone(&self.core), path)?;
        let (default_scene, scene_data) = gltf.create_default_scene()?;
        self.load_scene(&default_scene, scene_data)
    }

    /// Load a scene's assets into the resource manager. Returns the asset
    /// group index (usable with [`Self::unload_scene`] to free everything this
    /// call created in bulk) and the scene's instances as the
    /// `(blas key, world transforms)` vector. The instance list is *not*
    /// retained anywhere — the caller owns it, mutates it, and passes it to
    /// [`Self::render`] / [`Self::render_to_swapchain`] every frame.
    pub fn load_scene(&mut self, scene: &Scene, scene_data: SceneData) -> SrResult<(u64, Vec<(K, Vec<vk::TransformMatrixKHR>)>)>
    where
        K: From<ResourceKey>,
    {
        // Wait for all in-flight GPU work before invalidating descriptor sets that reference
        // buffers which will be reallocated (e.g. emissive_indirection_gpu).
        unsafe { self.core.device().inner().device_wait_idle() }?;

        let group = self.next_group;
        self.next_group += 1;

        let LoadedScene {
            blases,
            instances,
            textures,
            sampler_descs,
            images,
        } = scene.load_into_gpu(&self.core, scene_data)?;

        let mut group_keys: Vec<K> = Vec::new();
        let mut next_index = 0u64;
        let mut make_key = || {
            let key = K::from(ResourceKey {
                group,
                index: next_index,
            });
            next_index += 1;
            group_keys.push(key);
            key
        };

        let blas_keys = self
            .resource_manager
            .add_scene_assets(blases, textures, sampler_descs, images, &mut make_key)?;
        drop(make_key);
        self.scene_groups.insert(group, group_keys);

        // Group the per-instance transforms by BLAS key, preserving order.
        let mut grouped: Vec<(K, Vec<vk::TransformMatrixKHR>)> = blas_keys.iter().map(|&k| (k, Vec::new())).collect();
        for (blas_index, transform) in instances {
            grouped[blas_index].1.push(transform);
        }

        self.clear_image_dependent_data();
        if let Some(sc) = &self.swapchain_data {
            let images = sc.swapchain.images().to_vec();
            self.build_image_dependent_data(&images)?;
        }

        Ok((group, grouped))
    }

    /// Free every asset created by the `load_scene` call that returned `group`.
    /// Allows loading a scene repeatedly without leaking GPU memory. Instances
    /// referencing the freed keys must no longer be passed to `render`.
    pub fn unload_scene(&mut self, group: u64) -> SrResult<()> {
        unsafe { self.core.device().inner().device_wait_idle() }?;
        if let Some(keys) = self.scene_groups.remove(&group) {
            for key in &keys {
                self.resource_manager.remove(key);
            }
        }
        Ok(())
    }

    /// Render to dst_image. All per-frame inputs are parameters: the camera and
    /// the instance list (`(blas key, world transforms of its instances)` —
    /// keys come from scene loading). The user may also pass a Semaphore which the user should signal when the image is
    /// ready to be written to (for example after being acquired from a swapchain).
    ///
    /// Returns the frame's **absolute frame number**: the frame timeline
    /// semaphore is signaled with it when the frame's GPU work (including the
    /// final blit) completes, so "wait for this render to end" is
    /// `wait_frame(value)` — there is no per-frame fence to track.
    pub fn render(
        &mut self,
        dst_image: vk::Image,
        wait_sem: vk::Semaphore,
        camera: &Camera,
        instances: &[(K, Vec<vk::TransformMatrixKHR>)],
    ) -> SrResult<u64> {
        // ── Start of frame: scheduled callbacks + deferred deallocation of the
        // per-frame resources of frames the timeline reported complete.
        let upcoming_frame = *self.core.absolute_frame_count.borrow() as u64 + 1;
        self.run_start_of_frame_callbacks(upcoming_frame);
        self.run_due_end_of_frame_callbacks();

        // Waiting for device idle serializes the per-frame work (TLAS rebuild,
        // arena copy flush) against in-flight GPU work. Minimum viable fix —
        // the frame-local buffers below don't need it (each frame gets fresh
        // ones), but the TLAS and arena buffers are still shared across frames.
        unsafe {
            self.core.device().inner().device_wait_idle()?;
        }

        self.resource_manager.start_of_frame(upcoming_frame)?;

        // ── Per-frame GPU data: CpuToGpu buffers created on the spot, local to
        // this frame. They're moved into an end-of-frame callback at the end of
        // this function and freed once the frame timeline passes this frame.
        let mut matrices = camera.as_matrices(self.image_extent);
        // Inject the history matrix saved from the last frame; save the current
        // one to use as history NEXT frame.
        matrices.prev_view_proj = self.prev_view_proj;
        self.prev_view_proj = matrices.view_proj;

        // nalgebra's Matrix4 is column-major in memory. HLSL/Slang's
        // `float4x4(v0, v1, v2, v3)` constructor reads each float4 as a ROW.
        // Transposing here means each on-disk float4 (which the shader reads as
        // a member of `Matrices`) is a ROW of the intended matrix, so the
        // shader's `float4x4(m.vi0, m.vi1, m.vi2, m.vi3)` reconstructs the
        // matrix correctly without any per-shader `transpose()` call.
        let mut matrices_buffer =
            vulkan_abstraction::UniformBuffer::<CameraMatrices>::new(Rc::clone(&self.core), 1 as vk::DeviceSize)?;
        // Destructure-copy first: `CameraMatrices` is `repr(C, packed)`, so
        // taking references to its fields (which a method call would) is UB.
        let CameraMatrices {
            view_inverse,
            proj_inverse,
            view_proj,
            prev_view_proj,
        } = matrices;
        matrices_buffer.map_mut()?[0] = CameraMatrices {
            view_inverse: view_inverse.transpose(),
            proj_inverse: proj_inverse.transpose(),
            view_proj: view_proj.transpose(),
            prev_view_proj: prev_view_proj.transpose(),
        };

        let frame_data = self.resource_manager.frame_instance_data(instances)?;
        let instance_count = frame_data.as_instances.len() as u32;

        // Empty slices would produce null buffers (and so invalid heap
        // descriptors / build inputs); pad each with one dummy element. The
        // TLAS build reads 0 instances from the dummy and the shader sees the
        // dummy emissive entry only through the (matching, pre-rework) "no
        // lights" behavior.
        let mut as_instances = frame_data.as_instances;
        if as_instances.is_empty() {
            as_instances.push(vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR {
                    matrix: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                },
                instance_custom_index_and_mask: vk::Packed24_8::new(0, 0),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, 0),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR { device_handle: 0 },
            });
        }
        let mut transforms = frame_data.transforms;
        if transforms.is_empty() {
            transforms.push(vk::TransformMatrixKHR {
                matrix: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            });
        }
        let mut emissive_entries = frame_data.emissive_entries;
        if emissive_entries.is_empty() {
            emissive_entries.push(vulkan_abstraction::gltf::EmissiveIndirectionEntry {
                blas_tri_index: 0,
                entity_id: 0,
            });
        }

        let instances_buffer = vulkan_abstraction::StagingBuffer::new_from_data(
            Rc::clone(&self.core),
            &as_instances,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            "per-frame TLAS instances",
        )?;
        let transforms_buffer = vulkan_abstraction::StagingBuffer::new_from_data(
            Rc::clone(&self.core),
            &transforms,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "per-frame instance transforms",
        )?;
        // Exactly sized: the shader reads num_lights via `GetDimensions`.
        let emissive_indirection_buffer = vulkan_abstraction::StagingBuffer::new_from_data(
            Rc::clone(&self.core),
            &emissive_entries,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "per-frame emissive indirection",
        )?;

        self.resource_manager.rebuild_tlas(instance_count, &instances_buffer)?;

        let frame_gpu_data = FrameGpuData {
            matrices_address: matrices_buffer.get_device_address(),
            entity_transforms_slot: transforms_buffer.raw().storage_slot(),
            emissive_indirection_slot: emissive_indirection_buffer.raw().storage_slot(),
        };

        if !self.image_dependant_data.contains_key(&dst_image) {
            self.build_image_dependent_data(&[dst_image])?;
        }

        // Wait the render graph's previous frame BEFORE we touch graph state:
        // build_unified_graph -> compile() resets and re-records the persistent
        // cmd buffer, which is UB while the GPU may still be reading it.
        self.render_graph_fence.wait()?;

        let result_extent = self.image_extent;
        let postprocess_out = {
            let idd = self.image_dependant_data.get_mut(&dst_image).unwrap();
            idd.blit_cmd_buf.fence_mut().wait()?;
            Arc::clone(&idd.postprocess_result_image)
        };

        // Build + compile the unified render graph: RT (RIS + final), temporal
        // accumulation, the 8 a-trous denoise passes, and postprocess. Every pass
        // is heap + Slang; the intermediate G-buffer / RT-output images are
        // graph-internal (transient) resources.
        self.build_unified_graph(&postprocess_out, result_extent, &frame_gpu_data)?;

        // build_unified_graph advanced the absolute frame count: that's this
        // frame's number on the frame timeline.
        let frame_value = *self.core.absolute_frame_count.borrow() as u64;
        debug_assert_eq!(frame_value, upcoming_frame);

        // The graph submission must wait on any pending transfer work (e.g. the
        // TLAS build) before the RT pass runs.
        let wait_semaphores = self.core.transfer_semaphores_mut().drain(..).collect::<Vec<_>>();
        let wait_stages = wait_semaphores
            .iter()
            .map(|_| vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR | vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR)
            .collect::<Vec<_>>();

        unsafe {
            self.core.device().inner().device_wait_idle()?;
        }
        self.render_graph
            .run(&mut self.render_graph_fence, &wait_semaphores, &wait_stages)?;
        self.render_graph_fence.wait()?;

        // === Blit postprocess result -> caller's target (outside the graph) ===
        // The blit submission signals the frame timeline with this frame's
        // absolute number — that's the frame's "render end" signal.
        let single_stage = [vk::PipelineStageFlags::ALL_GRAPHICS];
        let (wait_sems, wait_dst_stages): (&[vk::Semaphore], &[vk::PipelineStageFlags]) = if wait_sem == vk::Semaphore::null() {
            (&[], &[])
        } else {
            (std::slice::from_ref(&wait_sem), &single_stage)
        };

        let idd = self.image_dependant_data.get_mut(&dst_image).unwrap();
        let blit_fence = idd.blit_cmd_buf.fence_mut().submit()?;
        let blit_cmd = idd.blit_cmd_buf.inner();
        self.core.graphics_queue().submit_async_with_timeline(
            blit_cmd,
            wait_sems,
            wait_dst_stages,
            (self.frame_timeline.inner(), frame_value),
            blit_fence,
        )?;

        // ── End of frame: the frame-local buffers stay alive until the GPU is
        // done with this frame, then get dropped on the render thread.
        self.end_of_frame_callbacks.push((
            frame_value,
            Box::new(move || {
                drop(matrices_buffer);
                drop(instances_buffer);
                drop(transforms_buffer);
                drop(emissive_indirection_buffer);
            }),
        ));

        Ok(frame_value)
    }

    /// Block until frame `frame_value` (as returned by [`Self::render`]) has
    /// completed on the GPU.
    pub fn wait_frame(&self, frame_value: u64) -> SrResult<()> {
        self.frame_timeline.wait(frame_value)
    }

    /// Read-only access to the internal swapchain (present only when the
    /// renderer was built with a surface). Useful for overlay passes that need
    /// the swapchain format / image count.
    pub fn swapchain(&self) -> Option<&Swapchain> {
        self.swapchain_data.as_ref().map(|sc| &sc.swapchain)
    }

    /// Render one frame to the internal swapchain and present it: waits the
    /// in-flight fence, acquires an image, calls [`Self::render`], transitions
    /// the image to `PRESENT_SRC` with the pre-recorded barrier, and presents.
    /// All per-frame inputs (camera + instances) come from the caller.
    pub fn render_to_swapchain(
        &mut self,
        camera: &Camera,
        instances: &[(K, Vec<vk::TransformMatrixKHR>)],
    ) -> SrResult<()> {
        self.render_to_swapchain_with(camera, instances, None)
    }

    /// Like [`Self::render_to_swapchain`], but lets the caller replace the
    /// final GENERAL -> PRESENT_SRC transition with its own pass (e.g. an egui
    /// overlay drawn straight onto the swapchain image). The `finalize` closure
    /// must leave the image in `PRESENT_SRC_KHR` and signal
    /// [`SwapchainFrame::ready_to_present_sem`]; the renderer presents after.
    pub fn render_to_swapchain_with(
        &mut self,
        camera: &Camera,
        instances: &[(K, Vec<vk::TransformMatrixKHR>)],
        finalize: Option<&mut dyn FnMut(&SwapchainFrame) -> SrResult<()>>,
    ) -> SrResult<()> {
        let (frame_index, img_acquired_sem, img_rendered_frame) = {
            let sc = self
                .swapchain_data
                .as_mut()
                .ok_or_else(|| SrError::new_custom("render_to_swapchain: renderer was built without a surface".to_string()))?;
            let frame_index = (sc.frame_count as usize) % MAX_FRAMES_IN_FLIGHT;
            sc.frame_count += 1;
            (
                frame_index,
                sc.img_acquired_sems[frame_index].inner(),
                sc.img_rendered_frames[frame_index],
            )
        };

        // Wait (on the frame timeline) for the frame that used this in-flight
        // slot MAX_FRAMES_IN_FLIGHT frames ago before reusing its semaphore.
        self.frame_timeline.wait(img_rendered_frame)?;

        let frame = {
            let sc = self.swapchain_data.as_ref().unwrap();
            let (image_index, suboptimal) = unsafe {
                sc.swapchain
                    .device()
                    .acquire_next_image(sc.swapchain.inner(), u64::MAX, img_acquired_sem, vk::Fence::null())
            }?;
            if suboptimal {
                log::warn!("VkAcquireNextImageKHR: swapchain is suboptimal for the surface");
            }
            let image_index = image_index as usize;
            SwapchainFrame {
                image: sc.swapchain.images()[image_index],
                image_view: sc.swapchain.image_views()[image_index],
                extent: sc.swapchain.extent(),
                image_index,
                ready_to_present_sem: sc.ready_to_present_sems[image_index].inner(),
            }
        };

        let rendered_frame = self.render(frame.image, img_acquired_sem, camera, instances)?;
        self.swapchain_data.as_mut().unwrap().img_rendered_frames[frame_index] = rendered_frame;

        match finalize {
            Some(finalize) => finalize(&frame)?,
            None => {
                let sc = self.swapchain_data.as_mut().unwrap();
                let barrier_cmd_buf = &mut sc.present_barrier_cmd_bufs[frame.image_index];
                let barrier_fence = barrier_cmd_buf.fence_mut().submit()?;
                let barrier_cmd = barrier_cmd_buf.inner();
                self.core.graphics_queue().submit_async(
                    barrier_cmd,
                    &[],
                    &[],
                    std::slice::from_ref(&frame.ready_to_present_sem),
                    barrier_fence,
                )?;
            }
        }

        // Present, waiting on the PRESENT_SRC transition.
        let sc = self.swapchain_data.as_ref().unwrap();
        let swapchains = [sc.swapchain.inner()];
        let image_indices = [frame.image_index as u32];
        let wait_semaphores = [frame.ready_to_present_sem];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        let queue = self.core.graphics_queue().inner();
        unsafe { sc.swapchain.device().queue_present(queue, &present_info) }?;

        Ok(())
    }

    /// Build + compile the unified render graph for this frame: ray tracing
    /// (RIS + final in one node), temporal accumulation, the 8 a-trous denoise
    /// passes, and postprocess. Every pass is heap-mode + Slang. The G-buffer /
    /// RT-output images are created as graph-internal (transient) resources; the
    /// cross-frame accumulation ping-pong, the denoise ping-pong, and the
    /// post-process output are imported.
    //TODO finni
    fn build_unified_graph(
        &mut self,
        postprocess_out: &Arc<vulkan_abstraction::Image>,
        extent: vk::Extent3D,
        frame_gpu_data: &FrameGpuData,
    ) -> SrResult<()> {
        let frame_count = self.relative_frame_count;
        let width = extent.width;
        let height = extent.height;
        // Ping-pong: TAA writes accum[accum_idx] (which denoise then reads) and
        // reads accum[history_idx] (last frame's output).
        let accum_idx = (frame_count % 2) as usize;
        let history_idx = ((frame_count + 1) % 2) as usize;

        let pack = |i: u32| -> [u32; 2] { [i, 0] };

        // Non-image fields of the RT push constant: stable slots come from the
        // resource manager, the per-frame ones (matrices address, transforms /
        // emissive indirection slots) from this frame's local buffers. The five
        // RT-output image slots are filled inside the closure from the graph's
        // transient resources (they're created per frame).
        let rt_pc_base = vulkan_abstraction::RaytracingHeapPushConstant {
            tlas: self.resource_manager.tlas().device_address(),
            matrices: frame_gpu_data.matrices_address,
            meshes_info: pack(self.resource_manager.meshes_info_storage_slot()),
            emissive_triangles: pack(self.resource_manager.emissive_triangles_storage_slot()),
            emissive_indirection: pack(frame_gpu_data.emissive_indirection_slot),
            entity_transforms: pack(frame_gpu_data.entity_transforms_slot),
            blue_noise_tex: pack(self.blue_noise_image.sampled_slot()),
            blue_noise_sampler: pack(self.blue_noise_sampler.slot()),
            reservoirs: [
                self.reservoir_buffers[0].device_address(),
                self.reservoir_buffers[1].device_address(),
            ],
            reservoirs_gi: [
                self.reservoir_gi_buffers[0].device_address(),
                self.reservoir_gi_buffers[1].device_address(),
            ],
            frame_count,
            use_srgb: if self.image_format == vk::Format::R8G8B8A8_SRGB {
                1
            } else {
                0
            },
            ..Default::default()
        };

        // RT passes now describe themselves with their per-stage SPIR-V; the
        // graph's pipeline cache builds/reuses the pipeline + SBT (RIS and final
        // share miss/closest-hit/any-hit and differ only in the ray-gen blob, so
        // they intern as two distinct entries). The shader list is ordered
        // [ray_gen, miss, closest_hit, any_hit] and each stage's entry point is
        // selected by its index — see `RayTracingShaders`.
        let make_rt_shaders = |ray_gen: &'static [u8]| -> RayTracingShaders {
            RayTracingShaders::new(
                vec![
                    ShaderSource::Spirv(ray_gen.to_vec()),
                    ShaderSource::Spirv(self.ray_miss_spirv.to_vec()),
                    ShaderSource::Spirv(self.closest_hit_spirv.to_vec()),
                    ShaderSource::Spirv(self.any_hit_spirv.to_vec()),
                ],
                (0, "main"),
                (1, "main"),
                (2, "main"),
                (3, "main"),
            )
        };
        let ris_shaders = make_rt_shaders(self.ray_gen_ris_spirv);
        let final_shaders = make_rt_shaders(self.ray_gen_final_spirv);

        // Compute passes now describe themselves with their SPIR-V; the graph's
        // pipeline cache builds/reuses the pipeline. Snapshot the bytes into
        // locals so the `&mut self.render_graph` borrow below stays disjoint.
        let taa_spirv = self.temporal_accumulation_spirv.clone();
        let denoise_spirv = self.denoise_spirv.clone();
        let postprocess_spirv = self.postprocess_spirv.clone();

        let accum_arc0 = Arc::clone(&self.accumulation_images[0]);
        let accum_arc1 = Arc::clone(&self.accumulation_images[1]);
        let denoise_arc0 = Arc::clone(&self.denoising_images[0]);
        let denoise_arc1 = Arc::clone(&self.denoising_images[1]);
        let postprocess_out_arc = Arc::clone(postprocess_out);

        // Ping-pong reservoir buffers, cloned so they can be imported into the
        // graph for hazard tracking (RIS writes → final reads). The shader still
        // reaches them by device-address (already baked into `rt_pc_base`); the
        // import only governs the RIS→final hand-off barrier.
        let reservoir_arc0 = Arc::clone(&self.reservoir_buffers[0]);
        let reservoir_arc1 = Arc::clone(&self.reservoir_buffers[1]);
        let reservoir_gi_arc0 = Arc::clone(&self.reservoir_gi_buffers[0]);
        let reservoir_gi_arc1 = Arc::clone(&self.reservoir_gi_buffers[1]);

        // Advance the frame counters for the next frame (after snapshotting
        // `frame_count` for this one).
        self.relative_frame_count += 1;
        *self.core.absolute_frame_count.borrow_mut() += 1;

        let rg = &mut self.render_graph;
        rg.reset();

        let mk_img = |format: vk::Format, usage: vk::ImageUsageFlags, name: &'static str| ImageDesc {
            extent,
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            usage,
            name,
        };

        // Internal (transient) RT outputs.
        let raw_color_h = rg.create_resource(mk_img(
            vk::Format::B10G11R11_UFLOAT_PACK32,
            vk::ImageUsageFlags::STORAGE,
            "rg_rt_raw_color",
        ));
        let depth_h = rg.create_resource(mk_img(
            vk::Format::R16_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            "rg_depth",
        ));
        let normal_h = rg.create_resource(mk_img(
            vk::Format::R8G8B8A8_SNORM,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            "rg_normal",
        ));
        let diffuse_h = rg.create_resource(mk_img(
            vk::Format::B10G11R11_UFLOAT_PACK32,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            "rg_diffuse",
        ));
        let motion_h = rg.create_resource(mk_img(
            vk::Format::R16G16_SFLOAT,
            vk::ImageUsageFlags::STORAGE,
            "rg_motion_vec",
        ));

        // Imported (persistent) resources.
        let accum0_h = rg.import::<ImageDesc>(accum_arc0);
        let accum1_h = rg.import::<ImageDesc>(accum_arc1);
        let denoise_a_h = rg.import::<ImageDesc>(denoise_arc0);
        let denoise_b_h = rg.import::<ImageDesc>(denoise_arc1);
        let postprocess_out_h = rg.import::<ImageDesc>(postprocess_out_arc);

        // Reservoir ping-pong buffers, imported so the graph tracks the RIS→final
        // hand-off and emits the barrier between the two RT passes itself.
        let reservoir0_h = rg.import::<BufferDesc>(reservoir_arc0);
        let reservoir1_h = rg.import::<BufferDesc>(reservoir_arc1);
        let reservoir_gi0_h = rg.import::<BufferDesc>(reservoir_gi_arc0);
        let reservoir_gi1_h = rg.import::<BufferDesc>(reservoir_gi_arc1);
        let reservoir_handles = [reservoir0_h, reservoir1_h, reservoir_gi0_h, reservoir_gi1_h];

        let accum_target_h = if accum_idx == 0 { accum0_h.clone() } else { accum1_h.clone() };
        let accum_history_h = if history_idx == 0 {
            accum0_h.clone()
        } else {
            accum1_h.clone()
        };

        // 1. Ray tracing as two heap-mode passes built through the standard
        // `RaytracingRenderPassBuilder::generate_render` path: RIS audition then
        // final shading, each interning its own pipeline + SBT in the graph cache.
        // They're ordered by the shared G-buffer write-after-write hazard; the
        // reservoir hand-off (RIS writes, final reads) is now a real graph edge on
        // the imported reservoir buffers — no manual barrier.
        Self::add_raytracing_ris_pass(
            rg,
            ris_shaders,
            rt_pc_base,
            raw_color_h.clone(),
            depth_h.clone(),
            normal_h.clone(),
            diffuse_h.clone(),
            motion_h.clone(),
            reservoir_handles.clone(),
            extent,
        )?;
        Self::add_raytracing_final_pass(
            rg,
            final_shaders,
            rt_pc_base,
            raw_color_h.clone(),
            depth_h.clone(),
            normal_h.clone(),
            diffuse_h.clone(),
            motion_h.clone(),
            reservoir_handles,
            extent,
        )?;

        // 2. Temporal accumulation.
        Self::add_temporal_pass(
            rg,
            &taa_spirv,
            raw_color_h.clone(),
            motion_h.clone(),
            accum_history_h,
            accum_target_h.clone(),
            frame_count,
            width,
            height,
        )?;

        // 3. Denoise (8 a-trous passes). Pass 0 reads the TAA output (accum_target).
        Self::add_denoise_passes(
            rg,
            &denoise_spirv,
            accum_target_h,
            depth_h,
            normal_h,
            diffuse_h,
            denoise_a_h.clone(),
            denoise_b_h.clone(),
            frame_count,
            width,
            height,
        )?;

        // 4. Postprocess: read the final denoise output, tonemap into the output.
        let final_idx = ((DENOISE_PASSES - 1) % 2) as usize;
        let denoise_input_h = if final_idx == 0 { denoise_a_h } else { denoise_b_h };
        Self::add_postprocess_pass(
            rg,
            &postprocess_spirv,
            denoise_input_h,
            postprocess_out_h,
            width,
            height,
            EXPOSURE,
        )?;

        rg.compile()?;
        Ok(())
    }

    /// Builds the heap-mode raytracing push constant for a frame: the non-image
    /// fields come from `pc_base`, the five G-buffer / RT-output image slots are
    /// resolved from the graph's transient resources, then the whole struct is
    /// serialized to the raw bytes `generate_render` pushes via `cmd_push_data`.
    /// Shared by both RT passes so they push identical bindings.
    fn rt_push_constant_bytes(
        pc_base: &vulkan_abstraction::RaytracingHeapPushConstant,
        tr: &render_graph::graph::TransientResources,
        raw_color_h: &Handle<vulkan_abstraction::Image>,
        depth_h: &Handle<vulkan_abstraction::Image>,
        normal_h: &Handle<vulkan_abstraction::Image>,
        diffuse_h: &Handle<vulkan_abstraction::Image>,
        motion_h: &Handle<vulkan_abstraction::Image>,
    ) -> SrResult<Vec<u8>> {
        let pack = |i: u32| -> [u32; 2] { [i, 0] };
        let mut pc = *pc_base;
        pc.raw_color = pack(tr.image(raw_color_h)?.storage_slot());
        pc.depth_img = pack(tr.image(depth_h)?.storage_slot());
        pc.normal_img = pack(tr.image(normal_h)?.storage_slot());
        pc.diffuse_img = pack(tr.image(diffuse_h)?.storage_slot());
        pc.motion_vec_img = pack(tr.image(motion_h)?.storage_slot());
        // `RaytracingHeapPushConstant` is `#[repr(C)]` plain data, so a verbatim
        // byte copy matches the shader's push-constant layout.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &pc as *const vulkan_abstraction::RaytracingHeapPushConstant as *const u8,
                size_of::<vulkan_abstraction::RaytracingHeapPushConstant>(),
            )
        }
        .to_vec();
        Ok(bytes)
    }

    /// RIS audition ray-tracing pass, built through the standard
    /// `RaytracingRenderPassBuilder::generate_render` path (pipeline + SBT interned
    /// in the graph's cache). Writes the G-buffer / RT-output images and the ReSTIR
    /// reservoirs; the reservoir writes are declared on the imported reservoir
    /// buffers so the graph emits the RIS→final hand-off barrier itself. Ordering
    /// against the final pass also comes from the shared G-buffer write-after-write
    /// hazard.
    #[allow(clippy::too_many_arguments)]
    fn add_raytracing_ris_pass(
        rg: &mut RenderGraph,
        shaders: RayTracingShaders,
        pc_base: vulkan_abstraction::RaytracingHeapPushConstant,
        raw_color_h: Handle<vulkan_abstraction::Image>,
        depth_h: Handle<vulkan_abstraction::Image>,
        normal_h: Handle<vulkan_abstraction::Image>,
        diffuse_h: Handle<vulkan_abstraction::Image>,
        motion_h: Handle<vulkan_abstraction::Image>,
        reservoir_handles: [Handle<vulkan_abstraction::RawBuffer>; 4],
        extent: vk::Extent3D,
    ) -> SrResult<()> {
        let mut common = PassCommonDataBuilder::new(rg, "raytracing_ris");
        // Writes all five outputs via storage descriptors (GENERAL). `General` is
        // used because vk_sync has no ray-tracing-specific write access type.
        common.write(&raw_color_h, vk_sync::AccessType::General)?;
        common.write(&depth_h, vk_sync::AccessType::General)?;
        common.write(&normal_h, vk_sync::AccessType::General)?;
        common.write(&diffuse_h, vk_sync::AccessType::General)?;
        common.write(&motion_h, vk_sync::AccessType::General)?;
        // Declare the reservoir SSBO writes so the graph orders the final pass's
        // reservoir reads after this pass (the RIS→final hand-off barrier).
        for h in &reservoir_handles {
            common.write(h, vk_sync::AccessType::AnyShaderWrite)?;
        }

        let pass = RaytracingRenderPassBuilder::default()
            .common(common.build())
            .shaders(shaders)
            .trace_extent([extent.width, extent.height, extent.depth])
            .generate_render(rg, move |tr| {
                Self::rt_push_constant_bytes(&pc_base, tr, &raw_color_h, &depth_h, &normal_h, &diffuse_h, &motion_h)
            })?
            .build()
            .map_err(|e| SrError::new_custom(format!("raytracing RIS pass builder failed: {e}")))?;
        rg.add_render_pass(AnyRenderPass::Rt(pass));
        Ok(())
    }

    /// Final shading ray-tracing pass, built through the standard
    /// `RaytracingRenderPassBuilder::generate_render` path (pipeline + SBT interned
    /// in the graph's cache). Reads the reservoirs the RIS pass produced
    /// (visibility established by the graph-emitted reservoir barrier) and writes
    /// the final color into the G-buffer outputs. Ordered after the RIS pass by the
    /// shared G-buffer write-after-write hazard the graph tracks.
    #[allow(clippy::too_many_arguments)]
    fn add_raytracing_final_pass(
        rg: &mut RenderGraph,
        shaders: RayTracingShaders,
        pc_base: vulkan_abstraction::RaytracingHeapPushConstant,
        raw_color_h: Handle<vulkan_abstraction::Image>,
        depth_h: Handle<vulkan_abstraction::Image>,
        normal_h: Handle<vulkan_abstraction::Image>,
        diffuse_h: Handle<vulkan_abstraction::Image>,
        motion_h: Handle<vulkan_abstraction::Image>,
        reservoir_handles: [Handle<vulkan_abstraction::RawBuffer>; 4],
        extent: vk::Extent3D,
    ) -> SrResult<()> {
        let mut common = PassCommonDataBuilder::new(rg, "raytracing_final");
        // Re-declares the same writes as the RIS pass: this is what creates the
        // write-after-write hazard edge that orders this pass after RIS.
        common.write(&raw_color_h, vk_sync::AccessType::General)?;
        common.write(&depth_h, vk_sync::AccessType::General)?;
        common.write(&normal_h, vk_sync::AccessType::General)?;
        common.write(&diffuse_h, vk_sync::AccessType::General)?;
        common.write(&motion_h, vk_sync::AccessType::General)?;
        // Read the reservoirs the RIS pass wrote — this read against the RIS pass's
        // declared writes is the graph edge that becomes the hand-off barrier.
        for h in &reservoir_handles {
            common.read(h, vk_sync::AccessType::RayTracingShaderReadOther)?;
        }

        let pass = RaytracingRenderPassBuilder::default()
            .common(common.build())
            .shaders(shaders)
            .trace_extent([extent.width, extent.height, extent.depth])
            .generate_render(rg, move |tr| {
                Self::rt_push_constant_bytes(&pc_base, tr, &raw_color_h, &depth_h, &normal_h, &diffuse_h, &motion_h)
            })?
            .build()
            .map_err(|e| SrError::new_custom(format!("raytracing final pass builder failed: {e}")))?;
        rg.add_render_pass(AnyRenderPass::Rt(pass));
        Ok(())
    }

    /// Temporal accumulation graph node (heap + Slang). Reads the RT raw color +
    /// motion vectors and the history accumulation image, writes the target
    /// accumulation image.
    #[allow(clippy::too_many_arguments)]
    fn add_temporal_pass(
        rg: &mut RenderGraph,
        spirv: &[u8],
        raw_color_h: Handle<vulkan_abstraction::Image>,
        motion_h: Handle<vulkan_abstraction::Image>,
        history_h: Handle<vulkan_abstraction::Image>,
        accum_target_h: Handle<vulkan_abstraction::Image>,
        frame_count: u32,
        width: u32,
        height: u32,
    ) -> SrResult<()> {
        let mut common = PassCommonDataBuilder::new(rg, "temporal_accumulation");
        common.read(&raw_color_h, vk_sync::AccessType::ComputeShaderReadOther)?;
        common.read(&motion_h, vk_sync::AccessType::ComputeShaderReadOther)?;
        common.read(&history_h, vk_sync::AccessType::ComputeShaderReadOther)?;
        common.write(&accum_target_h, vk_sync::AccessType::ComputeShaderWrite)?;

        // Only the shaders + a push-data closure: `generate_render` interns the
        // pipeline in the graph cache and installs the bind/push/dispatch closure.
        let pass = ComputeRenderPassBuilder::default()
            .common(common.build())
            .shaders(ComputeShaders::new(vec![ShaderSource::Spirv(spirv.to_vec())], 0, "main"))
            .generate_render(rg, [width.div_ceil(16), height.div_ceil(16), 1], move |tr| {
                let pack = |i: u32| -> [u32; 2] { [i, 0] };
                Ok(vulkan_abstraction::TemporalAccumulationHeapPushConstant {
                    raw_rt_color: pack(tr.image(&raw_color_h)?.storage_slot()),
                    motion_vector: pack(tr.image(&motion_h)?.storage_slot()),
                    history: pack(tr.image(&history_h)?.storage_slot()),
                    accum_output: pack(tr.image(&accum_target_h)?.storage_slot()),
                    frame_count,
                    width,
                    height,
                })
            })
            .map_err(|e| SrError::new_custom(format!("temporal accumulation pass builder failed: {e}")))?;
        rg.add_render_pass(AnyRenderPass::Compute(pass));
        Ok(())
    }

    /// The 8 a-trous denoise passes (heap + Slang). depth/normal/diffuse are read
    /// (sampled) only in pass 0 to register the GENERAL->SHADER_READ transition;
    /// later passes read the same stable slots directly without re-registering.
    #[allow(clippy::too_many_arguments)]
    fn add_denoise_passes(
        rg: &mut RenderGraph,
        spirv: &[u8],
        accum_in_h: Handle<vulkan_abstraction::Image>,
        depth_h: Handle<vulkan_abstraction::Image>,
        normal_h: Handle<vulkan_abstraction::Image>,
        diffuse_h: Handle<vulkan_abstraction::Image>,
        denoise_a_h: Handle<vulkan_abstraction::Image>,
        denoise_b_h: Handle<vulkan_abstraction::Image>,
        frame_count: u32,
        width: u32,
        height: u32,
    ) -> SrResult<()> {
        for pass_index in 0..DENOISE_PASSES {
            let step_width = 1i32 << pass_index;
            let (read_h, write_h) = if pass_index == 0 {
                (accum_in_h.clone(), denoise_a_h.clone())
            } else if pass_index % 2 == 1 {
                (denoise_a_h.clone(), denoise_b_h.clone())
            } else {
                (denoise_b_h.clone(), denoise_a_h.clone())
            };

            let mut common = PassCommonDataBuilder::new(rg, format!("denoise_{pass_index}"));
            common.read(&read_h, vk_sync::AccessType::ComputeShaderReadOther)?;
            common.write(&write_h, vk_sync::AccessType::ComputeShaderWrite)?;
            if pass_index == 0 {
                common.read(
                    &depth_h,
                    vk_sync::AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer,
                )?;
                common.read(
                    &normal_h,
                    vk_sync::AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer,
                )?;
                common.read(
                    &diffuse_h,
                    vk_sync::AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer,
                )?;
            }

            let read_h_c = read_h.clone();
            let write_h_c = write_h.clone();
            let depth_c = depth_h.clone();
            let normal_c = normal_h.clone();
            let diffuse_c = diffuse_h.clone();
            // The same SPIR-V is handed to every a-trous pass; the graph's
            // pipeline cache dedups them to one `vk::Pipeline`.
            let pass = ComputeRenderPassBuilder::default()
                .common(common.build())
                .shaders(ComputeShaders::new(vec![ShaderSource::Spirv(spirv.to_vec())], 0, "main"))
                .generate_render(rg, [width.div_ceil(16), height.div_ceil(16), 1], move |tr| {
                    let pack = |i: u32| -> [u32; 2] { [i, 0] };
                    Ok(vulkan_abstraction::DenoiseHeapPushConstant {
                        temporal_result: pack(tr.image(&read_h_c)?.storage_slot()),
                        depth: pack(tr.image(&depth_c)?.sampled_slot()),
                        normal: pack(tr.image(&normal_c)?.sampled_slot()),
                        diffuse: pack(tr.image(&diffuse_c)?.sampled_slot()),
                        spatial_output: pack(tr.image(&write_h_c)?.storage_slot()),
                        frame_count,
                        step_width,
                        width,
                        height,
                    })
                })
                .map_err(|e| SrError::new_custom(format!("denoise pass builder failed: {e}")))?;
            rg.add_render_pass(AnyRenderPass::Compute(pass));
        }
        Ok(())
    }

    /// Postprocess graph node (heap + Slang): tonemap + gamma the final denoise
    /// output into the post-process image.
    #[allow(clippy::too_many_arguments)]
    fn add_postprocess_pass(
        rg: &mut RenderGraph,
        spirv: &[u8],
        denoise_in_h: Handle<vulkan_abstraction::Image>,
        postprocess_out_h: Handle<vulkan_abstraction::Image>,
        width: u32,
        height: u32,
        exposure: f32,
    ) -> SrResult<()> {
        let mut common = PassCommonDataBuilder::new(rg, "postprocess");
        common.read(&denoise_in_h, vk_sync::AccessType::ComputeShaderReadOther)?;
        common.write(&postprocess_out_h, vk_sync::AccessType::ComputeShaderWrite)?;

        let pass = ComputeRenderPassBuilder::default()
            .common(common.build())
            .shaders(ComputeShaders::new(vec![ShaderSource::Spirv(spirv.to_vec())], 0, "main"))
            .generate_render(rg, [width.div_ceil(16), height.div_ceil(16), 1], move |tr| {
                Ok(PostprocessPushConstant {
                    input_idx: tr.image(&denoise_in_h)?.storage_slot(),
                    _input_pad: 0,
                    output_idx: tr.image(&postprocess_out_h)?.storage_slot(),
                    _output_pad: 0,
                    exposure,
                })
            })
            .map_err(|e| SrError::new_custom(format!("postprocess pass builder failed: {e}")))?;
        rg.add_render_pass(AnyRenderPass::Compute(pass));
        Ok(())
    }

    pub fn render_to_host_memory(
        &mut self,
        camera: &Camera,
        instances: &[(K, Vec<vk::TransformMatrixKHR>)],
    ) -> SrResult<Vec<u8>> {
        let mut dst_image = vulkan_abstraction::Image::new(
            Rc::clone(&self.core),
            self.image_extent,
            self.image_format,
            vk::ImageTiling::LINEAR,
            gpu_allocator::MemoryLocation::GpuToCpu,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
            "mapped sunray output image",
        )?;

        // Warm-up frames: ReSTIR temporal reuse + the a-trous denoise need a few
        // frames of accumulated history before the output is meaningful — a single
        // frame produces near-black because the initial temporal history is empty
        // and the first ReSTIR audition is just one RIS candidate per pixel.
        const WARMUP_FRAMES: u32 = 16;
        for _ in 0..WARMUP_FRAMES {
            let frame = self.render(dst_image.inner(), vk::Semaphore::null(), camera, instances)?;
            self.wait_frame(frame)?;
        }

        dst_image.get_raw_image_data_with_no_padding()
    }
    //TODO this needs to be reworked for a better integration with the graph or kept as default last pass
    fn cmd_blit_image(
        core: &vulkan_abstraction::Core,
        cmd_buf: vk::CommandBuffer,
        src_image: vk::Image,
        extent: vk::Extent3D,
        dst_image: vk::Image,
        image_subresource_range: &vk::ImageSubresourceRange,
    ) -> SrResult<()> {
        let device = core.device().inner();

        let image_subresource_layer = vk::ImageSubresourceLayers::default()
            .aspect_mask(image_subresource_range.aspect_mask)
            .base_array_layer(image_subresource_range.base_array_layer)
            .layer_count(image_subresource_range.layer_count)
            .mip_level(image_subresource_range.base_mip_level);
        let zero_offset = vk::Offset3D { x: 0, y: 0, z: 0 };
        let src_whole_image_offset = vk::Offset3D::default()
            .x(extent.width as i32)
            .y(extent.height as i32)
            .z(extent.depth as i32);
        let dst_whole_image_offset = vk::Offset3D::default()
            .x(extent.width as i32)
            .y(extent.height as i32)
            .z(extent.depth as i32);
        let src_offsets = [zero_offset, src_whole_image_offset];
        let dst_offsets = [zero_offset, dst_whole_image_offset];
        let image_blit = vk::ImageBlit::default()
            .src_subresource(image_subresource_layer)
            .src_offsets(src_offsets)
            .dst_subresource(image_subresource_layer)
            .dst_offsets(dst_offsets);

        unsafe {
            //transition src_image from general to transfer source layout
            vulkan_abstraction::cmd_image_memory_barrier(
                core,
                cmd_buf,
                src_image,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::SHADER_WRITE,
                vk::AccessFlags2::TRANSFER_READ,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );

            //transition dst_image to transfer destination layout
            vulkan_abstraction::cmd_image_memory_barrier(
                core,
                cmd_buf,
                dst_image,
                vk::PipelineStageFlags2::NONE,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::empty(),
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            device.cmd_blit_image(
                cmd_buf,
                src_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[image_blit],
                vk::Filter::NEAREST,
            );

            //transition dst_image to general layout which is required for mapping the image
            vulkan_abstraction::cmd_image_memory_barrier(
                core,
                cmd_buf,
                dst_image,
                vk::PipelineStageFlags2::TRANSFER,
                vk::PipelineStageFlags2::ALL_GRAPHICS, // the image should already be transitioned when the user makes use of it
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::AccessFlags2::MEMORY_READ,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::GENERAL,
            );

            //transition back src_image to general layout
            vulkan_abstraction::cmd_image_memory_barrier(
                core,
                cmd_buf,
                src_image,
                vk::PipelineStageFlags2::TRANSFER,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::TRANSFER_READ,
                vk::AccessFlags2::empty(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::GENERAL,
            );
        }

        Ok(())
    }

    pub fn core(&self) -> &Rc<vulkan_abstraction::Core> {
        &self.core
    }

}

// useful environment variables, set to 1 or 0
const ENABLE_VALIDATION_LAYER_ENV_VAR: &str = "ENABLE_VALIDATION_LAYER"; // defaults to 0 in debug build, to 1 in release build
const ENABLE_GPUAV_ENV_VAR_NAME: &str = "ENABLE_GPUAV"; // does nothing unless validation layer is enabled, defaults to 0
const ENABLE_NVIDIA_AFTERMATH_VAR_NAME: &str = "ENABLE_NVIDIA_AFTERMATH"; // does nothing unless validation layer is enabled, defaults to 0
const ENABLE_SHADER_DEBUG_SYMBOLS_ENV_VAR: &str = "ENABLE_SHADER_DEBUG_SYMBOLS"; // defaults to 0 in debug build, to 1 in release build
const IS_DEBUG_BUILD: bool = cfg!(debug_assertions);

impl<K: Hash + Eq + Copy + 'static> Drop for Renderer<K> {
    fn drop(&mut self) {
        // Stop the frame watcher before any Vulkan object it touches (the
        // timeline semaphore, the device) can be destroyed by the field drops
        // that follow this body.
        self.frame_watcher_shutdown.store(true, Ordering::Release);
        if let Some(handle) = self.frame_watcher.take() {
            let _ = handle.join();
        }

        match self.core().graphics_queue().wait_idle() {
            Ok(()) => {}
            Err(e) => match e.get_source() {
                ErrorSource::Vulkan(e) => {
                    log::warn!("VkQueueWaitIdle s returned {e:?} in sunray::Renderer::drop")
                }
                _ => log::warn!("VkQueueWaitIdle returned {e} in sunray::Renderer::drop"),
            },
        }

        // The queue is idle: every pending frame is complete, so all deferred
        // deallocation callbacks can run now.
        for (_, callback) in self.end_of_frame_callbacks.drain(..) {
            callback();
        }
    }
}
