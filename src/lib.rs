pub mod camera;
pub mod error;
pub mod render_graph;
pub mod scene;
pub mod shader_compiler;
pub mod utils;
pub mod vulkan_abstraction;

pub use camera::*;
use error::*;
pub use scene::*;

use std::{collections::HashMap, rc::Rc, sync::Arc};

use crate::render_graph::graph::AnyRenderPass;
use crate::render_graph::pass_builder::{ComputeRenderPassBuilder, PassCommonDataBuilder};
use crate::utils::{env_var_as_bool, na_mat4_to_vk_transform};
use crate::vulkan_abstraction::descriptor_sets::postprocess_descriptor_set::PostprocessDescriptorSetLayout;
use crate::vulkan_abstraction::descriptor_sets::temporal_accumulation_descriptor_set::TemporalAccumulationDescriptorSetLayout;
use crate::vulkan_abstraction::{
    DenoiseDescriptorSetLayout, DenoisePass, PostProcessDescriptorSets, PostprocessPass, PostprocessPushConstant, Reservoir,
    ReservoirGI, TemporalPass,
};
use ash::vk;
use vk_sync_fork as vk_sync;

pub const DENOISE_PASSES: u32 = 8;

pub const EXPOSURE: f32 = 1.0;

const MAX_TLAS_INSTANCES: usize = 10_000;

/// The number of concurrent frames that are processed (both by CPU and GPU).
///
/// Apparently 2 is the most common choice. Empirically it seems like the performance doesn't really
/// get any better with a higher number, but it does get measurably worse with only 1.
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

//TODO add a list of callbacks to call at the end of frames for cleanup or at start for setup
//TODO deferred deallocation for buffers and acceleration structures
struct ImageDependentData {
    pub raytracing_cmd_buf: vulkan_abstraction::CmdBuffer,
    pub blit_cmd_buf: vulkan_abstraction::CmdBuffer,
    #[allow(unused)]
    raytrace_result_image: vulkan_abstraction::Image,
    #[allow(unused)]
    postprocess_result_image: Arc<vulkan_abstraction::Image>,
    #[allow(unused)]
    depth_image: vulkan_abstraction::Image,
    #[allow(unused)]
    normal_image: vulkan_abstraction::Image,
    #[allow(unused)]
    diffuse_image: vulkan_abstraction::Image,
    #[allow(unused)]
    motion_vector_image: vulkan_abstraction::Image,

    pub raytracing_finished_semaphore: vulkan_abstraction::Semaphore,

    #[allow(unused)]
    pub raytracing_descriptor_sets: vulkan_abstraction::RaytracingDescriptorSets,
    #[allow(unused)]
    pub temporal_accumulation_descriptor_sets:
        vulkan_abstraction::descriptor_sets::temporal_accumulation_descriptor_set::TemporalAccumulationDescriptorSets,
    #[allow(unused)]
    pub denoise_descriptor_sets: vulkan_abstraction::DenoiseDescriptorSets,
    #[allow(unused)]
    pub postprocess_descriptor_sets: PostProcessDescriptorSets,
}

pub type CreateSurfaceFn = dyn Fn(&ash::Entry, &ash::Instance) -> SrResult<vk::SurfaceKHR>;

pub use crate::vulkan_abstraction::DiagnosticTool;

pub struct Renderer {
    image_dependant_data: HashMap<vk::Image, ImageDependentData>,

    resource_manager: vulkan_abstraction::ResourceManager,

    ///The first pipeline finds the best candidates for each pixel but doesn't trace many rays
    ray_tracing_pipeline_ris: vulkan_abstraction::RayTracingPipeline,
    shader_binding_table_ris: vulkan_abstraction::ShaderBindingTable,

    ///The second raytacing pipeline traces the rays based on the reservoirs created during the first pass
    ray_tracing_pipeline_final: vulkan_abstraction::RayTracingPipeline,
    shader_binding_table_final: vulkan_abstraction::ShaderBindingTable,

    ray_tracing_descriptor_set_layout: vulkan_abstraction::RaytracingDescriptorSetLayout,
    temporal_accumulation_descriptor_set_layout: TemporalAccumulationDescriptorSetLayout,
    denoise_descriptor_set_layout: DenoiseDescriptorSetLayout,
    postprocess_descriptor_set_layout: PostprocessDescriptorSetLayout,

    image_extent: vk::Extent3D,
    image_format: vk::Format,

    ///The first pass after raytracing merges the previous frame on the next one to reduce bias
    temporal_accumulation_pipeline: vulkan_abstraction::ComputePipeline<TemporalPass>,

    ///The denoise pass is run after the temporal accumulation to reduce noise even more (a-trous filter)
    denoise_pipeline: vulkan_abstraction::ComputePipeline<DenoisePass>,

    ///An extra pass to handle post-processing like exposure and color correction. Should be mathematically easy to calculate
    postprocess_pipeline: vulkan_abstraction::ComputePipeline<PostprocessPass>,

    blue_noise_image: vulkan_abstraction::Image,
    blue_noise_sampler: vulkan_abstraction::Sampler,

    /// Slang compiler held for the renderer's lifetime — owns a `GlobalSession` and
    /// is consulted when (re)building heap-mode pipelines.
    #[allow(unused)]
    shader_compiler: shader_compiler::ShaderCompiler,

    core: Rc<vulkan_abstraction::Core>,

    //2 images to avoid race conditions when reading/writing
    pub accumulation_images: [Arc<vulkan_abstraction::Image>; 2],
    pub denoising_images: [Arc<vulkan_abstraction::Image>; 2],
    ///this is used for temporal accumulation, there is an absolute frame counter in the core
    pub relative_frame_count: u32,

    prev_view_proj: nalgebra::Matrix4<f32>, //used to calculate motion vectors

    /// Persistent render graph used for the denoise + post-process pipeline.
    /// Re-populated each frame (passes / imports change because the ping-pong
    /// indices and per-frame descriptor sets do), but the underlying command
    /// buffer is reused across `compile` calls.
    render_graph: crate::render_graph::graph::RenderGraph,
    /// Fence signaled when the render graph's submission completes.
    render_graph_fence: vulkan_abstraction::Fence,

    reservoir_buffers: [vulkan_abstraction::GpuOnlyBuffer; 2],
    // Ping-pong pair of GI reservoir buffers for ReSTIR GI (Ouyang 2021); same lifetime/layout
    // contract as reservoir_buffers above, but storing surface samples (x2) instead of light samples.
    reservoir_gi_buffers: [vulkan_abstraction::GpuOnlyBuffer; 2],
}

impl Renderer {
    pub fn new(image_extent: (u32, u32), image_format: vk::Format) -> SrResult<Self> {
        Ok(Self::new_impl(image_extent, image_format, &[], None)?.0)
    }

    // It's necessary to pass a fn to create the surface, because it depends on instance, device depends on it (if present), and both device and
    // instance are created and owned inside Renderer (in Core) so this was deemed a good approach to allow the user to build their own surface
    pub fn new_with_surface(
        image_extent: (u32, u32),
        image_format: vk::Format,
        instance_exts: &'static [*const i8],
        create_surface: &CreateSurfaceFn,
    ) -> SrResult<(Self, vk::SurfaceKHR)> {
        let (r, s) = Self::new_impl(image_extent, image_format, instance_exts, Some(create_surface))?;
        Ok((r, s.unwrap()))
    }

    fn new_impl(
        image_extent: (u32, u32),
        image_format: vk::Format,
        instance_exts: &'static [*const i8],
        create_surface: Option<&CreateSurfaceFn>,
    ) -> SrResult<(Self, Option<vk::SurfaceKHR>)> {
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

        let image_extent = utils::tuple_to_extent3d(image_extent);

        //must be filled by loading a scene
        let resource_manager = vulkan_abstraction::ResourceManager::new_empty(Rc::clone(&core))?;

        let ray_tracing_descriptor_set_layout = vulkan_abstraction::RaytracingDescriptorSetLayout::new(Rc::clone(&core))?;
        let temporal_accumulation_descriptor_set_layout =
            vulkan_abstraction::TemporalAccumulationDescriptorSetLayout::new(Rc::clone(&core))?;
        let denoise_descriptor_set_layout = vulkan_abstraction::DenoiseDescriptorSetLayout::new(Rc::clone(&core))?;
        let postprocess_descriptor_set_layout = PostprocessDescriptorSetLayout::new(Rc::clone(&core))?;

        // Heap-mode RT pipelines built from the Slang-compiled SPIR-V. Both
        // pipelines share the same miss / closest-hit / any-hit stages — only
        // the ray-gen stage differs (RIS audition vs. final shading pass).
        let ray_miss_spirv = include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/ray_miss_slang.spirv"));
        let closest_hit_spirv = include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/closest_hit_slang.spirv"));
        let any_hit_spirv = include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/any_hit_slang.spirv"));

        let ray_tracing_pipeline_ris = vulkan_abstraction::RayTracingPipeline::new_heap(
            Rc::clone(&core),
            include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/ray_gen_ris_slang.spirv")),
            ray_miss_spirv,
            closest_hit_spirv,
            any_hit_spirv,
        )?;

        let shader_binding_table_ris = vulkan_abstraction::ShaderBindingTable::new(&core, &ray_tracing_pipeline_ris)?;

        let ray_tracing_pipeline_final = vulkan_abstraction::RayTracingPipeline::new_heap(
            Rc::clone(&core),
            include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/ray_gen_final_slang.spirv")),
            ray_miss_spirv,
            closest_hit_spirv,
            any_hit_spirv,
        )?;

        let shader_binding_table_final = vulkan_abstraction::ShaderBindingTable::new(&core, &ray_tracing_pipeline_final)?;

        let temporal_accumulation_pipeline = vulkan_abstraction::ComputePipeline::<TemporalPass>::new(
            Rc::clone(&core),
            temporal_accumulation_descriptor_set_layout.inner(),
        )?;

        let denoise_pipeline =
            vulkan_abstraction::ComputePipeline::<DenoisePass>::new(Rc::clone(&core), denoise_descriptor_set_layout.inner())?;

        let shaders_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
        let shader_compiler = shader_compiler::ShaderCompiler::new(shaders_dir)?;

        let postprocess_spirv = shader_compiler.compile("postprocess", "main")?;
        let postprocess_pipeline =
            vulkan_abstraction::ComputePipeline::<PostprocessPass>::new_heap(Rc::clone(&core), &postprocess_spirv)?;

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
        let reservoir_buffer_a = vulkan_abstraction::GpuOnlyBuffer::new::<Reservoir>(
            Rc::clone(&core),
            num_pixels as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "ReSTIR Reservoir Buffer A",
        )?;

        let reservoir_buffer_b = vulkan_abstraction::GpuOnlyBuffer::new::<Reservoir>(
            Rc::clone(&core),
            num_pixels as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "ReSTIR Reservoir Buffer B",
        )?;
        let reservoir_buffers = [reservoir_buffer_a, reservoir_buffer_b];

        let reservoir_gi_buffer_a = vulkan_abstraction::GpuOnlyBuffer::new::<ReservoirGI>(
            Rc::clone(&core),
            num_pixels as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "ReSTIR GI Reservoir Buffer A",
        )?;

        let reservoir_gi_buffer_b = vulkan_abstraction::GpuOnlyBuffer::new::<ReservoirGI>(
            Rc::clone(&core),
            num_pixels as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "ReSTIR GI Reservoir Buffer B",
        )?;
        let reservoir_gi_buffers = [reservoir_gi_buffer_a, reservoir_gi_buffer_b];

        let blue_noise_bytes = include_bytes!("../src/util_files/noise.png");
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

        let render_graph = crate::render_graph::graph::RenderGraph::new(Rc::clone(&core))?;
        let render_graph_fence = vulkan_abstraction::Fence::new_unsignaled(Rc::clone(core.device()))?;

        // Discard-init accumulation + denoising images to GENERAL once at startup.
        // Frame-0 denoise descriptor bindings expect GENERAL layout — the in-cmd-buf
        // discard barriers that used to do this transition lived inside the old
        // `cmd_denoise_image` and are gone now that denoise is in the render graph.
        {
            let device = core.device().inner();
            let mut setup_cmd_buf = vulkan_abstraction::CmdBuffer::new(Rc::clone(&core))?;
            let create_barrier = |image: vk::Image| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
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
                create_barrier(accumulation_images[0].inner()),
                create_barrier(accumulation_images[1].inner()),
                create_barrier(denoising_images[0].inner()),
                create_barrier(denoising_images[1].inner()),
            ];
            let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe {
                device.begin_command_buffer(setup_cmd_buf.inner(), &begin_info)?;
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                device.cmd_pipeline_barrier2(setup_cmd_buf.inner(), &dep_info);
                device.end_command_buffer(setup_cmd_buf.inner())?;
                let fence = setup_cmd_buf.fence_mut().submit()?;
                core.graphics_queue()
                    .submit_async(setup_cmd_buf.inner(), &[], &[], &[], fence)?;
                setup_cmd_buf.fence_mut().wait()?;
            }
        }

        Ok((
            Self {
                image_dependant_data,

                render_graph,
                render_graph_fence,

                reservoir_buffers,

                shader_binding_table_ris,
                ray_tracing_pipeline_ris,
                shader_binding_table_final,
                ray_tracing_pipeline_final,

                ray_tracing_descriptor_set_layout,
                temporal_accumulation_descriptor_set_layout,
                denoise_descriptor_set_layout,
                postprocess_descriptor_set_layout,

                prev_view_proj: nalgebra::zero(),

                image_extent,
                image_format,

                denoise_pipeline,
                temporal_accumulation_pipeline,
                postprocess_pipeline,

                accumulation_images,
                denoising_images,
                relative_frame_count: 0,

                blue_noise_image,
                blue_noise_sampler,

                resource_manager,
                reservoir_gi_buffers,

                shader_compiler,

                core,
            },
            surface,
        ))
    }

    pub fn resize(&mut self, image_extent: (u32, u32)) -> SrResult<()> {
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
        let reservoir_buffer_a = vulkan_abstraction::GpuOnlyBuffer::new::<Reservoir>(
            self.core.clone(),
            num_pixels as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "ReSTIR Reservoir Buffer A",
        )?;

        let reservoir_buffer_b = vulkan_abstraction::GpuOnlyBuffer::new::<Reservoir>(
            self.core.clone(),
            num_pixels as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "ReSTIR Reservoir Buffer B",
        )?;

        self.reservoir_buffers = [reservoir_buffer_a, reservoir_buffer_b];

        let reservoir_gi_buffer_a = vulkan_abstraction::GpuOnlyBuffer::new::<ReservoirGI>(
            self.core.clone(),
            num_pixels as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "ReSTIR GI Reservoir Buffer A",
        )?;

        let reservoir_gi_buffer_b = vulkan_abstraction::GpuOnlyBuffer::new::<ReservoirGI>(
            self.core.clone(),
            num_pixels as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "ReSTIR GI Reservoir Buffer B",
        )?;

        self.reservoir_gi_buffers = [reservoir_gi_buffer_a, reservoir_gi_buffer_b];

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
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
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

    pub fn clear_image_dependent_data(&mut self) {
        self.image_dependant_data = HashMap::new();
    }

    pub fn build_image_dependent_data(&mut self, images: &[vk::Image]) -> SrResult<()> {
        for post_blit_image in images {
            let raytrace_result_image = vulkan_abstraction::Image::new(
                Rc::clone(&self.core),
                self.image_extent,
                vk::Format::B10G11R11_UFLOAT_PACK32,
                vk::ImageTiling::OPTIMAL,
                gpu_allocator::MemoryLocation::GpuOnly,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                "sunray (preprocess) raytrace result image",
            )?;

            let denoise_result_image = vulkan_abstraction::Image::new(
                Rc::clone(&self.core),
                self.image_extent,
                vk::Format::B10G11R11_UFLOAT_PACK32,
                vk::ImageTiling::OPTIMAL,
                gpu_allocator::MemoryLocation::GpuOnly,
                vk::ImageUsageFlags::STORAGE,
                "sunray (internal, pre-blit) denoise result image",
            )?;

            let postprocess_result_image = Arc::new(vulkan_abstraction::Image::new(
                Rc::clone(&self.core),
                self.image_extent,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageTiling::OPTIMAL,
                gpu_allocator::MemoryLocation::GpuOnly,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
                "sunray (internal, pre-blit) postprocess result image",
            )?);

            let depth_image = vulkan_abstraction::Image::new(
                Rc::clone(&self.core),
                self.image_extent,
                vk::Format::R16_SFLOAT,
                vk::ImageTiling::OPTIMAL,
                gpu_allocator::MemoryLocation::GpuOnly,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                "sunray depth image",
            )?;

            let normal_image = vulkan_abstraction::Image::new(
                Rc::clone(&self.core),
                self.image_extent,
                vk::Format::R8G8B8A8_SNORM,
                vk::ImageTiling::OPTIMAL,
                gpu_allocator::MemoryLocation::GpuOnly,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                "sunray normal image",
            )?;

            let diffuse_image = vulkan_abstraction::Image::new(
                Rc::clone(&self.core),
                self.image_extent,
                vk::Format::B10G11R11_UFLOAT_PACK32,
                vk::ImageTiling::OPTIMAL,
                gpu_allocator::MemoryLocation::GpuOnly,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                "sunray diffuse image",
            )?;

            let motion_vector_image = vulkan_abstraction::Image::new(
                Rc::clone(&self.core),
                self.image_extent,
                vk::Format::R16G16_SFLOAT, // rg16f in GLSL
                vk::ImageTiling::OPTIMAL,
                gpu_allocator::MemoryLocation::GpuOnly,
                vk::ImageUsageFlags::STORAGE,
                "sunray motion vector image",
            )?;

            //Initializer block for g buffer images
            {
                let device = self.core.device().inner();
                let mut setup_cmd_buf = vulkan_abstraction::CmdBuffer::new(Rc::clone(&self.core))?;

                unsafe {
                    let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                    device.begin_command_buffer(setup_cmd_buf.inner(), &begin_info)?;

                    let create_barrier = |image: vk::Image| {
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                            .dst_stage_mask(
                                vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
                                    | vk::PipelineStageFlags2::COMPUTE_SHADER
                                    | vk::PipelineStageFlags2::TRANSFER,
                            )
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

                    // Add all the newly created G-Buffer and output images
                    let barriers = [
                        create_barrier(raytrace_result_image.inner()),
                        create_barrier(denoise_result_image.inner()),
                        create_barrier(depth_image.inner()),
                        create_barrier(normal_image.inner()),
                        create_barrier(diffuse_image.inner()),
                        create_barrier(motion_vector_image.inner()),
                        create_barrier(postprocess_result_image.inner()),
                    ];

                    let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                    device.cmd_pipeline_barrier2(setup_cmd_buf.inner(), &dep_info);

                    device.end_command_buffer(setup_cmd_buf.inner())?;

                    // Submit to GPU and immediately wait for it to finish
                    let fence = setup_cmd_buf.fence_mut().submit()?;
                    self.core
                        .graphics_queue()
                        .submit_async(setup_cmd_buf.inner(), &[], &[], &[], fence)?;

                    // Block the CPU so we guarantee the transitions are done before rendering starts
                    setup_cmd_buf.fence_mut().wait()?;
                }
            }

            let raytracing_descriptor_sets = vulkan_abstraction::RaytracingDescriptorSets::new(
                Rc::clone(&self.core),
                &self.ray_tracing_descriptor_set_layout,
                self.resource_manager.tlas(),
                &raytrace_result_image, // Raw color output
                &depth_image,           // G-Buffer Depth
                &normal_image,          // G-Buffer Normals
                &diffuse_image,         // G-Buffer Diffuse
                &motion_vector_image,   // G-Buffer Motion
                &self.blue_noise_image,
                self.blue_noise_sampler.inner(),
                &self.reservoir_buffers,
                &self.reservoir_gi_buffers,
                &self.resource_manager,
            )?;

            let accum_refs = [&*self.accumulation_images[0], &*self.accumulation_images[1]];
            let denoise_refs = [&*self.denoising_images[0], &*self.denoising_images[1]];

            let temporal_accumulation_descriptor_sets = vulkan_abstraction::TemporalAccumulationDescriptorSets::new(
                &self.core, // Passed as &Rc<Core> based on our previous struct signature
                &self.temporal_accumulation_descriptor_set_layout,
                &raytrace_result_image, // Binding 0: Noisy Input
                &motion_vector_image,   // Binding 1: Motion Vectors
                accum_refs,             // Binding 2: Ping-Pong Output (Storage)
                accum_refs,             // Binding 3: Ping-Pong History (Samplers)
                self.resource_manager.default_sampler().inner(),
            )?;

            let denoise_descriptor_sets = vulkan_abstraction::DenoiseDescriptorSets::new(
                Rc::clone(&self.core),
                &self.denoise_descriptor_set_layout,
                accum_refs,
                &depth_image,
                &normal_image,
                &diffuse_image,
                denoise_refs,
                self.resource_manager.default_sampler().inner(),
            )?;

            let postprocess_descriptor_sets = vulkan_abstraction::PostProcessDescriptorSets::new(
                Rc::clone(&self.core),
                &self.postprocess_descriptor_set_layout,
                denoise_refs,
                &postprocess_result_image,
            )?;

            let blit_cmd_buf = vulkan_abstraction::CmdBuffer::new(Rc::clone(&self.core))?;
            let raytracing_cmd_buf = vulkan_abstraction::CmdBuffer::new(Rc::clone(&self.core))?;

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

            let raytracing_finished_semaphore = vulkan_abstraction::Semaphore::new(self.core.clone())?;

            self.image_dependant_data.insert(
                *post_blit_image,
                ImageDependentData {
                    raytrace_result_image,
                    postprocess_result_image,
                    depth_image,
                    normal_image,
                    diffuse_image,
                    motion_vector_image,
                    raytracing_cmd_buf,
                    blit_cmd_buf,
                    raytracing_descriptor_sets,
                    temporal_accumulation_descriptor_sets,
                    denoise_descriptor_sets,
                    postprocess_descriptor_sets,
                    raytracing_finished_semaphore,
                },
            );
        }

        Ok(())
    }

    pub fn load_gltf(&mut self, path: &str) -> SrResult<Vec<vulkan_abstraction::EntityId>> {
        let gltf = vulkan_abstraction::gltf::Gltf::new(Rc::clone(&self.core), path)?;
        let (default_scene, scene_data) = gltf.create_default_scene()?;
        self.load_scene(&default_scene, scene_data)
    }

    pub fn load_scene(&mut self, scene: &Scene, scene_data: SceneData) -> SrResult<Vec<vulkan_abstraction::EntityId>> {
        // Wait for all in-flight GPU work before invalidating descriptor sets that reference
        // buffers which will be reallocated (e.g. emissive_indirection_gpu).
        unsafe { self.core.device().inner().device_wait_idle() }?;
        let ids = self.resource_manager.load_scene(scene, scene_data)?;
        self.image_dependant_data = HashMap::new();
        Ok(ids)
    }

    /// Spawn a new instance that shares the BLAS and material of `src` with a new transform.
    /// Automatically rebuilds the TLAS.
    pub fn duplicate_entity(
        &mut self,
        src: vulkan_abstraction::EntityId,
        transform: nalgebra::Matrix4<f32>,
    ) -> SrResult<vulkan_abstraction::EntityId> {
        let vk_transform = na_mat4_to_vk_transform(transform);
        let id = self.resource_manager.clone_entity(src, vk_transform)?;
        self.resource_manager.rebuild_tlas()?;
        // rebuild_tlas calls AccelerationStructure::rebuild which creates a new
        // VkAccelerationStructureKHR handle, invalidating any descriptor sets that
        // reference the old one. Clear them so they are rebuilt on the next frame.

        self.clear_image_dependent_data();

        Ok(id)
    }

    /// Remove an entity from the scene. Automatically rebuilds the TLAS.
    pub fn destroy_entity(&mut self, id: vulkan_abstraction::EntityId) -> SrResult<()> {
        self.resource_manager.destroy_entity(id);
        self.resource_manager.rebuild_tlas()?;
        self.clear_image_dependent_data();
        Ok(())
    }

    /// Update an entity's world transform. Does NOT rebuild the TLAS — call `rebuild_tlas` afterwards.
    pub fn set_entity_transform(&mut self, id: vulkan_abstraction::EntityId, transform: nalgebra::Matrix4<f32>) -> SrResult<()> {
        let vk_transform = na_mat4_to_vk_transform(transform);
        self.resource_manager.set_entity_transform(id, vk_transform)
    }

    pub fn set_camera(&mut self, camera: crate::Camera) -> SrResult<()> {
        //TODO
        //
        // Waiting for device idle here serializes the UBO update against any
        // in-flight GPU work. Minimum viable fix — the proper long-term fix is
        // per-frame UBOs (double/triple buffering) so we never overwrite bytes
        // a running frame might still read.
        unsafe {
            self.core.device().inner().device_wait_idle().unwrap();
        }

        let mut matrices = camera.as_matrices(self.image_extent);

        // Inject the history matrix saved from the last frame
        matrices.prev_view_proj = self.prev_view_proj;
        let tmp = matrices.view_proj;

        // Upload the struct to the uniform buffer
        self.resource_manager.set_matrices(matrices)?;

        // Save the current frame's matrix to use as history NEXT frame
        self.prev_view_proj = tmp;

        Ok(())
    }

    /// Render to dst_image. the user may also pass a Semaphore which the user should signal when the image is
    /// ready to be written to (for example after being acquired from a swapchain) and a Fence will be returned
    /// that will be signaled when the rendering is finished (which can be used to know when the Semaphore has no pending operations left).
    pub fn render_to_image(&mut self, dst_image: vk::Image, wait_sem: vk::Semaphore) -> SrResult<vk::Fence> {
        self.resource_manager.start_of_frame()?;

        unsafe {
            self.core.device().inner().device_wait_idle()?;
        }

        if !self.image_dependant_data.contains_key(&dst_image) {
            self.build_image_dependent_data(&[dst_image])?;
        }

        // Wait the render graph's previous frame BEFORE we touch render_graph state
        // (compile() will reset+re-record the persistent cmd buffer, which is UB if
        // the GPU is still using it).
        self.render_graph_fence.wait()?;

        let this_ptr = self as *mut Self;

        let img_dependent_data = self.image_dependant_data.get_mut(&dst_image).unwrap();

        // Raytracing
        img_dependent_data.raytracing_cmd_buf.fence_mut().wait()?;
        img_dependent_data.blit_cmd_buf.fence_mut().wait()?;

        let cmd_buf = img_dependent_data.raytracing_cmd_buf.inner();
        let result_image = img_dependent_data.raytrace_result_image.inner();
        let motion_vector_image = img_dependent_data.motion_vector_image.inner();
        let result_extent = img_dependent_data.raytrace_result_image.extent();

        // Raw-pointer alias keeps the borrow-checker happy: the unsafe block
        // below borrows `*this_ptr` mutably for `cmd_raytracing_render` while
        // we still hold the per-image data reference. The function only reads
        // through the pointer, so this is sound.
        let img_dependent_data_ptr = img_dependent_data as *const ImageDependentData;
        let temporal_accumulation_descriptor_sets_ptr = &img_dependent_data.temporal_accumulation_descriptor_sets
            as *const vulkan_abstraction::TemporalAccumulationDescriptorSets;

        // === Phase 1: record the RT + temporal accumulation cmd buf ===
        unsafe {
            let device = (*this_ptr).core.device().inner();

            let wait_semaphores = (*this_ptr).core.transfer_semaphores_mut().drain(..).collect::<Vec<_>>();
            let wait_stages = wait_semaphores
                .iter()
                .map(|_semaphore| {
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR | vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
                })
                .collect::<Vec<_>>();

            let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(cmd_buf, &begin_info)?;

            (*this_ptr).cmd_raytracing_render(cmd_buf, &*img_dependent_data_ptr, result_image, result_extent)?;

            // RT → TAA / denoise read barrier.
            let memory_barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ);
            let dep_info = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&memory_barrier));
            device.cmd_pipeline_barrier2(cmd_buf, &dep_info);

            // depth/normal/diffuse transition GENERAL → SHADER_READ_ONLY for the denoise
            // descriptor bindings. They stay in SHADER_READ_ONLY after this cmd buf
            // ends; next frame's cmd_raytracing_render discards them back to GENERAL
            // via its pre-RT barriers.
            let read_only_barriers = [
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(img_dependent_data.depth_image.inner())
                    .subresource_range(*img_dependent_data.depth_image.image_subresource_range()),
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(img_dependent_data.normal_image.inner())
                    .subresource_range(*img_dependent_data.normal_image.image_subresource_range()),
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(img_dependent_data.diffuse_image.inner())
                    .subresource_range(*img_dependent_data.diffuse_image.image_subresource_range()),
            ];

            let dep_info = vk::DependencyInfo::default().image_memory_barriers(&read_only_barriers);
            device.cmd_pipeline_barrier2(cmd_buf, &dep_info);

            (*this_ptr).cmd_temporal_accumulation(
                cmd_buf,
                &*temporal_accumulation_descriptor_sets_ptr,
                result_extent.width,
                result_extent.height,
                result_image,
                motion_vector_image,
                &self.accumulation_images,
            )?;

            device.end_command_buffer(cmd_buf)?;
        }

        // === Phase 2: build & compile the render graph for denoise + postprocess ===
        // `cmd_raytracing_render` has already advanced `relative_frame_count`, so the
        // ping-pong indices below match the post-increment values the old in-cmd-buf
        // denoise code used.
        let temporal_accum_idx = ((self.relative_frame_count + 1) % 2) as usize;

        // Re-aim the denoise descriptor set's pass-0 input at this frame's accum slot.
        img_dependent_data
            .denoise_descriptor_sets
            .update_initial_input(&self.accumulation_images[temporal_accum_idx]);

        unsafe {
            (*this_ptr).build_denoise_postprocess_graph(temporal_accum_idx, &*img_dependent_data_ptr, result_extent)?;
        }

        // === Phase 3: submit RT+temporal, wait_idle, then run the render graph ===
        let (raytracing_cmd, raytracing_fence_handle) = {
            let f = img_dependent_data.raytracing_cmd_buf.fence_mut().submit()?;
            (img_dependent_data.raytracing_cmd_buf.inner(), f)
        };
        unsafe {
            let wait_semaphores = (*this_ptr).core.transfer_semaphores_mut().drain(..).collect::<Vec<_>>();
            let wait_stages = wait_semaphores
                .iter()
                .map(|_| vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR | vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR)
                .collect::<Vec<_>>();
            (*this_ptr).core.graphics_queue().submit_async(
                raytracing_cmd,
                &wait_semaphores,
                &wait_stages,
                &[],
                raytracing_fence_handle,
            )?;
            (*this_ptr).core.device().inner().device_wait_idle()?;

            (*this_ptr).render_graph.run(&mut (*this_ptr).render_graph_fence)?;
            (*this_ptr).render_graph_fence.wait()?;
        }

        // === Phase 4: blit ===
        let (wait_sems, wait_dst_stages) = (
            [wait_sem, vk::Semaphore::null()],
            [vk::PipelineStageFlags::ALL_GRAPHICS, vk::PipelineStageFlags::TRANSFER],
        );
        let (wait_sems, wait_dst_stages) = if wait_sem == vk::Semaphore::null() {
            ([].as_slice(), [].as_slice())
        } else {
            (&wait_sems[..1], &wait_dst_stages[..1])
        };

        let signal_fence = img_dependent_data.blit_cmd_buf.fence_mut().submit()?;

        unsafe {
            (*this_ptr).core.graphics_queue().submit_async(
                img_dependent_data.blit_cmd_buf.inner(),
                wait_sems,
                wait_dst_stages,
                &[],
                signal_fence,
            )?;
        }

        Ok(signal_fence)
    }

    /// Populate `self.render_graph` with the denoise (8 a-trous passes) and the
    /// postprocess pass, then compile. Splits out of `render_to_image` to keep
    /// the closure-heavy bookkeeping isolated.
    fn build_denoise_postprocess_graph(
        &mut self,
        temporal_accum_idx: usize,
        img_dependent_data: &ImageDependentData,
        result_extent: vk::Extent3D,
    ) -> SrResult<()> {
        // Capture-by-value snapshot for the 'static closures.
        let core_for_closures = Rc::clone(&self.core);
        let denoise_pipeline = self.denoise_pipeline.inner();
        let denoise_layout = self.denoise_pipeline.layout();
        let postprocess_pipeline = self.postprocess_pipeline.inner();
        let frame_count = self.relative_frame_count;
        let width = result_extent.width;
        let height = result_extent.height;
        let dset_handles = [
            img_dependent_data.denoise_descriptor_sets.inner()[0],
            img_dependent_data.denoise_descriptor_sets.inner()[1],
            img_dependent_data.denoise_descriptor_sets.inner()[2],
        ];
        let denoising_images = [
            Arc::clone(&self.denoising_images[0]),
            Arc::clone(&self.denoising_images[1]),
        ];
        let postprocess_out_arc = Arc::clone(&img_dependent_data.postprocess_result_image);
        let accum_input_arc = Arc::clone(&self.accumulation_images[temporal_accum_idx]);

        let rg = &mut self.render_graph;
        rg.reset();

        let accum_handle = rg.import::<crate::render_graph::graph::ImageDesc>(accum_input_arc);
        let denoise_a_handle = rg.import::<crate::render_graph::graph::ImageDesc>(Arc::clone(&denoising_images[0]));
        let denoise_b_handle = rg.import::<crate::render_graph::graph::ImageDesc>(Arc::clone(&denoising_images[1]));
        let postprocess_out_handle = rg.import::<crate::render_graph::graph::ImageDesc>(Arc::clone(&postprocess_out_arc));

        for pass_index in 0..DENOISE_PASSES {
            let step_width = 1u32 << pass_index;
            let descriptor_idx = if pass_index == 0 {
                0usize
            } else if pass_index % 2 == 1 {
                1
            } else {
                2
            };
            let (read_handle, write_handle) = if pass_index == 0 {
                (&accum_handle, &denoise_a_handle)
            } else if pass_index % 2 == 1 {
                (&denoise_a_handle, &denoise_b_handle)
            } else {
                (&denoise_b_handle, &denoise_a_handle)
            };

            let mut builder = PassCommonDataBuilder::new(rg, format!("denoise_{pass_index}"));
            builder.read(read_handle, vk_sync::AccessType::ComputeShaderReadOther)?;
            builder.write(write_handle, vk_sync::AccessType::ComputeShaderWrite)?;

            let core = Rc::clone(&core_for_closures);
            let dset = dset_handles[descriptor_idx];
            let push = vulkan_abstraction::DenoisePushConstant {
                frame_count,
                step_width,
                width,
                height,
            };
            builder.render(move |cb, _tr| {
                let device = core.device().inner();
                unsafe {
                    device.cmd_bind_pipeline(*cb, vk::PipelineBindPoint::COMPUTE, denoise_pipeline);
                    let sets = [dset];
                    let bind_info = vk::BindDescriptorSetsInfo::default()
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .layout(denoise_layout)
                        .first_set(0)
                        .descriptor_sets(&sets)
                        .dynamic_offsets(&[]);
                    device.cmd_bind_descriptor_sets2(*cb, &bind_info);

                    let push_bytes: [u8; std::mem::size_of::<vulkan_abstraction::DenoisePushConstant>()] =
                        std::mem::transmute(push);
                    let push_info = vk::PushConstantsInfo::default()
                        .layout(denoise_layout)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .offset(0)
                        .values(&push_bytes);
                    device.cmd_push_constants2(*cb, &push_info);

                    device.cmd_dispatch(*cb, width.div_ceil(16), height.div_ceil(16), 1);
                }
                Ok(())
            });

            let pass = ComputeRenderPassBuilder::default()
                .common(builder.build())
                .shaders(vec![])
                .entry_point("main".to_string())
                .build()
                .map_err(|e| SrError::new_custom(format!("denoise pass builder failed: {e}")))?;
            rg.add_render_pass(AnyRenderPass::Compute(pass));
        }

        // === Postprocess ===
        let final_idx = ((DENOISE_PASSES - 1) % 2) as usize;
        let denoise_input_handle = if final_idx == 0 { &denoise_a_handle } else { &denoise_b_handle };

        let mut builder = PassCommonDataBuilder::new(rg, "postprocess");
        builder.read(denoise_input_handle, vk_sync::AccessType::ComputeShaderReadOther)?;
        builder.write(&postprocess_out_handle, vk_sync::AccessType::ComputeShaderWrite)?;

        let core = Rc::clone(&core_for_closures);
        let denoise_input_arc = Arc::clone(&denoising_images[final_idx]);
        let postprocess_out_arc_for_closure = Arc::clone(&postprocess_out_arc);
        let exposure = EXPOSURE;
        builder.render(move |cb, _tr| {
            let push_constants = PostprocessPushConstant {
                input_idx: denoise_input_arc.storage_slot(),
                _input_pad: 0,
                output_idx: postprocess_out_arc_for_closure.storage_slot(),
                _output_pad: 0,
                exposure,
            };
            let device = core.device().inner();
            unsafe {
                device.cmd_bind_pipeline(*cb, vk::PipelineBindPoint::COMPUTE, postprocess_pipeline);
                core.descriptor_heap().cmd_bind(*cb);
                let push_info = vk::PushDataInfoEXT::default().offset(0).data(vk::HostAddressRangeConstEXT {
                    address: &push_constants as *const _ as *const std::ffi::c_void,
                    size: std::mem::size_of::<PostprocessPushConstant>(),
                    _marker: Default::default(),
                });
                core.descriptor_heap_device().cmd_push_data(*cb, &push_info);
                device.cmd_dispatch(*cb, width.div_ceil(16), height.div_ceil(16), 1);
            }
            Ok(())
        });

        let pass = ComputeRenderPassBuilder::default()
            .common(builder.build())
            .shaders(vec![])
            .entry_point("main".to_string())
            .build()
            .map_err(|e| SrError::new_custom(format!("postprocess pass builder failed: {e}")))?;
        rg.add_render_pass(AnyRenderPass::Compute(pass));

        rg.compile()?;
        Ok(())
    }

    pub fn render_to_host_memory(&mut self) -> SrResult<Vec<u8>> {
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
            let wait_fence = self.render_to_image(dst_image.inner(), vk::Semaphore::null())?;
            vulkan_abstraction::wait_fence(self.core.device(), wait_fence)?;
        }

        dst_image.get_raw_image_data_with_no_padding()
    }

    fn cmd_raytracing_render(
        &mut self,
        cmd_buf: vk::CommandBuffer,
        img_dependent_data: &ImageDependentData,
        image: vk::Image,
        extent: vk::Extent3D,
    ) -> SrResult<()> {
        let device = self.core.device().inner();

        //ping pong to avoid errors when using accumulation images
        let history_idx = (self.relative_frame_count % 2) as usize;
        let accum_idx = ((self.relative_frame_count + 1) % 2) as usize;

        // Build the heap-mode push constant. Every field corresponds 1:1 to a
        // `DescriptorHandle<…>` in `shaders/rt_types.slang::RaytracingPC`; the
        // unused high word of each handle stays zero. Each `.storage_slot()` /
        // `.sampled_slot()` / `.slot()` call lazily allocates (and writes)
        // a heap descriptor on first use.
        use vulkan_abstraction::Buffer;
        let pack = |idx: u32| -> [u32; 2] { [idx, 0] };
        let push_constants = vulkan_abstraction::RaytracingHeapPushConstant {
            tlas: self.resource_manager.tlas().device_address(),
            raw_color: pack(img_dependent_data.raytrace_result_image.storage_slot()),
            depth_img: pack(img_dependent_data.depth_image.storage_slot()),
            normal_img: pack(img_dependent_data.normal_image.storage_slot()),
            diffuse_img: pack(img_dependent_data.diffuse_image.storage_slot()),
            motion_vec_img: pack(img_dependent_data.motion_vector_image.storage_slot()),
            matrices: self.resource_manager.matrices_buffer_address(),
            meshes_info: pack(self.resource_manager.meshes_info_storage_slot()),
            emissive_triangles: pack(self.resource_manager.emissive_triangles_storage_slot()),
            emissive_indirection: pack(self.resource_manager.emissive_indirection_storage_slot()),
            entity_transforms: pack(self.resource_manager.entity_transforms_storage_slot()),
            blue_noise_tex: pack(self.blue_noise_image.sampled_slot()),
            blue_noise_sampler: pack(self.blue_noise_sampler.slot()),
            reservoirs: [
                self.reservoir_buffers[0].get_device_address(),
                self.reservoir_buffers[1].get_device_address(),
            ],
            reservoirs_gi: [
                self.reservoir_gi_buffers[0].get_device_address(),
                self.reservoir_gi_buffers[1].get_device_address(),
            ],
            textures_lookup: pack(self.resource_manager.textures_lookup_slot()),
            frame_count: self.relative_frame_count,
            use_srgb: if self.image_format == vk::Format::R8G8B8A8_SRGB {
                1
            } else {
                0
            },
        };

        self.relative_frame_count += 1;
        *self.core.absolute_frame_count.borrow_mut() += 1;
        unsafe {
            let subresource_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(vk::REMAINING_MIP_LEVELS)
                .base_array_layer(0)
                .layer_count(vk::REMAINING_ARRAY_LAYERS);

            let src_stage = if self.relative_frame_count == 1 {
                vk::PipelineStageFlags2::TOP_OF_PIPE
            } else {
                vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
            };
            let dst_stage = vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR;

            let make_barrier = |img: vk::Image, old: vk::ImageLayout, new: vk::ImageLayout, src_a, dst_a| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(src_stage)
                    .dst_stage_mask(dst_stage)
                    .src_access_mask(src_a)
                    .dst_access_mask(dst_a)
                    .old_layout(old)
                    .new_layout(new)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(img)
                    .subresource_range(subresource_range)
            };

            let b_swap = make_barrier(
                image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags2::empty(),
                vk::AccessFlags2::SHADER_WRITE,
            );

            let (hist_old, hist_src) = if self.relative_frame_count == 1 {
                (vk::ImageLayout::UNDEFINED, vk::AccessFlags2::empty())
            } else {
                (vk::ImageLayout::GENERAL, vk::AccessFlags2::SHADER_WRITE)
            };

            let b_hist = make_barrier(
                self.accumulation_images[history_idx].inner(),
                hist_old,
                vk::ImageLayout::GENERAL,
                hist_src,
                vk::AccessFlags2::SHADER_READ,
            );

            let (accum_old, accum_src) = if self.relative_frame_count == 1 {
                (vk::ImageLayout::UNDEFINED, vk::AccessFlags2::empty())
            } else {
                (vk::ImageLayout::GENERAL, vk::AccessFlags2::SHADER_READ)
            };

            let b_accum = make_barrier(
                self.accumulation_images[accum_idx].inner(),
                accum_old,
                vk::ImageLayout::GENERAL,
                accum_src,
                vk::AccessFlags2::SHADER_WRITE,
            );

            // Depth / normal / diffuse get left in SHADER_READ_ONLY after the
            // previous frame's denoise descriptor binding (the cmd buf doesn't
            // transition them back any more — `return_to_general_barriers` was
            // dropped when denoise + postprocess moved into the render graph).
            // RT writes them fresh every frame, so a discard transition
            // (UNDEFINED → GENERAL) is sufficient and avoids per-frame
            // layout-tracking bookkeeping.
            let b_depth = make_barrier(
                img_dependent_data.depth_image.inner(),
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags2::empty(),
                vk::AccessFlags2::SHADER_WRITE,
            );
            let b_normal = make_barrier(
                img_dependent_data.normal_image.inner(),
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags2::empty(),
                vk::AccessFlags2::SHADER_WRITE,
            );
            let b_diffuse = make_barrier(
                img_dependent_data.diffuse_image.inner(),
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags2::empty(),
                vk::AccessFlags2::SHADER_WRITE,
            );

            let pre_rt_barriers = [b_swap, b_hist, b_accum, b_depth, b_normal, b_diffuse];
            let dep_info = vk::DependencyInfo::default().image_memory_barriers(&pre_rt_barriers);
            device.cmd_pipeline_barrier2(cmd_buf, &dep_info);

            // Bind the descriptor heaps once for both RT dispatches. Heap
            // bindings persist across pipeline binds so we don't repeat this
            // between the two trace_rays calls.
            self.core.descriptor_heap().cmd_bind(cmd_buf);

            // --- PASS 1: RIS audition (heap mode, push-data) ---
            device.cmd_bind_pipeline(
                cmd_buf,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.ray_tracing_pipeline_ris.inner(),
            );

            let push_info = vk::PushDataInfoEXT::default().offset(0).data(vk::HostAddressRangeConstEXT {
                address: &push_constants as *const _ as *const std::ffi::c_void,
                size: std::mem::size_of::<vulkan_abstraction::RaytracingHeapPushConstant>(),
                _marker: Default::default(),
            });
            self.core.descriptor_heap_device().cmd_push_data(cmd_buf, &push_info);

            self.core.cmd_set_checkpoint(cmd_buf, c"rt_pass_ris::before_trace_rays");
            self.core.rt_pipeline_device().cmd_trace_rays(
                cmd_buf,
                self.shader_binding_table_ris.raygen_region(),
                self.shader_binding_table_ris.miss_region(),
                self.shader_binding_table_ris.hit_region(),
                self.shader_binding_table_ris.callable_region(),
                extent.width,
                extent.height,
                extent.depth,
            );
            self.core.cmd_set_checkpoint(cmd_buf, c"rt_pass_ris::after_trace_rays");

            // Reservoir A->B handoff between the two RT dispatches.
            let reservoir_barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ);
            let dep_info = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&reservoir_barrier));
            device.cmd_pipeline_barrier2(cmd_buf, &dep_info);

            // --- PASS 2: final shading (heap mode, push-data) ---
            device.cmd_bind_pipeline(
                cmd_buf,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.ray_tracing_pipeline_final.inner(),
            );

            // Same push constant bytes — `cmd_push_data` is per-pipeline so we
            // re-issue it after the rebind.
            self.core.descriptor_heap_device().cmd_push_data(cmd_buf, &push_info);

            self.core.cmd_set_checkpoint(cmd_buf, c"rt_pass_final::before_trace_rays");
            self.core.rt_pipeline_device().cmd_trace_rays(
                cmd_buf,
                self.shader_binding_table_final.raygen_region(),
                self.shader_binding_table_final.miss_region(),
                self.shader_binding_table_final.hit_region(),
                self.shader_binding_table_final.callable_region(),
                extent.width,
                extent.height,
                extent.depth,
            );
            self.core.cmd_set_checkpoint(cmd_buf, c"rt_pass_final::after_trace_rays");
        }

        Ok(())
    }

    fn cmd_temporal_accumulation(
        &self,
        cmd_buf: vk::CommandBuffer,
        descriptor_sets: &vulkan_abstraction::TemporalAccumulationDescriptorSets,
        width: u32,
        height: u32,
        raw_rt_image: vk::Image,
        motion_vector_image: vk::Image,
        accumulation_images: &[Arc<vulkan_abstraction::image::Image>; 2],
    ) -> SrResult<()> {
        let device = self.core.device().inner();

        let history_idx = (self.relative_frame_count % 2) as usize;
        let accum_idx = ((self.relative_frame_count + 1) % 2) as usize;

        // 1. Prepare inputs (RT Image and Motion Vectors)
        let rt_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(raw_rt_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let mv_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE) // Assuming written in G-Buffer pass
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(motion_vector_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        // 2. Prepare Ping-Pong Images
        // The one we write to:
        let write_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::empty()) // Don't care what it was doing before
            .dst_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED) // Discard old contents
            .new_layout(vk::ImageLayout::GENERAL)
            .image(accumulation_images[accum_idx].inner())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let history_old_layout = if self.relative_frame_count == 0 {
            vk::ImageLayout::UNDEFINED
        } else {
            vk::ImageLayout::GENERAL
        };

        let history_src_access = if self.relative_frame_count == 0 {
            vk::AccessFlags2::empty()
        } else {
            vk::AccessFlags2::SHADER_WRITE
        };

        // The one we read from (History):
        let read_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(history_src_access) // Written to last frame
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .old_layout(history_old_layout)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(accumulation_images[history_idx].inner())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        unsafe {
            let pre_temporal_barriers = [rt_barrier, mv_barrier, write_barrier, read_barrier];
            let dep_info = vk::DependencyInfo::default().image_memory_barriers(&pre_temporal_barriers);
            device.cmd_pipeline_barrier2(cmd_buf, &dep_info);

            device.cmd_bind_pipeline(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.temporal_accumulation_pipeline.inner(),
            );

            let bind_info = vk::BindDescriptorSetsInfo::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .layout(self.temporal_accumulation_pipeline.layout())
                .first_set(0)
                .descriptor_sets(descriptor_sets.inner()) // Has all bindings combined
                .dynamic_offsets(&[]);
            device.cmd_bind_descriptor_sets2(cmd_buf, &bind_info);

            let push_constants = vulkan_abstraction::TemporalAccumulationPushConstant {
                frame_count: self.relative_frame_count,
                width,
                height,
            };
            let push_bytes = std::mem::transmute::<
                vulkan_abstraction::TemporalAccumulationPushConstant,
                [u8; std::mem::size_of::<vulkan_abstraction::TemporalAccumulationPushConstant>()],
            >(push_constants);
            let push_info = vk::PushConstantsInfo::default()
                .layout(self.temporal_accumulation_pipeline.layout())
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .values(&push_bytes);
            device.cmd_push_constants2(cmd_buf, &push_info);

            let group_x = width.div_ceil(16);
            let group_y = height.div_ceil(16);
            device.cmd_dispatch(cmd_buf, group_x, group_y, 1);
        }

        Ok(())
    }

    #[allow(dead_code)]
    fn cmd_denoise_image(
        &self,
        cmd_buf: vk::CommandBuffer,
        descriptor_sets: &vulkan_abstraction::DenoiseDescriptorSets,
        width: u32,
        height: u32,
        input_images: &[Arc<vulkan_abstraction::Image>; 2], //used in the first pass (getting from TAA to denoise)
        denoise_pingpong_images: &[Arc<vulkan_abstraction::Image>; 2],
    ) -> SrResult<()> {
        let device = self.core.device().inner();
        let total_passes = DENOISE_PASSES;

        // Temporal accumulation writes to `accumulation_images[(frame_count + 1) % 2]`
        // (its `accum_idx`), so that's the slot denoise must read on pass 0. Rebind
        // set 0's input each frame so the descriptor tracks the live slot — the
        // hardcoded `temp_0` binding from descriptor-set creation only matches half
        // the frames and yields stale-or-zero data the other half.
        let temporal_accum_idx = ((self.relative_frame_count + 1) % 2) as usize;
        descriptor_sets.update_initial_input(&input_images[temporal_accum_idx]);
        let history_idx = temporal_accum_idx;

        for pass_index in 0..total_passes {
            // Step width follows a-trous wavelet pattern: 1, 2, 4, 8...
            let step_width = 1 << pass_index;

            // Logic for choosing images and descriptors:
            // Pass 0: Read Input[hist] -> Write Denoise[0] (Set 0)
            // Pass 1: Read Denoise[0] -> Write Denoise[1] (Set 1)
            // Pass 2: Read Denoise[1] -> Write Denoise[0] (Set 2)
            // Pass 3: Read Denoise[0] -> Write Denoise[1] (Set 1)
            let (read_img, write_img, descriptor_idx) = if pass_index == 0 {
                (
                    input_images[history_idx].inner(),
                    denoise_pingpong_images[0].inner(),
                    0, // Set 0: Input -> Denoise 0
                )
            } else if pass_index % 2 == 1 {
                (
                    denoise_pingpong_images[0].inner(),
                    denoise_pingpong_images[1].inner(),
                    1, // Set 1: Denoise 0 -> Denoise 1
                )
            } else {
                (
                    denoise_pingpong_images[1].inner(),
                    denoise_pingpong_images[0].inner(),
                    2, // Set 2: Denoise 1 -> Denoise 0
                )
            };

            // --- BARRIERS ---
            // Synchronize the image we are about to read (ensure previous write is done)
            let read_barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(read_img)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                });

            // Synchronize the image we are about to write to
            let write_barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::empty()) // Or SHADER_READ if it was a source in a previous pass
                .dst_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(write_img)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                });

            let push_constants = vulkan_abstraction::DenoisePushConstant {
                frame_count: self.relative_frame_count,
                step_width,
                width,
                height,
            };

            unsafe {
                // Compute-to-Compute barrier ensures execution order and cache visibility
                let denoise_barriers = [read_barrier, write_barrier];
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&denoise_barriers);
                device.cmd_pipeline_barrier2(cmd_buf, &dep_info);

                device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, self.denoise_pipeline.inner());

                let descriptor_set = [descriptor_sets.inner()[descriptor_idx]];
                let bind_info = vk::BindDescriptorSetsInfo::default()
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .layout(self.denoise_pipeline.layout())
                    .first_set(0)
                    .descriptor_sets(&descriptor_set)
                    .dynamic_offsets(&[]);
                device.cmd_bind_descriptor_sets2(cmd_buf, &bind_info);

                let push_bytes = std::mem::transmute::<
                    vulkan_abstraction::DenoisePushConstant,
                    [u8; std::mem::size_of::<vulkan_abstraction::DenoisePushConstant>()],
                >(push_constants);
                let push_info = vk::PushConstantsInfo::default()
                    .layout(self.denoise_pipeline.layout())
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .offset(0)
                    .values(&push_bytes);
                device.cmd_push_constants2(cmd_buf, &push_info);

                let group_x = width.div_ceil(16);
                let group_y = height.div_ceil(16);
                device.cmd_dispatch(cmd_buf, group_x, group_y, 1);
            }
        }

        Ok(())
    }

    #[allow(dead_code)]
    fn cmd_postprocess_image(
        &self,
        cmd_buf: vk::CommandBuffer,
        width: u32,
        height: u32,
        input_image: &vulkan_abstraction::Image,
        output_image: &vulkan_abstraction::Image,
    ) -> SrResult<()> {
        let device = self.core.device().inner();

        let push_constants = vulkan_abstraction::PostprocessPushConstant {
            input_idx: input_image.storage_slot(),
            _input_pad: 0,
            output_idx: output_image.storage_slot(),
            _output_pad: 0,
            exposure: EXPOSURE,
        };

        let input_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(input_image.inner())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let output_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::empty())
            .dst_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(output_image.inner())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        unsafe {
            let postprocess_pre_barriers = [input_barrier, output_barrier];
            let dep_info = vk::DependencyInfo::default().image_memory_barriers(&postprocess_pre_barriers);
            device.cmd_pipeline_barrier2(cmd_buf, &dep_info);

            device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, self.postprocess_pipeline.inner());

            // Bind the descriptor heaps (resource + sampler) for this dispatch.
            self.core.descriptor_heap().cmd_bind(cmd_buf);

            // Heap-mode pipelines have no VkPipelineLayout, so push constants go through
            // vkCmdPushDataEXT — the descriptor-heap replacement for vkCmdPushConstants.
            let push_info = vk::PushDataInfoEXT::default().offset(0).data(vk::HostAddressRangeConstEXT {
                address: &push_constants as *const _ as *const std::ffi::c_void,
                size: size_of::<PostprocessPushConstant>(),
                _marker: Default::default(),
            });

            self.core.descriptor_heap_device().cmd_push_data(cmd_buf, &push_info);

            let group_x = width.div_ceil(16);
            let group_y = height.div_ceil(16);
            device.cmd_dispatch(cmd_buf, group_x, group_y, 1);

            // Final barrier: Ensure post-process is done before the Blit/Transfer starts
            let final_barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(output_image.inner())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let dep_info = vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&final_barrier));
            device.cmd_pipeline_barrier2(cmd_buf, &dep_info);
        }

        Ok(())
    }
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
                vk::PipelineStageFlags2::TOP_OF_PIPE,
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
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
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

    /// Updates the local CPU copy of an object's transform
    #[deprecated = "use set_entity_transform with a proper EntityId"]
    pub fn set_object_transform(&mut self, instance_id: usize, transform: nalgebra::Matrix4<f32>) {
        // Vulkan expects a 3x4 row-major matrix for raytracing transforms
        let vk_transform = na_mat4_to_vk_transform(transform);

        let entity_id = vulkan_abstraction::EntityId(instance_id as u64);
        let _ = self.resource_manager.set_entity_transform(entity_id, vk_transform);
    }

    /// Call this ONCE per frame before `render_to_image` to update blasses that needs it
    pub fn rebuild_blasses(&mut self) -> SrResult<()> {
        self.resource_manager.update_tlas()
    }

    /// Call this ONCE per frame before `render_to_image`
    pub fn rebuild_tlas(&mut self) -> SrResult<()> {
        self.resource_manager.update_tlas()
    }
}

// useful environment variables, set to 1 or 0
const ENABLE_VALIDATION_LAYER_ENV_VAR: &str = "ENABLE_VALIDATION_LAYER"; // defaults to 0 in debug build, to 1 in release build
const ENABLE_GPUAV_ENV_VAR_NAME: &str = "ENABLE_GPUAV"; // does nothing unless validation layer is enabled, defaults to 0
const ENABLE_NVIDIA_AFTERMATH_VAR_NAME: &str = "ENABLE_NVIDIA_AFTERMATH"; // does nothing unless validation layer is enabled, defaults to 0

const ENABLE_SHADER_DEBUG_SYMBOLS_ENV_VAR: &str = "ENABLE_SHADER_DEBUG_SYMBOLS"; // defaults to 0 in debug build, to 1 in release build
const IS_DEBUG_BUILD: bool = cfg!(debug_assertions);

impl Drop for Renderer {
    fn drop(&mut self) {
        match self.core().graphics_queue().wait_idle() {
            Ok(()) => {}
            Err(e) => match e.get_source() {
                ErrorSource::Vulkan(e) => {
                    log::warn!("VkQueueWaitIdle s returned {e:?} in sunray::Renderer::drop")
                }
                _ => log::warn!("VkQueueWaitIdle returned {e} in sunray::Renderer::drop"),
            },
        }
    }
}
