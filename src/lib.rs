pub mod camera;
pub mod error;
pub mod scene;
pub mod utils;
pub mod vulkan_abstraction;
pub mod render_graph;

pub use camera::*;
use error::*;
pub use scene::*;

use std::{collections::HashMap, rc::Rc};

use ash::vk;

use crate::utils::env_var_as_bool;
use crate::vulkan_abstraction::descriptor_sets::postprocess_descriptor_set::PostprocessDescriptorSetLayout;
use crate::vulkan_abstraction::descriptor_sets::temporal_accumulation_descriptor_set::TemporalAccumulationDescriptorSetLayout;
use crate::vulkan_abstraction::{DenoiseDescriptorSetLayout, DenoisePass, PostProcessDescriptorSets, PostprocessPass, Reservoir, ReservoirGI, TemporalPass};

pub const DENOISE_PASSES: u32 = 8;

pub const EXPOSURE: f32 = 1.0;

const MAX_TLAS_INSTANCES: usize = 10_000;

/// The number of concurrent frames that are processed (both by CPU and GPU).
///
/// Apparently 2 is the most common choice. Empirically it seems like the performance doesn't really
/// get any better with a higher number, but it does get measurably worse with only 1.
pub const MAX_FRAMES_IN_FLIGHT: usize = 4;

//TODO add a list of callbacks to call at the end of frames for cleanup or at start for setup
//TODO deferred deallocation for buffers and acceleration structures
struct ImageDependentData {
    pub raytracing_cmd_buf: vulkan_abstraction::CmdBuffer,
    pub blit_cmd_buf: vulkan_abstraction::CmdBuffer,
    #[allow(unused)]
    raytrace_result_image: vulkan_abstraction::Image,
    #[allow(unused)]
    postprocess_result_image: vulkan_abstraction::Image,
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

    core: Rc<vulkan_abstraction::Core>,

    //2 images to avoid race conditions when reading/writing
    pub accumulation_images: [vulkan_abstraction::Image; 2],
    pub denoising_images: [vulkan_abstraction::Image; 2],
    ///this is used for temporal accumulation, there is an absolute frame counter in the core
    pub relative_frame_count: u32,

    prev_view_proj: nalgebra::Matrix4<f32>, //used to calculate motion vectors
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
        let (core, surface) = vulkan_abstraction::Core::new_with_surface(
            with_validation_layer,
            with_gpuav,
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

        let ray_tracing_pipeline_ris = vulkan_abstraction::RayTracingPipeline::new(
            Rc::clone(&core),
            &ray_tracing_descriptor_set_layout,
            env_var_as_bool(ENABLE_SHADER_DEBUG_SYMBOLS_ENV_VAR).unwrap_or(IS_DEBUG_BUILD),
            include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/ray_gen_ris.spirv")),
        )?;

        let shader_binding_table_ris = vulkan_abstraction::ShaderBindingTable::new(&core, &ray_tracing_pipeline_ris)?;

        let ray_tracing_pipeline_final = vulkan_abstraction::RayTracingPipeline::new(
            Rc::clone(&core),
            &ray_tracing_descriptor_set_layout,
            env_var_as_bool(ENABLE_SHADER_DEBUG_SYMBOLS_ENV_VAR).unwrap_or(IS_DEBUG_BUILD),
            include_bytes_align_as!(u32, concat!(env!("OUT_DIR"), "/ray_gen_final.spirv")),
        )?;

        let shader_binding_table_final = vulkan_abstraction::ShaderBindingTable::new(&core, &ray_tracing_pipeline_final)?;

        let temporal_accumulation_pipeline = vulkan_abstraction::ComputePipeline::<TemporalPass>::new(
            Rc::clone(&core),
            temporal_accumulation_descriptor_set_layout.inner(),
        )?;

        let denoise_pipeline =
            vulkan_abstraction::ComputePipeline::<DenoisePass>::new(Rc::clone(&core), denoise_descriptor_set_layout.inner())?;

        let postprocess_pipeline = vulkan_abstraction::ComputePipeline::<PostprocessPass>::new(
            Rc::clone(&core),
            postprocess_descriptor_set_layout.inner(),
        )?;

        let image_dependant_data = HashMap::new();

        let create_accum_image = |name: &'static str| -> SrResult<vulkan_abstraction::Image> {
            vulkan_abstraction::Image::new(
                core.clone(),
                image_extent, // <--- USE THIS (it's already a vk::Extent3D)
                vk::Format::B10G11R11_UFLOAT_PACK32,
                vk::ImageTiling::OPTIMAL,
                gpu_allocator::MemoryLocation::GpuOnly,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                name,
            )
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
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "ReSTIR Reservoir Buffer A",
        )?;

        let reservoir_buffer_b = vulkan_abstraction::GpuOnlyBuffer::new::<Reservoir>(
            Rc::clone(&core),
            num_pixels as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "ReSTIR Reservoir Buffer B",
        )?;
        let reservoir_buffers = [reservoir_buffer_a, reservoir_buffer_b];

        let reservoir_gi_buffer_a = vulkan_abstraction::GpuOnlyBuffer::new::<ReservoirGI>(
            Rc::clone(&core),
            num_pixels  as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "ReSTIR GI Reservoir Buffer A",
        )?;

        let reservoir_gi_buffer_b = vulkan_abstraction::GpuOnlyBuffer::new::<ReservoirGI>(
            Rc::clone(&core),
            num_pixels as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
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

        Ok((
            Self {
                image_dependant_data,

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
        self.clear_image_dependent_data();

        let num_pixels = (new_extent.width * new_extent.height) as usize;
        let reservoir_buffer_a = vulkan_abstraction::GpuOnlyBuffer::new::<Reservoir>(
            self.core.clone(),
            num_pixels as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "ReSTIR Reservoir Buffer A",
        )?;

        let reservoir_buffer_b = vulkan_abstraction::GpuOnlyBuffer::new::<Reservoir>(
            self.core.clone(),
            num_pixels as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "ReSTIR Reservoir Buffer B",
        )?;

        self.reservoir_buffers = [reservoir_buffer_a, reservoir_buffer_b];

        let reservoir_gi_buffer_a = vulkan_abstraction::GpuOnlyBuffer::new::<ReservoirGI>(
            self.core.clone(),
            num_pixels  as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "ReSTIR GI Reservoir Buffer A",
        )?;

        let reservoir_gi_buffer_b = vulkan_abstraction::GpuOnlyBuffer::new::<ReservoirGI>(
            self.core.clone(),
            num_pixels as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "ReSTIR GI Reservoir Buffer B",
        )?;

        self.reservoir_gi_buffers = [reservoir_gi_buffer_a, reservoir_gi_buffer_b];

        self.image_extent = new_extent;

        let create_accum_image = |name: &'static str| -> SrResult<vulkan_abstraction::Image> {
            vulkan_abstraction::Image::new(
                self.core.clone(),
                new_extent,
                vk::Format::B10G11R11_UFLOAT_PACK32,
                vk::ImageTiling::OPTIMAL,
                gpu_allocator::MemoryLocation::GpuOnly,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                name,
            )
        };

        self.accumulation_images = [create_accum_image("Accumulation_1")?, create_accum_image("Accumulation_2")?];

        self.denoising_images = [create_accum_image("Denoising_1")?, create_accum_image("Denoising_2")?];

        let device = self.core.device().inner();
        let mut setup_cmd_buf = vulkan_abstraction::CmdBuffer::new(self.core.clone())?;

        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(setup_cmd_buf.inner(), &begin_info)?;

            let create_barrier = |image: vk::Image| {
                vk::ImageMemoryBarrier::default()
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
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ)
            };

            let barriers = [
                create_barrier(self.accumulation_images[0].inner()),
                create_barrier(self.accumulation_images[1].inner()),
                create_barrier(self.denoising_images[0].inner()),
                create_barrier(self.denoising_images[1].inner()),
            ];

            device.cmd_pipeline_barrier(
                setup_cmd_buf.inner(),
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );

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

            let postprocess_result_image = vulkan_abstraction::Image::new(
                Rc::clone(&self.core),
                self.image_extent,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageTiling::OPTIMAL,
                gpu_allocator::MemoryLocation::GpuOnly,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
                "sunray (internal, pre-blit) postprocess result image",
            )?;

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
                        vk::ImageMemoryBarrier::default()
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
                            .src_access_mask(vk::AccessFlags::empty())
                            .dst_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ)
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

                    device.cmd_pipeline_barrier(
                        setup_cmd_buf.inner(),
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR
                            | vk::PipelineStageFlags::COMPUTE_SHADER
                            | vk::PipelineStageFlags::TRANSFER, // Added TRANSFER for the blit cmd buf later
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &barriers,
                    );

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

            let temporal_accumulation_descriptor_sets = vulkan_abstraction::TemporalAccumulationDescriptorSets::new(
                &self.core, // Passed as &Rc<Core> based on our previous struct signature
                &self.temporal_accumulation_descriptor_set_layout,
                &raytrace_result_image,    // Binding 0: Noisy Input
                &motion_vector_image,      // Binding 1: Motion Vectors
                &self.accumulation_images, // Binding 2: Ping-Pong Output (Storage)
                &self.accumulation_images, // Binding 3: Ping-Pong History (Samplers)
                self.resource_manager.default_sampler().inner(),
            )?;

            let denoise_descriptor_sets = vulkan_abstraction::DenoiseDescriptorSets::new(
                Rc::clone(&self.core),
                &self.denoise_descriptor_set_layout,
                &self.accumulation_images,
                &depth_image,
                &normal_image,
                &diffuse_image,
                &self.denoising_images,
                self.resource_manager.default_sampler().inner(),
            )?;

            let postprocess_descriptor_sets = vulkan_abstraction::PostProcessDescriptorSets::new(
                Rc::clone(&self.core),
                &self.postprocess_descriptor_set_layout,
                &self.denoising_images,
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

    fn na_mat4_to_vk_transform(m: nalgebra::Matrix4<f32>) -> vk::TransformMatrixKHR {
        vk::TransformMatrixKHR {
            matrix: [
                m.m11, m.m12, m.m13, m.m14,
                m.m21, m.m22, m.m23, m.m24,
                m.m31, m.m32, m.m33, m.m34,
            ],
        }
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
        let vk_transform = Self::na_mat4_to_vk_transform(transform);
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
    pub fn set_entity_transform(
        &mut self,
        id: vulkan_abstraction::EntityId,
        transform: nalgebra::Matrix4<f32>,
    ) -> SrResult<()> {
        let vk_transform = Self::na_mat4_to_vk_transform(transform);
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

        let this_ptr = self as *mut Self;

        let img_dependent_data = self.image_dependant_data.get_mut(&dst_image).unwrap();

        // Raytracing
        img_dependent_data.raytracing_cmd_buf.fence_mut().wait()?;
        img_dependent_data.blit_cmd_buf.fence_mut().wait()?;

        let cmd_buf = img_dependent_data.raytracing_cmd_buf.inner();
        let result_image = img_dependent_data.raytrace_result_image.inner();
        let motion_vector_image = img_dependent_data.motion_vector_image.inner();
        let postprocessed_image = img_dependent_data.postprocess_result_image.inner();
        let result_extent = img_dependent_data.raytrace_result_image.extent();

        let raytracing_descriptor_sets_ptr =
            &img_dependent_data.raytracing_descriptor_sets as *const vulkan_abstraction::RaytracingDescriptorSets;
        let temporal_accumulation_descriptor_sets_ptr = &img_dependent_data.temporal_accumulation_descriptor_sets
            as *const vulkan_abstraction::TemporalAccumulationDescriptorSets;
        let denoise_descriptor_sets_ptr =
            &img_dependent_data.denoise_descriptor_sets as *const vulkan_abstraction::DenoiseDescriptorSets;
        let postprocess_descriptor_sets_ptr =
            &img_dependent_data.postprocess_descriptor_sets as *const vulkan_abstraction::PostProcessDescriptorSets;

        unsafe {
            // Use (*this_ptr).core because 'self.core' is locked by 'img_dependent_data'
            let device = (*this_ptr).core.device().inner();

            // Supponiamo che tu abbia salvato il semaforo restituito dalla transfer queue
            // in un campo opzionale `self.pending_transfer_semaphore`
            let wait_semaphores = (*this_ptr).core.transfer_semaphores_mut().drain(..).collect::<Vec<_>>();
            // IMPORTANTE: Diciamo alla GPU di aspettare PRIMA di lanciare i Ray Tracing Shader
            // Se la TLAS viene aggiornata in questo cmd buffer, aggiungi anche ACCELERATION_STRUCTURE_BUILD_KHR
            let wait_stages = wait_semaphores
                .iter()
                .map(|semaphore| {
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR | vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
                })
                .collect::<Vec<_>>();

            let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            device.begin_command_buffer(cmd_buf, &begin_info)?;

            (*this_ptr).cmd_raytracing_render(
                cmd_buf,
                &*raytracing_descriptor_sets_ptr, // Dereference back to reference
                result_image,
                result_extent,
            )?;

            let memory_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE) // Wait for RT to finish writing
                .dst_access_mask(vk::AccessFlags::SHADER_READ); // Make sure Denoise can see the data

            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR, // Source Stage
                vk::PipelineStageFlags::COMPUTE_SHADER,         // Destination Stage (Denoising is usually compute)
                vk::DependencyFlags::empty(),
                &[memory_barrier], // Use a global memory barrier for simplicity
                &[],
                &[],
            );

            let read_only_barriers = [
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(img_dependent_data.depth_image.inner())
                    .subresource_range(*img_dependent_data.depth_image.image_subresource_range()),
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(img_dependent_data.normal_image.inner())
                    .subresource_range(*img_dependent_data.normal_image.image_subresource_range()),
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(img_dependent_data.diffuse_image.inner())
                    .subresource_range(*img_dependent_data.diffuse_image.image_subresource_range()),
            ];

            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[], // Remove the global memory_barrier you had here
                &[],
                &read_only_barriers, // Add these image barriers
            );

            (*this_ptr).cmd_temporal_accumulation(
                cmd_buf,
                &*temporal_accumulation_descriptor_sets_ptr,
                result_extent.width,
                result_extent.height,
                result_image,        // Raw RT output
                motion_vector_image, // From G-buffer
                &self.accumulation_images,
            )?;

            //let temporal_barrier_1 = vk::MemoryBarrier::

            // 5. Denoise Ping-Pong Loop
            (*this_ptr).cmd_denoise_image(
                cmd_buf,
                &*denoise_descriptor_sets_ptr,
                result_extent.width,
                result_extent.height,
                &self.accumulation_images,
                &self.denoising_images,
            )?;

            // 6. Post-process
            // Use result of the last denoise pass
            let final_denoise_idx = (DENOISE_PASSES % 2) as usize;

            (*this_ptr).cmd_postprocess_image(
                cmd_buf,
                &*postprocess_descriptor_sets_ptr,
                result_extent.width,
                result_extent.height,
                //result_image,
                self.denoising_images[0].inner(),
                postprocessed_image,
                final_denoise_idx,
            )?;

            let return_to_general_barriers = [
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_READ)
                    .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(img_dependent_data.depth_image.inner())
                    .subresource_range(*img_dependent_data.depth_image.image_subresource_range()),
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_READ)
                    .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(img_dependent_data.normal_image.inner())
                    .subresource_range(*img_dependent_data.normal_image.image_subresource_range()),
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_READ)
                    .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(img_dependent_data.diffuse_image.inner())
                    .subresource_range(*img_dependent_data.diffuse_image.image_subresource_range()),
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_READ)
                    .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(img_dependent_data.raytrace_result_image.inner())
                    .subresource_range(*img_dependent_data.raytrace_result_image.image_subresource_range()),
            ];

            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::COMPUTE_SHADER, // Wait for Denoise to finish reading
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR, // Block next frame's RT from writing early
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &return_to_general_barriers,
            );

            device.end_command_buffer(cmd_buf)?;

            // Submit using (*this_ptr)
            (*this_ptr).core.graphics_queue().submit_async(
                img_dependent_data.raytracing_cmd_buf.inner(),
                &wait_semaphores,
                &wait_stages,
                &[img_dependent_data.raytracing_finished_semaphore.inner()],
                img_dependent_data.raytracing_cmd_buf.fence_mut().submit()?,
            )?;
        }

        // Blitting
        let (wait_sems, wait_dst_stages) = (
            [wait_sem, img_dependent_data.raytracing_finished_semaphore.inner()],
            [vk::PipelineStageFlags::ALL_GRAPHICS, vk::PipelineStageFlags::TRANSFER],
        );
        let (wait_sems, wait_dst_stages) = if wait_sem == vk::Semaphore::null() {
            ([].as_slice(), [].as_slice())
        } else {
            (wait_sems.as_slice(), wait_dst_stages.as_slice())
        };

        let signal_fence = img_dependent_data.blit_cmd_buf.fence_mut().submit()?;

        unsafe {
            // Again, use (*this_ptr) because img_dependent_data is still alive here
            (*this_ptr).core.graphics_queue().submit_async(
                img_dependent_data.blit_cmd_buf.inner(),
                &wait_sems,
                &wait_dst_stages,
                &[],
                signal_fence,
            )?;
        }

        Ok(signal_fence)
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

        let wait_fence = self.render_to_image(dst_image.inner(), vk::Semaphore::null())?;
        vulkan_abstraction::wait_fence(self.core.device(), wait_fence)?;

        Ok(dst_image.get_raw_image_data_with_no_padding()?)
    }

    fn cmd_raytracing_render(
        &mut self,
        cmd_buf: vk::CommandBuffer,
        descriptor_sets: &vulkan_abstraction::RaytracingDescriptorSets,
        image: vk::Image,
        extent: vk::Extent3D,
    ) -> SrResult<()> {
        let device = self.core.device().inner();

        //ping pong to avoid errors when using accumulation images
        let history_idx = (self.relative_frame_count % 2) as usize;
        let accum_idx = ((self.relative_frame_count + 1) % 2) as usize;

        //TODO finni removed this in his branch

        //  // Use GENERAL for everything to rule out layout mismatches
        // let (old_layout, src_stage, src_access) = if self.frame_count == 0 {
        //     (vk::ImageLayout::UNDEFINED, vk::PipelineStageFlags::TOP_OF_PIPE, vk::AccessFlags::empty())
        // } else {
        //     (vk::ImageLayout::GENERAL, vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR, vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ)
        // };

        // Initializing push constant values
        let push_constants = vulkan_abstraction::RaytracingPushConstant {
            frame_count: self.relative_frame_count,
            use_srgb: self.image_format == vk::Format::R8G8B8A8_SRGB,
            _padding: [0; 3],
        };

        // Extract the bytes once to reuse for both pipelines
        let push_constant_bytes = unsafe {
            std::mem::transmute::<
                vulkan_abstraction::RaytracingPushConstant,
                [u8; std::mem::size_of::<vulkan_abstraction::RaytracingPushConstant>()],
            >(push_constants)
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

            let make_barrier = |img: vk::Image, old: vk::ImageLayout, new: vk::ImageLayout, src_a, dst_a| {
                vk::ImageMemoryBarrier::default()
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
                vk::AccessFlags::empty(),
                vk::AccessFlags::SHADER_WRITE,
            );

            let (hist_old, hist_src) = if self.relative_frame_count == 1 {
                (vk::ImageLayout::UNDEFINED, vk::AccessFlags::empty())
            } else {
                (vk::ImageLayout::GENERAL, vk::AccessFlags::SHADER_WRITE)
            };

            let b_hist = make_barrier(
                self.accumulation_images[history_idx].inner(),
                hist_old,
                vk::ImageLayout::GENERAL,
                hist_src,
                vk::AccessFlags::SHADER_READ,
            );

            let (accum_old, accum_src) = if self.relative_frame_count == 1 {
                (vk::ImageLayout::UNDEFINED, vk::AccessFlags::empty())
            } else {
                (vk::ImageLayout::GENERAL, vk::AccessFlags::SHADER_READ)
            };

            let b_accum = make_barrier(
                self.accumulation_images[accum_idx].inner(),
                accum_old,
                vk::ImageLayout::GENERAL,
                accum_src,
                vk::AccessFlags::SHADER_WRITE,
            );

            device.cmd_pipeline_barrier(
                cmd_buf,
                // Frame 1: Wait for nothing. Frame 2+: Wait for previous Ray Tracing.
                if self.relative_frame_count == 1 {
                    vk::PipelineStageFlags::TOP_OF_PIPE
                } else {
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR
                },
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[b_swap, b_hist, b_accum],
            );

            // --- PASS 1: RIS Audition ---
            device.cmd_bind_pipeline(
                cmd_buf,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.ray_tracing_pipeline_ris.inner(),
            );

            device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.ray_tracing_pipeline_ris.layout(),
                0,
                descriptor_sets.inner(),
                &[],
            );

            device.cmd_push_constants(
                cmd_buf,
                self.ray_tracing_pipeline_ris.layout(),
                vk::ShaderStageFlags::RAYGEN_KHR
                    | vk::ShaderStageFlags::CLOSEST_HIT_KHR
                    | vk::ShaderStageFlags::MISS_KHR
                    | vk::ShaderStageFlags::ANY_HIT_KHR,
                0,
                &push_constant_bytes,
            );

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

            let reservoir_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);

            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                &[reservoir_barrier],
                &[],
                &[],
            );

            device.cmd_bind_pipeline(
                cmd_buf,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.ray_tracing_pipeline_final.inner(),
            );

            device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.ray_tracing_pipeline_final.layout(),
                0,
                descriptor_sets.inner(),
                &[],
            );

            device.cmd_push_constants(
                cmd_buf,
                self.ray_tracing_pipeline_final.layout(),
                vk::ShaderStageFlags::RAYGEN_KHR
                    | vk::ShaderStageFlags::CLOSEST_HIT_KHR
                    | vk::ShaderStageFlags::MISS_KHR
                    | vk::ShaderStageFlags::ANY_HIT_KHR,
                0,
                &push_constant_bytes,
            );

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
        accumulation_images: &[vulkan_abstraction::image::Image; 2],
    ) -> SrResult<()> {
        let device = self.core.device().inner();

        let history_idx = (self.relative_frame_count % 2) as usize;
        let accum_idx = ((self.relative_frame_count + 1) % 2) as usize;

        // 1. Prepare inputs (RT Image and Motion Vectors)
        let rt_barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
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

        let mv_barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE) // Assuming written in G-Buffer pass
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
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
        let write_barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty()) // Don't care what it was doing before
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
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
            vk::AccessFlags::empty()
        } else {
            vk::AccessFlags::SHADER_WRITE
        };

        // The one we read from (History):
        let read_barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(history_src_access) // Written to last frame
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
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
            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR, // Wait for RT
                vk::PipelineStageFlags::COMPUTE_SHADER,         // Block Temporal
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[rt_barrier, mv_barrier, write_barrier, read_barrier],
            );

            device.cmd_bind_pipeline(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.temporal_accumulation_pipeline.inner(),
            );

            device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.temporal_accumulation_pipeline.layout(),
                0,
                descriptor_sets.inner(), // Has all bindings combined
                &[],
            );

            device.cmd_push_constants(
                cmd_buf,
                self.temporal_accumulation_pipeline.layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                &self.relative_frame_count.to_ne_bytes(),
            );

            let group_x = (width + 15) / 16;
            let group_y = (height + 15) / 16;
            device.cmd_dispatch(cmd_buf, group_x, group_y, 1);
        }

        Ok(())
    }

    ///This function takes 2 arrays of 2 images: the first are the temporal accumulation results, from which it only takes the image the first time
    /// (accumulated_image -> first denoise pass on denoise_pingpong_images\[0])
    /// All the next steps are done in the denoise_pingpong_images

    fn cmd_denoise_image(
        &self,
        cmd_buf: vk::CommandBuffer,
        descriptor_sets: &vulkan_abstraction::DenoiseDescriptorSets,
        width: u32,
        height: u32,
        input_images: &[vulkan_abstraction::Image; 2], //used in the first pass (getting from TAA to denoise)
        denoise_pingpong_images: &[vulkan_abstraction::Image; 2],
    ) -> SrResult<()> {
        let device = self.core.device().inner();
        let total_passes = DENOISE_PASSES;

        // Determine which accumulation image is the "latest" history to read from
        let history_idx = (self.relative_frame_count % 2) as usize;

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
            let read_barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
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
            let write_barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty()) // Or SHADER_READ if it was a source in a previous pass
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
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
            };

            unsafe {
                // Compute-to-Compute barrier ensures execution order and cache visibility
                device.cmd_pipeline_barrier(
                    cmd_buf,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[read_barrier, write_barrier],
                );

                device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, self.denoise_pipeline.inner());

                device.cmd_bind_descriptor_sets(
                    cmd_buf,
                    vk::PipelineBindPoint::COMPUTE,
                    self.denoise_pipeline.layout(),
                    0,
                    &[descriptor_sets.inner()[descriptor_idx]],
                    &[],
                );

                device.cmd_push_constants(
                    cmd_buf,
                    self.denoise_pipeline.layout(),
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &std::mem::transmute::<
                        vulkan_abstraction::DenoisePushConstant,
                        [u8; std::mem::size_of::<vulkan_abstraction::DenoisePushConstant>()],
                    >(push_constants),
                );

                let group_x = (width + 15) / 16;
                let group_y = (height + 15) / 16;
                device.cmd_dispatch(cmd_buf, group_x, group_y, 1);
            }
        }

        Ok(())
    }

    fn cmd_postprocess_image(
        &self,
        cmd_buf: vk::CommandBuffer,
        descriptor_sets: &vulkan_abstraction::PostProcessDescriptorSets,
        width: u32,
        height: u32,
        input_image: vk::Image,
        output_image: vk::Image,
        final_denoise_idx: usize, // ADDED: to track which descriptor set to bind
    ) -> SrResult<()> {
        let device = self.core.device().inner();

        let push_constants = vulkan_abstraction::PostprocessPushConstant { exposure: EXPOSURE };

        // 1. Synchronize: Wait for Denoise (Compute) to finish writing to input_image
        let input_barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(input_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        // Ensure final output image is ready
        let output_barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED) // CHANGED: Blit leaves this in GENERAL, so match it here
            .new_layout(vk::ImageLayout::GENERAL)
            .image(output_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        unsafe {
            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::COMPUTE_SHADER, // Wait for Denoise
                vk::PipelineStageFlags::COMPUTE_SHADER, // Block Post-process
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[input_barrier, output_barrier],
            );

            device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, self.postprocess_pipeline.inner());

            // CHANGED: Bind specifically Set 0 or Set 1 based on the denoise ping-pong result
            device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.postprocess_pipeline.layout(),
                0,
                &[descriptor_sets.inner()[final_denoise_idx]],
                &[],
            );

            device.cmd_push_constants(
                cmd_buf,
                self.denoise_pipeline.layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                &std::mem::transmute::<
                    vulkan_abstraction::PostprocessPushConstant,
                    [u8; std::mem::size_of::<vulkan_abstraction::PostprocessPushConstant>()],
                >(push_constants),
            );

            // Dispatch post-process shader
            let group_x = (width + 15) / 16;
            let group_y = (height + 15) / 16;
            device.cmd_dispatch(cmd_buf, group_x, group_y, 1);

            // Final barrier: Ensure post-process is done before the Blit/Transfer starts
            let final_barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(output_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[final_barrier],
            );
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
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::TRANSFER_READ,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );

            //transition dst_image to transfer destination layout
            vulkan_abstraction::cmd_image_memory_barrier(
                core,
                cmd_buf,
                dst_image,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
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
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_GRAPHICS, // the image should already be transitioned when the user makes use of it
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::MEMORY_READ,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::GENERAL,
            );

            //transition back src_image to general layout
            vulkan_abstraction::cmd_image_memory_barrier(
                core,
                cmd_buf,
                src_image,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::AccessFlags::TRANSFER_READ,
                vk::AccessFlags::empty(),
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
        let vk_transform = vk::TransformMatrixKHR {
            matrix: [
                transform.m11,
                transform.m12,
                transform.m13,
                transform.m14,
                transform.m21,
                transform.m22,
                transform.m23,
                transform.m24,
                transform.m31,
                transform.m32,
                transform.m33,
                transform.m34,
            ],
        };

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
const ENABLE_VALIDATION_LAYER_ENV_VAR: &'static str = "ENABLE_VALIDATION_LAYER"; // defaults to 0 in debug build, to 1 in release build
const ENABLE_GPUAV_ENV_VAR_NAME: &'static str = "ENABLE_GPUAV"; // does nothing unless validation layer is enabled, defaults to 0
const ENABLE_SHADER_DEBUG_SYMBOLS_ENV_VAR: &'static str = "ENABLE_SHADER_DEBUG_SYMBOLS"; // defaults to 0 in debug build, to 1 in release build
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
