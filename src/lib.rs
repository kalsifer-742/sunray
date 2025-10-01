pub mod camera;
pub mod error;
pub mod scene;
pub mod vulkan_abstraction;

pub use camera::*;
use error::*;
pub use scene::*;

use nalgebra as na;
use std::{collections::HashMap, rc::Rc};

use ash::vk;

struct UniformBufferContents {
    pub view_inverse: na::Matrix4<f32>,
    pub proj_inverse: na::Matrix4<f32>,
}

struct ImageDependentData {
    pub raytracing_cmd_buf: vulkan_abstraction::CmdBuffer,
    pub blit_cmd_buf: vulkan_abstraction::CmdBuffer,
    #[allow(unused)]
    pub raytrace_result_image: vulkan_abstraction::Image,

    #[allow(unused)]
    descriptor_sets: vulkan_abstraction::DescriptorSets,
}

pub struct Renderer {
    image_dependant_data: HashMap<vk::Image, ImageDependentData>,

    uniform_buffer: vulkan_abstraction::Buffer,
    blases: Vec<vulkan_abstraction::BLAS>,
    #[allow(unused)]
    tlas: vulkan_abstraction::TLAS,
    #[allow(unused)]
    shader_binding_table: vulkan_abstraction::ShaderBindingTable,
    #[allow(unused)]
    ray_tracing_pipeline: vulkan_abstraction::RayTracingPipeline,
    descriptor_set_layout: vulkan_abstraction::DescriptorSetLayout,
    image_extent: vk::Extent3D,
    image_format: vk::Format,

    core: Rc<vulkan_abstraction::Core>,
}

impl Drop for Renderer {
    fn drop(&mut self) {
        match self.core().queue().wait_idle() {
            Ok(()) => {}
            Err(e) => match e.get_source() {
                Some(ErrorSource::VULKAN(e)) => {
                    log::warn!("VkQueueWaitIdle s returned {e:?} in sunray::Renderer::drop")
                }
                _ => log::warn!("VkQueueWaitIdle returned {e} in sunray::Renderer::drop"),
            },
        }
    }
}

fn get_env_var_as_bool(name: &str) -> Option<bool> {
    match std::env::var(name) {
        Ok(s) => match s.parse::<i32>() {
            Ok(v) => Some(v != 0),
            Err(_) => None,
        },
        Err(_) => None,
    }
}

pub type CreateSurfaceFn = dyn Fn(&ash::Entry, &ash::Instance) -> SrResult<vk::SurfaceKHR>;

impl Renderer {
    // useful environment variables, set to 1 or 0
    const ENABLE_VALIDATION_LAYER_ENV_VAR: &'static str = "ENABLE_VALIDATION_LAYER"; // defaults to 0 in debug build, to 1 in release build
    const ENABLE_GPUAV_ENV_VAR_NAME: &'static str = "ENABLE_GPUAV"; // does nothing unless validation layer is enabled, defaults to 0
    const ENABLE_SHADER_DEBUG_SYMBOLS_ENV_VAR: &'static str = "ENABLE_SHADER_DEBUG_SYMBOLS"; // defaults to 0 in debug build, to 1 in release build
    const IS_DEBUG_BUILD: bool = cfg!(debug_assertions);

    pub fn new(image_extent: (u32, u32), image_format: vk::Format) -> SrResult<Self> {
        Ok(Self::new_impl(image_extent, image_format, &[], None)?.0)
    }

    pub fn resize(&mut self, image_extent: (u32, u32)) -> SrResult<()> {
        if image_extent.0 == self.image_extent.width && image_extent.1 == self.image_extent.height {
            return Ok(());
        }
        self.clear_image_dependent_data();
        self.image_extent.width = image_extent.0;
        self.image_extent.height = image_extent.1;

        Ok(())
    }

    // It is necessary to pass a function to create the surface, because surface depends on instance,
    // device depends on surface (if present), and both device and instance are created and owned inside
    // Renderer (in Core) so this seems to be the best approach to allow the user to build its own surface
    pub fn new_with_surface(
        image_extent: (u32, u32),
        image_format: vk::Format,
        instance_exts: &'static [*const i8],
        create_surface: &CreateSurfaceFn,
    ) -> SrResult<(Self, vk::SurfaceKHR)> {
        let (r, s) = Self::new_impl(
            image_extent,
            image_format,
            instance_exts,
            Some(create_surface),
        )?;

        return Ok((r, s.unwrap()));
    }

    fn new_impl(
        image_extent: (u32, u32),
        image_format: vk::Format,
        instance_exts: &'static [*const i8],
        create_surface: Option<&CreateSurfaceFn>,
    ) -> SrResult<(Self, Option<vk::SurfaceKHR>)> {
        let with_validation_layer = get_env_var_as_bool(Self::ENABLE_VALIDATION_LAYER_ENV_VAR)
            .unwrap_or(Self::IS_DEBUG_BUILD);
        let with_gpuav = get_env_var_as_bool(Self::ENABLE_GPUAV_ENV_VAR_NAME).unwrap_or(false);
        let (core, surface) = vulkan_abstraction::Core::new_with_surface(
            with_validation_layer,
            with_gpuav,
            image_format,
            instance_exts,
            create_surface,
        )?;
        let core = Rc::new(core);

        let image_extent = vk::Extent2D {
            width: image_extent.0,
            height: image_extent.1,
        }
        .into();

        let blases = vec![];
        let tlas = vulkan_abstraction::TLAS::new(Rc::clone(&core), &[])?;

        let uniform_buffer =
            vulkan_abstraction::Buffer::new_uniform::<UniformBufferContents>(Rc::clone(&core))?;

        let descriptor_set_layout = vulkan_abstraction::DescriptorSetLayout::new(Rc::clone(&core))?;

        let ray_tracing_pipeline = vulkan_abstraction::RayTracingPipeline::new(
            Rc::clone(&core),
            &descriptor_set_layout,
            get_env_var_as_bool(Self::ENABLE_SHADER_DEBUG_SYMBOLS_ENV_VAR)
                .unwrap_or(Self::IS_DEBUG_BUILD),
        )?;

        let shader_binding_table =
            vulkan_abstraction::ShaderBindingTable::new(&core, &ray_tracing_pipeline)?;

        let image_dependant_data = HashMap::new();

        Ok((
            Self {
                image_dependant_data,

                shader_binding_table,
                ray_tracing_pipeline,
                descriptor_set_layout,
                blases,
                tlas,
                uniform_buffer,
                image_extent,
                image_format,

                core,
            },
            surface,
        ))
    }

    pub fn clear_image_dependent_data(&mut self) {
        self.image_dependant_data = HashMap::new();
    }

    pub fn build_image_dependent_data(&mut self, images: &[vk::Image]) -> SrResult<()> {
        for post_blit_image in images {
            let raytrace_result_image = vulkan_abstraction::Image::new(
                Rc::clone(&self.core),
                self.image_extent,
                self.image_format,
                vk::ImageTiling::OPTIMAL,
                gpu_allocator::MemoryLocation::GpuOnly,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                "sunray (internal, pre-blit) raytrace result image",
            )?;

            let descriptor_sets = vulkan_abstraction::DescriptorSets::new(
                Rc::clone(&self.core),
                &self.descriptor_set_layout,
                &self.tlas,
                raytrace_result_image.image_view(),
                &self.uniform_buffer,
            )?;

            let blit_cmd_buf = vulkan_abstraction::CmdBuffer::new(Rc::clone(&self.core))?;
            let raytracing_cmd_buf = vulkan_abstraction::CmdBuffer::new(Rc::clone(&self.core))?;

            // record raytracing
            {
                let cmd_buf_begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);

                unsafe {
                    self.core
                        .device()
                        .inner()
                        .begin_command_buffer(raytracing_cmd_buf.inner(), &cmd_buf_begin_info)
                }?;

                Self::cmd_raytracing_render(
                    &self.core,
                    raytracing_cmd_buf.inner(),
                    &self.ray_tracing_pipeline,
                    &descriptor_sets,
                    &self.shader_binding_table,
                    raytrace_result_image.inner(),
                    raytrace_result_image.extent(),
                )?;

                unsafe {
                    self.core
                        .device()
                        .inner()
                        .end_command_buffer(raytracing_cmd_buf.inner())
                }?;
            }

            //record blit
            {
                let cmd_buf_begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);
                unsafe {
                    self.core
                        .device()
                        .inner()
                        .begin_command_buffer(blit_cmd_buf.inner(), &cmd_buf_begin_info)
                }?;

                Self::cmd_blit_image(
                    &self.core,
                    blit_cmd_buf.inner(),
                    raytrace_result_image.inner(),
                    raytrace_result_image.extent(),
                    *post_blit_image,
                    raytrace_result_image.image_subresource_range(),
                )?;

                unsafe {
                    self.core
                        .device()
                        .inner()
                        .end_command_buffer(blit_cmd_buf.inner())
                }?;
            }

            self.image_dependant_data.insert(
                *post_blit_image,
                ImageDependentData {
                    raytrace_result_image,
                    raytracing_cmd_buf,
                    blit_cmd_buf,
                    descriptor_sets,
                },
            );
        }

        Ok(())
    }

    pub fn load_gltf(&mut self, path: &str) -> SrResult<()> {
        let gltf = vulkan_abstraction::gltf::Gltf::new(Rc::clone(&self.core), path)?;
        let (default_scene_index, scenes, mut scenes_data) = gltf.create_scenes()?;
        let scene_data = scenes_data.get_mut(default_scene_index).unwrap();
        let default_scene = scenes.get(default_scene_index).unwrap();

        self.load_scene(default_scene, scene_data)?;
        Ok(())
    }

    pub fn load_scene(
        &mut self,
        scene: &Scene,
        scene_data: &mut vulkan_abstraction::gltf::PrimitiveDataMap,
    ) -> SrResult<()> {
        scene.load(&self.core, &mut self.tlas, &mut self.blases, scene_data)?;

        //TODO: update insted of recreating
        self.clear_image_dependent_data();

        Ok(())
    }

    pub fn set_camera(&mut self, camera: crate::Camera) -> SrResult<()> {
        let eye = camera.position();
        let target = camera.target();
        let up = &na::vector![0.0, 1.0, 0.0];

        //view-space: camera pov
        let view = na::Isometry3::look_at_rh(&eye, &target, &up);
        //clip_space: normalised coordinates adding perspective
        let projection = na::Perspective3::new(
            self.image_extent.width as f32 / self.image_extent.height as f32,
            camera.fov() * 3.14 / std::f32::consts::PI,
            0.1,   //render everything after this distance
            100.0, //discard everything after this distance
        );

        let mem = self.uniform_buffer.map::<crate::UniformBufferContents>()?;
        mem[0].view_inverse = view.to_homogeneous().try_inverse().unwrap(); //view_space -> world_space
        mem[0].proj_inverse = projection.to_homogeneous().try_inverse().unwrap(); //clip_space -> view_space

        Ok(())
    }

    /// Render to dst_image. the user may also pass a Semaphore which the user should signal when the image is
    /// ready to be written to (for example after being acquired from a swapchain) and a Fence will be returned
    /// that will be signaled when the rendering is finished (which can be used to know when the Semaphore has no pending operations left).
    pub fn render_to_image(
        &mut self,
        dst_image: vk::Image,
        wait_sem: vk::Semaphore,
    ) -> SrResult<vk::Fence> {
        if !self.image_dependant_data.contains_key(&dst_image) {
            // gracefully handle cache misses
            self.build_image_dependent_data(&[dst_image])?;
        }
        let img_dependent_data = self.image_dependant_data.get_mut(&dst_image).unwrap();

        // raytracing
        img_dependent_data.raytracing_cmd_buf.fence_mut().wait()?;
        img_dependent_data.raytracing_cmd_buf.fence_mut().reset()?;

        self.core.queue().submit_async(
            img_dependent_data.raytracing_cmd_buf.inner(),
            &[],
            &[],
            &[],
            img_dependent_data.raytracing_cmd_buf.fence_mut().submit(),
        )?;

        // blitting
        let wait_semaphores = [wait_sem];
        // ALL_GRAPHICS is fine, since literally all graphics (both barriers and blit) have to wait for the image to be available
        let wait_dst_stages = [vk::PipelineStageFlags::ALL_GRAPHICS];

        let (wait_sems, wait_dst_stages) = if wait_sem != vk::Semaphore::null() {
            (wait_semaphores.as_slice(), wait_dst_stages.as_slice())
        } else {
            ([].as_slice(), [].as_slice())
        };

        img_dependent_data.blit_cmd_buf.fence_mut().wait()?;
        img_dependent_data.blit_cmd_buf.fence_mut().reset()?;
        let signal_fence = img_dependent_data.blit_cmd_buf.fence_mut().submit();

        self.core.queue().submit_async(
            img_dependent_data.blit_cmd_buf.inner(),
            &wait_sems,
            &wait_dst_stages,
            &[],
            signal_fence,
        )?;

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

        unsafe {
            self.core
                .device()
                .inner()
                .wait_for_fences(&[wait_fence], true, u64::MAX)
        }?;

        let mem = Self::fix_image_memory_alignment(&self.core, &mut dst_image)?;

        Ok(mem)
    }

    fn cmd_raytracing_render(
        core: &vulkan_abstraction::Core,
        cmd_buf: vk::CommandBuffer,
        rt_pipeline: &vulkan_abstraction::RayTracingPipeline,
        descriptor_sets: &vulkan_abstraction::DescriptorSets,
        shader_binding_table: &vulkan_abstraction::ShaderBindingTable,
        image: vk::Image,
        extent: vk::Extent3D,
    ) -> SrResult<()> {
        let device = core.device().inner();
        // Initializing push constant values
        let push_constants = vulkan_abstraction::PushConstant {
            clear_color: [1.0, 0.0, 0.0, 1.0],
        };

        unsafe {
            vulkan_abstraction::cmd_image_memory_barrier(
                core,
                cmd_buf,
                image,
                vk::PipelineStageFlags::TOP_OF_PIPE, //wait nothing
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::AccessFlags::empty(),      //no writes to flush out
                vk::AccessFlags::SHADER_WRITE, //maybe also shader read is needed
                vk::ImageLayout::UNDEFINED,    //input is garbage
                vk::ImageLayout::GENERAL, //great for flexibility, and it should have good performance in all cases
            );

            device.cmd_bind_pipeline(
                cmd_buf,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                rt_pipeline.get_handle(),
            );
            device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                rt_pipeline.get_layout(),
                0,
                descriptor_sets.get_handles(),
                &[],
            );
            device.cmd_push_constants(
                cmd_buf,
                rt_pipeline.get_layout(),
                vk::ShaderStageFlags::RAYGEN_KHR
                    | vk::ShaderStageFlags::CLOSEST_HIT_KHR
                    | vk::ShaderStageFlags::MISS_KHR,
                0,
                &std::mem::transmute::<
                    vulkan_abstraction::PushConstant,
                    [u8; std::mem::size_of::<vulkan_abstraction::PushConstant>()],
                >(push_constants), //TODO: comment this transmute
            );
            core.rt_pipeline_device().cmd_trace_rays(
                cmd_buf,
                shader_binding_table.get_raygen_region(),
                shader_binding_table.get_miss_region(),
                shader_binding_table.get_hit_region(),
                shader_binding_table.get_callable_region(),
                extent.width,
                extent.height,
                extent.depth, //for now it's one because of the Extent2D.into()
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
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
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
                vk::PipelineStageFlags::ALL_GRAPHICS, // the image should already be transitioned when the used makes use of it
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

    fn fix_image_memory_alignment(
        core: &vulkan_abstraction::Core,
        image: &mut vulkan_abstraction::Image,
    ) -> SrResult<Vec<u8>> {
        //transform dst_image to bytes(correctly aligned)
        let image_sub = image.image_subresource_range();
        let image_subresource = vk::ImageSubresource {
            aspect_mask: image_sub.aspect_mask,
            mip_level: image_sub.base_mip_level,
            array_layer: image_sub.base_array_layer,
        };
        let subresource_layout = unsafe {
            core.device()
                .inner()
                .get_image_subresource_layout(image.inner(), image_subresource)
        };

        let size = image.extent().width as usize
            * image.extent().height as usize
            * std::mem::size_of::<u32>();
        let row_byte_size = image.extent().width as usize * std::mem::size_of::<u32>();
        let height = image.extent().height as usize;

        let mem = image.map()?;
        let mut row_pitch_corrected_mem: Vec<u8> = vec![0; size];

        let mut index = 0;
        let mut fixed_pitch_index = 0;

        for _ in 0..height {
            row_pitch_corrected_mem[index..index + row_byte_size]
                .copy_from_slice(&mem[fixed_pitch_index..fixed_pitch_index + row_byte_size]);

            fixed_pitch_index += subresource_layout.row_pitch as usize;
            index += row_byte_size;
        }

        Ok(row_pitch_corrected_mem)
    }

    pub fn core(&self) -> &Rc<vulkan_abstraction::Core> {
        &self.core
    }
}
