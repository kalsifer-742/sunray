extern crate shaderc;

use std::{rc::Rc};

use crate::error::*;
use ash::{ khr, vk };
use winit::raw_window_handle_05::{RawDisplayHandle, RawWindowHandle};

pub mod error;
pub mod utils;
mod vulkan_abstraction;

#[allow(dead_code)]
struct UniformBufferContents {
    pub view_inverse: nalgebra::Matrix4<f32>,
    pub proj_inverse: nalgebra::Matrix4<f32>,
}

fn make_view_inverse_matrix() -> nalgebra::Matrix4<f32> {
    let eye = nalgebra::geometry::Point3::new(0.0, 0.0, 3.0);
    let target = nalgebra::geometry::Point3::new(0.0, 0.0, 0.0);
    let up = nalgebra::Vector3::new(0.0, -1.0, 0.0);
    //apparently vulkan uses right-handed coordinates
    let view = nalgebra::Isometry3::look_at_rh(&eye, &target, &up);

    let view_matrix : nalgebra::Matrix4<f32> = view.to_homogeneous();

    view_matrix.try_inverse().unwrap()
}

fn make_proj_inverse_matrix(dimensions: (u32, u32)) -> nalgebra::Matrix4<f32> {
    let proj = nalgebra::geometry::Perspective3::new(dimensions.0 as f32 / dimensions.1 as f32, 3.14 / 2.0, 0.1, 1000.0);

    let proj = proj.to_homogeneous();

    proj.try_inverse().unwrap()
}

#[allow(dead_code)]
pub struct Renderer {
    core: Rc<vulkan_abstraction::Core>,

    vertex_buffer: vulkan_abstraction::VertexBuffer,
    index_buffer: vulkan_abstraction::IndexBuffer,
    blas: vulkan_abstraction::BLAS,
    tlas: vulkan_abstraction::TLAS,
    image: vk::Image,
    image_device_memory: vk::DeviceMemory,
    image_view: vk::ImageView,
    uniform_buffer: vulkan_abstraction::Buffer,
    descriptor_sets: vulkan_abstraction::DescriptorSets,
    ray_tracing_pipeline: vulkan_abstraction::RayTracingPipeline,
    shader_binding_table: vulkan_abstraction::ShaderBindingTable,
}

fn get_env_var_as_bool(name: &str) -> Option<bool> {
    match std::env::var(name) {
        Ok(s) => {
            match s.parse::<i32>() {
                Ok(v) => Some(v != 0),
                Err(_) => None,
            }
        }
        Err(_) => None,
    }
}

impl Renderer {
    // useful environment variables, set to 1 or 0
    // TODO: switch to program arguments (safeguards against typos, and allows for a short explanation in --help)
    const ENABLE_VALIDATION_LAYER_ENV_VAR: &'static str = "ENABLE_VALIDATION_LAYER"; // defaults to 0 in debug build, to 1 in release build
    const ENABLE_GPUAV_ENV_VAR_NAME: &'static str = "ENABLE_GPUAV"; // does nothing unless validation layer is enabled, defaults to 0
    const ENABLE_SHADER_DEBUG_SYMBOLS_ENV_VAR: &'static str = "ENABLE_SHADER_DEBUG_SYMBOLS"; // defaults to 0 in debug build, to 1 in release build
    const IS_DEBUG_BUILD: bool = cfg!(debug_assertions);

    // TODO: currently take for granted that the user has a window, no support for offline rendering
    pub fn new(window_extent: [u32; 2], raw_window_handle: RawWindowHandle, raw_display_handle: RawDisplayHandle) -> SrResult<Self> {

        let core = Rc::new(vulkan_abstraction::Core::new(vulkan_abstraction::CoreCreateInfo {
            instance_exts: crate::utils::enumerate_required_extensions(raw_display_handle)?,
            device_exts: &[ khr::swapchain::NAME.as_ptr() ],

            with_swapchain: true,
            window_extent: Some(window_extent),
            raw_window_handle: Some(raw_window_handle),
            raw_display_handle: Some(raw_display_handle),

            with_validation_layer: get_env_var_as_bool(Self::ENABLE_VALIDATION_LAYER_ENV_VAR).unwrap_or(Self::IS_DEBUG_BUILD),
            with_gpu_assisted_validation: get_env_var_as_bool(Self::ENABLE_GPUAV_ENV_VAR_NAME).unwrap_or(false),
        })?);
        let device = core.device().inner();


        let vertex_buffer = {
            #[derive(Clone, Copy)]
            struct Vertex {
                #[allow(unused)]
                pos: [f32; 3],
            }

            let verts = [
                Vertex { pos: [-1.0, -0.5, 0.0] },
                Vertex { pos: [1.0, -0.5, 0.0] },
                Vertex { pos: [0.0, 1.0, 0.0] },
            ];
            let staging_buffer = vulkan_abstraction::Buffer::new_staging_from_data::<Vertex>(
                Rc::clone(&core),
                &verts,
            )?;
            let vertex_buffer = vulkan_abstraction::VertexBuffer::new_for_blas::<Vertex>(
                Rc::clone(&core),
                verts.len(),
            )?;
            vulkan_abstraction::Buffer::clone_buffer(&core, &staging_buffer, &vertex_buffer)?;

            vertex_buffer
        };
        let index_buffer = {
            let indices : [u32; 3] = [0, 1, 2];
            let staging_buffer = vulkan_abstraction::Buffer::new_staging_from_data::<u32>(
                Rc::clone(&core),
                &indices,
            )?;
            let index_buffer = vulkan_abstraction::IndexBuffer::new_for_blas::<u32>(
                Rc::clone(&core),
                indices.len(),
            )?;
            vulkan_abstraction::Buffer::clone_buffer(&core, &staging_buffer, &index_buffer)?;

            index_buffer
        };

        let blas = vulkan_abstraction::BLAS::new(
            Rc::clone(&core),
            &vertex_buffer,
            &index_buffer,
        )?;


        let tlas = vulkan_abstraction::TLAS::new(
            Rc::clone(&core),
            &[&blas],
        )?;

        const OUT_IMAGE_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

        // the image we will do the rendering on; before every frame it will be copied to the swapchain
        let (image, image_device_memory, image_view) = {
            let image = {
                let image_create_info = vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(OUT_IMAGE_FORMAT)
                    .extent(core.swapchain().image_extent().into())
                    .flags(vk::ImageCreateFlags::empty())
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(
                        vk::ImageUsageFlags::STORAGE
                        | vk::ImageUsageFlags::TRANSFER_SRC,
                    )
                    .initial_layout(vk::ImageLayout::UNDEFINED);

                unsafe { device.create_image(&image_create_info, None) }.unwrap()
            };

            let image_device_memory = {
                let mem_reqs = unsafe { device.get_image_memory_requirements(image) };
                let mem_alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_reqs.size)
                .memory_type_index(vulkan_abstraction::get_memory_type_index(
                    &core,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    &mem_reqs,
                )?);

                unsafe { device.allocate_memory(&mem_alloc_info, None) }.unwrap()
            };

            unsafe { device.bind_image_memory(image, image_device_memory, 0) }.unwrap();

            let image_view = {
                let image_view_create_info = vk::ImageViewCreateInfo::default()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(OUT_IMAGE_FORMAT)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image(image);

                unsafe { device.create_image_view(&image_view_create_info, None) }.unwrap()
            };

            //switch the ImageLayout from UNDEFINED TO GENERAL
            {
                let image_barrier_cmd_buf = vulkan_abstraction::cmd_buffer::new(&*core.cmd_pool(), core.device())?;

                let begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

                //record command buffer
                unsafe {
                    device.begin_command_buffer(image_barrier_cmd_buf, &begin_info)?;

                    let stage_all = vk::PipelineStageFlags::ALL_COMMANDS;
                    Self::cmd_image_memory_barrier(&core, image_barrier_cmd_buf, image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL, stage_all, stage_all, vk::AccessFlags::empty(), vk::AccessFlags::empty());

                    device.end_command_buffer(image_barrier_cmd_buf)?;
                }

                core.queue().submit_sync(image_barrier_cmd_buf)?;

                unsafe { device.free_command_buffers(**core.cmd_pool(), &[image_barrier_cmd_buf]) };
            }

            (image, image_device_memory, image_view)
        };

        let uniform_buffer = {
            let mut uniform_buffer = vulkan_abstraction::Buffer::new::<u8>(
                Rc::clone(&core),
                std::mem::size_of::<UniformBufferContents>(),
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::MemoryAllocateFlags::empty(),
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?;

            let mem = uniform_buffer.map::<UniformBufferContents>()?;
            mem[0].proj_inverse = make_proj_inverse_matrix((core.swapchain().image_extent().width, core.swapchain().image_extent().height));
            mem[0].view_inverse = make_view_inverse_matrix();

            {
                let origin    = mem[0].view_inverse * nalgebra::Vector4::new(0.0, 0.0, 0.0, 1.0);
                let target    = mem[0].proj_inverse * nalgebra::Vector4::new(0.0, 0.0, 1.0, 1.0);
                let target_normalized = target.normalize();
                let direction = mem[0].view_inverse * nalgebra::Vector4::new(target_normalized.x, target_normalized.y, target_normalized.z, 0.0);

                let origin = origin.xyz();
                let direction = direction.xyz().normalize();

                let fmt_vec = |v: nalgebra::Vector3<f32>| format!("({}, {}, {})", v.x, v.y, v.z);
                println!("for screen center, ray origin={}, direction={}", fmt_vec(origin), fmt_vec(direction));
            }

            uniform_buffer.unmap();

            uniform_buffer
        };

        let descriptor_sets = vulkan_abstraction::DescriptorSets::new(Rc::clone(&core), &tlas, &image_view, &uniform_buffer)?;


        let ray_tracing_pipeline = vulkan_abstraction::RayTracingPipeline::new(
            Rc::clone(&core),
            &descriptor_sets,
            get_env_var_as_bool(Self::ENABLE_SHADER_DEBUG_SYMBOLS_ENV_VAR).unwrap_or(Self::IS_DEBUG_BUILD),
        )?;

        let shader_binding_table = vulkan_abstraction::ShaderBindingTable::new(&core, &ray_tracing_pipeline)?;

        Self::record_render_command_buffers(
            &core,
            &core.cmd_pool().get_buffers()[..core.swapchain().images().len()],
            &ray_tracing_pipeline,
            &descriptor_sets,
            &shader_binding_table,
            image,
        )?;


        Ok(Self {
            core,
            vertex_buffer,
            index_buffer,
            blas,
            tlas,
            image,
            image_device_memory,
            image_view,
            uniform_buffer,
            descriptor_sets,
            ray_tracing_pipeline,
            shader_binding_table,
        })
    }

    pub fn render(&mut self) -> SrResult<()> {
        let img_index = self.core.queue().acquire_next_image(self.core.swapchain().inner())?;

        let cmd_buf = self.core.cmd_pool().get_buffers()[img_index as usize];

        self.core.queue().submit_async(cmd_buf)?;
        self.core.queue().wait_idle()?;

        self.core.queue_mut().present(self.core.swapchain().inner(), img_index)?;
        self.core.queue().wait_idle()?;

        Ok(())
    }


    unsafe fn cmd_image_memory_barrier (core: &vulkan_abstraction::Core, cmd_buf: vk::CommandBuffer, image: vk::Image, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout, src_stage: vk::PipelineStageFlags, dst_stage: vk::PipelineStageFlags, src_access_mask: vk::AccessFlags, dst_access_mask: vk::AccessFlags) {
        let image_memory_barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
            );
        unsafe {
            core.device().inner().cmd_pipeline_barrier(
                cmd_buf,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[], // memory barriers
                &[], // buffer memory barriers
                &[image_memory_barrier]
            );
        }
    }

    fn record_render_command_buffers(
        core: &vulkan_abstraction::Core,
        cmd_bufs: &[vk::CommandBuffer],
        rt_pipeline: &vulkan_abstraction::RayTracingPipeline,
        descriptor_sets: &vulkan_abstraction::DescriptorSets,
        shader_binding_table: &vulkan_abstraction::ShaderBindingTable,
        image: vk::Image,
    ) -> SrResult<()> {
        let device = core.device().inner();
        let cmd_buf_usage_flags = vk::CommandBufferUsageFlags::SIMULTANEOUS_USE;
        let cmd_buf_begin_info = vk::CommandBufferBeginInfo::default()
        .flags(cmd_buf_usage_flags);

        for (i, cmd_buf) in cmd_bufs.iter().cloned().enumerate() {
            let sc_image = core.swapchain().images()[i];
            // Initializing push constant values
            let push_constants = vulkan_abstraction::PushConstant {
                clear_color: [1.0, 0.0, 0.0, 1.0],
            };

            unsafe {
                device.begin_command_buffer(cmd_buf, &cmd_buf_begin_info)?;

                device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::RAY_TRACING_KHR, rt_pipeline.get_handle());
                device.cmd_bind_descriptor_sets(
                    cmd_buf,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    rt_pipeline.get_layout(),
                    0,
                    descriptor_sets.get_handles(), &[]
                );
                device.cmd_push_constants(
                    cmd_buf,
                    rt_pipeline.get_layout(),
                    vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR | vk::ShaderStageFlags::MISS_KHR,
                    0, &std::mem::transmute::<vulkan_abstraction::PushConstant, [u8;std::mem::size_of::<vulkan_abstraction::PushConstant>()]>(push_constants)
                );
                core.rt_pipeline_device().cmd_trace_rays(
                    cmd_buf,
                    shader_binding_table.get_raygen_region(),
                    shader_binding_table.get_miss_region(),
                    shader_binding_table.get_hit_region(),
                    shader_binding_table.get_callable_region(),
                    core.swapchain().image_extent().width,
                    core.swapchain().image_extent().height,
                    1
                );

                let stage_rt = vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR;
                let stage_tx = vk::PipelineStageFlags::TRANSFER;
                let stage_pipetop = vk::PipelineStageFlags::TOP_OF_PIPE;
                let stage_pipebtm = vk::PipelineStageFlags::BOTTOM_OF_PIPE;

                let layout_undef = vk::ImageLayout::UNDEFINED;
                let layout_general = vk::ImageLayout::GENERAL;
                let layout_tx_src = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                let layout_tx_dst = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                let layout_present = vk::ImageLayout::PRESENT_SRC_KHR;

                Self::cmd_image_memory_barrier(&core, cmd_buf, image, layout_general, layout_tx_src, stage_rt, stage_tx, vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::TRANSFER_READ);
                Self::cmd_image_memory_barrier(&core, cmd_buf, sc_image, layout_undef, layout_tx_dst, stage_pipetop, stage_tx, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE);


                //now blit the image onto the swapchain image
                let image_extent = core.swapchain().image_extent();
                let image_subresource_layers = vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1);
                let whole_img_offsets = [
                    ash::vk::Offset3D { x: 0, y: 0, z: 0 },
                    ash::vk::Offset3D { x: image_extent.width as i32, y: image_extent.height as i32, z: 1}
                ];
                let image_blit = vk::ImageBlit::default()
                    .src_subresource(image_subresource_layers)
                    .src_offsets(whole_img_offsets)
                    .dst_subresource(image_subresource_layers)
                    .dst_offsets(whole_img_offsets);
                let filter = vk::Filter::NEAREST;

                device.cmd_blit_image(cmd_buf, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, core.swapchain().images()[i], vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[image_blit], filter);

                Self::cmd_image_memory_barrier(&core, cmd_buf, image, layout_tx_src, layout_general, stage_tx, stage_pipebtm, vk::AccessFlags::TRANSFER_READ, vk::AccessFlags::empty());
                Self::cmd_image_memory_barrier(&core, cmd_buf, sc_image, layout_tx_dst, layout_present, stage_tx, stage_pipebtm, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::empty());

                device.end_command_buffer(cmd_buf)?;
            }
        }

        Ok(())
    }



}

impl Drop for Renderer {
    fn drop(&mut self) {
        let device = self.core.device().inner();
        unsafe { device.destroy_image_view(self.image_view, None); }
        unsafe { device.destroy_image(self.image, None); }
        unsafe { device.free_memory(self.image_device_memory, None); }
    }
}
