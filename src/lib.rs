extern crate shaderc;

use std::rc::Rc;

use crate::camera::*;
use crate::error::*;
use ash::vk;

pub mod camera;
pub mod error;
mod vulkan_abstraction;

struct UniformBufferContents {
    pub view_inverse: nalgebra::Matrix4<f32>,
    pub proj_inverse: nalgebra::Matrix4<f32>,
}

/// A lot of fields are options because at the moment
/// the code does not provide a way to update existing structure
/// and so at every edit we have to recreate everything from scratch
pub struct Renderer {
    core: Rc<vulkan_abstraction::Core>,
    image_extent: vk::Extent3D,
    image_format: vk::Format,
    scene: Option<vulkan_abstraction::Scene>,
    blas: Option<vulkan_abstraction::BLAS>,
    tlas: Option<vulkan_abstraction::TLAS>,
    image: vk::Image,
    image_device_memory: vk::DeviceMemory,
    image_view: vk::ImageView,
    uniform_buffer: vulkan_abstraction::Buffer,
    descriptor_sets: Option<vulkan_abstraction::DescriptorSets>,
    ray_tracing_pipeline: Option<vulkan_abstraction::RayTracingPipeline>,
    shader_binding_table: Option<vulkan_abstraction::ShaderBindingTable>,
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

impl Renderer {
    // useful environment variables, set to 1 or 0
    // TODO: switch to program arguments (safeguards against typos, and allows for a short explanation in --help)
    const ENABLE_VALIDATION_LAYER_ENV_VAR: &'static str = "ENABLE_VALIDATION_LAYER"; // defaults to 0 in debug build, to 1 in release build
    const ENABLE_GPUAV_ENV_VAR_NAME: &'static str = "ENABLE_GPUAV"; // does nothing unless validation layer is enabled, defaults to 0
    const ENABLE_SHADER_DEBUG_SYMBOLS_ENV_VAR: &'static str = "ENABLE_SHADER_DEBUG_SYMBOLS"; // defaults to 0 in debug build, to 1 in release build
    const IS_DEBUG_BUILD: bool = cfg!(debug_assertions);

    pub fn new(image_extent: (u32, u32), image_format: vk::Format) -> SrResult<Self> {
        let core = Rc::new(vulkan_abstraction::Core::new(
            get_env_var_as_bool(Self::ENABLE_VALIDATION_LAYER_ENV_VAR)
                .unwrap_or(Self::IS_DEBUG_BUILD),
            get_env_var_as_bool(Self::ENABLE_GPUAV_ENV_VAR_NAME).unwrap_or(false),
        )?);
        let device = core.device().inner();

        let image_extent = vk::Extent2D {
            width: image_extent.0,
            height: image_extent.1,
        }
        .into();

        /*
           This code about the image needs to be moved into the image struct, wich at the moment is a placeholder
           So that then the image can be passed around with its fields
        */

        // the image we will do the rendering on
        let (image, image_device_memory, image_view) = {
            let image = {
                let image_create_info = vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(image_format)
                    .extent(image_extent)
                    .flags(vk::ImageCreateFlags::empty())
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
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
                    .format(image_format)
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
                let image_barrier_cmd_buf =
                    vulkan_abstraction::cmd_buffer::new(&*core.cmd_pool(), core.device())?;

                let begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

                //record command buffer
                unsafe {
                    device.begin_command_buffer(image_barrier_cmd_buf, &begin_info)?;

                    let stage_all = vk::PipelineStageFlags::ALL_COMMANDS;
                    Self::cmd_image_memory_barrier(
                        &core,
                        image_barrier_cmd_buf,
                        image,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::GENERAL,
                        stage_all,
                        stage_all,
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::empty(),
                    );

                    device.end_command_buffer(image_barrier_cmd_buf)?;
                }

                core.queue().submit_sync(image_barrier_cmd_buf)?;

                unsafe { device.free_command_buffers(**core.cmd_pool(), &[image_barrier_cmd_buf]) };
            }

            (image, image_device_memory, image_view)
        };

        let uniform_buffer = vulkan_abstraction::Buffer::new::<u8>(
            Rc::clone(&core),
            std::mem::size_of::<UniformBufferContents>(),
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryAllocateFlags::empty(),
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        )?;

        Ok(Self {
            core,
            image_extent,
            scene: Default::default(),
            blas: Default::default(),
            tlas: Default::default(),
            image,
            image_device_memory,
            image_view,
            image_format,
            uniform_buffer,
            descriptor_sets: Default::default(),
            ray_tracing_pipeline: Default::default(),
            shader_binding_table: Default::default(),
        })
    }

    /// https://github.com/SaschaWillems/Vulkan/blob/master/base/VulkanTools.cpp#L318
    unsafe fn cmd_image_memory_barrier(
        core: &vulkan_abstraction::Core,
        cmd_buf: vk::CommandBuffer,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
    ) {
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
                    .layer_count(1),
            );
        unsafe {
            core.device().inner().cmd_pipeline_barrier(
                cmd_buf,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[], // memory barriers
                &[], // buffer memory barriers
                &[image_memory_barrier],
            );
        }
    }

    /// https://github.com/SaschaWillems/Vulkan/blob/02f8f1017091edb4e5a531c92bde07c78d061df6/examples/screenshot/screenshot.cpp#L143
    fn record_render_command_buffers(
        core: &vulkan_abstraction::Core,
        cmd_buf: &vk::CommandBuffer,
        rt_pipeline: &vulkan_abstraction::RayTracingPipeline,
        descriptor_sets: &vulkan_abstraction::DescriptorSets,
        shader_binding_table: &vulkan_abstraction::ShaderBindingTable,
        image: vk::Image,
        image_extent: vk::Extent3D,
        image_format: vk::Format,
        dst_image: vk::Image,
    ) -> SrResult<()> {
        let device = core.device().inner();
        let cmd_buf_usage_flags = vk::CommandBufferUsageFlags::SIMULTANEOUS_USE;
        let cmd_buf_begin_info = vk::CommandBufferBeginInfo::default().flags(cmd_buf_usage_flags);

        // Initializing push constant values
        let push_constants = vulkan_abstraction::PushConstant {
            clear_color: [1.0, 0.0, 0.0, 1.0],
        };

        unsafe {
            device.begin_command_buffer(*cmd_buf, &cmd_buf_begin_info)?;

            device.cmd_bind_pipeline(
                *cmd_buf,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                rt_pipeline.get_handle(),
            );
            device.cmd_bind_descriptor_sets(
                *cmd_buf,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                rt_pipeline.get_layout(),
                0,
                descriptor_sets.get_handles(),
                &[],
            );
            device.cmd_push_constants(
                *cmd_buf,
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
                *cmd_buf,
                shader_binding_table.get_raygen_region(),
                shader_binding_table.get_miss_region(),
                shader_binding_table.get_hit_region(),
                shader_binding_table.get_callable_region(),
                image_extent.width,
                image_extent.height,
                1,
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

            //Ray tracing memory barrier
            //This is the only memory barried that i saved from the refactor
            //This should also already prepare correctly the image for transfer
            Self::cmd_image_memory_barrier(
                &core,
                *cmd_buf,
                image,
                layout_general,
                layout_tx_src,
                stage_rt,
                stage_tx,
                vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::TRANSFER_READ,
            );

            // Transition image from present to transfer source layout
            // Self::cmd_image_memory_barrier(
            //     &core,
            //     *cmd_buf,
            //     image,
            //     vk::ImageLayout::PRESENT_SRC_KHR,
            //     vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            //     vk::PipelineStageFlags::TRANSFER,
            //     vk::PipelineStageFlags::TRANSFER,
            //     vk::AccessFlags::MEMORY_READ,
            //     vk::AccessFlags::TRANSFER_READ,
            // );

            // Transition destination image to transfer destination layout
            Self::cmd_image_memory_barrier(
                &core,
                *cmd_buf,
                dst_image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
            );

            //TODO: check for blit support, if blit is not supported there is image copy
            let image_subresource_layers = vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1);
            //blit size
            let whole_img_offsets = [
                ash::vk::Offset3D { x: 0, y: 0, z: 0 }, //shouldn't the two offsets be the same?
                ash::vk::Offset3D {
                    x: image_extent.width as i32,
                    y: image_extent.height as i32,
                    z: 1,
                },
            ];
            let image_blit = vk::ImageBlit::default()
                .src_subresource(image_subresource_layers)
                .src_offsets(whole_img_offsets)
                .dst_subresource(image_subresource_layers)
                .dst_offsets(whole_img_offsets);

            device.cmd_blit_image(
                *cmd_buf,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[image_blit],
                vk::Filter::NEAREST,
            );

            // Transition destination image to general layout, which is the required layout for mapping the image memory later on
            Self::cmd_image_memory_barrier(
                &core,
                *cmd_buf,
                dst_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::MEMORY_READ,
            );

            // Transition back the image after the blit is done
            Self::cmd_image_memory_barrier(
                &core,
                *cmd_buf,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                layout_present,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_READ,
                vk::AccessFlags::MEMORY_READ,
            );

            device.end_command_buffer(*cmd_buf)?;
        }

        Ok(())
    }

    /// # This is a mock function
    pub fn load_file(&mut self) -> SrResult<()> {
        self.descriptor_sets = Some(vulkan_abstraction::DescriptorSets::new(
            Rc::clone(&self.core),
            self.tlas.as_ref().unwrap(),
            &self.image_view,
            &self.uniform_buffer,
        )?);
        //TODO: and so on

        self.ray_tracing_pipeline = Some(vulkan_abstraction::RayTracingPipeline::new(
            Rc::clone(&self.core),
            self.descriptor_sets.as_ref().unwrap(),
            get_env_var_as_bool(Self::ENABLE_SHADER_DEBUG_SYMBOLS_ENV_VAR)
                .unwrap_or(Self::IS_DEBUG_BUILD),
        )?);

        self.shader_binding_table = Some(vulkan_abstraction::ShaderBindingTable::new(
            &self.core,
            self.ray_tracing_pipeline.as_ref().unwrap(),
        )?);

        Ok(())
    }

    /// # This is a mock function
    pub fn set_camera(&mut self, camera: Camera) -> SrResult<()> {
        let proj = nalgebra::geometry::Perspective3::new(
            self.image_extent.width as f32 / self.image_extent.height as f32,
            3.14 / 2.0,
            0.1,
            1000.0,
        );

        let eye = nalgebra::geometry::Point3::new(0.0, 0.0, 3.0);
        let target = nalgebra::geometry::Point3::new(0.0, 0.0, 0.0);
        let up = nalgebra::Vector3::new(0.0, -1.0, 0.0);
        //apparently vulkan uses right-handed coordinates
        let view = nalgebra::Isometry3::look_at_rh(&eye, &target, &up);

        let mem = self.uniform_buffer.map::<crate::UniformBufferContents>()?;
        mem[0].proj_inverse = proj.to_homogeneous().try_inverse().unwrap();
        mem[0].view_inverse = view.to_homogeneous().try_inverse().unwrap();

        let origin = mem[0].view_inverse * nalgebra::Vector4::new(0.0, 0.0, 0.0, 1.0);
        let target = mem[0].proj_inverse * nalgebra::Vector4::new(0.0, 0.0, 1.0, 1.0);
        let target_normalized = target.normalize();
        let direction = mem[0].view_inverse
            * nalgebra::Vector4::new(
                target_normalized.x,
                target_normalized.y,
                target_normalized.z,
                0.0,
            );

        let origin = origin.xyz();
        let direction = direction.xyz().normalize();

        let fmt_vec = |v: nalgebra::Vector3<f32>| format!("({}, {}, {})", v.x, v.y, v.z);
        println!(
            "for screen center, ray origin={}, direction={}",
            fmt_vec(origin),
            fmt_vec(direction)
        );

        self.uniform_buffer.unmap();

        Ok(())
    }

    pub fn render(&self) -> SrResult<*mut std::ffi::c_void> {
        let dst_image = {
            let image_create_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(self.image_format)
                .extent(self.image_extent)
                .flags(vk::ImageCreateFlags::empty())
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::LINEAR)
                .usage(vk::ImageUsageFlags::TRANSFER_DST)
                .initial_layout(vk::ImageLayout::UNDEFINED);

            unsafe {
                self.core
                    .device()
                    .inner()
                    .create_image(&image_create_info, None)
            }
            .unwrap()
        };
        let dst_image_device_memory = {
            let mem_reqs = unsafe {
                self.core
                    .device()
                    .inner()
                    .get_image_memory_requirements(dst_image)
            };
            let mem_alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_reqs.size)
                .memory_type_index(vulkan_abstraction::get_memory_type_index(
                    &self.core,
                    vk::MemoryPropertyFlags::HOST_VISIBLE //the memory must be host visible to copy from
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    &mem_reqs,
                )?);

            unsafe {
                self.core
                    .device()
                    .inner()
                    .allocate_memory(&mem_alloc_info, None)
            }
            .unwrap()
        };
        unsafe {
            self.core
                .device()
                .inner()
                .bind_image_memory(dst_image, dst_image_device_memory, 0)
        }
        .unwrap();

        /*
           This should be in the new
           as there is no need to recreate the command buffer every time
           but it depends on a lof of stuff already existing.

           I'm parking the function here for the moment
        */
        Self::record_render_command_buffers(
            &self.core,
            self.core.cmd_pool().get_buffer(),
            self.ray_tracing_pipeline.as_ref().unwrap(),
            self.descriptor_sets.as_ref().unwrap(),
            self.shader_binding_table.as_ref().unwrap(),
            self.image,
            self.image_extent,
            self.image_format,
            dst_image,
        )?;

        let cmd_buf = self.core.cmd_pool().get_buffer();

        self.core.queue().submit_async(*cmd_buf)?;
        self.core.queue().wait_idle()?;

        let dst_image_subresource = vk::ImageSubresource {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            array_layer: 0,
        };
        let dst_image_subresource_layout = unsafe {
            self.core
                .device()
                .inner()
                .get_image_subresource_layout(dst_image, dst_image_subresource)
        };

        //should we just return this?
        let mapped_memory = unsafe {
            self.core.device().inner().map_memory(
                dst_image_device_memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )?
        };
        // Looking at Sascha Willems code i get the idea that it knows that the type is u32

        //possibility: Allocation of gpu_allocator

        // let raw_mem = unsafe {
        //     vulkan_abstraction::mapped_memory::RawMappedMemory::new(p, vk::WHOLE_SIZE as usize)
        // };
        // let mut mapped_memory = Some(raw_mem);
        // let ret = mapped_memory.as_mut().unwrap().borrow(); //i have not figured out yet what it the type of this thing

        //I think we should return a reference
        Ok(mapped_memory)
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        let device = self.core.device().inner();
        unsafe {
            device.destroy_image_view(self.image_view, None);
        }
        unsafe {
            device.destroy_image(self.image, None);
        }
        unsafe {
            device.free_memory(self.image_device_memory, None);
        }
    }
}
