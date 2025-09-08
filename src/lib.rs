extern crate shaderc;

use std::{collections::HashSet, ffi::CStr};

use crate::error::*;
use ash::vk::{
    AccessFlags, CommandBuffer, CommandBufferBeginInfo, CommandBufferUsageFlags, DependencyFlags, Image, ImageCopy, ImageCreateFlags, ImageLayout, ImageMemoryBarrier, ImageSubresourceLayers, PhysicalDeviceProperties2, PhysicalDeviceRayTracingPipelinePropertiesKHR, PipelineBindPoint, PipelineStageFlags, ShaderStageFlags
};
use ash::{
    Device, Entry, Instance,
    khr::{self, acceleration_structure, surface, swapchain},
    vk::{
        ApplicationInfo, ColorSpaceKHR, CommandPoolCreateFlags, ComponentMapping, ComponentSwizzle,
        CompositeAlphaFlagsKHR, DeviceCreateInfo, DeviceQueueCreateInfo, Extent2D, Format,
        ImageAspectFlags, ImageSubresourceRange, ImageUsageFlags, ImageView, ImageViewCreateInfo,
        ImageViewType, InstanceCreateInfo, PhysicalDevice,
        PhysicalDeviceAccelerationStructureFeaturesKHR,
        PhysicalDeviceRayTracingPipelineFeaturesKHR, PhysicalDeviceType,
        PhysicalDeviceVulkan12Features, PresentModeKHR, QueueFlags, SharingMode,
        SurfaceCapabilitiesKHR, SurfaceFormatKHR, SurfaceKHR, SwapchainCreateInfoKHR, SwapchainKHR,
        make_api_version,
    },
};
use winit::raw_window_handle_05::{RawDisplayHandle, RawWindowHandle};

pub mod error;
pub mod utils;
mod vulkan_abstraction;

struct SwapchainSupportDetails {
    surface_capabilities: SurfaceCapabilitiesKHR,
    surface_formats: Vec<SurfaceFormatKHR>,
    surface_present_modes: Vec<PresentModeKHR>,
}

impl SwapchainSupportDetails {
    /*
    TODO: bad error handling.
    many different phases can cause vulkan errors in choosing a physical device.
    we don't keep track of which errors occur because if no suitable device is found we'd need to tell the user what error makes each device unsuitable
    */
    fn new(
        surface: SurfaceKHR,
        surface_instance: &surface::Instance,
        physical_device: PhysicalDevice,
    ) -> SrResult<Self> {
        let surface_capabilities = unsafe {
            surface_instance.get_physical_device_surface_capabilities(physical_device, surface)
        }
        .to_sr_result()?;

        let surface_formats = unsafe {
            surface_instance.get_physical_device_surface_formats(physical_device, surface)
        }
        .to_sr_result()?;

        let surface_present_modes = unsafe {
            surface_instance.get_physical_device_surface_present_modes(physical_device, surface)
        }
        .to_sr_result()?;

        Ok(Self {
            surface_capabilities,
            surface_formats,
            surface_present_modes,
        })
    }

    fn check_swapchain_support(&self) -> bool {
        !self.surface_formats.is_empty() && !self.surface_present_modes.is_empty()
    }
}

//TODO: impl Drop

#[allow(dead_code)]
pub struct Core {
    entry: Entry,
    instance: Instance,
    surface_instance: khr::surface::Instance,
    device: Device,
    swapchain_device: khr::swapchain::Device,
    acceleration_structure_device: khr::acceleration_structure::Device,

    queue: vulkan_abstraction::Queue,
    surface: SurfaceKHR,
    swapchain_image_extent: Extent2D,
    swapchain: SwapchainKHR,
    swapchain_images: Vec<Image>,
    swapchain_image_views: Vec<ImageView>,
    physical_device_rt_pipeline_properties: PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>, // 'static because pNext is null
    cmd_pool: vulkan_abstraction::CmdPool,
    vertex_buffer: vulkan_abstraction::VertexBuffer,
    index_buffer: vulkan_abstraction::IndexBuffer,
    // blas: vulkan_abstraction::BLAS,
    tlas: vulkan_abstraction::TLAS,
    image: Image,
    descriptor_sets: vulkan_abstraction::DescriptorSets,
    ray_tracing_pipeline_device: khr::ray_tracing_pipeline::Device,
    ray_tracing_pipeline: vulkan_abstraction::RayTracingPipeline,
    shader_binding_table: vulkan_abstraction::ShaderBindingTable,
}

impl Core {
    const VALIDATION_LAYER_NAME: &'static CStr = c"VK_LAYER_KHRONOS_validation";

    // TODO: currently take for granted that the user has a window, no support for offline rendering
    pub fn new(window_extent: [u32; 2], raw_window_handle: RawWindowHandle, raw_display_handle: RawDisplayHandle) -> SrResult<Self> {
        let entry = Entry::linked();
        let application_info = ApplicationInfo::default().api_version(make_api_version(0, 1, 4, 0));

        let enabled_layer_names =
            if cfg!(debug_assertions) && Self::check_validation_layer_support(&entry)? {
                &[Self::VALIDATION_LAYER_NAME.as_ptr()]
            } else {
                [].as_slice()
            };

        let required_extensions = crate::utils::enumerate_required_extensions(raw_display_handle)?;

        let instance = {
            let instance_create_info = InstanceCreateInfo::default()
                .application_info(&application_info)
                .enabled_layer_names(enabled_layer_names)
                .enabled_extension_names(required_extensions);

            unsafe { entry.create_instance(&instance_create_info, None) }.to_sr_result()?
        };

        let surface_instance = ash::khr::surface::Instance::new(&entry, &instance);

        let surface = unsafe {
            crate::utils::create_surface(
                &entry,
                &instance,
                raw_display_handle,
                raw_window_handle,
                None,
            )
        }?;

        let required_device_extensions = &[
            khr::swapchain::NAME, // TODO: not needed for offline rendering

            //ray tracing extensions
            khr::ray_tracing_pipeline::NAME,
            khr::acceleration_structure::NAME,
            khr::deferred_host_operations::NAME,
        ]
        .map(CStr::as_ptr);

        let physical_devices = unsafe { instance.enumerate_physical_devices() }.to_sr_result()?;

        let (physical_device, queue_family_index, swapchain_support_details) = physical_devices
            .into_iter()
            //only allow devices which support all required extensions
            .filter(|physical_device| {
                Self::check_device_extension_support(
                    &instance,
                    *physical_device,
                    required_device_extensions,
                ).unwrap_or(false)
            })
            //only allow devices with swapchain support, and acquire swapchain support details
            .filter_map(|physical_device| {
                let swapchain_support_details =
                    SwapchainSupportDetails::new(surface, &surface_instance, physical_device)
                        .ok()?; // currently ignoring devices which return errors while querying their swapchain support

                if swapchain_support_details.check_swapchain_support() {
                    Some((physical_device, swapchain_support_details))
                } else {
                    None
                }
            })
            //choose a suitable queue family, and filter out devices without one
            .filter_map(|(physical_device, swapchain_support_details)| {
                Some((
                    physical_device,
                    Self::select_queue_family(
                        &instance,
                        &surface_instance,
                        physical_device,
                        surface,
                    )?,
                    swapchain_support_details,
                ))
            })
            // try to get a discrete or at least integrated gpu
            .max_by_key(|(physical_device, _, _)| {
                let device_type =
                    unsafe { instance.get_physical_device_properties(*physical_device) }
                        .device_type;

                match device_type {
                    PhysicalDeviceType::DISCRETE_GPU => 2,
                    PhysicalDeviceType::INTEGRATED_GPU => 1,
                    _ => 0,
                }
            })
            .ok_or(SrError::new(None, String::from("No suitable GPU found!")))?;

        let device = {
            let queue_priorities = vec![1.0; 1]; // TODO: use more than 1 queue
            let queue_create_infos = [DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&queue_priorities)];

            // enable some device features necessary for ray-tracing
            let mut vk12_features =
                PhysicalDeviceVulkan12Features::default().buffer_device_address(true);
            let mut physical_device_rt_pipeline_features =
                PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);
            let mut physical_device_acceleration_structure_features =
                PhysicalDeviceAccelerationStructureFeaturesKHR::default()
                    .acceleration_structure(true);

            let device_create_info = DeviceCreateInfo::default()
                .enabled_extension_names(required_device_extensions)
                .push_next(&mut vk12_features)
                .push_next(&mut physical_device_rt_pipeline_features)
                .push_next(&mut physical_device_acceleration_structure_features)
                .queue_create_infos(&queue_create_infos);

            unsafe { instance.create_device(physical_device, &device_create_info, None) }
                .to_sr_result()?
        };

        let swapchain_device = swapchain::Device::new(&instance, &device);
        let acceleration_structure_device = acceleration_structure::Device::new(&instance, &device);

        let queue = vulkan_abstraction::Queue::new(
            device.clone(),
            swapchain_device.clone(),
            queue_family_index,
            0,
        )?;

        // for creating swapchain and swapchain_image_views
        let surface_format = {
            let formats = swapchain_support_details.surface_formats;

            //find the BGRA8 SRGB nonlinear surface format
            let bgra8_srgb_nonlinear = formats.iter().find(|surface_format| {
                surface_format.format == Format::B8G8R8A8_SRGB
                    && surface_format.color_space == ColorSpaceKHR::SRGB_NONLINEAR
            });

            if let Some(format) = bgra8_srgb_nonlinear {
                *format
            } else {
                //or else get the first format the device offers
                *formats.first().ok_or(SrError::new(
                    None,
                    String::from("Physical device does not support any surface formats"),
                ))?
            }
        };

        let swapchain_image_extent = if swapchain_support_details.surface_capabilities.current_extent.width != u32::MAX {
            swapchain_support_details.surface_capabilities.current_extent
        } else {
            Extent2D {
                width: window_extent[0].clamp(
                    swapchain_support_details.surface_capabilities.min_image_extent.width,
                    swapchain_support_details.surface_capabilities.max_image_extent.width,
                ),
                height: window_extent[1].clamp(
                    swapchain_support_details.surface_capabilities.min_image_extent.height,
                    swapchain_support_details.surface_capabilities.max_image_extent.height,
                ),
            }
        };

        let swapchain = {
            let present_modes = &swapchain_support_details.surface_present_modes;
            let present_mode = if present_modes.contains(&PresentModeKHR::MAILBOX) {
                PresentModeKHR::MAILBOX
            } else if present_modes.contains(&PresentModeKHR::IMMEDIATE) {
                PresentModeKHR::IMMEDIATE
            } else {
                PresentModeKHR::FIFO // fifo is guaranteed to exist
            };

            let surface_capabilities = &swapchain_support_details.surface_capabilities;

            // Sticking to this minimum means that we may sometimes have to wait on the driver to
            // complete internal operations before we can acquire another image to render to.
            // Therefore it is recommended to request at least one more image than the minimum
            let mut image_count = surface_capabilities.min_image_count + 1;

            if surface_capabilities.max_image_count > 0
                && image_count > surface_capabilities.max_image_count
            {
                image_count = surface_capabilities.max_image_count;
            }

            let swapchain_create_info = SwapchainCreateInfoKHR::default()
                .surface(surface)
                .min_image_count(image_count)
                .image_format(surface_format.format)
                .image_color_space(surface_format.color_space)
                .image_extent(swapchain_image_extent)
                .image_array_layers(1)
                .image_usage(ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::TRANSFER_DST)
                .image_sharing_mode(SharingMode::EXCLUSIVE)
                .pre_transform(surface_capabilities.current_transform)
                .composite_alpha(CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .old_swapchain(SwapchainKHR::null());

            unsafe { swapchain_device.create_swapchain(&swapchain_create_info, None) }
                .to_sr_result()?
        };

        let swapchain_images = unsafe { swapchain_device.get_swapchain_images(swapchain) }.to_sr_result()?;

        let swapchain_image_views = swapchain_images
            .iter()
            .map(|image| {
                let image_view_create_info = ImageViewCreateInfo::default()
                    .image(*image)
                    .view_type(ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .components(ComponentMapping {
                        r: ComponentSwizzle::IDENTITY,
                        g: ComponentSwizzle::IDENTITY,
                        b: ComponentSwizzle::IDENTITY,
                        a: ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(
                        ImageSubresourceRange::default()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );

                unsafe { device.create_image_view(&image_view_create_info, None) }.to_sr_result()
            })
            .collect::<Result<Vec<_>, _>>()?;

        //necessary for memory allocations
        let physical_device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let physical_device_rt_pipeline_properties = {
            let mut physical_device_rt_pipeline_properties =
                PhysicalDeviceRayTracingPipelinePropertiesKHR::default();

            let mut physical_device_properties = PhysicalDeviceProperties2::default()
                .push_next(&mut physical_device_rt_pipeline_properties);

            unsafe {
                instance.get_physical_device_properties2(
                    physical_device,
                    &mut physical_device_properties,
                )
            };

            physical_device_rt_pipeline_properties
        };
        let cmd_pool = {
            let mut cmd_pool = vulkan_abstraction::CmdPool::new(
                device.clone(),
                CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                queue_family_index,
            )?;

            // add render command buffers to cmd_pool
            cmd_pool.append_buffers(vulkan_abstraction::cmd_buffer::new_vec(&cmd_pool.as_raw(), &device, swapchain_images.len())?);

            cmd_pool
        };

        let vertex_buffer = {
            #[derive(Clone, Copy)]
            struct Vertex {
                #[allow(dead_code)]
                pos: [f32; 3],
            }

            let verts = [
                Vertex {
                    pos: [-1.0, -0.5, 0.0],
                },
                Vertex {
                    pos: [1.0, -0.5, 0.0],
                },
                Vertex {
                    pos: [0.0, 1.0, 0.0],
                },
            ];
            let staging_buffer = vulkan_abstraction::Buffer::new_staging_from_data::<Vertex>(
                device.clone(),
                &verts,
                &physical_device_memory_properties,
            )?;
            let vertex_buffer = vulkan_abstraction::VertexBuffer::new_for_blas::<Vertex>(
                device.clone(),
                verts.len(),
                &physical_device_memory_properties,
            )?;
            vulkan_abstraction::Buffer::clone_buffer(
                &device,
                &queue,
                &cmd_pool,
                &staging_buffer,
                &vertex_buffer,
            )?;

            vertex_buffer
        };
        let index_buffer = {
            let indices : [u32; 3] = [0, 1, 2];
            let staging_buffer = vulkan_abstraction::Buffer::new_staging_from_data::<u32>(
                device.clone(),
                &indices,
                &physical_device_memory_properties,
            )?;
            let index_buffer = vulkan_abstraction::IndexBuffer::new_for_blas::<u32>(
                device.clone(),
                indices.len(),
                &physical_device_memory_properties,
            )?;
            vulkan_abstraction::Buffer::clone_buffer(
                &device,
                &queue,
                &cmd_pool,
                &staging_buffer,
                &index_buffer,
            )?;

            index_buffer
        };

        let blas = vulkan_abstraction::BLAS::new(
            &device,
            acceleration_structure_device.clone(),
            &physical_device_memory_properties,
            &cmd_pool,
            &queue,
            &vertex_buffer,
            &index_buffer,
        )?;

        let blases = vec![blas];

        let tlas = vulkan_abstraction::TLAS::new(
            &device,
            acceleration_structure_device.clone(),
            &physical_device_memory_properties,
            &cmd_pool,
            &queue,
            &blases,
        )?;

        const OUT_IMAGE_FORMAT: Format = Format::R8G8B8A8_UNORM;

        // the image we will do the rendering on; before every frame it will be copied to the swapchain
        // TODO: dispose of these respources in drop(), maybe even abstract them into a struct
        let (image, _image_device_memory, image_view) = {
            let image = {
                let image_create_info = ash::vk::ImageCreateInfo::default()
                    .image_type(ash::vk::ImageType::TYPE_2D)
                    .format(OUT_IMAGE_FORMAT)
                    .extent(swapchain_image_extent.into())
                    .flags(ImageCreateFlags::empty())
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(ash::vk::SampleCountFlags::TYPE_1)
                    .tiling(ash::vk::ImageTiling::OPTIMAL)
                    .usage(
                        ash::vk::ImageUsageFlags::STORAGE
                        | ash::vk::ImageUsageFlags::TRANSFER_SRC,
                    )
                    .initial_layout(ImageLayout::UNDEFINED);

                unsafe { device.create_image(&image_create_info, None) }.unwrap()
            };

            let image_device_memory = {
                let mem_reqs = unsafe { device.get_image_memory_requirements(image) };
                let mem_alloc_info = ash::vk::MemoryAllocateInfo::default()
                .allocation_size(mem_reqs.size)
                .memory_type_index(vulkan_abstraction::get_memory_type_index(
                    ash::vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    &mem_reqs,
                    &physical_device_memory_properties,
                )?);

                unsafe { device.allocate_memory(&mem_alloc_info, None) }.unwrap()
            };

            unsafe { device.bind_image_memory(image, image_device_memory, 0) }.unwrap();

            let image_view = {
                let image_view_create_info = ash::vk::ImageViewCreateInfo::default()
                .view_type(ash::vk::ImageViewType::TYPE_2D)
                .format(OUT_IMAGE_FORMAT)
                .subresource_range(ash::vk::ImageSubresourceRange {
                    aspect_mask: ash::vk::ImageAspectFlags::COLOR,
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
                let image_barrier_cmd_buf = vulkan_abstraction::cmd_buffer::new(&cmd_pool.as_raw(), &device)?;

                let begin_info = CommandBufferBeginInfo::default()
                    .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);

                //record command buffer
                unsafe {
                    device.begin_command_buffer(image_barrier_cmd_buf, &begin_info)
                        .to_sr_result()?;

                    Self::cmd_image_memory_barrier(&device, image_barrier_cmd_buf, image, ImageLayout::UNDEFINED, ImageLayout::GENERAL);

                    device.end_command_buffer(image_barrier_cmd_buf).to_sr_result()?;
                }

                queue.submit_sync(image_barrier_cmd_buf)?;

                unsafe { device.device_wait_idle() }.to_sr_result()?;

                unsafe { device.free_command_buffers(*cmd_pool, &[image_barrier_cmd_buf]) };
            }

            (image, image_device_memory, image_view)
        };

        let descriptor_sets = vulkan_abstraction::DescriptorSets::new(&device, &tlas, &image_view)?;

        let ray_tracing_pipeline_device =
            khr::ray_tracing_pipeline::Device::new(&instance, &device);

        let ray_tracing_pipeline = vulkan_abstraction::RayTracingPipeline::new(
            device.clone(),
            &ray_tracing_pipeline_device,
            &descriptor_sets,
        )?;

        let shader_binding_table = vulkan_abstraction::ShaderBindingTable::new(
            &device,
            &ray_tracing_pipeline_device,
            &ray_tracing_pipeline,
            &physical_device_rt_pipeline_properties,
            &physical_device_memory_properties,
        )?;

        Self::record_render_command_buffers(
            &device,
            &cmd_pool.get_buffers()[..swapchain_images.len()],
            &ray_tracing_pipeline_device,
            &ray_tracing_pipeline,
            &descriptor_sets,
            &shader_binding_table,
            swapchain_image_extent,
            &swapchain_images,
            image,
        )?;


        Ok(Self {
            entry,
            instance,
            surface_instance,
            device,
            swapchain_device,
            acceleration_structure_device,
            queue,
            surface,
            swapchain_image_extent,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            physical_device_rt_pipeline_properties,
            cmd_pool,
            vertex_buffer,
            index_buffer,
            tlas,
            image,
            descriptor_sets,
            ray_tracing_pipeline_device,
            ray_tracing_pipeline,
            shader_binding_table,
        })
    }

    pub fn render(&mut self) -> SrResult<()> {
        let img_index = self.queue.acquire_next_image(self.swapchain)?;

        let cmd_buf = self.cmd_pool.get_buffers()[img_index as usize];

        self.queue.submit_async(cmd_buf)?;
        self.queue.wait_idle()?;

        self.queue.present(self.swapchain, img_index)?;
        self.queue.wait_idle()?;

        Ok(())
    }


    unsafe fn cmd_image_memory_barrier (device: &Device, cmd_buf: CommandBuffer, image: Image, old_layout: ImageLayout, new_layout: ImageLayout) {
        let image_memory_barrier = ImageMemoryBarrier::default()
            .src_access_mask(AccessFlags::empty())
            .dst_access_mask(AccessFlags::empty())
            .old_layout(old_layout)
            .new_layout(new_layout)
            .image(image)
            .subresource_range(
                ImageSubresourceRange::default()
                    .aspect_mask(ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
            );
        unsafe {
            device.cmd_pipeline_barrier(
                cmd_buf,
                PipelineStageFlags::ALL_COMMANDS, // wait for all commands from any pipeline stage unconditionally
                PipelineStageFlags::ALL_COMMANDS, // block all commands from any pipeline stage unconditionally
                DependencyFlags::empty(),
                &[], // memory barriers
                &[], // buffer memory barriers
                &[image_memory_barrier]
            );
        }
    }

    fn record_render_command_buffers(
        device: &Device,
        cmd_bufs: &[CommandBuffer],
        rt_pipeline_device : &khr::ray_tracing_pipeline::Device,
        rt_pipeline: &vulkan_abstraction::RayTracingPipeline,
        descriptor_sets: &vulkan_abstraction::DescriptorSets,
        shader_binding_table: &vulkan_abstraction::ShaderBindingTable,
        swapchain_image_extent: Extent2D,
        swapchain_images: &[Image],
        image: Image,
    ) -> SrResult<()> {
        let cmd_buf_usage_flags = CommandBufferUsageFlags::SIMULTANEOUS_USE;
        let cmd_buf_begin_info = CommandBufferBeginInfo::default()
        .flags(cmd_buf_usage_flags);

        for (i, cmd_buf) in cmd_bufs.iter().cloned().enumerate() {
            // Initializing push constant values
            let push_constants = vulkan_abstraction::PushConstant {
                clear_color: [1.0, 0.0, 0.0, 1.0],
            };

            unsafe {
                device.begin_command_buffer(cmd_buf, &cmd_buf_begin_info).to_sr_result()?;

                device.cmd_bind_pipeline(cmd_buf, PipelineBindPoint::RAY_TRACING_KHR, rt_pipeline.get_handle());
                device.cmd_bind_descriptor_sets(
                    cmd_buf,
                    PipelineBindPoint::RAY_TRACING_KHR,
                    rt_pipeline.get_layout(),
                    0,
                    descriptor_sets.get_handles(), &[]
                );
                device.cmd_push_constants(
                    cmd_buf,
                    rt_pipeline.get_layout(),
                    ShaderStageFlags::RAYGEN_KHR | ShaderStageFlags::CLOSEST_HIT_KHR | ShaderStageFlags::MISS_KHR,
                    0, &std::mem::transmute::<vulkan_abstraction::PushConstant, [u8;16]>(push_constants)
                );
                rt_pipeline_device.cmd_trace_rays(
                    cmd_buf,
                    shader_binding_table.get_raygen_region(),
                    shader_binding_table.get_miss_region(),
                    shader_binding_table.get_hit_region(),
                    shader_binding_table.get_callable_region(),
                    swapchain_image_extent.width,
                    swapchain_image_extent.height,
                    1
                );

                Self::cmd_image_memory_barrier(device, cmd_buf, image, ImageLayout::UNDEFINED, ImageLayout::TRANSFER_SRC_OPTIMAL);
                Self::cmd_image_memory_barrier(device, cmd_buf, swapchain_images[i], ImageLayout::UNDEFINED, ImageLayout::TRANSFER_DST_OPTIMAL);

                //now copy the image onto the swapchain image
                let image_subresource_layers = ImageSubresourceLayers::default()
                    .aspect_mask(ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1);
                let image_copy_info = ImageCopy::default()
                    .src_subresource(image_subresource_layers)
                    .src_offset(ash::vk::Offset3D { x: 0, y: 0, z: 0 })
                    .dst_subresource(image_subresource_layers)
                    .dst_offset(ash::vk::Offset3D { x: 0, y: 0, z: 0 })
                    .extent(swapchain_image_extent.into());
                device.cmd_copy_image(cmd_buf, image, ImageLayout::TRANSFER_SRC_OPTIMAL, swapchain_images[i], ImageLayout::TRANSFER_DST_OPTIMAL, &[image_copy_info]);

                Self::cmd_image_memory_barrier(device, cmd_buf, image, ImageLayout::UNDEFINED, ImageLayout::GENERAL);
                Self::cmd_image_memory_barrier(device, cmd_buf, swapchain_images[i], ImageLayout::UNDEFINED, ImageLayout::PRESENT_SRC_KHR);

                device.end_command_buffer(cmd_buf).to_sr_result()?;
            }
        }

        Ok(())
    }

    fn check_validation_layer_support(entry: &Entry) -> SrResult<bool> {
        let layers_props = unsafe { entry.enumerate_instance_layer_properties() }.to_sr_result()?;

        let supports_validation_layer = layers_props
            .iter()
            .any(|p| p.layer_name_as_c_str().unwrap() == Self::VALIDATION_LAYER_NAME); //TODO unwrap

        Ok(supports_validation_layer)
    }

    // TODO:
    //     This takes for granted that we want to render to a surface.
    //     How would this work for off-screen rendering?
    fn select_queue_family(
        instance: &Instance,
        surface_instance: &surface::Instance,
        physical_device: PhysicalDevice,
        surface: SurfaceKHR,
    ) -> Option<u32> {
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
            .into_iter()
            .enumerate()
            .filter(|(queue_family_index, queue_family_props)| {
                queue_family_props
                    .queue_flags
                    .contains(QueueFlags::GRAPHICS)
                    && unsafe {
                        surface_instance.get_physical_device_surface_support(
                            physical_device,
                            *queue_family_index as u32,
                            surface,
                        )
                    }
                    .unwrap_or(false)
            })
            .map(|(queue_family_index, _)| queue_family_index as u32)
            .next()
    }

    fn check_device_extension_support(
        instance: &Instance,
        physical_device: PhysicalDevice,
        required_device_extensions: &[*const i8],
    ) -> SrResult<bool> {
        let required_exts_set: HashSet<&CStr> = required_device_extensions
            .iter()
            .map(|p| unsafe { CStr::from_ptr(*p) })
            .collect();

        let available_exts =
            unsafe { instance.enumerate_device_extension_properties(physical_device) }.to_sr_result()?;

        let available_exts_set: HashSet<&CStr> = available_exts
            .iter()
            .map(|props| props.extension_name_as_c_str()) 
            .collect::<Result<_,_>>()
            .map_err(|e| {
                SrError::new(None, format!("Error while checking device extension support. Could not convert extension name to CStr with message: {e}"))
            })?;

        let all_exts_are_available = required_exts_set.is_subset(&available_exts_set);

        Ok(all_exts_are_available)
    }
}
