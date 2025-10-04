use std::rc::Rc;

use ash::vk;

use crate::{error::SrResult, vulkan_abstraction};

pub struct Image {
    core: Rc<vulkan_abstraction::Core>,
    image: vk::Image,
    allocation: gpu_allocator::vulkan::Allocation,
    byte_size: u64,
    image_view: vk::ImageView,
    image_subresource_range: vk::ImageSubresourceRange,
    extent: vk::Extent3D,
    format: vk::Format,

    // can be null
    sampler: vk::Sampler,
}

impl Image {
    pub fn new_from_data<T: Copy>(
        core: Rc<vulkan_abstraction::Core>,
        image_data: Vec<T>,
        extent: vk::Extent3D,
        format: vk::Format,
        tiling: vk::ImageTiling,
        location: gpu_allocator::MemoryLocation,
        usage_flags: vk::ImageUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        let staging_buffer = vulkan_abstraction::Buffer::new_staging_from_data(Rc::clone(&core), &image_data)?;
        let usage_flags = vk::ImageUsageFlags::TRANSFER_DST | usage_flags;
        let mut image = Self::new(core, extent, format, tiling, location, usage_flags, name)?;

        image.copy_from_buffer(&staging_buffer)?;

        Ok(image)
    }
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        extent: vk::Extent3D,
        format: vk::Format,
        tiling: vk::ImageTiling,
        location: gpu_allocator::MemoryLocation,
        usage_flags: vk::ImageUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        let image = {
            let image_create_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(extent)
                .flags(vk::ImageCreateFlags::empty())
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(tiling)
                .usage(usage_flags)
                .initial_layout(vk::ImageLayout::UNDEFINED);

            unsafe { core.device().inner().create_image(&image_create_info, None) }.unwrap()
        };

        let (allocation, byte_size) = {
            let mem_reqs = unsafe { core.device().inner().get_image_memory_requirements(image) };

            let byte_size = mem_reqs.size;

            let allocation = core.allocator_mut().allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name,
                requirements: mem_reqs,
                location,
                linear: tiling == vk::ImageTiling::LINEAR,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })?;

            (allocation, byte_size)
        };

        unsafe {
            core.device()
                .inner()
                .bind_image_memory(image, allocation.memory(), allocation.offset())
        }
        .unwrap();

        let image_subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let image_view = {
            let image_view_create_info = vk::ImageViewCreateInfo::default()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(image_subresource_range)
                .image(image);

            unsafe { core.device().inner().create_image_view(&image_view_create_info, None) }.unwrap()
        };

        Ok(Self {
            core,
            image,
            allocation,
            byte_size,
            image_view,
            image_subresource_range,
            extent,
            format,

            sampler: vk::Sampler::null(),
        })
    }

    pub fn map(&mut self) -> SrResult<&[u8]> {
        Ok(self.allocation.mapped_slice().unwrap())
    }

    // copies from a staging buffer mainly useful to copy from a staging buffer to a device buffer
    // note that this function internally changes the image's layout to TRANSFER_DST_OPTIMAL
    pub fn copy_from_buffer(&mut self, src: &vulkan_abstraction::Buffer) -> SrResult<()> {
        if src.is_null() {
            return Ok(());
        }

        let device = self.core.device().inner();
        let cmd_buf = vulkan_abstraction::cmd_buffer::new_command_buffer(self.core.cmd_pool(), device)?;

        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device.begin_command_buffer(cmd_buf, &begin_info)?;

            vulkan_abstraction::synchronization::cmd_image_memory_barrier(
                &self.core,
                cmd_buf,
                self.image,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            let region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(self.extent);

            device.cmd_copy_buffer_to_image(
                cmd_buf,
                src.inner(),
                self.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );

            device.end_command_buffer(cmd_buf)?;
        }

        self.core.queue().submit_sync(cmd_buf)?;

        unsafe { device.free_command_buffers(self.core.cmd_pool().inner(), &[cmd_buf]) };

        Ok(())
    }

    pub fn get_raw_image_data_with_no_padding(&mut self) -> SrResult<Vec<u8>> {
        //transform dst_image to bytes(correctly aligned)
        let image_sub = self.image_subresource_range();
        let image_subresource = vk::ImageSubresource {
            aspect_mask: image_sub.aspect_mask,
            mip_level: image_sub.base_mip_level,
            array_layer: image_sub.base_array_layer,
        };
        let subresource_layout = unsafe {
            self.core
                .device()
                .inner()
                .get_image_subresource_layout(self.inner(), image_subresource)
        };

        let size = self.extent().width as usize * self.extent().height as usize * std::mem::size_of::<u32>();
        let row_byte_size = self.extent().width as usize * std::mem::size_of::<u32>();
        let height = self.extent().height as usize;

        let mem = self.map()?;
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

    pub fn with_sampler(mut self, filter: vk::Filter) -> SrResult<Self> {
        let create_info = vk::SamplerCreateInfo::default()
            .flags(vk::SamplerCreateFlags::empty())
            // linear filtering both for magnification and minification
            .min_filter(filter)
            .mag_filter(filter)
            // repeat (tile) the texture on all axes
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            // use supported anisotropy
            // TODO: does this make sense for raytracing?
            .anisotropy_enable(true)
            .max_anisotropy(self.core.device().properties().limits.max_sampler_anisotropy)
            // use normalized ([0,1] range) coordinates
            .unnormalized_coordinates(false)
            // no need for a comparison function ("mainly used for percentage-closer filtering on shadow maps")
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            // mipmapping
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(0.0);
        self.sampler = unsafe { self.core.device().inner().create_sampler(&create_info, None) }?;
        Ok(self)
    }

    pub fn image_view(&self) -> vk::ImageView {
        self.image_view
    }

    pub fn byte_size(&self) -> u64 {
        self.byte_size
    }

    pub fn extent(&self) -> vk::Extent3D {
        self.extent
    }

    pub fn image_subresource_range(&self) -> &vk::ImageSubresourceRange {
        &self.image_subresource_range
    }

    pub fn format(&self) -> vk::Format {
        self.format
    }

    pub fn inner(&self) -> vk::Image {
        self.image
    }

    pub fn sampler(&self) -> vk::Sampler {
        assert_ne!(self.sampler, vk::Sampler::null());
        self.sampler
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        let device = self.core.device().inner();

        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.image_view, None);
            device.destroy_image(self.image, None);
        }

        //need to take ownership to pass to free
        let allocation = std::mem::replace(&mut self.allocation, gpu_allocator::vulkan::Allocation::default());
        match self.core.allocator_mut().free(allocation) {
            Ok(()) => {}
            Err(e) => {
                log::error!("gpu_allocator::vulkan::Allocator::free returned {e} in sunray::vulkan_abstraction::Image::drop")
            }
        }
    }
}
