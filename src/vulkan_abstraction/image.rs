use std::rc::Rc;

use ash::vk;

use crate::{error::SrResult, vulkan_abstraction};

pub struct Image {
    core: Rc<vulkan_abstraction::Core>,
    image: vk::Image,
    allocation: gpu_allocator::vulkan::Allocation,
    byte_size: usize,
    image_view: vk::ImageView,
    image_subresource_range: vk::ImageSubresourceRange,
    extent: vk::Extent3D,
    format: vk::Format,
}

impl Image {
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

            let byte_size = mem_reqs.size as usize;

            let allocation =
                core.allocator_mut()
                    .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                        name,
                        requirements: mem_reqs,
                        location,
                        linear: true, // Should be ok?
                        allocation_scheme:
                            gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
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

            unsafe {
                core.device()
                    .inner()
                    .create_image_view(&image_view_create_info, None)
            }
            .unwrap()
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
        })
    }

    pub fn map(&mut self) -> SrResult<&[u8]> {
        Ok(self.allocation.mapped_slice().unwrap())
    }

    pub fn image_view(&self) -> &vk::ImageView {
        &self.image_view
    }

    pub fn byte_size(&self) -> usize {
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
}

impl Drop for Image {
    fn drop(&mut self) {
        let device = self.core.device().inner();
        unsafe {
            device.destroy_image_view(self.image_view, None);
        }
        unsafe {
            device.destroy_image(self.image, None);
        }

        //need to take ownership to pass to free
        let allocation = std::mem::replace(
            &mut self.allocation,
            gpu_allocator::vulkan::Allocation::default(),
        );
        match self.core.allocator_mut().free(allocation) {
            Ok(()) => {}
            Err(e) => log::error!(
                "gpu_allocator::vulkan::Allocator::free returned {e} in sunray::vulkan_abstraction::Image::drop"
            ),
        }
    }
}
