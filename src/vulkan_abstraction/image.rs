use std::rc::Rc;

use ash::vk;

use crate::{error::SrResult, vulkan_abstraction};

pub struct Image {
    core: Rc<vulkan_abstraction::Core>,
    image: vk::Image,
    device_memory: vk::DeviceMemory,
    byte_size: usize,
    image_view: vk::ImageView,
    image_subresource_range: vk::ImageSubresourceRange,
    extent: vk::Extent3D,
    format: vk::Format,
    mapped_memory: Option<vulkan_abstraction::mapped_memory::RawMappedMemory>,
}

impl Image {
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        extent: vk::Extent3D,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage_flags: vk::ImageUsageFlags,
        memory_flags: vk::MemoryPropertyFlags,
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

        let (device_memory, byte_size) = {
            let mem_reqs = unsafe { core.device().inner().get_image_memory_requirements(image) };
            let mem_alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_reqs.size)
                .memory_type_index(vulkan_abstraction::get_memory_type_index(
                    &core,
                    memory_flags,
                    &mem_reqs,
                )?);

            let byte_size = mem_reqs.size as usize;
            let device_memory =
                unsafe { core.device().inner().allocate_memory(&mem_alloc_info, None) }.unwrap();

            (device_memory, byte_size)
        };

        unsafe {
            core.device()
                .inner()
                .bind_image_memory(image, device_memory, 0)
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
            device_memory,
            byte_size,
            image_view,
            image_subresource_range,
            extent,
            format,
            mapped_memory: None,
        })
    }

    pub fn device_memory(&self) -> &vk::DeviceMemory {
        &self.device_memory
    }

    //TODO: somehow unify this with Buffer's implementation
    pub fn map(&mut self, offset: usize) -> SrResult<&[u8]> {
        let p = unsafe {
            self.core.device().inner().map_memory(
                self.device_memory,
                offset as u64,
                self.byte_size as u64,
                vk::MemoryMapFlags::empty(),
            )
        }?;
        // let p = unsafe { p.add(offset as usize) };

        let raw_mem = unsafe {
            vulkan_abstraction::mapped_memory::RawMappedMemory::new(p, self.byte_size as usize)
        };
        self.mapped_memory = Some(raw_mem);
        let ret = self.mapped_memory.as_mut().unwrap().borrow();

        Ok(ret)
    }

    // correctness of unmap is checked by the borrow checker: it only works if the previous
    // mut borrow of self was already dropped. drop() calls unmap() if necessary
    pub fn unmap(&mut self) {
        self.mapped_memory = None;

        unsafe { self.core.device().inner().unmap_memory(self.device_memory) };
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
        if let Some(_mapped_memory) = &self.mapped_memory {
            self.unmap();
        }

        let device = self.core.device().inner();
        unsafe {
            device.destroy_image_view(self.image_view, None);
        }
        unsafe {
            device.destroy_image(self.image, None);
        }
        unsafe {
            device.free_memory(self.device_memory, None);
        }
    }
}
