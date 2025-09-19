use ash::vk;

use crate::error::SrResult;

struct Image {
    image: vk::Image,
    image_format: vk::Format,
    image_extent: vk::Extent3D,
    //image_memory
    //image_view
}

impl Image {
    pub fn new(
        image: vk::Image,
        image_format: vk::Format,
        image_extent: (u32, u32),
    ) -> SrResult<Self> {
        todo!()
    }
}
