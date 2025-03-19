use std::error::Error;
use ash::vk;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use crate::demo_runner::Demo;
use crate::vkal;

#[allow(dead_code)]
pub struct TriangleDemo {
    vk_core: vkal::VulkanCore,
}
impl TriangleDemo {
    pub fn new(app_name: &str, w: &winit::window::Window) -> Result<Self, Box<dyn Error>> {
        let params = vkal::InstanceParams {
            app_name,
            ..Default::default()
        };
        let mut vk_core = vkal::VulkanCore::new(params, w.display_handle()?.as_raw(), w.window_handle()?.as_raw())?;

        let cmd_pool = vk_core.get_cmd_pool().as_raw();

        let n = vk_core.get_swapchain().get_images().len() as u32;

        let bufs = vkal::cmd_buffer::new_vec(cmd_pool, vk_core.get_device(), n)?;
        vk_core.get_cmd_pool_mut().append_buffers(bufs);


        Self::record_command_buffers(&vk_core)?;

        Ok(Self { vk_core })
    }
    fn record_command_buffers(vk_core: &vkal::VulkanCore) -> Result<(), Box<dyn Error>> {
        let mut clear_color = vk::ClearColorValue::default();
        unsafe { clear_color.float32[0] = 1.0; }

        let image_subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        let image_subresource_ranges = [image_subresource_range];

        let device = vk_core.get_device();

        let cmd_buf_usage_flags = vk::CommandBufferUsageFlags::empty();
        let cmd_buf_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(cmd_buf_usage_flags);

        let cmd_bufs = vk_core.get_cmd_pool().get_buffers();
        let imgs = vk_core.get_swapchain().get_images();

        for (cmd_buf, img) in cmd_bufs.iter().cloned().zip(imgs.iter().cloned()) {
            unsafe {
                device.begin_command_buffer(cmd_buf, &cmd_buf_begin_info)?;

                device.cmd_clear_color_image(cmd_buf, img, vk::ImageLayout::GENERAL, &clear_color, &image_subresource_ranges);

                device.end_command_buffer(cmd_buf)?;
            }
        }
        Ok(())
    }

}
impl Demo for TriangleDemo {
    fn render(&mut self) -> Result<(), Box<dyn Error>> {
        let image_index = self.vk_core.get_queue().acquire_next_image()?;
        self.vk_core.get_queue().submit_async(self.vk_core.get_cmd_pool().get_buffers()[image_index as usize])?;
        self.vk_core.get_queue().present(image_index)?;
        Ok(())
    }
}