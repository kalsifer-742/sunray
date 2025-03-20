use std::error::Error;
use std::ops::Deref;
use std::rc::Rc;
use ash::vk;
use crate::vkal;

pub struct RenderPass {
    device: Rc<vkal::Device>,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
}
impl RenderPass {
    pub fn new(device: Rc<vkal::Device>, swapchain: &vkal::Swapchain, num_imgs: usize) -> Result<Self, Box<dyn Error>> {
        let attachment_descriptions = [
            vk::AttachmentDescription::default()
            .format(device.get_physical_device_info().format.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        ];

        let attach_refs = [
            vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::GENERAL)
        ];

        let subpass_descriptions = [
            vk::SubpassDescription::default()
            .flags(vk::SubpassDescriptionFlags::empty())
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            // .input_attachments(&[])
            // .resolve_attachments(&[])
            // .depth_stencil_attachment(&[])
            // .preserve_attachments(&[])
            .color_attachments(&attach_refs)
        ];

        let render_pass_flags = vk::RenderPassCreateFlags::empty();
        let render_pass_info = vk::RenderPassCreateInfo::default()
            .flags(render_pass_flags)
            // .dependencies(&[])
            .attachments(&attachment_descriptions)
            .subpasses(&subpass_descriptions);
        let render_pass = unsafe { device.create_render_pass(&render_pass_info, vkal::NO_ALLOCATOR) }?;

        let wsize = device.get_physical_device_info().surface_capabilities.min_image_extent;

        let framebuffers = Self::build_framebuffers(&device, swapchain, num_imgs, wsize, &render_pass)?;

        Ok(Self { device, render_pass, framebuffers })
    }

    fn build_framebuffers(device: &vkal::Device, swapchain: &vkal::Swapchain, num_imgs: usize, size: vk::Extent2D, render_pass: &vk::RenderPass) -> Result<Vec<vk::Framebuffer>, Box<dyn Error>> {
        let fb_flags = vk::FramebufferCreateFlags::empty();
        let mut fb_info = vk::FramebufferCreateInfo::default()
            .flags(fb_flags)
            .render_pass(*render_pass)
            .layers(1)
            .width(size.width)
            .height(size.height);

        let framebuffers = (0..num_imgs).map(|i| unsafe {
            fb_info = fb_info.attachments(&swapchain.get_image_views()[i..=i]);
            device.create_framebuffer(&fb_info, vkal::NO_ALLOCATOR)
        }).collect::<Result<_, _>>()?;

        Ok(framebuffers)
    }
    unsafe fn destroy_framebuffers(&self) {
        for framebuffer in &self.framebuffers {
            self.device.destroy_framebuffer(*framebuffer, vkal::NO_ALLOCATOR);
        }
    }

    pub fn get_framebuffer(&self, i: usize) -> &vk::Framebuffer { &self.framebuffers[i] }
}
impl Drop for RenderPass {
    fn drop(&mut self) {
        unsafe {
            self.destroy_framebuffers();
            self.device.destroy_render_pass(self.render_pass, vkal::NO_ALLOCATOR);
        }
    }
}
impl Deref for RenderPass {
    type Target = vk::RenderPass;
    fn deref(&self) -> &Self::Target { &self.render_pass }
}