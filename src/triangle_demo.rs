use std::error::Error;
use std::rc::Rc;
use ash::vk;
// use glsl_to_spirv_macros::glsl_vs;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use crate::demo_runner::Demo;
use crate::vkal;


#[allow(dead_code)]
pub struct TriangleDemo {
    render_pass: vkal::RenderPass,
    pipeline: vkal::Pipeline,
    shader: vkal::Shader,
    vk_core: vkal::VulkanCore,
}


fn u32_slice_to_u8_slice<'a>(s: &'a[u8]) -> &'a[u32] {
    let (pre, ret, post) = unsafe { s.align_to::<u32>() };
    assert_eq!(pre.len(), 0);
    assert_eq!(post.len(), 0);
    ret
}


impl TriangleDemo {
    pub fn new(app_name: &str, w: &winit::window::Window) -> Result<Self, Box<dyn Error>> {
        let params = vkal::InstanceParams {
            app_name,
            ..Default::default()
        };
        let mut vk_core = vkal::VulkanCore::new(params, w.display_handle()?.as_raw(), w.window_handle()?.as_raw())?;

        let cmd_pool = vk_core.get_cmd_pool().as_raw();

        let num_imgs = vk_core.get_swapchain().get_images().len();

        let bufs = vkal::cmd_buffer::new_vec(cmd_pool, vk_core.get_device(), num_imgs)?;
        vk_core.get_cmd_pool_mut().append_buffers(bufs);

        let render_pass = vkal::RenderPass::new(Rc::clone(vk_core.get_device()), vk_core.get_swapchain(), num_imgs)?;

        let shader = vkal::Shader::new(Rc::clone(vk_core.get_device()), u32_slice_to_u8_slice(glsl_vs!{r#"
            #version 460
            vec2 positions[3] = vec2[3](vec2(-0.7, 0.7), vec2(0.7, 0.7), vec2(0.0, -0.7));
            vec3 colors[3] = vec3[3](vec3(1,0,0), vec3(0,1,0), vec3(0,0,1));
            layout(location = 0) out vec3 color;
            void main() {
                gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
                color = colors[gl_VertexIndex];
            }
        "#}), u32_slice_to_u8_slice(glsl_fs!{r#"
            #version 460
            layout(location = 0) in vec3 incolor;
            layout(location=0) out vec4 outcolor;
            void main() {
                outcolor = vec4(incolor, 1.0);
            }
        "#}))?;
        let pipeline = vkal::Pipeline::new(Rc::clone(vk_core.get_device()), &render_pass, &shader)?;

        Self::record_command_buffers(&vk_core, &render_pass, &pipeline)?;

        Ok(Self { vk_core, render_pass, pipeline, shader })
    }
    fn record_command_buffers(vk_core: &vkal::VulkanCore, render_pass: &vkal::RenderPass, pipeline: &vkal::Pipeline) -> Result<(), Box<dyn Error>> {
        let device = vk_core.get_device();

        let cmd_buf_usage_flags = vk::CommandBufferUsageFlags::SIMULTANEOUS_USE;
        let cmd_buf_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(cmd_buf_usage_flags);

        let cmd_bufs = vk_core.get_cmd_pool().get_buffers();

        let clear_values = {
            let mut clear_value = vk::ClearValue::default();
            let mut clear_color = vk::ClearColorValue::default();
            for i in 0..3 {
                unsafe { clear_color.float32[i] = 0.04; }
            }
            clear_value.color = clear_color;
            [clear_value]
        };

        let render_area_size = vk_core.get_device().get_physical_device_info().surface_capabilities.min_image_extent;

        let mut render_pass_begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(**render_pass)
            .clear_values(&clear_values)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: render_area_size,
            })
        ;

        for (i, cmd_buf) in cmd_bufs.iter().cloned().enumerate() {
            render_pass_begin_info.framebuffer = *render_pass.get_framebuffer(i);

            unsafe {
                device.begin_command_buffer(cmd_buf, &cmd_buf_begin_info)?;

                device.cmd_begin_render_pass(cmd_buf, &render_pass_begin_info, vk::SubpassContents::INLINE);

                device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::GRAPHICS, **pipeline);

                device.cmd_draw(cmd_buf, 3, 1, 0,0);

                device.cmd_end_render_pass(cmd_buf);

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
    fn on_exit(&mut self) -> Result<(), Box<dyn Error>> {
        self.vk_core.get_queue().wait_idle()?;
        Ok(())
    }
}