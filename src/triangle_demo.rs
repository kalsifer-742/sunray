use std::error::Error;
use std::rc::Rc;
use ash::vk;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use crate::demo_runner::Demo;
use crate::vkal;


#[derive(Clone, Copy)]
#[allow(dead_code)]
struct Vertex {
    pub pos: [f32; 3],
}

const VERTICES: [Vertex;3] = [
    Vertex { pos: [ 0.0,  0.7, 1.0 ] },
    Vertex { pos: [ 0.7, -0.7, 1.0 ] },
    Vertex { pos: [-0.7, -0.7, 1.0 ] },
];



#[allow(dead_code)]
pub struct TriangleDemo {
    bufcpy_cmd_buf: vk::CommandBuffer,
    vertex_buffer: vkal::Buffer,
    render_pass: vkal::RenderPass,
    pipeline: vkal::Pipeline,
    shader: vkal::Shader,
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

        let num_imgs = vk_core.get_swapchain().get_images().len();

        let bufs = vkal::cmd_buffer::new_vec(cmd_pool, vk_core.get_device(), num_imgs)?;
        vk_core.get_cmd_pool_mut().append_buffers(bufs);

        let render_pass = vkal::RenderPass::new(Rc::clone(vk_core.get_device()), vk_core.get_swapchain(), num_imgs)?;

        let shader = vkal::Shader::new(Rc::clone(vk_core.get_device()),
            glsl_vs!{r#"
                #version 460
                vec2 positions[3] = vec2[3](vec2(-0.7, 0.7), vec2(0.7, 0.7), vec2(0.0, -0.7));
                vec3 colors[3] = vec3[3](vec3(1,0,0), vec3(0,1,0), vec3(0,0,1));
                layout(location = 0) out vec3 color;
                void main() {
                    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
                    color = colors[gl_VertexIndex];
                }
            "#}.to_u32_slice().unwrap(),
            glsl_fs!{r#"
                #version 460
                layout(location = 0) in vec3 incolor;
                layout(location=0) out vec4 outcolor;
                void main() {
                    outcolor = vec4(incolor, 1.0);
                }
            "#}.to_u32_slice().unwrap())?;
        let pipeline = vkal::Pipeline::new(Rc::clone(vk_core.get_device()), &render_pass, &shader)?;

        Self::record_command_buffers(&vk_core, &render_pass, &pipeline)?;

        let bufcpy_cmd_buf = vkal::cmd_buffer::new(cmd_pool, vk_core.get_device())?;
        vk_core.get_cmd_pool_mut().get_buffers_mut().push(bufcpy_cmd_buf);

        let staging_buffer = vkal::Buffer::new_staging_from_data::<Vertex>(Rc::clone(vk_core.get_device()), &VERTICES)?;
        let vertex_buffer = vkal::Buffer::new_vertex::<Vertex>(Rc::clone(vk_core.get_device()), VERTICES.len())?;
        Self::copy_buffer(&vk_core, bufcpy_cmd_buf, &staging_buffer, &vertex_buffer)?;

        Ok(Self { bufcpy_cmd_buf, vk_core, render_pass, pipeline, shader, vertex_buffer })
    }

    fn copy_buffer(vk_core: &vkal::VulkanCore, bufcpy_cmd_buf: vk::CommandBuffer, src: &vkal::Buffer, dst: &vkal::Buffer) -> vkal::Result<()>{
        let device = vk_core.get_device();
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { device.begin_command_buffer(bufcpy_cmd_buf, &begin_info) }?;

        let regions = [
            vk::BufferCopy::default()
                .size(src.byte_size())
                .src_offset(0)
                .dst_offset(0)
        ];

        unsafe { device.cmd_copy_buffer(bufcpy_cmd_buf, **src, **dst, &regions) };

        unsafe { device.end_command_buffer(bufcpy_cmd_buf) }?;

        vk_core.get_queue().submit_sync(bufcpy_cmd_buf)?;
        vk_core.get_queue().wait_idle()?;

        Ok(())
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

trait ToU32Slice {
    fn to_u32_slice(&self) -> Option<&[u32]>;
}
impl ToU32Slice for [u8] {
    fn to_u32_slice(&self) -> Option<&[u32]> {
        let (pre, ret, post) = unsafe { self.align_to::<u32>() };
        if !pre.is_empty() {
            None
        } else if !post.is_empty() {
            None
        } else {
            Some(ret)
        }
    }
}