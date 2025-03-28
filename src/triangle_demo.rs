use std::ops::Range;
use std::rc::Rc;
use std::time::SystemTime;
use ash::vk;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use crate::demo_runner::Demo;
use crate::vkal;
use crate::vec3::Vec3;
use nalgebra as na;

const VERTEX_SHADER : &[u32] = include_spirv!{"shaders/vert.glsl", vert, glsl};
const FRAGMENT_SHADER : &[u32] = include_spirv!{"shaders/frag.glsl", frag, glsl};

#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct Vertex {
    pub pos: Vec3,
    pub color: Vec3,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Uniforms {
    transform: na::Matrix4<f32>
}


#[allow(dead_code)]
pub struct TriangleDemo {
    bufcpy_cmd_buf: vk::CommandBuffer,
    vertex_buffer: vkal::Buffer,
    uniform_buffers: Vec<vkal::Buffer>,
    render_pass: vkal::RenderPass,
    pipeline: vkal::Pipeline,
    shader: vkal::Shader,
    vk_core: vkal::VulkanCore,

    start_time: SystemTime,
    frame_count: usize,
    last_lap_time: SystemTime,
    translation: na::Matrix4<f32>,
    projection: na::Matrix4<f32>,
}

impl TriangleDemo {
    pub fn new(app_name: &str, w: &winit::window::Window) -> vkal::Result<Self> {
        let params = vkal::InstanceParams {
            app_name,
            ..Default::default()
        };
        let mut vk_core = vkal::VulkanCore::new(params,
            w.display_handle().unwrap().as_raw(),
            w.window_handle().unwrap().as_raw())?;

        let cmd_pool = vk_core.get_cmd_pool().as_raw();

        let num_imgs = vk_core.get_swapchain().get_images().len();

        let bufs = vkal::cmd_buffer::new_vec(cmd_pool, vk_core.get_device(), num_imgs)?;
        vk_core.get_cmd_pool_mut().append_buffers(bufs);

        let render_pass = vkal::RenderPass::new(Rc::clone(vk_core.get_device()), vk_core.get_swapchain(), num_imgs)?;

        let shader = vkal::Shader::new(Rc::clone(vk_core.get_device()), &VERTEX_SHADER, &FRAGMENT_SHADER)?;

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX)
        ];
        let layout = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(vk::DescriptorSetLayoutCreateFlags::default())
            .bindings(&bindings);
        let layouts = vec![layout; num_imgs];

        let pipeline = vkal::Pipeline::new(Rc::clone(vk_core.get_device()), &render_pass, &shader, &layouts)?;

        let bufcpy_cmd_buf = vkal::cmd_buffer::new(cmd_pool, vk_core.get_device())?;
        vk_core.get_cmd_pool_mut().get_buffers_mut().push(bufcpy_cmd_buf);

        let vertices : [Vertex;3] = [
            Vertex { pos: Vec3::new( 0.0,  0.7, 0.0), color: Vec3::new(1.0, 0.0, 0.0), }, // top center
            Vertex { pos: Vec3::new( 0.7, -0.7, 0.0), color: Vec3::new(0.0, 1.0, 0.0), }, // bottom right
            Vertex { pos: Vec3::new(-0.7, -0.7, 0.0), color: Vec3::new(0.0, 0.0, 1.0), }, // bottom left
        ];
        let staging_buffer = vkal::Buffer::new_staging_from_data::<Vertex>(Rc::clone(vk_core.get_device()), &vertices)?;
        let vertex_buffer = vkal::Buffer::new_vertex::<Vertex>(Rc::clone(vk_core.get_device()), vertices.len())?;
        Self::copy_buffer(&vk_core, bufcpy_cmd_buf, &staging_buffer, &vertex_buffer)?;
        drop(staging_buffer);

        let uniform_buffers = (0..num_imgs).map(|_|
            vkal::Buffer::new_uniform::<Uniforms>(Rc::clone(vk_core.get_device()))
        ).collect::<Result<Vec<_>, _>>()?;

        //update the descriptor sets with the vertex buffer
        let buffer_info_vb = vk::DescriptorBufferInfo::default()
            .buffer(*vertex_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let buffer_infos_vb = [buffer_info_vb];

        let buffer_infos_ub = (0..num_imgs).map(|i| {
            vk::DescriptorBufferInfo::default()
                .buffer(*uniform_buffers[i])
                .offset(0)
                .range(vk::WHOLE_SIZE)
        }).collect::<Vec<_>>();

        let write_desc_sets = pipeline.get_descriptor_sets().iter().map(|desc_set| {
            vk::WriteDescriptorSet::default()
                .dst_set(*desc_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffer_infos_vb)
        }).chain(
            pipeline.get_descriptor_sets().iter().enumerate().map(|(idx, desc_set)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(*desc_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&buffer_infos_ub[idx..=idx])
            })
        ).collect::<Vec<_>>();

        // do this before recording the command buffers
        unsafe { vk_core.get_device().update_descriptor_sets(&write_desc_sets, &[]) };

        Self::record_command_buffers(&vk_core, &render_pass, &pipeline, 0..num_imgs)?;
        let translation = na::Matrix4::new_translation(&na::Vector3::new(0.0, 0.0, -4.0));
        let aspect = {
            let extent = vk_core.get_device().get_physical_device_info().surface_capabilities.min_image_extent;
            extent.width as f32 / extent.height as f32
        };
        let projection = na::Matrix4::new_perspective(aspect, 100.0, 0.01, 100.0);

        let start_time = SystemTime::now();
        let last_lap_time = start_time;


        Ok(Self {
            start_time, last_lap_time, frame_count: 0,
            translation, projection,
            bufcpy_cmd_buf, vk_core, render_pass, pipeline, shader,
            vertex_buffer, uniform_buffers
        })
    }

    fn copy_buffer(vk_core: &vkal::VulkanCore, bufcpy_cmd_buf: vk::CommandBuffer, src: &vkal::Buffer, dst: &vkal::Buffer) -> vkal::Result<()>{
        let device = vk_core.get_device();
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { device.begin_command_buffer(bufcpy_cmd_buf, &begin_info) }.unwrap();

        debug_assert!(src.byte_size() <= dst.byte_size());

        let regions = [
            vk::BufferCopy::default()
                .size(src.byte_size())
                .src_offset(0)
                .dst_offset(0)
        ];

        unsafe { device.cmd_copy_buffer(bufcpy_cmd_buf, **src, **dst, &regions) };

        unsafe { device.end_command_buffer(bufcpy_cmd_buf) }.unwrap();

        vk_core.get_queue().submit_sync(bufcpy_cmd_buf)?;
        // vk_core.get_queue().wait_idle()?;

        Ok(())
    }
    fn record_command_buffers(vk_core: &vkal::VulkanCore, render_pass: &vkal::RenderPass, pipeline: &vkal::Pipeline, buf_indices: Range<usize>) -> vkal::Result<()> {
        let device = vk_core.get_device();

        let cmd_buf_usage_flags = vk::CommandBufferUsageFlags::SIMULTANEOUS_USE;
        let cmd_buf_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(cmd_buf_usage_flags);

        let cmd_bufs = &vk_core.get_cmd_pool().get_buffers()[buf_indices];

        let clear_values = {
            let mut clear_value = vk::ClearValue::default();
            let mut clear_color = vk::ClearColorValue::default();
            //set the clear color to black
            for i in 0..3 {
                unsafe { clear_color.float32[i] = 0.; }
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
                device.begin_command_buffer(cmd_buf, &cmd_buf_begin_info).unwrap();

                device.cmd_begin_render_pass(cmd_buf, &render_pass_begin_info, vk::SubpassContents::INLINE);

                pipeline.cmd_bind(cmd_buf, vk::PipelineBindPoint::GRAPHICS, i);

                device.cmd_draw(cmd_buf, 3, 1, 0,0);

                device.cmd_end_render_pass(cmd_buf);

                device.end_command_buffer(cmd_buf).unwrap();
            }
        }
        Ok(())
    }

    fn update_uniform_buffer(&mut self, img_idx: u32, uniforms: &Uniforms) -> vkal::Result<()>{
        let buf = &mut self.uniform_buffers[img_idx as usize];

        let mapped_memory = buf.map::<Uniforms>()?;
        mapped_memory[0] = uniforms.clone();
        buf.unmap::<Uniforms>();
        Ok(())
    }

}
impl Demo for TriangleDemo {
    fn render(&mut self) -> vkal::Result<()> {
        let img_idx = self.vk_core.get_queue().acquire_next_image().unwrap();
        let time = SystemTime::now().duration_since(self.start_time).unwrap().as_millis() as f32 / 1000.0;

        // at every lap (a few frames) print avg framerate over the last lap
        const LAP_FRAMES: usize = 5_000; // the number of frames in a lap
        if (self.frame_count+1) % LAP_FRAMES == 0 {
            let now = SystemTime::now();
            let elapsed = now.duration_since(self.last_lap_time).unwrap().as_millis() as f32 / 1000.0;
            let framerate = LAP_FRAMES as f32 / elapsed;
            println!("current framerate: {framerate} fps");

            self.last_lap_time = now;
        }

        let uniforms = {
            let angle = time * 3.1415926 / 4.0;
            // let up = na::Vector3::y_axis();
            // let rotation = na::Matrix4::<f32>::from_axis_angle(&up, time); // rotate

            let sin = angle.sin();
            let cos = angle.cos();
            let rotation = na::Matrix4::<f32>::new(
                cos,    0.0,    sin,    0.0,
                0.0,    1.0,    0.0,    0.0,
                sin,    0.0,    cos,    0.0,
                0.0,    0.0,    0.0,    1.0,
            );
            let model = self.translation * rotation;

            let transform = self.projection * model;

            Uniforms { transform }
        };


        self.update_uniform_buffer(img_idx, &uniforms)?;
        let cmd_buf = self.vk_core.get_cmd_pool().get_buffers()[img_idx as usize];
        self.vk_core.get_queue().submit_async(cmd_buf)?;

        self.vk_core.get_queue().present(img_idx)?;

        self.frame_count += 1;

        Ok(())

    }
    fn on_exit(&mut self) -> vkal::Result<()> {
        let end_time = SystemTime::now();
        let frame_count = self.frame_count;
        let runtime = end_time.duration_since(self.start_time).unwrap().as_millis() as f64 / 1000.0;
        let avg_framerate = self.frame_count as f64 / runtime;


        println!("{frame_count} frames / {runtime} seconds = {avg_framerate} fps");

        self.vk_core.get_queue().wait_idle()?;
        Ok(())
    }
}