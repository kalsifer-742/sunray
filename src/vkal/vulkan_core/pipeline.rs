use std::error::Error;
use std::ops::Deref;
use std::rc::Rc;
use ash::vk;
use crate::vkal;

pub struct Pipeline {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    device: Rc<vkal::Device>,
}
impl Pipeline {
    pub fn new(device: Rc<vkal::Device>, render_pass: &vkal::RenderPass, shader: &vkal::Shader) -> Result<Self, Box<dyn Error>> {
        let cache = vk::PipelineCache::null();

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(shader.get_vertex())
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(shader.get_fragment())
                .name(c"main"),
        ];

        let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let surface_size = device.get_physical_device_info().surface_capabilities.min_image_extent;
        let scissor = [vk::Rect2D {
            offset: vk::Offset2D{ x: 0, y: 0 },
            extent: surface_size
        }];
        let viewports = [vk::Viewport {
            x: 0., y: 0.,
            width: surface_size.width as f32, height: surface_size.height as f32,
            min_depth: 0.0, max_depth: 1.0
        }];
        let viewport_state_info = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissor);

        let rasterization_info = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)//TODO: replace with BACK to enable backface culling
            .front_face(vk::FrontFace::CLOCKWISE)
            .line_width(1.0);

        let multisample_state_info = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false)
            .min_sample_shading(1.0);

        let color_blend_attach_states = [
            vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA)
        ];

        let color_blend_state_info = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&color_blend_attach_states);

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&[]);
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(&layout_info, vkal::NO_ALLOCATOR)
        }?;


        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input_state_info)// ?
            .input_assembly_state(&input_assembly_state_info)
            .viewport_state(&viewport_state_info)
            .rasterization_state(&rasterization_info)
            .multisample_state(&multisample_state_info)
            .color_blend_state(&color_blend_state_info)
            .layout(pipeline_layout)
            .render_pass(**render_pass)
            .subpass(0)
            .base_pipeline_handle(vk::Pipeline::null())
            .base_pipeline_index(0);

        let pipeline_result = unsafe {
            device.create_graphics_pipelines(cache, &[info], vkal::NO_ALLOCATOR)
        };
        let pipeline = match pipeline_result {
            Ok(pipeline) => pipeline,
            Err((pipeline, result)) => {
                println!("Pipeline creation reported error: {result}");
                pipeline
            }
        };
        assert_eq!(pipeline.len(), 1);
        let pipeline = pipeline.into_iter().next().unwrap();

        Ok(Self {
            pipeline,
            pipeline_layout,
            device,
        })
    }
}
impl Deref for Pipeline {
    type Target = vk::Pipeline;
    fn deref(&self) -> &Self::Target { &self.pipeline }
}
impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline_layout(self.pipeline_layout, vkal::NO_ALLOCATOR);
            self.device.destroy_pipeline(self.pipeline, vkal::NO_ALLOCATOR);
        }
    }
}