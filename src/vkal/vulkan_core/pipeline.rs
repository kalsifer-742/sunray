use std::ops::Deref;
use std::rc::Rc;
use ash::vk;
use ash::vk::PipelineBindPoint;
use crate::vkal;

pub struct Pipeline {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    descriptor_sets: Vec<vk::DescriptorSet>,
    device: Rc<vkal::Device>,
}
impl Pipeline {
    pub fn new(
            device: Rc<vkal::Device>, render_pass: &vkal::RenderPass,
            shader: &vkal::Shader, layouts: &[vk::DescriptorSetLayoutCreateInfo]
        ) -> vkal::Result<Self>
    {
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

        let (descriptor_pool, layouts, sets) = Self::create_descriptor_sets(&device, layouts)?;

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts);
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
            descriptor_pool,
            descriptor_set_layouts: layouts,
            descriptor_sets: sets,
            device,
        })
    }

    pub fn get_descriptor_sets(&self) -> &[vk::DescriptorSet] { &self.descriptor_sets }

    pub fn cmd_bind(&self, cmd_buf: vk::CommandBuffer, pipeline_bind_point: PipelineBindPoint, img_idx: usize) {
        unsafe { self.device.cmd_bind_pipeline(cmd_buf, pipeline_bind_point, self.pipeline) };

        if self.descriptor_sets.len() > 0 {
            unsafe {
                self.device.cmd_bind_descriptor_sets(cmd_buf, pipeline_bind_point, self.pipeline_layout,
                                                     0, &self.descriptor_sets[img_idx..=img_idx], &[])
            };
        }
    }

    fn create_descriptor_sets(device: &vkal::Device, layouts_info: &[vk::DescriptorSetLayoutCreateInfo])
            -> vkal::Result<(
                vk::DescriptorPool,
                Vec<vk::DescriptorSetLayout>,
                Vec<vk::DescriptorSet>
            )>
    {
        let pool=
            if layouts_info.len() > 0 {
                let storage_pool_sizes = [
                    vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(layouts_info.len() as u32),

                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(layouts_info.len() as u32),
                ];
                let info = vk::DescriptorPoolCreateInfo::default()
                    .max_sets(layouts_info.len() as u32)
                    .pool_sizes(&storage_pool_sizes)
                    .flags(vk::DescriptorPoolCreateFlags::empty());

                unsafe { device.create_descriptor_pool(&info, vkal::NO_ALLOCATOR) }?
            } else {
                println!("0 descriptor set layouts passed to vkal::Pipeline::new; pool = vk::DescriptorPool::null()");
                vk::DescriptorPool::null()
            };

        let layouts =
            layouts_info.iter().map(|info|
                unsafe { device.create_descriptor_set_layout(info, vkal::NO_ALLOCATOR) }
            ).collect::<Result<Vec<_>, _>>()?;

        let sets = {
            let info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);
            unsafe { device.allocate_descriptor_sets(&info) }?
        };

        Ok((pool, layouts, sets))
    }
}
impl Deref for Pipeline {
    type Target = vk::Pipeline;
    fn deref(&self) -> &Self::Target { &self.pipeline }
}
impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            for layout in &self.descriptor_set_layouts {
                self.device.destroy_descriptor_set_layout(*layout, vkal::NO_ALLOCATOR);
            }
            self.device.destroy_pipeline_layout(self.pipeline_layout, vkal::NO_ALLOCATOR);
            self.device.destroy_descriptor_pool(self.descriptor_pool, vkal::NO_ALLOCATOR); // ok for it to be null here
            self.device.destroy_pipeline(self.pipeline, vkal::NO_ALLOCATOR);
        }
    }
}
