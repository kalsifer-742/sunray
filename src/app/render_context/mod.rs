use std::sync::Arc;

use vulkano::{
    device::Device,
    image::{view::ImageView, Image},
    instance::Instance,
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::ShaderModule,
    single_pass_renderpass,
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
    sync::{self, GpuFuture},
};
use vulkano::image::ImageUsage;
use winit::window::Window;

pub struct RenderContext {
    pub window: Arc<Window>,
    pub swapchain: Arc<Swapchain>,
    pub render_pass: Arc<RenderPass>,
    pub framebuffers: Vec<Arc<Framebuffer>>,
    pub pipeline: Arc<GraphicsPipeline>,
    pub recreate_swapchain: bool,
    pub previous_future: Option<Box<dyn GpuFuture>>,
}

impl RenderContext {
    pub fn new(device: Arc<Device>, instance: Arc<Instance>, window: Arc<Window>) -> Self {
        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();
        let (swapchain, images) = Self::get_swapchain(device.clone(), surface, window.clone());
        let render_pass = Self::get_render_pass(device.clone(), swapchain.clone());
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window.inner_size().into(),
            depth_range: 0.0..=1.0,
        };

        let vs = super::vs::load(device.clone()).unwrap();
        let fs = super::fs::load(device.clone()).unwrap();

        let pipeline = Self::get_pipeline(device.clone(), render_pass.clone(), viewport, vs, fs);

        let framebuffers = Self::get_framebuffers(&images, render_pass.clone());
        let previous_future = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);

        Self {
            window,
            swapchain,
            render_pass,
            framebuffers,
            pipeline,
            recreate_swapchain: false,
            previous_future,
        }
    }

    fn get_swapchain(
        device: Arc<Device>,
        surface: Arc<Surface>,
        window: Arc<Window>,
    ) -> (Arc<Swapchain>, Vec<Arc<Image>>) {
        let caps = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();
        let alpha = caps.supported_composite_alpha.into_iter().next().unwrap(); //another strange type wich i don't understand

        let image_format = device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;

        // circled_square code
        // let surface_formats = device.physical_device().surface_formats(&surface, Default::default()).unwrap();
        // let mut image_format= surface_formats[0].0;
        // for (format, color_space) in surface_formats.iter() {
        //     if *format == Format::B8G8R8A8_SRGB && *color_space == ColorSpace::SrgbNonLinear {
        //         image_format = *format;
        //     }
        // }

        Swapchain::new(
            device,
            surface,
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count,
                image_format,
                image_extent: window.inner_size().into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha: alpha,
                ..Default::default()
            },
        )
        .unwrap()
    }

    fn get_render_pass(device: Arc<Device>, swapchain: Arc<Swapchain>) -> Arc<RenderPass> {
        single_pass_renderpass!(
            device,
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap()
    }

    fn get_pipeline(
        device: Arc<Device>,
        render_pass: Arc<RenderPass>,
        viewport: Viewport,
        vs: Arc<ShaderModule>,
        fs: Arc<ShaderModule>,
    ) -> Arc<GraphicsPipeline> {
        let vs = vs.entry_point("main").unwrap();
        let fs = fs.entry_point("main").unwrap();

        let vertex_input_state = super::MyVertex::per_vertex().definition(&vs).unwrap();

        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        //i don't understand completely the black background
        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                //tell vulkan ho input should be assembled into privates...
                //The dafault is a triangle list which is what we are passing
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState {
                    viewports: [viewport].into_iter().collect(),
                    ..Default::default()
                }),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    }

    fn get_framebuffers(
        images: &Vec<Arc<Image>>,
        render_pass: Arc<RenderPass>,
    ) -> Vec<Arc<Framebuffer>> {
        images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>()
    }

    pub fn recreate_swapchain(&mut self) {
        //omitted match case to handle errors
        let (new_swapchain, new_images) = self
            .swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: self.window.inner_size().into(),
                ..self.swapchain.create_info()
            })
            .unwrap();
        self.swapchain = new_swapchain;
        self.framebuffers = Self::get_framebuffers(&new_images, self.render_pass.clone());
        self.recreate_swapchain = false;
    }
}
