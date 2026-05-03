use ash::vk;
use crate::render_graph::graph::{Handle, RenderGraph, RenderPass, Resource, Setup};

pub struct RenderPassBuilder {
    render_pass: RenderPass,
}
impl RenderPassBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        let render_pass = RenderPass {
            read: vec![],
            write: vec![],
            render_fn: None,
            name: name.into(),
            idx: 0,
        };
        Self { render_pass }
    }

    fn submit(mut self, render_graph: &mut RenderGraph<Setup>) -> RenderPass {
        //TODO possible drop trait to submit as well
        self.render_pass.idx = render_graph.passes.len();
        self.render_pass
    }

    pub fn read<Res : Resource>(&mut self, resource : Handle<Res>, access_type : vk_sync_fork::AccessType   ) {
        if access_type.

        self.render_pass.read.push()

    }

}