use crate::error::SrResult;
use crate::render_graph::graph::{
    Handle, PassResourceAccessSyncType, PassResourceAccessType, RenderGraph, RenderPass, Resource, ResourceRef, Setup,
};
use crate::render_graph::graph_error::GraphError;

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

    pub(crate) fn submit(mut self, render_graph: &mut RenderGraph<Setup>) -> RenderPass {
        //TODO possible drop trait to submit as well
        self.render_pass.idx = render_graph.passes.len();
        self.render_pass
    }

    pub fn read<Res: Resource>(&mut self, resource: &Handle<Res>, access_type: vk_sync_fork::AccessType) -> SrResult<()> {
        if !access_type.is_write() {
            self.render_pass.read.push(ResourceRef {
                raw: resource.raw,
                usage: PassResourceAccessType {
                    access_type,
                    sync_type: PassResourceAccessSyncType::SkipSyncIfSameAccessType,
                },
            });
            Ok(())
        } else {
            Err(GraphError::IncorrectRenderAccessFlags)
        }
    }

    pub fn write<Res: Resource>(&mut self, resource: &Handle<Res>, access_type: vk_sync_fork::AccessType) -> SrResult<()> {
        //TODO more complex not always sync write+write and read+write
        if access_type.is_write() {
            self.render_pass.read.push(ResourceRef {
                raw: resource.raw,
                usage: PassResourceAccessType {
                    access_type,
                    sync_type: PassResourceAccessSyncType::AlwaysSync,
                },
            });
            Ok(())
        } else {
            Err(GraphError::IncorrectRenderAccessFlags)
        }
    }
}



