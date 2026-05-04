use crate::error::{ErrorSource, SrError, SrResult};

#[derive(Debug)]
pub enum GraphError{
    IncorrectRenderAccessFlags,

}

impl Into<ErrorSource> for GraphError {
    fn into(self) -> ErrorSource {
        ErrorSource::RenderGraph(self)
    }
}
