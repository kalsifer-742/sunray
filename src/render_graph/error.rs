use crate::error::ErrorSource;
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum GraphError {
    IncorrectRenderAccessFlags,
    InvalidResourceRef,
    SwapchainAlreadyImported,
    MissingCoreForCompile,
}

impl Display for GraphError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Graph error {self:?} ")
    }
}

impl Error for GraphError {}

impl From<GraphError> for ErrorSource {
    fn from(val: GraphError) -> Self {
        ErrorSource::RenderGraph(val)
    }
}
