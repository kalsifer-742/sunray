use crate::error::{ErrorSource, SrError, SrResult};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum GraphError {
    IncorrectRenderAccessFlags,
    InvalidResourceRef,
}

impl Display for GraphError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Graph error {self:?} ")
    }
}

impl Error for GraphError {}

impl Into<ErrorSource> for GraphError {
    fn into(self) -> ErrorSource {
        ErrorSource::RenderGraph(self)
    }
}
