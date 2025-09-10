use ash::vk;
use std::{backtrace::BacktraceStatus, fmt::Display};

pub type SrResult<T> = std::result::Result<T, SrError>;

#[derive(Debug)]
pub struct SrError {
    source: Option<vk::Result>,
    description: String,
}

impl SrError {
    pub fn new(source: Option<vk::Result>, description: String) -> Self {
        Self {
            source,
            description,
        }
    }
    pub fn from_vk_with_backtrace(vk_result: vk::Result, bt: std::backtrace::Backtrace) -> Self {
        let description = match vk_result {
            //TODO: provide description for some errors
            e => format!("UNEXPECTED VULKAN ERROR: {e}"),
        };
        let description = if bt.status() == BacktraceStatus::Captured { format!("{description}\n{bt}") } else { format!("{description} (set RUST_BACKTRACE=1 to get a backtrace)") };

        SrError::new(Some(vk_result), description)
    }
    pub fn get_source(&self) -> Option<vk::Result> { self.source }
}

impl From<vk::Result> for SrError {
    fn from(value: vk::Result) -> Self {
        Self::from_vk_with_backtrace(value, std::backtrace::Backtrace::capture())
    }
}

impl Display for SrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.description.fmt(f)
    }
}

impl std::error::Error for SrError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.source {
            Some(src) => Some(src),
            None => None,
        }
    }
}
