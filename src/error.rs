use ash::vk;
use std::{backtrace::BacktraceStatus, fmt::Display};

pub type SrResult<T> = std::result::Result<T, SrError>;

#[derive(Debug)]
pub enum ErrorSource {
    VULKAN(vk::Result),
    GLTF(gltf::Error),
    ALLOCATOR(gpu_allocator::AllocationError),
    CUSTOM(String),
}

#[derive(Debug)]
pub struct SrError {
    source: Option<ErrorSource>,
    description: String,
}

impl SrError {
    pub fn new(source: Option<ErrorSource>, description: String) -> Self {
        Self::new_with_backtrace(source, description, std::backtrace::Backtrace::capture())
    }

    pub fn new_with_backtrace(source: Option<ErrorSource>, description: String, bt: std::backtrace::Backtrace) -> Self {
        let description = if bt.status() == BacktraceStatus::Captured {
            format!("{description}\n{bt}")
        } else {
            format!("{description} (set RUST_BACKTRACE=1 to get a backtrace)")
        };

        Self { source, description }
    }
    pub fn get_source(&self) -> Option<&ErrorSource> {
        self.source.as_ref()
    }
}

impl From<gltf::Error> for SrError {
    fn from(value: gltf::Error) -> Self {
        let description = match &value {
            //TODO: provide description for some errors
            e => format!("UNEXPECTED GLTF ERROR: {e}"),
        };

        Self::new(Some(ErrorSource::GLTF(value)), description)
    }
}

impl From<vk::Result> for SrError {
    fn from(value: vk::Result) -> Self {
        let description = match value {
            //TODO: provide description for some errors
            e => format!("UNEXPECTED VULKAN ERROR: {e}"),
        };

        Self::new(Some(ErrorSource::VULKAN(value)), description)
    }
}

impl From<gpu_allocator::AllocationError> for SrError {
    fn from(value: gpu_allocator::AllocationError) -> Self {
        let description = match value {
            //TODO: provide description for some errors
            ref e => format!("UNEXPECTED GPU_ALLOCATOR ERROR: {e}"),
        };

        Self::new(Some(ErrorSource::ALLOCATOR(value)), description)
    }
}

impl Display for ErrorSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorSource::VULKAN(e) => e.fmt(f),
            ErrorSource::GLTF(e) => e.fmt(f),
            ErrorSource::ALLOCATOR(e) => e.fmt(f),
            ErrorSource::CUSTOM(e) => e.fmt(f),
        }
    }
}

impl Display for SrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.description.fmt(f)
    }
}

impl std::error::Error for ErrorSource {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
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
