use ash::vk;
use std::{backtrace::BacktraceStatus, fmt::Display};

pub type SrResult<T> = std::result::Result<T, SrError>;

#[derive(Debug)]
pub enum ErrorSource {
    Vulkan(vk::Result),
    Gltf(gltf::Error),
    GpuAllocator(gpu_allocator::AllocationError),
    Custom(String),
}

#[derive(Debug)]
pub struct SrError {
    source: ErrorSource,
    description: String,
}

impl SrError {
    pub fn new_custom(error: String) -> Self {
        let description = format!("UNEXPECTED CUSTOM ERROR: {error}");

        Self::new(ErrorSource::Custom(error), description)
    }
    fn new(source: ErrorSource, description: String) -> Self {
        Self::new_with_backtrace(source, description, std::backtrace::Backtrace::capture())
    }

    fn new_with_backtrace(source: ErrorSource, description: String, bt: std::backtrace::Backtrace) -> Self {
        let description = if bt.status() == BacktraceStatus::Captured {
            format!("{description}\n{bt}")
        } else {
            format!("{description} (set RUST_BACKTRACE=1 to get a backtrace)")
        };

        Self { source, description }
    }
    pub fn get_source(&self) -> &ErrorSource {
        &self.source
    }
}

impl From<gltf::Error> for SrError {
    fn from(value: gltf::Error) -> Self {
        let description = match &value {
            //TODO: provide description for some errors
            e => format!("UNEXPECTED GLTF ERROR: {e}"),
        };

        Self::new(ErrorSource::Gltf(value), description)
    }
}

impl From<vk::Result> for SrError {
    fn from(value: vk::Result) -> Self {
        let description = match value {
            //TODO: provide description for some errors
            e => format!("UNEXPECTED VULKAN ERROR: {e}"),
        };

        Self::new(ErrorSource::Vulkan(value), description)
    }
}

impl From<gpu_allocator::AllocationError> for SrError {
    fn from(value: gpu_allocator::AllocationError) -> Self {
        let description = format!("UNEXPECTED GPU_ALLOCATOR ERROR: {value}");

        Self::new(ErrorSource::GpuAllocator(value), description)
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
            ErrorSource::Vulkan(error) => Some(error),
            ErrorSource::Gltf(error) => Some(error),
            ErrorSource::GpuAllocator(error) => Some(error),
            ErrorSource::Custom(_string) => None,
        }
    }
}
