use ash::vk;
use std::fmt::Display;

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

    pub fn from_vk_result(vk_result: vk::Result) -> Self {
        let description = match vk_result {
            //TODO: provide description for some errors
            e => format!("UNEXPECTED VULKAN ERROR: {}", e),
        };

        SrError::new(Some(vk_result), description)
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
