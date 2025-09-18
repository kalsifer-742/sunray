use ash::vk;

use crate::{error::SrResult, vulkan_abstraction};

/*
Define what a camera is
In the raygen we use view_inverse, proj_inverse
is that all we need?
What is the interface for the user
*/
pub struct Camera {}

impl Default for Camera {
    fn default() -> Self {
        Self {}
    }
}

impl Camera {
    pub fn new() -> SrResult<Self> {
        Ok(Self {})
    }
}
