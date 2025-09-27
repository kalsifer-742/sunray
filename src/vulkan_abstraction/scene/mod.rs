pub mod model;
pub use model::*;

use crate::vulkan_abstraction::{self};

pub struct Scene {
    pub models: Vec<vulkan_abstraction::Model>,
}
