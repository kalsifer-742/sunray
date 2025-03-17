use std::error::Error;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use crate::demo_runner::Demo;
use crate::vkal;

#[allow(dead_code)]
pub struct TriangleDemo {
    vk_resources: vkal::VulkanResources,
}
impl TriangleDemo {
    pub fn new(app_name: &str, w: &winit::window::Window) -> Result<Self, Box<dyn Error>> {
        let params = vkal::OwnedInstanceParams {
            app_name,
            ..Default::default()
        };
        let vk_resources = vkal::VulkanResources::new(params, w.display_handle()?.as_raw(), w.window_handle()?.as_raw())?;

        Ok(Self { vk_resources, })
    }
}
impl Demo for TriangleDemo {
    fn render(&mut self) {
    }
}