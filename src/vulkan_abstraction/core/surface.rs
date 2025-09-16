use ash::{khr, vk};
use winit::raw_window_handle_05::{RawDisplayHandle, RawWindowHandle};

use crate::{error::SrResult, vulkan_abstraction};

pub struct Surface {
    surface_instance: khr::surface::Instance,
    surface: vk::SurfaceKHR,
}

impl Surface {
    pub fn new(entry: &ash::Entry, instance: &vulkan_abstraction::Instance, raw_display_handle: RawDisplayHandle, raw_window_handle: RawWindowHandle) -> SrResult<Self> {
        let surface_instance = ash::khr::surface::Instance::new(&entry, instance.inner());

        let surface = unsafe {
            crate::utils::create_surface(
                &entry,
                instance.inner(),
                raw_display_handle,
                raw_window_handle,
                None,
            )
        }?;
        Ok(Self { surface_instance, surface })
    }
    pub fn inner(&self) -> vk::SurfaceKHR { self.surface }
    pub fn instance(&self) -> &khr::surface::Instance { &self.surface_instance }
}

impl Drop for Surface {
   fn drop(&mut self) {
        unsafe { self.surface_instance.destroy_surface(self.surface, None); }
    }
}
