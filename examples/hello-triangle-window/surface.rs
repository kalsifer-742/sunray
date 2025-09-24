
use ash::{khr, vk};

pub struct Surface {
    surface_instance: khr::surface::Instance,
    surface: vk::SurfaceKHR,
}

impl Surface {
    pub fn new(entry: &ash::Entry, instance: &ash::Instance, surface: vk::SurfaceKHR) -> Self {
        let surface_instance = ash::khr::surface::Instance::new(&entry, &instance);

        Self { surface_instance, surface }
    }
    pub fn inner(&self) -> vk::SurfaceKHR { self.surface }
    #[allow(unused)]
    pub fn instance(&self) -> &khr::surface::Instance { &self.surface_instance }
}

impl Drop for Surface {
   fn drop(&mut self) {
        unsafe { self.surface_instance.destroy_surface(self.surface, None); }
    }
}
