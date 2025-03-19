use std::error::Error;
use std::ops::Deref;
use std::rc::Rc;
use ash::vk;
use winit::raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use crate::vkal;

pub struct Surface {
    surface: vk::SurfaceKHR,
    instance: Rc<vkal::Instance>,
}
impl Surface {
    pub fn new(entry: &ash::Entry, instance: Rc<vkal::Instance>, display_handle: RawDisplayHandle, window_handle: RawWindowHandle) -> Result<Self, Box<dyn Error>> {
        let surface = unsafe {
            ash_window::create_surface(&entry, &instance, display_handle, window_handle, vkal::NO_ALLOCATOR)
        }?;
        Ok(Self { surface, instance })
    }
    pub fn inner(&self) -> vk::SurfaceKHR { self.surface }
}
impl Drop for Surface {
    fn drop(&mut self) {
        unsafe { self.instance.surface_instance().destroy_surface(self.surface, vkal::NO_ALLOCATOR) };
    }
}
impl Deref for Surface {
    type Target = vk::SurfaceKHR;
    fn deref(&self) -> &Self::Target { &self.surface }
}