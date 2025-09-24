use std::rc::Rc;
use ash::vk;
use crate::{error::SrResult, vulkan_abstraction};


pub struct Fence {
    device: Rc<vulkan_abstraction::Device>,
    handle: vk::Fence,
}

impl Fence {
    pub fn new_signaled(device: Rc<vulkan_abstraction::Device>) -> SrResult<Self> {
        Self::new(device, vk::FenceCreateFlags::SIGNALED)
    }
    pub fn new_unsignaled(device: Rc<vulkan_abstraction::Device>) -> SrResult<Self> {
        Self::new(device, vk::FenceCreateFlags::empty())
    }
    pub fn new(device: Rc<vulkan_abstraction::Device>, flags: vk::FenceCreateFlags) -> SrResult<Self> {
        let fence_info = vk::FenceCreateInfo::default().flags(flags);

        let handle = unsafe { device.inner().create_fence(&fence_info, None) }?;

        Ok(Self{
            device, handle
        })
    }
    pub fn inner(&self) -> vk::Fence { self.handle }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe { self.device.inner().destroy_fence(self.handle, None) };
    }
}
