use crate::{
    error::{ErrorSource, SrResult},
    vulkan_abstraction,
};
use ash::vk;
use std::rc::Rc;

pub struct Fence {
    device: Rc<vulkan_abstraction::Device>,
    handle: vk::Fence,
    fence_waited: bool,
}

impl Fence {
    pub fn new_signaled(device: Rc<vulkan_abstraction::Device>) -> SrResult<Self> {
        Self::new(device, vk::FenceCreateFlags::SIGNALED)
    }
    pub fn new_unsignaled(device: Rc<vulkan_abstraction::Device>) -> SrResult<Self> {
        Self::new(device, vk::FenceCreateFlags::empty())
    }
    pub fn new(
        device: Rc<vulkan_abstraction::Device>,
        flags: vk::FenceCreateFlags,
    ) -> SrResult<Self> {
        let fence_info = vk::FenceCreateInfo::default().flags(flags);

        let handle = unsafe { device.inner().create_fence(&fence_info, None) }?;

        Ok(Self {
            device,
            handle,
            fence_waited: true,
        })
    }
    pub fn reset(&mut self) -> SrResult<()> {
        self.fence_waited = true;
        unsafe { self.device.inner().reset_fences(&[self.handle]) }?;

        Ok(())
    }
    pub fn submit(&mut self) -> vk::Fence {
        self.fence_waited = false;
        self.handle
    }
    pub fn get_fence_for_wait(&mut self) -> SrResult<vk::Fence> {
        self.fence_waited = true;

        Ok(self.handle)
    }
    pub fn wait(&mut self) -> SrResult<()> {
        if !self.fence_waited {
            unsafe {
                self.device
                    .inner()
                    .wait_for_fences(&[self.handle], true, u64::MAX)?;
            }
            self.fence_waited = true;
        }

        Ok(())
    }
    pub unsafe fn inner(&self) -> vk::Fence {
        self.handle
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        // don't panic in drop, if possible
        match self.wait() {
            Ok(()) => {}
            Err(e) => match e.get_source() {
                Some(ErrorSource::VULKAN(e)) => {
                    log::warn!("VkWaitForFences returned {e:?} in Fence::drop")
                }
                _ => log::error!("VkWaitForFences returned {e} in Fence::drop"),
            },
        }
        unsafe { self.device.inner().destroy_fence(self.handle, None) };
    }
}
