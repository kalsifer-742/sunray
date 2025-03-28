use std::rc::Rc;
use ash::vk;
use ash::vk::Fence;
use crate::vkal;


pub struct Queue {
    queue: vk::Queue,

    render_complete_sem: vk::Semaphore,
    present_complete_sem: vk::Semaphore,
    submit_complete_fence: vk::Fence,
    device: Rc<vkal::Device>,
    swapchain: vk::SwapchainKHR,
}
impl Queue {
    pub fn new(device: Rc<vkal::Device>, swapchain: &vkal::Swapchain, q_family: u32, q_index: u32) -> vkal::Result<Self> {
        let swapchain = **swapchain;
        let queue = unsafe { device.get_device_queue(q_family, q_index) };

        let create_semaphore =  || unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), vkal::NO_ALLOCATOR) };
        let render_complete_sem = create_semaphore()?;
        let present_complete_sem = create_semaphore()?;

        let create_fence = || {
            let fence_flags = vk::FenceCreateFlags::SIGNALED; // SIGNALED flag to start with a flag that's already signaled
            let fence_info = vk::FenceCreateInfo::default()
                .flags(fence_flags);

            unsafe { device.create_fence(&fence_info, vkal::NO_ALLOCATOR) }
        };
        let submit_complete_fence = create_fence()?;

        Ok(Self { queue, render_complete_sem, present_complete_sem, device, swapchain, submit_complete_fence })
    }

    #[allow(dead_code)]
    pub fn wait_idle(&self) -> vkal::Result<()> {
        unsafe { self.device.queue_wait_idle(self.queue) }?;
        Ok(())
    }

    pub fn acquire_next_image(&self) -> vkal::Result<u32> {
        unsafe {
            self.device.wait_for_fences(&[self.submit_complete_fence], true, u64::MAX)?;
            self.device.reset_fences(&[self.submit_complete_fence])?;
        }

        // acquire signals the fence it itself must wait, so only one acquire runs at a time
        // and we don't risk running with a signaled present_complete_sem
        let fence = self.submit_complete_fence;

        let dev = self.device.get_swapchain_device();
        let (index, _suboptimal_surface) = unsafe {
            dev.acquire_next_image(self.swapchain, u64::MAX, self.present_complete_sem, fence)
        }?;
        Ok(index)
    }

    pub fn submit_async(&self, command_buffer: vk::CommandBuffer) -> vkal::Result<()> {
        let wait_flags = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let wait_semaphores = [self.present_complete_sem];
        let command_buffers = [command_buffer];
        let signal_semaphores = &[self.render_complete_sem];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_flags)
            .command_buffers(&command_buffers)
            .signal_semaphores(signal_semaphores);

        unsafe { self.device.queue_submit(self.queue, &[submit_info], Fence::null()) }?;

        Ok(())
    }

    #[allow(dead_code)]
    pub fn submit_sync(&self, command_buffer: vk::CommandBuffer) -> vkal::Result<()> {
        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&[])
            .wait_dst_stage_mask(&[])
            .command_buffers(&command_buffers)
            .signal_semaphores(&[]);

        unsafe { self.device.queue_submit(self.queue, &[submit_info], Fence::null()) }?;
        Ok(())
    }

    pub fn present(&self, img_idx: u32) -> vkal::Result<()> {
        let wait_semaphores = &[self.render_complete_sem];
        let swapchains = [self.swapchain];
        let image_indices = [img_idx];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let dev = self.device.get_swapchain_device();
        unsafe { dev.queue_present(self.queue, &present_info) }?;
        Ok(())
    }
}
impl Drop for Queue {
    fn drop(&mut self) {
        self.wait_idle().unwrap();

        unsafe {
            self.device.destroy_semaphore(self.render_complete_sem, vkal::NO_ALLOCATOR);
            self.device.destroy_semaphore(self.present_complete_sem, vkal::NO_ALLOCATOR);
            self.device.destroy_fence(self.submit_complete_fence, vkal::NO_ALLOCATOR);
        }
    }
}