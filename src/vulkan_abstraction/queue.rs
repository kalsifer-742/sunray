use std::rc::Rc;

use crate::{error::*, vulkan_abstraction};
use ash::{vk};

pub const MAX_FRAMES_IN_FLIGHT: usize = 1;

pub struct Queue {
    queue: vk::Queue,

    device: Rc<vulkan_abstraction::Device>,
}
impl Queue {
    pub fn new(device: Rc<vulkan_abstraction::Device>, q_index: u32) -> SrResult<Self> {
        let queue = unsafe {
            device
                .inner()
                .get_device_queue(device.queue_family_index(), q_index)
        };

        Ok(Self {
            queue,
            device,
        })
    }

    pub fn wait_idle(&self) -> SrResult<()> {
        unsafe { self.device.inner().queue_wait_idle(self.queue) }?;
        Ok(())
    }

    pub fn submit_async(
        &self,
        command_buffer: vk::CommandBuffer,
        signal_fence: vk::Fence,
        wait_semaphores: &[vk::Semaphore],
        signal_semaphores: &[vk::Semaphore],
        wait_dst_stages: &[vk::PipelineStageFlags]
    ) -> SrResult<()> {
        // NOTE: consider using VkQueueSubmit2 from the extension VK_KHR_synchronization2 which adds more dst stages (VkPipelineStageFlags2) like BLIT
        if cfg!(debug_assertions) && wait_semaphores.len() != wait_dst_stages.len() {
            return Err(SrError::new(None, "Incorrect parameters to Queue::submit_async: wait_semaphores.len() != wait_dst_stages.len()".to_string()));
        }

        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(&wait_dst_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(signal_semaphores);

        unsafe {
            self.device
                .inner()
                .queue_submit(self.queue, &[submit_info], signal_fence)
        }?;

        Ok(())
    }

    pub fn submit_sync(&self, command_buffer: vk::CommandBuffer) -> SrResult<()> {
        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&[])
            .wait_dst_stage_mask(&[])
            .command_buffers(&command_buffers)
            .signal_semaphores(&[]);

        let fence = vulkan_abstraction::Fence::new_unsignaled(Rc::clone(&self.device))?;

        unsafe {
            self.device
                .inner()
                .queue_submit(self.queue, &[submit_info], fence.inner())
        }?;
        unsafe {
            self.device
                .inner()
                .wait_for_fences(&[fence.inner()], true, u64::MAX)
        }?;

        Ok(())
    }

    #[allow(dead_code)]
    pub fn inner(&self) -> vk::Queue { self.queue }
}
