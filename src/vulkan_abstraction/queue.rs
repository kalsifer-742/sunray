use std::rc::Rc;

use crate::{error::*, vulkan_abstraction};
use ash::{vk};

pub const MAX_FRAMES_IN_FLIGHT: usize = 1;

pub struct Queue {
    queue: vk::Queue,

    render_complete_sems: Vec<vk::Semaphore>,
    img_available_sem: Vec<vk::Semaphore>,
    render_complete_fences: Vec<vulkan_abstraction::Fence>,
    device: Rc<vulkan_abstraction::Device>,
}
impl Queue {
    pub fn new(device: Rc<vulkan_abstraction::Device>, q_index: u32) -> SrResult<Self> {
        let queue = unsafe {
            device
                .inner()
                .get_device_queue(device.queue_family_index(), q_index)
        };

        let create_semaphore = || unsafe {
            device
                .inner()
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
        };
        let render_complete_sems = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| create_semaphore())
            .collect::<Result<_, _>>()?;
        let img_available_sem = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| create_semaphore())
            .collect::<Result<_, _>>()?;

        let create_fence = || {
            vulkan_abstraction::Fence::new_signaled(Rc::clone(&device))
        };
        let render_complete_fences = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| create_fence())
            .collect::<Result<_, _>>()?;

        Ok(Self {
            queue,
            render_complete_sems,
            img_available_sem,
            device,
            render_complete_fences,
        })
    }

    pub fn wait_idle(&self) -> SrResult<()> {
        unsafe { self.device.inner().queue_wait_idle(self.queue) }?;
        Ok(())
    }

    //TODO: fix synchronization
    pub fn submit_async(&self, command_buffer: vk::CommandBuffer) -> SrResult<()> {
        // let wait_flags = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let wait_flags = [vk::PipelineStageFlags::ALL_COMMANDS];
        // let wait_sem = &[self.img_available_sem[0]];
        let wait_sem = &[];
        let command_buffers = [command_buffer];
        // let signal_sem = &[self.render_complete_sems[0]];
        let signal_sem = &[];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(wait_sem)
            .wait_dst_stage_mask(&wait_flags)
            .command_buffers(&command_buffers)
            .signal_semaphores(signal_sem);
        // let signal_fence = self.render_complete_fences[0];
        let signal_fence = vk::Fence::null();

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

impl Drop for Queue {
    fn drop(&mut self) {
        match self.wait_idle() {
            Ok(()) => {}
            // do not panic: drop should not panic, since it is invoked for all objects after a panic; for example
            // if the logical device is lost all queues will be dropped on panic and they will all panic themselves and make the backtrace unreadable
            Err(e) => log::error!("Queue::wait_idle (inside Queue::drop) returned '{}'", e.get_source().unwrap()),
        }

        unsafe {
            for s in self.render_complete_sems.iter() {
                self.device.inner().destroy_semaphore(*s, None);
            }
            for s in self.img_available_sem.iter() {
                self.device.inner().destroy_semaphore(*s, None);
            }
        }
    }
}
