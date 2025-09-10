#![allow(dead_code)]
use ash::{ Device, vk, khr };
use ash::vk::Fence;
use crate::error::*;

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct Queue {
    queue: vk::Queue,

    render_complete_sems: Vec<vk::Semaphore>,
    img_available_sem: Vec<vk::Semaphore>,
    render_complete_fences: Vec<vk::Fence>,
    device: Device,
    swapchain_device: khr::swapchain::Device,

    current_frame: usize,
}
impl Queue {
    pub fn new(device: Device, swapchain_device: khr::swapchain::Device, q_family: u32, q_index: u32) -> SrResult<Self> {
        let queue = unsafe { device.get_device_queue(q_family, q_index) };

        let create_semaphore =  || unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)  };
        let render_complete_sems = (0..MAX_FRAMES_IN_FLIGHT).map(|_| create_semaphore()).collect::<Result<_, _>>()?;
        let img_available_sem = (0..MAX_FRAMES_IN_FLIGHT).map(|_| create_semaphore()).collect::<Result<_, _>>()?;

        let create_fence = || {
            let fence_flags = vk::FenceCreateFlags::SIGNALED; // SIGNALED flag to start with a flag that's already signaled
            let fence_info = vk::FenceCreateInfo::default()
                .flags(fence_flags);

            unsafe { device.create_fence(&fence_info, None)  }
        };
        let render_complete_fences = (0..MAX_FRAMES_IN_FLIGHT).map(|_| create_fence()).collect::<Result<_,_>>()?;

        Ok(Self { 
            queue, render_complete_sems, img_available_sem, device, swapchain_device,
             render_complete_fences, current_frame: 0
        })
    }

    #[allow(dead_code)]
    pub fn wait_idle(&self) -> SrResult<()> {
        unsafe { self.device.queue_wait_idle(self.queue)  }?;
        Ok(())
    }

    pub fn acquire_next_image(&self, swapchain: vk::SwapchainKHR) -> SrResult<u32> {
        let wait_fence = &self.render_complete_fences[self.current_frame..=self.current_frame];
        unsafe { self.device.wait_for_fences(wait_fence, true, u64::MAX)  }?;
        unsafe { self.device.reset_fences(wait_fence)  }?;
        

        let image_available_sem = self.img_available_sem[self.current_frame];
        let (index, _suboptimal_surface) = unsafe {
            self.swapchain_device.acquire_next_image(swapchain, u64::MAX, image_available_sem, Fence::null())
        }?;
        Ok(index)
    }

    pub fn submit_async(&self, command_buffer: vk::CommandBuffer) -> SrResult<()> {
        let wait_flags = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let wait_sem = &self.img_available_sem[self.current_frame..=self.current_frame];
        let command_buffers = [command_buffer];
        let signal_sem = &self.render_complete_sems[self.current_frame..=self.current_frame];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(wait_sem)
            .wait_dst_stage_mask(&wait_flags)
            .command_buffers(&command_buffers)
            .signal_semaphores(signal_sem);
        let signal_fence = self.render_complete_fences[self.current_frame];

        unsafe { self.device.queue_submit(self.queue, &[submit_info], signal_fence)  }?;

        Ok(())
    }

    #[allow(dead_code)]
    pub fn submit_sync(&self, command_buffer: vk::CommandBuffer) -> SrResult<()> {
        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&[])
            .wait_dst_stage_mask(&[])
            .command_buffers(&command_buffers)
            .signal_semaphores(&[]);

        unsafe { self.device.queue_submit(self.queue, &[submit_info], Fence::null())  }?;
        Ok(())
    }

    pub fn present(&mut self, swapchain: vk::SwapchainKHR, img_idx: u32) -> SrResult<()> {
        let wait_semaphores = &self.render_complete_sems[self.current_frame..=self.current_frame];
        let swapchains = [swapchain];
        let image_indices = [img_idx];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe { self.swapchain_device.queue_present(self.queue, &present_info)  }?;

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        Ok(())
    }
}
impl Drop for Queue {
    fn drop(&mut self) {
        match self.wait_idle() {
            Ok(()) => {}
            // do not panic: drop should not panic, since it is invoked for all objects after a panic; for example
            // if the logical device is lost all queues will be dropped on panic and they will all panic themselves and make the backtrace unreadable
            Err(e) => eprintln!("Queue::wait_idle (inside Queue::drop) returned {}", e.get_source().unwrap()),
        }

        unsafe {
            for s in self.render_complete_sems.iter() {
                self.device.destroy_semaphore(*s, None);
            }
            for s in self.img_available_sem.iter() {
                self.device.destroy_semaphore(*s, None);
            }
            for f in self.render_complete_fences.iter() {
                self.device.destroy_fence(*f, None);
            }
        }
    }
}
