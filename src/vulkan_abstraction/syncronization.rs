//! # What a memory barrier does:
//! - Wait `src_pipeline_stage` to complete\
//! - Make **available** the writes perfomed in the `src_pipeline_stage` + `src_access_mask combination`\
//! - Make memory **visible** to the `dst_pipeline_stage` + `dst_access_mask` combination\
//! - Start `dst_pipeline_stage`
//!
//! # Theory:
//! Execution order and memory order are two different things and we have to manage them individually.\
//! Also GPUs have different caches that we need to make coherent to avoid errors.
//!
//! ## Memory concepts:
//! - available (flush caches) - the top level cache contains the most up-to-date data
//! - visible (invalidate caches) - the memory is **available** and is **visible** to the pipeline stage + access mask combination
//!
//! ### References:
//! - https://themaister.net/blog/2019/08/14/yet-another-blog-explaining-vulkan-synchronization/

use ash::vk;

use crate::vulkan_abstraction;

/// # Creates a memory barrier
///
/// # Parameters:
/// - `src_pipeline_stage` - The pipeline stage that must complete before the barrier
/// - `dst_pipeline_stage` - The pipeline stage that can start after the barrier
pub unsafe fn cmd_memory_barrier(
    core: &vulkan_abstraction::Core,
    cmd_buf: vk::CommandBuffer,
    src_pipeline_stage: vk::PipelineStageFlags,
    dst_pipeline_stage: vk::PipelineStageFlags,
    memory_barriers: &[vk::MemoryBarrier],
    buffer_memory_barriers: &[vk::BufferMemoryBarrier],
    image_memory_barriers: &[vk::ImageMemoryBarrier],
) {
    unsafe {
        core.device().inner().cmd_pipeline_barrier(
            cmd_buf,
            src_pipeline_stage,
            dst_pipeline_stage,
            vk::DependencyFlags::empty(),
            memory_barriers,
            buffer_memory_barriers,
            image_memory_barriers,
        );
    }
}

/// # Creates an image memory barrier
///
/// # Parameters
/// - old_layout
/// - new_layout
pub unsafe fn cmd_image_memory_barrier(
    core: &vulkan_abstraction::Core,
    cmd_buf: vk::CommandBuffer,
    image: vk::Image,
    src_pipeline_stage: vk::PipelineStageFlags,
    dst_pipeline_stage: vk::PipelineStageFlags,
    src_access_mask: vk::AccessFlags,
    dst_access_mask: vk::AccessFlags,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let image_memory_barrier = vk::ImageMemoryBarrier::default()
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );

    unsafe {
        cmd_memory_barrier(
            core,
            cmd_buf,
            src_pipeline_stage,
            dst_pipeline_stage,
            &[],
            &[],
            &[image_memory_barrier],
        );
    };
}
