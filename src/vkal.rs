mod vulkan_core;
pub use vulkan_core::*;

// currently no support for allocation callbacks, this constant is used to express that
// instead of passing None to every function that requires an allocator
const NO_ALLOCATOR : Option<&ash::vk::AllocationCallbacks> = None;