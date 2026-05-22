//! VK_EXT_descriptor_heap heap and slot allocator.
//!
//! Two heaps: one for resources (images, buffers, AS), one for samplers.
//! Each heap is a host-visible buffer; descriptors are written into host memory
//! at `index * stride` and the GPU reads them via the buffer's device address.
//!
//! Slots are allocated lazily by resources on first use and returned on Drop.

pub mod heap;
pub mod slot;
pub mod error;

pub use heap::*;
pub use slot::*;
