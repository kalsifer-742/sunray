//! Slang shader compiler.
//!
//! First iteration: shaders compile at application startup via the `shader-slang`
//! crate, which links against the Slang runtime (`slang.dll` from the Vulkan SDK
//! or a standalone Slang install).
//!
//! Slang is configured to emit SPIR-V with the `spvDescriptorHeapEXT` capability
//! enabled, so `DescriptorHandle<T>` accesses lower to `SPV_EXT_descriptor_heap`
//! ops and the resulting SPIR-V plugs straight into a `VK_EXT_descriptor_heap`
//! pipeline (no descriptor sets).
//!
//! A later iteration will broaden the option set (defines, debug info, target
//! profile, on-disk caching) and possibly move to ahead-of-time compilation.

pub mod compiler;

pub use compiler::ShaderCompiler;
