//! Optional profiling instrumentation.
//!
//! Everything in this module is gated behind the `profiling` Cargo feature. With
//! the feature **off** every entry point is an inlined no-op and the
//! instrumentation compiles away to nothing — a normal build pays no runtime cost
//! and pulls in no extra dependency.
//!
//! There are two layers, used together:
//!
//! * **GPU pass labels** ([`GpuProfiler`]) — `VK_EXT_debug_utils` command-buffer
//!   labels wrapped around each render-graph pass, so every pass shows up as a
//!   named region on the *GPU* timeline in any Vulkan profiler (NVIDIA Nsight
//!   Graphics, RenderDoc, AMD RGP). This is tool-neutral and is what `--features
//!   profiling` alone gives you. It relies on `VK_EXT_debug_utils`, which the
//!   instance enables automatically whenever the `profiling` feature is compiled
//!   in (see `Instance::new`).
//!
//! * **CPU scopes / marks** ([`Scope`], [`mark`]) — host-timeline ranges. The
//!   backend is chosen at compile time by the **vendor flag** `profiling-nvtx`:
//!   with it, scopes/marks are emitted as NVIDIA NVTX ranges so they land on the
//!   host timeline in Nsight Systems / Graphics; without it (plain `profiling`)
//!   they fall back to `log` (elapsed wall-time at `trace` level, allocation-free
//!   unless trace logging is actually enabled).
//!
//! `profiling-nvtx` is the "move the vendor-specific logging data" switch: it
//! `dep`-enables the `nvtx` crate and routes the CPU instrumentation into
//! NVIDIA's tooling. `--features profiling` keeps everything tool-neutral and
//! dependency-free; adding `--features profiling-nvtx` additionally streams the
//! NVIDIA NVTX data.
//!
//! Prefer the [`profile_scope!`](crate::profile_scope) macro for CPU scopes at
//! call sites — it expands to nothing when the feature is off, so the call sites
//! need no `#[cfg]` of their own.

/// One-time profiling setup. Cheap no-op unless a vendor backend is compiled in.
///
/// Under `profiling-nvtx` this names the calling (render) thread for NVTX so the
/// host timeline is labelled, and logs that the vendor backend is live.
#[inline]
pub fn init() {
    #[cfg(feature = "profiling-nvtx")]
    {
        nvtx::name_thread!("sunray-render");
        log::info!("[profiling] NVTX vendor instrumentation active (feature `profiling-nvtx`)");
    }
    #[cfg(all(feature = "profiling", not(feature = "profiling-nvtx")))]
    {
        log::info!(
            "[profiling] instrumentation active (feature `profiling`); enable `profiling-nvtx` for NVTX host-timeline data"
        );
    }
}

/// Emit an instantaneous marker on the host timeline (NVTX `mark`). No-op
/// without `profiling-nvtx`.
#[inline(always)]
pub fn mark(_name: &str) {
    #[cfg(feature = "profiling-nvtx")]
    nvtx::mark!("{}", _name);
}

// ─── CPU scope ──────────────────────────────────────────────────────────────

/// RAII host-timeline range. Created at the start of a region, ends the region
/// when dropped. Construct one with [`profile_scope!`](crate::profile_scope) or
/// directly with [`Scope::new`].
///
/// Backend by feature:
/// * `profiling-nvtx` → pushes/pops an NVTX thread range (properly nested per
///   thread, matching Rust's reverse-declaration drop order).
/// * `profiling` only → records the elapsed wall-time and logs it at `trace`.
#[cfg(feature = "profiling-nvtx")]
pub struct Scope {
    _priv: (),
}

#[cfg(feature = "profiling-nvtx")]
impl Scope {
    #[inline]
    pub fn new(name: &str) -> Self {
        nvtx::range_push!("{}", name);
        Self { _priv: () }
    }
}

#[cfg(feature = "profiling-nvtx")]
impl Drop for Scope {
    #[inline]
    fn drop(&mut self) {
        nvtx::range_pop!();
    }
}

#[cfg(all(feature = "profiling", not(feature = "profiling-nvtx")))]
pub struct Scope {
    // `None` (and so zero allocation / clock read) unless trace logging is on.
    timing: Option<(String, std::time::Instant)>,
}

#[cfg(all(feature = "profiling", not(feature = "profiling-nvtx")))]
impl Scope {
    #[inline]
    pub fn new(name: &str) -> Self {
        let timing = if log::log_enabled!(log::Level::Trace) {
            Some((name.to_owned(), std::time::Instant::now()))
        } else {
            None
        };
        Self { timing }
    }
}

#[cfg(all(feature = "profiling", not(feature = "profiling-nvtx")))]
impl Drop for Scope {
    #[inline]
    fn drop(&mut self) {
        if let Some((name, start)) = &self.timing {
            log::trace!("[profile] {name}: {:.3} ms", start.elapsed().as_secs_f64() * 1e3);
        }
    }
}

/// Open a CPU profiling [`Scope`] bound to the enclosing block. Expands to
/// nothing when the `profiling` feature is off, so it can be dropped into hot
/// paths unconditionally:
///
/// ```ignore
/// fn render_frame(&mut self) {
///     crate::profile_scope!("render frame");
///     // ... the whole function is one host-timeline range ...
/// }
/// ```
#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {
        #[cfg(feature = "profiling")]
        let _sr_profile_scope = $crate::profiling::Scope::new($name);
    };
}

// ─── GPU pass labels (VK_EXT_debug_utils) ────────────────────────────────────

/// Records named GPU-timeline regions (`vkCmdBeginDebugUtilsLabelEXT` /
/// `…End…`) around command-buffer work. One is held by the render graph and used
/// to bracket every pass, so a Vulkan profiler shows the passes by name on the
/// GPU timeline.
///
/// Construct with the instance/device handles; pass `enabled =
/// Core::debug_utils_enabled()` so it degrades to a no-op when the
/// `VK_EXT_debug_utils` extension isn't actually present (in which case calling
/// the label commands would be undefined behaviour).
#[cfg(feature = "profiling")]
pub struct GpuProfiler {
    debug_utils: Option<ash::ext::debug_utils::Device>,
}

#[cfg(feature = "profiling")]
impl GpuProfiler {
    #[inline]
    pub fn new(instance: &ash::Instance, device: &ash::Device, enabled: bool) -> Self {
        let debug_utils = enabled.then(|| ash::ext::debug_utils::Device::load(instance, device));
        Self { debug_utils }
    }

    /// Begin a GPU label on `cmd`; the returned guard ends it on drop. The label
    /// brackets every command recorded into `cmd` while the guard is alive.
    #[inline]
    pub fn scope(&self, cmd: ash::vk::CommandBuffer, name: &str) -> GpuScope<'_> {
        let Some(debug_utils) = &self.debug_utils else {
            return GpuScope { end: None };
        };
        // NUL-terminate for Vulkan; fall back to an empty label if `name` itself
        // contains an interior NUL (it never does — pass names are static).
        let label_name = std::ffi::CString::new(name).unwrap_or_default();
        let label = ash::vk::DebugUtilsLabelEXT::default().label_name(&label_name);
        unsafe { debug_utils.cmd_begin_debug_utils_label(cmd, &label) };
        GpuScope {
            end: Some((debug_utils, cmd)),
        }
    }
}

/// RAII guard that closes a [`GpuProfiler::scope`] GPU label on drop.
#[cfg(feature = "profiling")]
pub struct GpuScope<'a> {
    end: Option<(&'a ash::ext::debug_utils::Device, ash::vk::CommandBuffer)>,
}

#[cfg(feature = "profiling")]
impl Drop for GpuScope<'_> {
    #[inline]
    fn drop(&mut self) {
        if let Some((debug_utils, cmd)) = self.end {
            unsafe { debug_utils.cmd_end_debug_utils_label(cmd) };
        }
    }
}
