//! GPU diagnostic / crash-analysis tooling integration.
//!
//! `DiagnosticTool` selects which (if any) vendor tool the renderer should
//! cooperate with: enabling the right instance/device extensions, pushing
//! the matching `p_next` config structs, and emitting per-command-buffer
//! checkpoint markers that survive a `VK_ERROR_DEVICE_LOST`.
//!
//! Only NVIDIA Aftermath is wired up today; the other variants are accepted
//! so call sites stay stable. They currently behave the same as `None` — when
//! we add RenderDoc API loader or AMD RGP markers, only this module changes.

use std::ffi::{CStr, c_void};

use ash::{ext, nv, vk};

#[cfg(feature = "nvidia-aftermath")]
use aftermath_rs::{Aftermath, AftermathDelegate, DescriptionBuilder};

#[cfg(not(feature = "nvidia-aftermath"))]
type Aftermath = ();

/// Minimal Aftermath delegate that writes crash dumps + shader debug info to
/// disk next to the executable. Tweak the output dir / naming when integrating
/// with a real dump-management pipeline.
#[cfg(feature = "nvidia-aftermath")]
struct DumpToDiskDelegate;

#[cfg(feature = "nvidia-aftermath")]
impl AftermathDelegate for DumpToDiskDelegate {
    fn dumped(&mut self, dump_data: &[u8]) {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let path = format!("sunray_{}.nv-gpudmp", ts);
        match std::fs::write(&path, dump_data) {
            Ok(()) => log::error!("NVIDIA Aftermath: wrote crash dump to {}", path),
            Err(e) => log::error!("NVIDIA Aftermath: failed to write dump '{}': {}", path, e),
        }
    }

    fn shader_debug_info(&mut self, data: &[u8]) {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let path = format!("sunray_shader_{}.nvdbg", ts);
        let _ = std::fs::write(&path, data);
    }

    fn description(&mut self, describe: &mut DescriptionBuilder) {
        describe.set_application_name(c"sunray");
        describe.set_application_version(c"0.2.0");
    }
}

/// Which GPU diagnostic backend (if any) to enable for this `Core`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DiagnosticTool {
    /// No diagnostics — production default.
    None,
    /// NVIDIA Nsight Aftermath: writes a `.nv-gpudmp` crash dump on DEVICE_LOST
    /// using `VK_NV_device_diagnostics_config` + `VK_NV_device_diagnostic_checkpoints`.
    NvidiaAftermath,
    /// NVIDIA Nsight Graphics frame debugger. Nsight injects its own capture
    /// layer (launch the app *through* Nsight), so nothing vendor-specific is
    /// wired here — instead we force `VK_EXT_debug_utils` on and emit
    /// object names + per-pass command-buffer labels so the capture is
    /// readable and the descriptor heap / barriers are inspectable (which
    /// RenderDoc can't do for `VK_EXT_descriptor_heap`).
    NvidiaNsightGraphics,
    /// RenderDoc capture API. *Stub* — not wired up yet.
    RenderDoc,
    /// AMD Radeon GPU Profiler markers. *Stub* — not wired up yet.
    RadeonGpuProfiler,
}

impl DiagnosticTool {
    /// Device-level Vulkan extensions this tool needs enabled.
    pub fn device_extensions(self) -> &'static [&'static CStr] {
        match self {
            DiagnosticTool::NvidiaAftermath => &[nv::device_diagnostics_config::NAME, nv::device_diagnostic_checkpoints::NAME],
            DiagnosticTool::None
            | DiagnosticTool::NvidiaNsightGraphics
            | DiagnosticTool::RenderDoc
            | DiagnosticTool::RadeonGpuProfiler => &[],
        }
    }

    /// Instance-level Vulkan extensions this tool needs enabled. None at the
    /// moment — debug_utils is unconditional and handled elsewhere.
    pub fn instance_extensions(self) -> &'static [&'static CStr] {
        &[]
    }

    /// Whether this tool benefits from `VK_EXT_debug_utils` object names +
    /// command-buffer labels (so a frame capture is human-readable). Forces the
    /// instance extension on even when the validation layer is off — capture
    /// tools (Nsight Graphics, RenderDoc) are usually run without validation.
    pub fn wants_debug_labels(self) -> bool {
        matches!(
            self,
            DiagnosticTool::NvidiaNsightGraphics | DiagnosticTool::RenderDoc | DiagnosticTool::NvidiaAftermath
        )
    }
}

/// Owns per-tool runtime state (e.g. the Aftermath crash-dump handler) and
/// the loaded device-extension function pointers used for checkpoint markers.
///
/// Lives on `Instance` (the Aftermath handle must outlive every Vulkan call
/// that could fault). The device-side pointers are populated lazily once the
/// `ash::Device` exists.
pub struct DiagnosticsContext {
    tool: DiagnosticTool,
    aftermath: Option<Aftermath>,
    checkpoints: Option<nv::device_diagnostic_checkpoints::Device>,
    /// Loaded when `VK_EXT_debug_utils` is enabled; drives object naming and
    /// per-pass command-buffer labels for GPU captures (Nsight Graphics /
    /// RenderDoc). `None` == labels/naming are no-ops.
    debug_utils: Option<ext::debug_utils::Device>,
}

impl DiagnosticsContext {
    /// Initialize the per-instance bits. For Aftermath this also boots up the
    /// crash-dump handler; do this *before* `vkCreateInstance` so any device
    /// loss during initialization still produces a dump.
    pub fn new(tool: DiagnosticTool) -> Self {
        let aftermath: Option<Aftermath> = match tool {
            DiagnosticTool::NvidiaAftermath => {
                #[cfg(feature = "nvidia-aftermath")]
                {
                    log::info!("NVIDIA Aftermath: GPU crash dump handler enabled — dumps land in cwd as sunray_*.nv-gpudmp");
                    Some(Aftermath::new(DumpToDiskDelegate))
                }
                #[cfg(not(feature = "nvidia-aftermath"))]
                {
                    log::warn!(
                        "DiagnosticTool::NvidiaAftermath selected but the `nvidia-aftermath` cargo \
                         feature is disabled — Vulkan checkpoints will still be set, but no \
                         .nv-gpudmp dump file will be written. Rebuild with `--features nvidia-aftermath` \
                         to enable the dump handler."
                    );
                    None
                }
            }
            _ => None,
        };
        Self {
            tool,
            aftermath,
            checkpoints: None,
            debug_utils: None,
        }
    }

    pub fn tool(&self) -> DiagnosticTool {
        self.tool
    }

    /// Load the device-side function pointers once the `ash::Device` is
    /// created. Safe to call when the tool is `None` (it just skips).
    ///
    /// `debug_utils_available` must reflect whether the instance actually
    /// enabled `VK_EXT_debug_utils` — loading the device functions without the
    /// instance extension yields null pointers that crash on call.
    pub fn load_device(&mut self, instance: &ash::Instance, device: &ash::Device, debug_utils_available: bool) {
        if matches!(self.tool, DiagnosticTool::NvidiaAftermath) {
            self.checkpoints = Some(nv::device_diagnostic_checkpoints::Device::load(instance, device));
        }
        if debug_utils_available {
            self.debug_utils = Some(ext::debug_utils::Device::load(instance, device));
        }
    }

    /// Open a labeled region in `cmd` (shows as a named scope in an Nsight
    /// Graphics / RenderDoc capture). Balanced by [`Self::cmd_end_label`].
    /// No-op when `VK_EXT_debug_utils` isn't loaded. `label` must be a
    /// nul-terminated string; callers pass a `&CStr`.
    pub fn cmd_begin_label(&self, cmd: vk::CommandBuffer, label: &CStr) {
        if let Some(du) = &self.debug_utils {
            let info = vk::DebugUtilsLabelEXT::default().label_name(label);
            unsafe { du.cmd_begin_debug_utils_label(cmd, &info) };
        }
    }

    /// Close the most recently opened label region in `cmd`.
    pub fn cmd_end_label(&self, cmd: vk::CommandBuffer) {
        if let Some(du) = &self.debug_utils {
            unsafe { du.cmd_end_debug_utils_label(cmd) };
        }
    }

    /// Attach a human-readable name to a Vulkan object so it's identifiable in a
    /// capture (e.g. "ReSTIR GI Reservoir Buffer" instead of a raw handle).
    /// No-op without `VK_EXT_debug_utils`. `name` must be nul-terminated. The
    /// object type is derived from the handle's `vk::Handle::TYPE`.
    pub fn set_object_name<H: vk::Handle>(&self, handle: H, name: &CStr) {
        if let Some(du) = &self.debug_utils {
            let info = vk::DebugUtilsObjectNameInfoEXT::default()
                .object_handle(handle)
                .object_name(name);
            // Failure here is purely cosmetic (naming) — log and continue.
            if let Err(e) = unsafe { du.set_debug_utils_object_name(&info) } {
                log::debug!("set_debug_utils_object_name failed: {e:?}");
            }
        }
    }

    /// Whether debug-utils labels/naming are active this run.
    pub fn labels_enabled(&self) -> bool {
        self.debug_utils.is_some()
    }

    /// Insert a checkpoint into the command stream. After a DEVICE_LOST, the
    /// driver reports which checkpoints completed — pinpointing which dispatch
    /// crashed. No-op when the active tool doesn't support it.
    ///
    /// `label` is borrowed for the duration of the checkpoint marker; callers
    /// should use `'static` strings (the marker is just a `*const c_void` to
    /// the driver, and Aftermath uses it as a stable identifier).
    pub fn cmd_set_checkpoint(&self, cmd: vk::CommandBuffer, label: &'static CStr) {
        if let Some(chk) = &self.checkpoints {
            unsafe { chk.cmd_set_checkpoint(cmd, label.as_ptr() as *const c_void) };
        }
    }

    /// Query the driver for which checkpoints had completed on `queue` at the
    /// time of the last fault, and log them. Call this after `VK_ERROR_DEVICE_LOST`
    /// to narrow down which dispatch crashed. No-op when the active tool
    /// doesn't support checkpoints.
    pub fn log_queue_checkpoints(&self, queue: vk::Queue) {
        let Some(chk) = &self.checkpoints else { return };
        unsafe {
            let len = chk.get_queue_checkpoint_data_len(queue);
            if len == 0 {
                log::error!("Diagnostics: queue reported 0 checkpoints completed.");
                return;
            }
            let mut data = vec![vk::CheckpointDataNV::default(); len];
            chk.get_queue_checkpoint_data(queue, &mut data);
            log::error!("Diagnostics: {} checkpoints completed on faulting queue:", len);
            for cp in &data {
                let label = if cp.p_checkpoint_marker.is_null() {
                    "<null>".to_string()
                } else {
                    CStr::from_ptr(cp.p_checkpoint_marker as *const i8)
                        .to_string_lossy()
                        .into_owned()
                };
                log::error!("    stage={:?}  marker={:?}", cp.stage, label);
            }
        }
    }
}

impl Drop for DiagnosticsContext {
    fn drop(&mut self) {
        // If Aftermath is active and a crash dump is being collected by the driver,
        // block here until it finishes (or times out) — otherwise we'd drop the
        // dump handler mid-collection and lose the .nv-gpudmp file. Polling kept
        // short (5s) to avoid hanging clean shutdowns.
        #[cfg(feature = "nvidia-aftermath")]
        if self.aftermath.is_some() {
            use aftermath_rs::Status;
            let status = Status::get();
            log::error!("NVIDIA Aftermath: dump status on shutdown = {:?}", status);
            if matches!(status, Status::CollectingData | Status::InvokingCallback | Status::Unknown) {
                log::error!("NVIDIA Aftermath: waiting up to 5s for the dump to flush…");
                let final_status = Status::wait_for_status(Some(std::time::Duration::from_secs(5)));
                log::error!("NVIDIA Aftermath: final dump status = {:?}", final_status);
            }
        }
        self.aftermath.take();
    }
}

/// Helper for callers wiring the `DeviceDiagnosticsConfigCreateInfoNV` p_next
/// onto `VkDeviceCreateInfo`. Returns `None` when the tool doesn't need one.
pub fn device_diagnostics_p_next(tool: DiagnosticTool) -> Option<vk::DeviceDiagnosticsConfigCreateInfoNV<'static>> {
    match tool {
        DiagnosticTool::NvidiaAftermath => Some(vk::DeviceDiagnosticsConfigCreateInfoNV::default().flags(
            vk::DeviceDiagnosticsConfigFlagsNV::ENABLE_SHADER_DEBUG_INFO
                | vk::DeviceDiagnosticsConfigFlagsNV::ENABLE_RESOURCE_TRACKING
                | vk::DeviceDiagnosticsConfigFlagsNV::ENABLE_SHADER_ERROR_REPORTING
                | vk::DeviceDiagnosticsConfigFlagsNV::ENABLE_AUTOMATIC_CHECKPOINTS,
        )),
        _ => None,
    }
}

// Make sure ext is referenced so the import line doesn't get flagged when
// the other variants gain real extension lists later.
#[allow(dead_code)]
const _UNUSED_EXT_IMPORT: Option<&'static CStr> = {
    let _ = ext::debug_utils::NAME;
    None
};
