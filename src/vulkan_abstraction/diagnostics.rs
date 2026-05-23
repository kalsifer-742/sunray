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

use std::ffi::{c_void, CStr};

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
    /// RenderDoc capture API. *Stub* — not wired up yet.
    RenderDoc,
    /// AMD Radeon GPU Profiler markers. *Stub* — not wired up yet.
    RadeonGpuProfiler,
}

impl DiagnosticTool {
    /// Device-level Vulkan extensions this tool needs enabled.
    pub fn device_extensions(self) -> &'static [&'static CStr] {
        match self {
            DiagnosticTool::NvidiaAftermath => &[
                nv::device_diagnostics_config::NAME,
                nv::device_diagnostic_checkpoints::NAME,
            ],
            DiagnosticTool::None | DiagnosticTool::RenderDoc | DiagnosticTool::RadeonGpuProfiler => &[],
        }
    }

    /// Instance-level Vulkan extensions this tool needs enabled. None at the
    /// moment — debug_utils is unconditional and handled elsewhere.
    pub fn instance_extensions(self) -> &'static [&'static CStr] {
        &[]
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
        }
    }

    pub fn tool(&self) -> DiagnosticTool {
        self.tool
    }

    /// Load the device-side function pointers once the `ash::Device` is
    /// created. Safe to call when the tool is `None` (it just skips).
    pub fn load_device(&mut self, instance: &ash::Instance, device: &ash::Device) {
        if matches!(self.tool, DiagnosticTool::NvidiaAftermath) {
            self.checkpoints = Some(nv::device_diagnostic_checkpoints::Device::load(instance, device));
        }
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
            if matches!(
                status,
                Status::CollectingData | Status::InvokingCallback | Status::Unknown
            ) {
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
        DiagnosticTool::NvidiaAftermath => Some(
            vk::DeviceDiagnosticsConfigCreateInfoNV::default().flags(
                vk::DeviceDiagnosticsConfigFlagsNV::ENABLE_SHADER_DEBUG_INFO
                    | vk::DeviceDiagnosticsConfigFlagsNV::ENABLE_RESOURCE_TRACKING
                    | vk::DeviceDiagnosticsConfigFlagsNV::ENABLE_SHADER_ERROR_REPORTING
                    | vk::DeviceDiagnosticsConfigFlagsNV::ENABLE_AUTOMATIC_CHECKPOINTS,
            ),
        ),
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
