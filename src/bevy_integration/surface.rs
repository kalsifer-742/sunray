//! Surface creation from a Bevy `RawHandleWrapper`.
//!
//! Bevy hands us `raw-window-handle` **0.6** handles (see
//! `bevy_window::RawHandleWrapper::{get_window_handle, get_display_handle}`),
//! whereas `examples/window/utils` works with `raw-window-handle` 0.5. The two
//! helpers below are the 0.6 port of that logic: enumerate the instance
//! extensions a surface needs for the current display server, and create the
//! `vk::SurfaceKHR` from the raw handles.
//!
//! Platform arms other than Windows are `cfg`-gated so the Windows build (the
//! primary target) can never fail to compile on a field that only exists for
//! another windowing system.

use std::ffi::c_char;

use ash::{khr, vk};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::error::{SrError, SrResult};

/// Instance extensions required to create a surface for `display_handle`.
///
/// Returns a `'static` slice so it can be fed straight into
/// [`crate::Renderer::new_with_surface`], which wants `&'static [*const i8]`.
pub fn enumerate_required_extensions(display_handle: RawDisplayHandle) -> SrResult<&'static [*const c_char]> {
    let extensions: &'static [*const c_char] = match display_handle {
        RawDisplayHandle::Windows(_) => {
            const EXTS: [*const c_char; 2] = [khr::surface::NAME.as_ptr(), khr::win32_surface::NAME.as_ptr()];
            &EXTS
        }
        RawDisplayHandle::Wayland(_) => {
            const EXTS: [*const c_char; 2] = [khr::surface::NAME.as_ptr(), khr::wayland_surface::NAME.as_ptr()];
            &EXTS
        }
        RawDisplayHandle::Xlib(_) => {
            const EXTS: [*const c_char; 2] = [khr::surface::NAME.as_ptr(), khr::xlib_surface::NAME.as_ptr()];
            &EXTS
        }
        RawDisplayHandle::Xcb(_) => {
            const EXTS: [*const c_char; 2] = [khr::surface::NAME.as_ptr(), khr::xcb_surface::NAME.as_ptr()];
            &EXTS
        }
        RawDisplayHandle::Android(_) => {
            const EXTS: [*const c_char; 2] = [khr::surface::NAME.as_ptr(), khr::android_surface::NAME.as_ptr()];
            &EXTS
        }
        RawDisplayHandle::AppKit(_) | RawDisplayHandle::UiKit(_) => {
            const EXTS: [*const c_char; 2] = [khr::surface::NAME.as_ptr(), ash::ext::metal_surface::NAME.as_ptr()];
            &EXTS
        }
        _ => return Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT.into()),
    };

    Ok(extensions)
}

/// Create a `vk::SurfaceKHR` from raw-window-handle 0.6 handles.
///
/// # Safety
///
/// The `window_handle`/`display_handle` must be valid and must outlive the
/// returned surface. Caller is on a thread where using the handle is sound
/// (the single-threaded render SubApp runs on the main thread — see
/// `docs/bevy_integration.md`).
pub fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    display_handle: RawDisplayHandle,
    window_handle: RawWindowHandle,
) -> SrResult<vk::SurfaceKHR> {
    let result = match (display_handle, window_handle) {
        #[cfg(target_os = "windows")]
        (RawDisplayHandle::Windows(_), RawWindowHandle::Win32(window)) => {
            let surface_desc = vk::Win32SurfaceCreateInfoKHR::default()
                .hwnd(window.hwnd.get())
                .hinstance(window.hinstance.map_or(0, |h| h.get()));
            let surface_fn = khr::win32_surface::Instance::load(entry, instance);
            unsafe { surface_fn.create_win32_surface(&surface_desc, None) }
        }

        #[cfg(all(unix, not(any(target_os = "macos", target_os = "ios", target_os = "android"))))]
        (RawDisplayHandle::Wayland(display), RawWindowHandle::Wayland(window)) => {
            let surface_desc = vk::WaylandSurfaceCreateInfoKHR::default()
                .display(display.display.as_ptr())
                .surface(window.surface.as_ptr());
            let surface_fn = khr::wayland_surface::Instance::load(entry, instance);
            unsafe { surface_fn.create_wayland_surface(&surface_desc, None) }
        }

        #[cfg(all(unix, not(any(target_os = "macos", target_os = "ios", target_os = "android"))))]
        (RawDisplayHandle::Xlib(display), RawWindowHandle::Xlib(window)) => {
            let surface_desc = vk::XlibSurfaceCreateInfoKHR::default()
                .dpy(display.display.map_or(std::ptr::null_mut(), |d| d.as_ptr()).cast())
                .window(window.window);
            let surface_fn = khr::xlib_surface::Instance::load(entry, instance);
            unsafe { surface_fn.create_xlib_surface(&surface_desc, None) }
        }

        #[cfg(all(unix, not(any(target_os = "macos", target_os = "ios", target_os = "android"))))]
        (RawDisplayHandle::Xcb(display), RawWindowHandle::Xcb(window)) => {
            let surface_desc = vk::XcbSurfaceCreateInfoKHR::default()
                .connection(display.connection.map_or(std::ptr::null_mut(), |c| c.as_ptr()))
                .window(window.window.get());
            let surface_fn = khr::xcb_surface::Instance::load(entry, instance);
            unsafe { surface_fn.create_xcb_surface(&surface_desc, None) }
        }

        _ => return Err(SrError::new_custom("unsupported window/display handle for surface creation".to_string())),
    };

    result.map_err(SrError::from)
}
