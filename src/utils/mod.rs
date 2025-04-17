use std::ffi::c_char;

use ash::{
    ext::metal_surface,
    khr::{android_surface, surface, wayland_surface, win32_surface, xcb_surface, xlib_surface},
    vk,
};
use winit::raw_window_handle_05::RawDisplayHandle;

use crate::error::{SrError, SrResult};

/// Query the required instance extensions for creating a surface from a raw display handle.
///
/// This [`RawDisplayHandle`] can typically be acquired from a window, but is usually also
/// accessible earlier through an "event loop" concept to allow querying required instance
/// extensions and creation of a compatible Vulkan instance prior to creating a window.
///
/// The returned extensions will include all extension dependencies.
pub fn enumerate_required_extensions(
    display_handle: RawDisplayHandle,
) -> SrResult<&'static [*const c_char]> {
    let extensions = match display_handle {
        RawDisplayHandle::Windows(_) => {
            const WINDOWS_EXTS: [*const c_char; 2] =
                [surface::NAME.as_ptr(), win32_surface::NAME.as_ptr()];

            &WINDOWS_EXTS
        }

        RawDisplayHandle::Wayland(_) => {
            const WAYLAND_EXTS: [*const c_char; 2] =
                [surface::NAME.as_ptr(), wayland_surface::NAME.as_ptr()];

            &WAYLAND_EXTS
        }

        RawDisplayHandle::Xlib(_) => {
            const XLIB_EXTS: [*const c_char; 2] =
                [surface::NAME.as_ptr(), xlib_surface::NAME.as_ptr()];

            &XLIB_EXTS
        }

        RawDisplayHandle::Xcb(_) => {
            const XCB_EXTS: [*const c_char; 2] =
                [surface::NAME.as_ptr(), xcb_surface::NAME.as_ptr()];

            &XCB_EXTS
        }

        RawDisplayHandle::Android(_) => {
            const ANDROID_EXTS: [*const c_char; 2] =
                [surface::NAME.as_ptr(), android_surface::NAME.as_ptr()];

            &ANDROID_EXTS
        }

        RawDisplayHandle::AppKit(_) | RawDisplayHandle::UiKit(_) => {
            const METAL_EXTS: [*const c_char; 2] =
                [surface::NAME.as_ptr(), metal_surface::NAME.as_ptr()];

            &METAL_EXTS
        }

        _ => {
            return Err(SrError::from_vk_result(
                vk::Result::ERROR_EXTENSION_NOT_PRESENT,
            ));
        }
    };

    Ok(extensions)
}
