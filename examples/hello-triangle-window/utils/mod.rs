use std::ffi::c_char;

use ash::{ext, khr, vk};
use winit::raw_window_handle_05::{RawDisplayHandle, RawWindowHandle};

use sunray::error::*;

/// Query the required instance extensions for creating a surface from a raw display handle.
///
/// This [`RawDisplayHandle`] can typically be acquired from a window, but is usually also
/// accessible earlier through an "event loop" concept to allow querying required instance
/// extensions and creation of a compatible Vulkan instance prior to creating a window.
///
/// The returned extensions will include all extension dependencies.
pub fn enumerate_required_extensions(display_handle: RawDisplayHandle) -> SrResult<&'static [*const c_char]> {
    let extensions = match display_handle {
        RawDisplayHandle::Windows(_) => {
            const WINDOWS_EXTS: [*const c_char; 2] = [khr::surface::NAME.as_ptr(), khr::win32_surface::NAME.as_ptr()];

            &WINDOWS_EXTS
        }

        RawDisplayHandle::Wayland(_) => {
            const WAYLAND_EXTS: [*const c_char; 2] = [khr::surface::NAME.as_ptr(), khr::wayland_surface::NAME.as_ptr()];

            &WAYLAND_EXTS
        }

        RawDisplayHandle::Xlib(_) => {
            const XLIB_EXTS: [*const c_char; 2] = [khr::surface::NAME.as_ptr(), khr::xlib_surface::NAME.as_ptr()];

            &XLIB_EXTS
        }

        RawDisplayHandle::Xcb(_) => {
            const XCB_EXTS: [*const c_char; 2] = [khr::surface::NAME.as_ptr(), khr::xcb_surface::NAME.as_ptr()];

            &XCB_EXTS
        }

        RawDisplayHandle::Android(_) => {
            const ANDROID_EXTS: [*const c_char; 2] = [khr::surface::NAME.as_ptr(), khr::android_surface::NAME.as_ptr()];

            &ANDROID_EXTS
        }

        RawDisplayHandle::AppKit(_) | RawDisplayHandle::UiKit(_) => {
            const METAL_EXTS: [*const c_char; 2] = [khr::surface::NAME.as_ptr(), ext::metal_surface::NAME.as_ptr()];

            &METAL_EXTS
        }

        _ => {
            return Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT.into());
        }
    };

    Ok(extensions)
}

/// Create a surface from a raw display and window handle.
///
/// `instance` must have created with platform specific surface extensions enabled, acquired
/// through [`enumerate_required_extensions()`].
///
/// # Safety
///
/// There is a [parent/child relation] between [`Instance`] and [`Entry`], and the resulting
/// [`vk::SurfaceKHR`].  The application must not [destroy][Instance::destroy_instance()] these
/// parent objects before first [destroying][surface::Instance::destroy_surface()] the returned
/// [`vk::SurfaceKHR`] child object.  [`vk::SurfaceKHR`] does _not_ implement [drop][drop()]
/// semantics and can only be destroyed via [`destroy_surface()`][surface::Instance::destroy_surface()].
///
/// See the [`Entry::create_instance()`] documentation for more destruction ordering rules on
/// [`Instance`].
///
/// The window represented by `window_handle` must be associated with the display connection
/// in `display_handle`.
///
/// `window_handle` and `display_handle` must be associated with a valid window and display
/// connection, which must not be destroyed for the lifetime of the returned [`vk::SurfaceKHR`].
///
/// [parent/child relation]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#fundamentals-objectmodel-lifetime
pub fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    display_handle: RawDisplayHandle,
    window_handle: RawWindowHandle,
    allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
) -> SrResult<vk::SurfaceKHR> {
    match (display_handle, window_handle) {
        (RawDisplayHandle::Windows(_), RawWindowHandle::Win32(window)) => {
            let surface_desc = vk::Win32SurfaceCreateInfoKHR::default()
                .hwnd(window.hwnd as isize)
                .hinstance(window.hinstance as isize);

            let surface_fn = khr::win32_surface::Instance::new(entry, instance);

            unsafe { surface_fn.create_win32_surface(&surface_desc, allocation_callbacks) }
        }

        (RawDisplayHandle::Wayland(display), RawWindowHandle::Wayland(window)) => {
            let surface_desc = vk::WaylandSurfaceCreateInfoKHR::default()
                .display(display.display)
                .surface(window.surface);

            let surface_fn = khr::wayland_surface::Instance::new(entry, instance);

            unsafe { surface_fn.create_wayland_surface(&surface_desc, allocation_callbacks) }
        }

        (RawDisplayHandle::Xlib(display), RawWindowHandle::Xlib(window)) => {
            let surface_desc = vk::XlibSurfaceCreateInfoKHR::default()
                .dpy(display.display)
                .window(window.window);

            let surface_fn = khr::xlib_surface::Instance::new(entry, instance);

            unsafe { surface_fn.create_xlib_surface(&surface_desc, allocation_callbacks) }
        }

        (RawDisplayHandle::Xcb(display), RawWindowHandle::Xcb(window)) => {
            let surface_desc = vk::XcbSurfaceCreateInfoKHR::default()
                .connection(display.connection)
                .window(window.window);

            let surface_fn = khr::xcb_surface::Instance::new(entry, instance);

            unsafe { surface_fn.create_xcb_surface(&surface_desc, allocation_callbacks) }
        }

        (RawDisplayHandle::Android(_), RawWindowHandle::AndroidNdk(window)) => {
            let surface_desc = vk::AndroidSurfaceCreateInfoKHR::default().window(window.a_native_window);

            let surface_fn = khr::android_surface::Instance::new(entry, instance);

            unsafe { surface_fn.create_android_surface(&surface_desc, allocation_callbacks) }
        }

        #[cfg(target_os = "macos")]
        (RawDisplayHandle::AppKit(_), RawWindowHandle::AppKit(window)) => {
            use raw_window_metal::{Layer, appkit};

            let layer = match appkit::metal_layer_from_handle(window) {
                Layer::Existing(layer) | Layer::Allocated(layer) => layer.cast(),
            };

            let surface_desc = vk::MetalSurfaceCreateInfoEXT::default().layer(&*layer);

            let surface_fn = metal_surface::Instance::new(entry, instance);

            unsafe { surface_fn.create_metal_surface(&surface_desc, allocation_callbacks) }
        }

        #[cfg(target_os = "ios")]
        (RawDisplayHandle::UiKit(_), RawWindowHandle::UiKit(window)) => {
            use raw_window_metal::{Layer, uikit};

            let layer = match uikit::metal_layer_from_handle(window) {
                Layer::Existing(layer) | Layer::Allocated(layer) => layer.cast(),
            };

            let surface_desc = vk::MetalSurfaceCreateInfoEXT::default().layer(&*layer);

            let surface_fn = metal_surface::Instance::new(entry, instance);

            unsafe { surface_fn.create_metal_surface(&surface_desc, allocation_callbacks) }
        }

        _ => Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT),
    }
    .map_err(SrError::from)
}
