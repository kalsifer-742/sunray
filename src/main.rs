use app::App;
use winit::event_loop::{ControlFlow, EventLoop};

mod app;

// In Vulkan 1.3 there is a new way of drawing things

// https://github.com/vulkano-rs/vulkano/tree/v0.35.0/examples/triangle-v1_3

// This version of the triangle example is written using dynamic rendering instead of render pass
// and framebuffer objects. If your device does not support Vulkan 1.3 or the
// `khr_dynamic_rendering` extension, or if you want to see how to support older versions, see the
// original triangle example.

// I'm doing things in the old way because the tutorials follow this approach

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(&event_loop);
    event_loop.run_app(&mut app).unwrap();
}
