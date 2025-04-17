use sunray::Core;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{self, ControlFlow, EventLoop},
    raw_window_handle_05::HasRawDisplayHandle,
    window::Window,
};

#[derive(Default)]
struct App {
    window: Option<Window>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        self.window = Some(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );

        let required_extensions = sunray::utils::enumerate_required_extensions(
            self.window.as_ref().unwrap().raw_display_handle(),
        )
        .unwrap();

        let _core = Core::new(required_extensions).unwrap();
    }

    fn window_event(
        &mut self,
        event_loop: &event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {}
            _ => (),
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::default();
    let _ = event_loop.run_app(&mut app).unwrap();
}
