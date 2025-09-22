use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{self, ControlFlow, EventLoop},
    raw_window_handle_05::{HasRawDisplayHandle, HasRawWindowHandle},
    window::Window,
};

#[derive(Default)]
struct App {
    window: Option<Window>,
    renderer: Option<sunray::Renderer>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        self.window = Some(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );

        self.renderer = Some(
            sunray::Renderer::new(
                self.window.as_ref().unwrap().inner_size().into(),
                self.window.as_ref().unwrap().raw_window_handle(),
                self.window.as_ref().unwrap().raw_display_handle(),
            )
            .unwrap()
        );
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
            WindowEvent::RedrawRequested => {
                match self.renderer.as_mut().unwrap().render() {
                    Ok(()) => {}
                    Err(error) => {
                        //no need to panic, sunray already takes care of the backtrace
                        log::error!("Sunray error: {}", error);
                        event_loop.exit();
                    }
                }
            }
            _ => (),
        }
    }
}

fn main() {
    log4rs::config::init_file("examples/log4rs.yaml", log4rs::config::Deserializers::new()).unwrap();

    if cfg!(debug_assertions) {
        //stdlib unfortunately completely pollutes trace log level, TODO somehow config stdlib/log to fix this?
        log::set_max_level(log::LevelFilter::Debug);
    } else {
        log::set_max_level(log::LevelFilter::Warn);
    }

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::default();
    let _ = event_loop.run_app(&mut app).unwrap();
}
