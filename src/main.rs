use app::App;
use winit::event_loop::EventLoop;

mod app;

fn main() {
    let event_loop = EventLoop::new().unwrap();

    let mut app = App::new(&event_loop);
    event_loop.run_app(&mut app).unwrap();
}
