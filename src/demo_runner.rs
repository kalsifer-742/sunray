use std::cell::RefCell;
use std::error::Error;
use winit::event::{ElementState, Event, KeyEvent, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::window::WindowBuilder;

pub trait Demo {
    fn render(&mut self);

    fn on_suspend(&mut self) {}
    fn on_resume(&mut self) {}
    fn on_exit(&mut self) {}
}


pub struct DemoRunner {
    window: winit::window::Window,
    event_loop: RefCell<EventLoop<()>>,
}

impl DemoRunner {
    pub fn run_demo(&self, demo: &mut dyn Demo) -> Result<(), impl Error> {
        self.event_loop.borrow_mut().run_on_demand(|event, elwp| {
            elwp.set_control_flow(ControlFlow::Poll);
            match event {
                Event::WindowEvent {
                    event:
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        event: KeyEvent {
                            state: ElementState::Pressed,
                            logical_key: Key::Named(NamedKey::Escape), ..
                        }, ..
                    }, ..
                } => {
                    elwp.exit();
                }
                Event::AboutToWait => demo.render(),
                Event::LoopExiting => demo.on_exit(),
                Event::Resumed => demo.on_resume(),
                Event::Suspended => demo.on_suspend(),
                _ => {}
            }
        })
    }

    pub fn new(win_name: &str, window_width: u32, window_height: u32) -> Result<Self, Box<dyn Error>> {
        let event_loop = EventLoop::new()?;
        let window = WindowBuilder::new()
            .with_title(win_name)
            .with_inner_size(winit::dpi::LogicalSize::new(
                f64::from(window_width),
                f64::from(window_height),
            ))
            .build(&event_loop)
            .unwrap();

        let event_loop = RefCell::new(event_loop);

        Ok(Self { window, event_loop })
    }
    pub fn get_window(&self) -> &winit::window::Window { &self.window }
}
