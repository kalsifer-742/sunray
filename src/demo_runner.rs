use std::error::Error;
use winit::event::{ElementState, Event, KeyEvent, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::window::WindowBuilder;
use crate::vkal;

pub trait Demo {
    fn render(&mut self) -> vkal::Result<()>;

    fn on_suspend(&mut self) -> vkal::Result<()> { Ok(()) }
    fn on_resume(&mut self) -> vkal::Result<()> { Ok(()) }
    fn on_exit(&mut self) -> vkal::Result<()> { Ok(()) }

    #[allow(dead_code)]
    fn on_resize(&mut self) -> vkal::Result<()> { Ok(()) }
}


pub struct DemoRunner {
    window: winit::window::Window,
    event_loop: EventLoop<()>,
}

impl DemoRunner {
    pub fn run_demo(&mut self, demo: &mut dyn Demo) -> Result<(), impl Error> {
        self.event_loop.run_on_demand(|event, elwp| {
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
                Event::AboutToWait => demo.render().unwrap(),
                Event::LoopExiting => demo.on_exit().unwrap(),
                Event::Resumed     => demo.on_resume().unwrap(),
                Event::Suspended   => demo.on_suspend().unwrap(),
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
            .build(&event_loop)?;

        Ok(Self { window, event_loop })
    }
    pub fn get_window(&self) -> &winit::window::Window { &self.window }
}
