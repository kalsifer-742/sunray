#[macro_use] extern crate inline_spirv;

mod demo_runner;
mod triangle_demo;
mod vkal;
mod vec3;

use crate::demo_runner::DemoRunner;
use crate::triangle_demo::TriangleDemo;

fn main() -> vkal::Result<()> {
    let app_name = "sunray - demo";
    let mut runner = DemoRunner::new(app_name, 640, 480)?;
    let mut demo = TriangleDemo::new(app_name, runner.get_window())?;
    runner.run_demo(&mut demo)?;
    Ok(())
}