mod demo_runner;
mod triangle_demo;
mod vkal;

use std::error::Error;
use crate::demo_runner::DemoRunner;
use crate::triangle_demo::TriangleDemo;

fn main() -> Result<(), Box<dyn Error>> {
    let app_name = "sunray - demo";
    let runner = DemoRunner::new(app_name, 1920, 1080)?;
    let mut demo = TriangleDemo::new(app_name, runner.get_window())?;
    runner.run_demo(&mut demo)?;
    Ok(())
}