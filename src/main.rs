#[macro_use] extern crate inline_spirv;

mod demo_runner;
mod triangle_demo;
mod vkal;
mod vec3;

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
#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    struct SetsTrueOnDrop { v: Rc<RefCell<bool>> }
    impl Drop for SetsTrueOnDrop {
        fn drop(&mut self) { *self.v.borrow_mut() = true; }
    }

    #[test]
    fn test() {
        let mut drop_ran = Rc::new(RefCell::new(false));

        let s = SetsTrueOnDrop { v: drop_ran.clone() };

        assert!(!*drop_ran.borrow());

        drop(s);

        assert!(*drop_ran.borrow());
    }
}