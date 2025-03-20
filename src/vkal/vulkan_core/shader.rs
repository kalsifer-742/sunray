use std::error::Error;
use std::rc::Rc;
use crate::vkal;
use ash::vk;
pub struct Shader {
    device: Rc<vkal::Device>,
    vert: vk::ShaderModule,
    frag: vk::ShaderModule,
}

impl Shader {
    pub fn new(device: Rc<vkal::Device>, vert_code: &[u32], frag_code: &[u32]) -> Result<Self, Box<dyn Error>> {
        let frag_info  = vk::ShaderModuleCreateInfo::default()
            .code(&frag_code)
            .flags(vk::ShaderModuleCreateFlags::empty());
        let frag = unsafe { device.create_shader_module(&frag_info, vkal::NO_ALLOCATOR) }?;

        let vert_info  = frag_info.code(&vert_code);
        let vert = unsafe { device.create_shader_module(&vert_info, vkal::NO_ALLOCATOR) }?;

        Ok(Self{ device, vert, frag })
    }
    pub fn get_fragment(&self) -> vk::ShaderModule { self.frag }
    pub fn get_vertex(&self) -> vk::ShaderModule { self.vert }
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_shader_module(self.frag, vkal::NO_ALLOCATOR);
            self.device.destroy_shader_module(self.vert, vkal::NO_ALLOCATOR);
        }
    }
}