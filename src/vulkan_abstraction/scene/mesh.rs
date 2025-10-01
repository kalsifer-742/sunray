use crate::error::SrResult;

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct Vertex {
    #[allow(unused)]
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
}

pub struct Primitive {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

pub struct Mesh {
    primitives: Vec<Primitive>,
}

impl Mesh {
    pub fn new(primitives: Vec<Primitive>) -> SrResult<Self> {
        Ok(Self { primitives })
    }

    pub fn primitives(&self) -> &[Primitive] {
        &self.primitives
    }
}
