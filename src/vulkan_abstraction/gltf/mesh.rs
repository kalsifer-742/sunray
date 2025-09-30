use crate::error::SrResult;

#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    #[allow(unused)]
    pub position: [f32; 3],
}

pub struct Mesh {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> SrResult<Self> {
        Ok(Self { vertices, indices })
    }

    pub fn vertices(&self) -> &[Vertex] {
        &self.vertices
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }
}
