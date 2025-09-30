use std::rc::Rc;

use crate::{error::SrResult, vulkan_abstraction};

use nalgebra as na;

pub mod mesh;
pub mod node;
pub mod primitive;

pub use mesh::*;
pub use node::*;
pub use primitive::*;

#[derive(Debug, Clone, Copy)]
struct Vertex {
    #[allow(unused)]
    position: [f32; 3],
}

pub struct Gltf {
    core: Rc<vulkan_abstraction::Core>,
    document: gltf::Document,
    buffers: Vec<gltf::buffer::Data>,
    _images: Vec<gltf::image::Data>,
}

impl Gltf {
    pub fn new(core: Rc<vulkan_abstraction::Core>, path: &str) -> SrResult<Self> {
        let (document, buffers, _images) = gltf::import(path)?;

        Ok(Self {
            core,
            document,
            buffers,
            _images,
        })
    }

    pub fn create_scenes(&self) -> SrResult<(usize, Vec<crate::Scene>)> {
        // find the defualt scene index
        let default_scene_index = match self.document.default_scene() {
            Some(s) => s.index(),
            None => 0,
        };

        let mut scenes = vec![];
        // load all scenes by default
        for gltf_scene in self.document.scenes() {
            let mut nodes = vec![];
            for gltf_node in gltf_scene.nodes() {
                let node = self.explore(&gltf_node)?;
                nodes.push(node);
            }
            scenes.push(crate::Scene::new(nodes)?);
        }

        Ok((default_scene_index, scenes))
    }

    fn explore(&self, gltf_node: &gltf::Node) -> SrResult<vulkan_abstraction::gltf::Node> {
        let (transform, mesh) = self.process_node(gltf_node)?;

        let children = if gltf_node.children().len() == 0 {
            None
        } else {
            let mut children = vec![];
            for gltf_child in gltf_node.children() {
                let child = self.explore(&gltf_child)?;
                children.push(child);
            }

            Some(children)
        };

        Ok(vulkan_abstraction::gltf::Node::new(
            transform, mesh, children,
        )?)
    }

    fn process_node(
        &self,
        gltf_node: &gltf::Node,
    ) -> SrResult<(na::Matrix4<f32>, Option<vulkan_abstraction::gltf::Mesh>)> {
        // the trasnform can also be given decomposed in: translation, rotation and scale
        // but the gltf crate takes care of this:
        // "If the transform is Decomposed, then the matrix is generated with the equation matrix = translation * rotation * scale."
        let transform = na::Matrix4::from(gltf_node.transform().matrix());
        let mut mesh = None;

        // TODO: this code does not manage multiple nodes pointing to the same meshes
        // fix proposal: check for the mesh id
        if let Some(gltf_mesh) = gltf_node.mesh() {
            let mut primitives = vec![];

            for primitive in gltf_mesh.primitives() {
                let reader = primitive.reader(|buffer| Some(&self.buffers[buffer.index()]));

                // get vertices positions
                let vertices = reader
                    .read_positions()
                    .unwrap()
                    .map(|position| vulkan_abstraction::gltf::Vertex { position })
                    .collect::<Vec<_>>();

                // get vertices index
                let indices = reader
                    .read_indices()
                    .unwrap()
                    .into_u32()
                    .collect::<Vec<_>>();

                let vertex_buffer = vulkan_abstraction::VertexBuffer::new_for_blas_from_data::<
                    Vertex,
                >(Rc::clone(&self.core), &vertices)?;
                let index_buffer = vulkan_abstraction::IndexBuffer::new_for_blas_from_data::<u32>(
                    Rc::clone(&self.core),
                    &indices,
                )?;

                primitives.push(vulkan_abstraction::gltf::Primitive::new(
                    vertex_buffer,
                    index_buffer,
                )?);
            }

            mesh = Some(vulkan_abstraction::gltf::Mesh::new(primitives)?);
        }

        Ok((transform, mesh))
    }
}
