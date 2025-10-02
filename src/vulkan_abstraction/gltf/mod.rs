use std::{collections::HashMap, rc::Rc};

use crate::{error::SrResult, vulkan_abstraction};

use nalgebra as na;

pub mod mesh;
pub mod node;
pub mod primitive;

pub use mesh::*;
pub use node::*;
pub use primitive::*;

pub type PrimitiveDataMap =
    HashMap<vulkan_abstraction::gltf::PrimitiveUniqueKey, vulkan_abstraction::gltf::PrimitiveData>;


#[derive(Debug, Clone, Copy, Default)]
#[repr(C, packed)]
struct Vertex {
    // NOTE: don't move position or place any attributes before it:
    // the BLAS assumes that the vertex_buffer has a vec3 position attribute as its first (not necessarily the only) attribute in memory
    pub position:  [f32; 3],
    pub _padding0: [f32; 1],

    pub tex_coords:[f32; 2],
    pub _padding2: [f32; 2],
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

    pub fn create_scenes(&self) -> SrResult<(usize, Vec<crate::Scene>, Vec<PrimitiveDataMap>)> {
        // find the defualt scene index
        let default_scene_index = match self.document.default_scene() {
            Some(s) => s.index(),
            None => 0,
        };

        let mut scenes = vec![];
        let mut primitive_data_maps = vec![];
        // load all scenes by default
        for gltf_scene in self.document.scenes() {
            let mut nodes = vec![];
            let mut primitive_data_map: PrimitiveDataMap = PrimitiveDataMap::new();
            for gltf_node in gltf_scene.nodes() {
                let node = self.explore(&gltf_node, &mut primitive_data_map)?;
                nodes.push(node);
            }
            scenes.push(crate::Scene::new(nodes)?);
            primitive_data_maps.push(primitive_data_map);
        }

        Ok((default_scene_index, scenes, primitive_data_maps))
    }

    fn explore(
        &self,
        gltf_node: &gltf::Node,
        primitive_data_map: &mut PrimitiveDataMap,
    ) -> SrResult<vulkan_abstraction::gltf::Node> {
        let (transform, mesh) = self.process_node(gltf_node, primitive_data_map)?;

        let children = if gltf_node.children().len() == 0 {
            None
        } else {
            let mut children = vec![];
            for gltf_child in gltf_node.children() {
                let child = self.explore(&gltf_child, primitive_data_map)?;
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
        primitive_data_map: &mut PrimitiveDataMap,
    ) -> SrResult<(na::Matrix4<f32>, Option<vulkan_abstraction::gltf::Mesh>)> {
        // the trasnform can also be given decomposed in: translation, rotation and scale
        // but the gltf crate takes care of this:
        // "If the transform is Decomposed, then the matrix is generated with the equation matrix = translation * rotation * scale."
        let transform = na::Matrix4::from(gltf_node.transform().matrix());
        let mut mesh = None;

        // TODO: this code does not manage multiple nodes pointing to the same meshes
        // fix proposal: check for the primitive id
        if let Some(gltf_mesh) = gltf_node.mesh() {
            let mut primitives = vec![];

            //TODO: check for primitive support
            for (i, primitive) in gltf_mesh.primitives().enumerate() {
                let vertex_position_accessor_index = primitive
                    .attributes() // ATTRIBUTES are required in the spec
                    .filter(|(semantic, _)| *semantic == gltf::Semantic::Positions) // POSITION is always defined
                    .next()
                    .unwrap()
                    .1
                    .index();

                let indices_accessor_index = match primitive.indices() {
                    Some(accessor) => accessor.index(),
                    None => i, // this is a cheap fix in the case that the primitive is a non-indexed geometry
                };

                let primitive_unique_key = (vertex_position_accessor_index, indices_accessor_index);

                if !primitive_data_map.contains_key(&primitive_unique_key) {
                    let reader = primitive.reader(|buffer| Some(&self.buffers[buffer.index()]));

                    let vertex_buffer = {
                        // get vertices positions
                        let vertices =
                        std::iter::zip(
                            reader.read_positions().unwrap(),
                            reader.read_tex_coords(0).unwrap().into_f32(),
                        )
                            .map(|(position, tex_coords)| vulkan_abstraction::gltf::Vertex { position: position.into(), tex_coords:tex_coords.into(), ..Default::default() })
                            .collect::<Vec<_>>();

                        let vertex_buffer =
                            vulkan_abstraction::VertexBuffer::new_for_blas_from_data::<Vertex>(
                                Rc::clone(&self.core),
                                &vertices,
                            )?;

                        vertex_buffer
                    };

                    let index_buffer = {
                        let indices = if primitive.indices().is_some() {
                            // get vertices index
                            let indices = reader
                                .read_indices()
                                .unwrap()
                                .into_u32()
                                .collect::<Vec<_>>();

                            indices
                        } else {
                            // if the primitive is a non-indexed geometry we create the indices
                            let indices = (0..vertex_buffer.len() as u32 / 3).collect::<Vec<_>>();

                            indices
                        };

                        let index_buffer = vulkan_abstraction::IndexBuffer::new_for_blas_from_data::<u32>(
                            Rc::clone(&self.core), &indices
                        )?;

                        index_buffer
                    };

                    let primitive_data = vulkan_abstraction::gltf::PrimitiveData {
                        vertex_buffer,
                        index_buffer,
                    };

                    primitive_data_map.insert(primitive_unique_key, primitive_data);
                }

                primitives.push(vulkan_abstraction::gltf::Primitive {
                    unique_key: primitive_unique_key,
                });
            }
            mesh = Some(vulkan_abstraction::gltf::Mesh::new(primitives)?);
        }

        Ok((transform, mesh))
    }
}
