use std::rc::Rc;

use ash::vk;

use crate::{
    error::SrResult,
    vulkan_abstraction::{self, Scene, Vertex, scene},
};

struct GltfData {
    document: gltf::Document,
    buffers: Vec<gltf::buffer::Data>,
    _images: Vec<gltf::image::Data>,
}

pub struct Gltf {
    core: Rc<vulkan_abstraction::Core>,
    gltf_data: GltfData,
}

impl Gltf {
    pub fn new(core: Rc<vulkan_abstraction::Core>, path: &str) -> SrResult<Self> {
        let (document, buffers, _images) = gltf::import(path)?;
        let gltf_data = GltfData {
            document,
            buffers,
            _images,
        };

        Ok(Self { core, gltf_data })
    }

    pub fn create_scenes(&self) -> SrResult<(usize, Vec<vulkan_abstraction::Scene>)> {
        let mut scenes = vec![];

        // find the defualt scene - TODO: check if this index makes sense
        let default_scene_index = match self.gltf_data.document.default_scene() {
            Some(s) => s.index(),
            None => 0,
        };

        // load all scenes by default
        for scene in self.gltf_data.document.scenes() {
            let mut meshes = vec![];
            for node in scene.nodes() {
                self.explore(&node, &mut meshes);
            }
            scenes.push(vulkan_abstraction::Scene { meshes });
        }

        Ok((default_scene_index, scenes))
    }

    fn explore(&self, node: &gltf::Node, meshes: &mut Vec<vulkan_abstraction::Mesh>) {
        self.process_node(node, meshes);

        for child in node.children() {
            self.explore(&child, meshes);
        }
    }

    fn process_node(&self, node: &gltf::Node, meshes: &mut Vec<vulkan_abstraction::Mesh>) {
        // the trasnform can also be given decomposed in: translation, rotation and scale
        // but the gltf crate takes care of this:
        // "If the transform is Decomposed, then the matrix is generated with the equation matrix = translation * rotation * scale."
        let _transform = Self::to_vk_transform(node.transform().matrix());

        if let Some(mesh) = node.mesh() {
            let mut vertices = vec![];
            let mut indices = vec![];

            for (i, primitive) in mesh.primitives().enumerate() {
                let reader =
                    primitive.reader(|buffer| Some(&self.gltf_data.buffers[buffer.index()]));

                // get vertices positions
                reader
                    .read_positions()
                    .unwrap()
                    .for_each(|position| vertices.push(Vertex { position }));

                // get vertices index
                let indexes = reader.read_indices().unwrap().into_u32();
                indexes.clone().for_each(|i| indices.push(i));
            }
            meshes.push(vulkan_abstraction::Mesh::new(&self.core, &vertices, &indices).unwrap())
        }
    }

    fn to_vk_transform(transform: [[f32; 4]; 4]) -> vk::TransformMatrixKHR {
        let r0 = transform[0];
        let r1 = transform[1];
        let r2 = transform[2];
        let r3 = transform[3];

        #[rustfmt::skip]
    let matrix = [
        r0[0], r1[0], r2[0], r3[0],
        r0[1], r1[1], r2[1], r3[1],
        r0[2], r1[2], r2[2], r3[2],
    ];

        vk::TransformMatrixKHR { matrix }
    }
}

// // I think this is not optimal but i'm going to load all scenes from the start by default
// for (i, scene) in document.scenes().enumerate() {
//     log::debug!("scene #{}", i);

//     let mut models: Vec<vulkan_abstraction::Model> = Vec::new();

//     //maybe i should keep the node structure for the scene
//     for (j, node) in scene.nodes().enumerate() {
//         log::debug!("\tnode #{}", j);

//         let mut meshes: Vec<vulkan_abstraction::Mesh> = Vec::new();
//         let mut transforms: Vec<vk::TransformMatrixKHR> = Vec::new();
//         let mut vertices: Vec<vulkan_abstraction::Vertex> = Vec::new();
//         let mut indices: Vec<u32> = Vec::new();

//         if let Some(mesh) = node.mesh() {

//         }

//         for (z, child) in node.children().enumerate() {
//             log::debug!("\t\tchidlren #{} of node #{}", z, j);

//             // for now we are interested only in loading meshes
//             if child.mesh().is_some() {
//                 log::debug!("\t\t\tmesh #{}", z);

//                 let mesh = child.mesh().unwrap();

//                 // the trasnform can also be given decomposed in: translation, rotation and scale
//                 // but the gltf crate takes care of this:
//                 // "If the transform is Decomposed, then the matrix is generated with the equation matrix = translation * rotation * scale."
//                 transforms.push(to_vk_transform(child.transform().matrix()));
//                 log::debug!("\t\t\ttransform #{}", z);

//                 let vertex_offset = vertices.len();
//                 let index_offset = indices.len();
//                 let mut index_count = 0;
//                 for (k, primitive) in mesh.primitives().enumerate() {
//                     log::debug!("\t\t\t\tprimitive #{}", k);

//                     let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

//                     // get vertices positions
//                     reader
//                         .read_positions()
//                         .unwrap()
//                         .for_each(|position| vertices.push(Vertex { position }));

//                     // get vertices index
//                     let indexes = reader.read_indices().unwrap().into_u32();
//                     index_count = indexes.len();

//                     indexes.clone().for_each(|i| indices.push(i));
//                 }

//                 log::debug!(
//                     "\t\t\tmesh attributes: v_offset {} - i_offset {} - i_count - {}",
//                     vertex_offset,
//                     index_offset,
//                     index_count
//                 );

//                 meshes.push(vulkan_abstraction::Mesh {
//                     vertex_offset,
//                     index_offset,
//                     index_count,
//                 });
//             }
//         }

//         models.push(vulkan_abstraction::Model::new(
//             core,
//             &vertices,
//             &indices,
//             &transforms,
//             meshes,
//         )?);
//     }

//     scenes.push(vulkan_abstraction::Scene { models });
// }

// Ok((default_scene_index, scenes))
