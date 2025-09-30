use crate::{error::SrResult, vulkan_abstraction};

use nalgebra as na;

pub struct Gltf {
    document: gltf::Document,
    buffers: Vec<gltf::buffer::Data>,
    _images: Vec<gltf::image::Data>,
}

impl Gltf {
    pub fn new(path: &str) -> SrResult<Self> {
        let (document, buffers, _images) = gltf::import(path)?;

        Ok(Self {
            document,
            buffers,
            _images,
        })
    }

    pub fn create_scenes(&self) -> SrResult<(usize, Vec<vulkan_abstraction::Scene>)> {
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
            scenes.push(vulkan_abstraction::Scene::new(nodes)?);
        }

        Ok((default_scene_index, scenes))
    }

    fn explore(&self, gltf_node: &gltf::Node) -> SrResult<vulkan_abstraction::Node> {
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

        Ok(vulkan_abstraction::Node::new(transform, mesh, children)?)
    }

    fn process_node(
        &self,
        gltf_node: &gltf::Node,
    ) -> SrResult<(na::Matrix4<f32>, Option<vulkan_abstraction::Mesh>)> {
        // the trasnform can also be given decomposed in: translation, rotation and scale
        // but the gltf crate takes care of this:
        // "If the transform is Decomposed, then the matrix is generated with the equation matrix = translation * rotation * scale."
        let transform = na::Matrix4::from(gltf_node.transform().matrix());
        let mut mesh = None;

        // TODO: this code does not manage multiple nodes pointing to the same meshes
        // fix proposal: check for the mesh id
        if let Some(gltf_mesh) = gltf_node.mesh() {
            let mut vertices = vec![];
            let mut indices = vec![];

            for (_i, primitive) in gltf_mesh.primitives().enumerate() {
                let reader = primitive.reader(|buffer| Some(&self.buffers[buffer.index()]));

                // get vertices positions
                reader
                    .read_positions()
                    .unwrap()
                    .for_each(|position| vertices.push(vulkan_abstraction::Vertex { position }));

                // get vertices index
                let indexes = reader.read_indices().unwrap().into_u32();
                indexes.clone().for_each(|i| indices.push(i));
            }

            mesh = Some(vulkan_abstraction::Mesh::new(vertices, indices)?);
        }

        Ok((transform, mesh))
    }
}

    fn load_scene(&mut self, scene: &vulkan_abstraction::Scene) -> SrResult<()> {
        self.blases.clear();

        let mut blas_instances = vec![];
        for node in scene.nodes() {
            let local_transform = node.transform();
            self.load_node(node, local_transform, &mut blas_instances)?;
        }

        let blas_instances = blas_instances
            .into_iter()
            .map(|(index, transform)| vulkan_abstraction::BlasInstance {
                blas: &self.blases[index],
                transform: Self::to_vk_transform(transform),
            })
            .collect::<Vec<_>>();
        self.tlas.rebuild(&blas_instances)?;

        Ok(())
    }

    fn load_node(
        &mut self,
        node: &vulkan_abstraction::Node,
        transform: &na::Matrix4<f32>,
        blas_instances: &mut Vec<(usize, na::Matrix4<f32>)>,
    ) -> SrResult<()> {
        let local_transform = node.transform() * transform;

        // TODO: avoid creating new blas for alredy seen meshes
        if node.mesh().is_some() {
            let (vertex_buffer, index_buffer) = node.load_mesh_into_gpu_memory(&self.core)?;
            let blas =
                vulkan_abstraction::BLAS::new(Rc::clone(&self.core), vertex_buffer, index_buffer)?;
            self.blases.push(blas);

            blas_instances.push((self.blases.len() - 1, local_transform));
        }

        if let Some(children) = node.children() {
            for child in children {
                self.load_node(child, &local_transform, blas_instances)?
            }
        }

        Ok(())
    }

    fn to_vk_transform(transform: na::Matrix4<f32>) -> vk::TransformMatrixKHR {
        let r0 = transform.row(0);
        let r1 = transform.row(1);
        let r2 = transform.row(2);
        let r3 = transform.row(3);

        #[rustfmt::skip]
        let matrix = [
            r0[0], r1[0], r2[0], r3[0],
            r0[1], r1[1], r2[1], r3[1],
            r0[2], r1[2], r2[2], r3[2],
        ];

        vk::TransformMatrixKHR { matrix }
    }