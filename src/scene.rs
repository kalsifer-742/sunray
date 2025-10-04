use std::{collections::HashMap, rc::Rc};

use crate::{error::SrResult, vulkan_abstraction};

use ash::vk;
use nalgebra as na;

type BlasInstanceInfo = (usize, na::Matrix4<f32>);

pub struct Scene {
    nodes: Vec<vulkan_abstraction::gltf::Node>,
}

impl Scene {
    pub fn new(nodes: Vec<vulkan_abstraction::gltf::Node>) -> SrResult<Self> {
        Ok(Self { nodes })
    }

    pub fn nodes(&self) -> &[vulkan_abstraction::gltf::Node] {
        &self.nodes
    }

    pub fn load<'a>(
        &self,
        core: &Rc<vulkan_abstraction::Core>,
        blases: &'a mut Vec<vulkan_abstraction::BLAS>,
        materials: &mut Vec<vulkan_abstraction::gltf::Material>,
        scene_data: &mut vulkan_abstraction::gltf::PrimitiveDataMap,
    ) -> SrResult<Vec<vulkan_abstraction::BlasInstance<'a>>> {
        blases.clear();

        let mut blas_instances_info: Vec<BlasInstanceInfo> = vec![];
        let mut primitives_blas_index: HashMap<vulkan_abstraction::gltf::PrimitiveUniqueKey, usize> = HashMap::new();
        for node in self.nodes() {
            // the root nodes do not have a parent transform to apply
            let transform = na::Matrix4::identity();
            self.load_node(
                node,
                transform,
                core,
                blases,
                &mut blas_instances_info,
                &mut primitives_blas_index,
                materials,
                scene_data,
            )?;
        }

        let blas_instances = blas_instances_info
            .into_iter()
            .enumerate()
            .map(
                |(blas_instance_index, (blas_index, transform))| vulkan_abstraction::BlasInstance {
                    blas_instance_index: blas_instance_index as u32,
                    blas: &blases[blas_index],
                    transform: Self::to_vk_transform(transform),
                },
            )
            .collect::<Vec<_>>();

        Ok(blas_instances)
    }

    fn load_node(
        &self,
        node: &vulkan_abstraction::gltf::Node,
        parent_transform: na::Matrix4<f32>,
        core: &Rc<vulkan_abstraction::Core>,
        blases: &mut Vec<vulkan_abstraction::BLAS>,
        blas_instances_info: &mut Vec<BlasInstanceInfo>,
        primitives_blas_index: &mut HashMap<vulkan_abstraction::gltf::PrimitiveUniqueKey, usize>,
        materials: &mut Vec<vulkan_abstraction::gltf::Material>,
        scene_data: &mut vulkan_abstraction::gltf::PrimitiveDataMap,
    ) -> SrResult<()> {
        let transform = parent_transform * node.transform();

        // TODO: avoid creating new blas for alredy seen meshes
        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                let primitive_unique_key = primitive.unique_key;

                let blas_index = match primitives_blas_index.get(&primitive_unique_key) {
                    Some(blas_index) => *blas_index,
                    None => {
                        let primitive_data = scene_data.remove(&primitive_unique_key).unwrap();

                        let blas = vulkan_abstraction::BLAS::new(
                            core.clone(),
                            primitive_data.vertex_buffer,
                            primitive_data.index_buffer,
                        )?;
                        blases.push(blas);

                        let blas_index = blases.len() - 1;
                        primitives_blas_index.insert(primitive_unique_key, blas_index);

                        blas_index
                    }
                };

                materials.push(primitive.material.clone());

                // the first idea that could come to your mind is to create a BlasInstance here directly.
                // Apart from having to manage lifetimes it is still not going to work because:
                // - &blases[blases.len()]
                // creates an immutable borrow of blases when a mutable borrow already exist - compiler error!
                // - blases.last() - compiler error!
                // - blases.last_mut()
                // creates another mutable borrow when anoter mutable borrow already exists
                // but only one mutable borrow can exist at any time - compile error!
                //
                // tl;dr don't waste time making lifetimes work
                blas_instances_info.push((blas_index, transform));
            }
        }

        if let Some(children) = node.children() {
            for child in children {
                self.load_node(
                    child,
                    transform,
                    core,
                    blases,
                    blas_instances_info,
                    primitives_blas_index,
                    materials,
                    scene_data,
                )? // mut borrow
            }
        }

        Ok(())
    }

    fn to_vk_transform(transform: na::Matrix4<f32>) -> vk::TransformMatrixKHR {
        let c0 = transform.column(0);
        let c1 = transform.column(1);
        let c2 = transform.column(2);
        let c3 = transform.column(3);

        #[rustfmt::skip]
        let matrix = [
            c0[0], c1[0], c2[0], c3[0],
            c0[1], c1[1], c2[1], c3[1],
            c0[2], c1[2], c2[2], c3[2],
        ];

        vk::TransformMatrixKHR { matrix }
    }
}
