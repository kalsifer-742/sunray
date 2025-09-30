use std::rc::Rc;

use crate::{
    error::SrResult,
    vulkan_abstraction::{self},
};

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

    pub fn load(
        &self,
        core: &Rc<vulkan_abstraction::Core>,
        tlas: &mut vulkan_abstraction::TLAS,
        blases: &mut Vec<vulkan_abstraction::BLAS>,
    ) -> SrResult<()> {
        blases.clear();

        let mut blas_instances_info: Vec<BlasInstanceInfo> = vec![];
        for node in self.nodes() {
            // the root nodes do not have a parent transform to apply
            let transform = na::Matrix4::identity();
            self.load_node(node, transform, core, blases, &mut blas_instances_info)?;
        }

        let blas_instances = blas_instances_info
            .into_iter()
            .map(|(index, transform)| vulkan_abstraction::BlasInstance {
                blas: &blases[index],
                transform: Self::to_vk_transform(transform),
            })
            .collect::<Vec<_>>();
        tlas.rebuild(&blas_instances)?;

        Ok(())
    }

    fn load_node(
        &self,
        node: &vulkan_abstraction::gltf::Node,
        parent_transform: na::Matrix4<f32>,
        core: &Rc<vulkan_abstraction::Core>,
        blases: &mut Vec<vulkan_abstraction::BLAS>,
        blas_instances_info: &mut Vec<BlasInstanceInfo>,
    ) -> SrResult<()> {
        let transform = parent_transform * node.transform();

        // TODO: avoid creating new blas for alredy seen meshes
        if let Some(mesh) = node.mesh() {
            let blas = vulkan_abstraction::BLAS::new(
                core.clone(),
                mesh.vertex_buffer(),
                mesh.index_buffer(),
            )?;
            blases.push(blas);

            // the first idea that could come to your mind is to create a BlasInstance here directly.
            // Apart from having to manage lifetimes it is still not going to work because:
            // - &blases[blases.len()]
            // creates an immutable borrow of blases when a mutable borrow already exist - compiler error!
            // - blases.last() - compiler error!
            // - blases.last_mut()
            // creates another mutable borrow when anoter mutable borrow already exists
            // but only one mutable borrow can exist at every time - compile error!
            //
            // tl;dr don't waste time making lifetimes work
            let index = blases.len() - 1;
            blas_instances_info.push((index, transform));
        }

        if let Some(children) = node.children() {
            for child in children {
                self.load_node(child, transform, core, blases, blas_instances_info)? // mut borrow
            }
        }

        Ok(())
    }

    fn to_vk_transform(transform: na::Matrix4<f32>) -> vk::TransformMatrixKHR {
        let r0 = transform.column(0);
        let r1 = transform.column(1);
        let r2 = transform.column(2);
        let r3 = transform.column(3);

        #[rustfmt::skip]
        let matrix = [
            r0[0], r1[0], r2[0], r3[0],
            r0[1], r1[1], r2[1], r3[1],
            r0[2], r1[2], r2[2], r3[2],
        ];

        vk::TransformMatrixKHR { matrix }
    }
}
