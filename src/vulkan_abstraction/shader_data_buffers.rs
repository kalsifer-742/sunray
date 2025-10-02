
use std::rc::Rc;

use nalgebra as na;
use ash::vk;

use crate::{error::SrResult, vulkan_abstraction, CameraMatrices};

#[derive(Clone, Copy)]
#[repr(C)]
struct MatricesBufferContents {
    pub view_inverse: na::Matrix4<f32>,
    pub proj_inverse: na::Matrix4<f32>,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct MeshesInfoBufferContents {
    vertex_buffer: vk::DeviceAddress,
    index_buffer: vk::DeviceAddress,
    material_index: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct MaterialsBufferContents {
    texture_index: u32,
}

pub(crate) struct ShaderDataBuffers {
    matrices_uniform_buffer: vulkan_abstraction::Buffer,
    meshes_info_storage_buffer: vulkan_abstraction::Buffer,
    materials_storage_buffer: vulkan_abstraction::Buffer,

    textures: Vec<(vk::Sampler, vk::ImageView)>,

    core: Rc<vulkan_abstraction::Core>
}

impl ShaderDataBuffers {
    pub const NUMBER_OF_SAMPLERS : usize = 1024;

    pub fn new_empty(core: Rc<vulkan_abstraction::Core>) -> SrResult<Self> {
        let matrices_uniform_buffer = vulkan_abstraction::Buffer::new_uniform::<MatricesBufferContents>(Rc::clone(&core), 1)?;

        Ok(Self{
            matrices_uniform_buffer,
            meshes_info_storage_buffer: vulkan_abstraction::Buffer::new_null(Rc::clone(&core)),
            materials_storage_buffer: vulkan_abstraction::Buffer::new_null(Rc::clone(&core)),
            textures: Vec::new(),
            core,
        })
    }


    pub fn set_matrices(&mut self, CameraMatrices { view_inverse, proj_inverse }: CameraMatrices) -> SrResult<()> {
        let mem = self.matrices_uniform_buffer.map::<MatricesBufferContents>()?;
        mem[0] = MatricesBufferContents { view_inverse, proj_inverse };

        Ok(())
    }

    pub fn update(&mut self, blas_instances: &[vulkan_abstraction::BlasInstance], materials: &[()], textures: &[&vulkan_abstraction::Image], fallback: &vulkan_abstraction::Image) -> SrResult<()> {
        self.set_meshes_info(blas_instances)?;
        self.set_materials(materials)?;
        self.set_textures(textures, fallback);

        Ok(())
    }

    fn set_meshes_info(&mut self, blas_instances: &[vulkan_abstraction::BlasInstance]) -> SrResult<()> {
        let meshes_info_storage_buffer_contents =
            blas_instances.iter().map(|blas_instance| MeshesInfoBufferContents {
                vertex_buffer: blas_instance.blas.vertex_buffer().get_device_address(),
                index_buffer: blas_instance.blas.index_buffer().get_device_address(),
                //all meshes point to placeholder material
                material_index: 0,
            })
            .collect::<Vec<_>>();

        self.meshes_info_storage_buffer = vulkan_abstraction::Buffer::new_storage_from_data(
            Rc::clone(&self.core), &meshes_info_storage_buffer_contents, "meshes info storage buffer"
        )?;

        Ok(())
    }

    fn set_materials(&mut self, _materials: &[()]) -> SrResult<()> {
        //placeholder material
        self.materials_storage_buffer = vulkan_abstraction::Buffer::new_storage_from_data(
            Rc::clone(&self.core),
            &[ MaterialsBufferContents { texture_index: 0 } ],
            "materials storage buffer",
        )?;
        Ok(())
    }

    fn set_textures(&mut self, textures: &[&vulkan_abstraction::Image], fallback: &vulkan_abstraction::Image) {
        self.textures.clear();

        self.textures.reserve_exact(Self::NUMBER_OF_SAMPLERS);

        for i in 0..textures.len() {
            self.textures.push((textures[i].sampler(), textures[i].image_view()));
        }

        while self.textures.len() < Self::NUMBER_OF_SAMPLERS {
            self.textures.push((fallback.sampler(), fallback.image_view()));
        }

        assert_eq!(self.textures.len(), Self::NUMBER_OF_SAMPLERS);
    }

    pub fn get_matrices_uniform_buffer(&self) -> vk::Buffer {
        self.matrices_uniform_buffer.inner()
    }

    pub fn get_meshes_info_storage_buffer(&self) -> vk::Buffer {
        self.meshes_info_storage_buffer.inner()
    }

    pub fn get_materials_storage_buffer(&self) -> vk::Buffer {
        self.materials_storage_buffer.inner()
    }

    pub fn get_textures(&self) -> &[(vk::Sampler, vk::ImageView)] {
        &self.textures
    }
}
