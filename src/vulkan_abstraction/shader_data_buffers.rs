use std::rc::Rc;

use ash::vk;
use nalgebra as na;

use crate::{CameraMatrices, error::SrResult, vulkan_abstraction};

#[derive(Clone, Copy)]
#[repr(C, packed)]
struct MatricesBufferContents {
    pub view_inverse: na::Matrix4<f32>,
    pub proj_inverse: na::Matrix4<f32>,
}

#[derive(Clone, Copy)]
#[repr(C, packed)]
struct Material {
    base_color_value: [f32; 4],
    base_color_texture_index: u32,

    metallic_factor: f32,
    roughness_factor: f32,
    metallic_roughness_texture_index: u32,

    normal_texture_index: u32,
    occlusion_texture_index: u32,

    _padding: [f32; 2],

    emissive_factor: [f32; 3],
    emissive_texture_index: u32,
    //alpha mode and alpha cutoff are missing
}
impl Material {
    const NULL_TEXTURE_INDEX: u32 = u32::MAX;
}

impl From<&vulkan_abstraction::gltf::Material> for Material {
    fn from(material: &vulkan_abstraction::gltf::Material) -> Self {
        let to_texture_index = |i: Option<usize>| -> u32 {
            match i {
                Some(i) => i as u32,
                None => Self::NULL_TEXTURE_INDEX,
            }
        };

        Self {
            base_color_value: material.pbr_metallic_roughness_properties.base_color_factor,
            base_color_texture_index: to_texture_index(material.pbr_metallic_roughness_properties.base_color_texture_index),

            metallic_factor: material.pbr_metallic_roughness_properties.metallic_factor,
            roughness_factor: material.pbr_metallic_roughness_properties.roughness_factor,
            metallic_roughness_texture_index: to_texture_index(
                material.pbr_metallic_roughness_properties.base_color_texture_index,
            ),

            normal_texture_index: to_texture_index(material.normal_texture_index),
            occlusion_texture_index: to_texture_index(material.occlusion_texture_index),

            emissive_factor: material.emissive_factor,
            emissive_texture_index: to_texture_index(material.emissive_texture_index),

            _padding: [0.0; 2],
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C, packed)]
struct MeshesInfoBufferContents {
    vertex_buffer: vk::DeviceAddress,
    index_buffer: vk::DeviceAddress,

    material: Material,
}

pub(crate) struct ShaderDataBuffers {
    matrices_uniform_buffer: vulkan_abstraction::Buffer,
    meshes_info_storage_buffer: vulkan_abstraction::Buffer,

    textures: Vec<(vk::Sampler, vk::ImageView)>,

    core: Rc<vulkan_abstraction::Core>,
}

impl ShaderDataBuffers {
    pub const NUMBER_OF_SAMPLERS: usize = 1024;

    pub fn new_empty(core: Rc<vulkan_abstraction::Core>) -> SrResult<Self> {
        let matrices_uniform_buffer = vulkan_abstraction::Buffer::new_uniform::<MatricesBufferContents>(Rc::clone(&core), 1)?;

        Ok(Self {
            matrices_uniform_buffer,
            meshes_info_storage_buffer: vulkan_abstraction::Buffer::new_null(Rc::clone(&core)),
            textures: Vec::new(),
            core,
        })
    }

    pub fn set_matrices(
        &mut self,
        CameraMatrices {
            view_inverse,
            proj_inverse,
        }: CameraMatrices,
    ) -> SrResult<()> {
        let mem = self.matrices_uniform_buffer.map::<MatricesBufferContents>()?;
        mem[0] = MatricesBufferContents {
            view_inverse,
            proj_inverse,
        };

        Ok(())
    }

    pub fn update(
        &mut self,
        blas_instances: &[vulkan_abstraction::BlasInstance],
        materials: &[vulkan_abstraction::gltf::Material],
        textures: &[&vulkan_abstraction::Image],
        fallback: &vulkan_abstraction::Image,
    ) -> SrResult<()> {
        self.set_meshes_info(blas_instances, materials)?;
        self.set_textures(textures, fallback);

        Ok(())
    }

    fn set_meshes_info(
        &mut self,
        blas_instances: &[vulkan_abstraction::BlasInstance],
        materials: &[vulkan_abstraction::gltf::Material],
    ) -> SrResult<()> {
        let meshes_info_storage_buffer_contents = std::iter::zip(blas_instances.iter(), materials.iter())
            .map(|(blas_instance, material)| MeshesInfoBufferContents {
                vertex_buffer: blas_instance.blas.vertex_buffer().get_device_address(),
                index_buffer: blas_instance.blas.index_buffer().get_device_address(),
                material: Material::from(material),
            })
            .collect::<Vec<_>>();

        self.meshes_info_storage_buffer = vulkan_abstraction::Buffer::new_storage_from_data(
            Rc::clone(&self.core),
            &meshes_info_storage_buffer_contents,
            "meshes info storage buffer",
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

    pub fn get_textures(&self) -> &[(vk::Sampler, vk::ImageView)] {
        &self.textures
    }
}
