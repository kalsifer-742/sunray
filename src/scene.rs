use std::{collections::HashMap, rc::Rc};

use crate::{error::SrResult, vulkan_abstraction};

use ash::vk;
use nalgebra as na;

type BlasInstanceInfo = (usize, na::Matrix4<f32>);

pub struct SceneData {
    pub textures: Vec<vulkan_abstraction::gltf::Texture>,
    pub samplers: Vec<vulkan_abstraction::gltf::Sampler>,
    pub images: Vec<vulkan_abstraction::gltf::Image>,
    pub primitive_data_map: vulkan_abstraction::gltf::PrimitiveDataMap,
}

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

    pub fn load_into_gpu<'a>(
        &self,
        core: &Rc<vulkan_abstraction::Core>,
        blases: &'a mut Vec<vulkan_abstraction::BLAS>,
        mut scene_data: crate::SceneData,
    ) -> SrResult<(
        Vec<vulkan_abstraction::BlasInstance<'a>>,
        Vec<vulkan_abstraction::gltf::Material>,
        Vec<vulkan_abstraction::gltf::Texture>,
        Vec<vulkan_abstraction::image::Sampler>,
        Vec<vulkan_abstraction::Image>,
    )> {
        blases.clear();

        let mut blas_instances_info = vec![];
        let mut materials = vec![];

        let mut primitives_blas_index: HashMap<vulkan_abstraction::gltf::PrimitiveUniqueKey, usize> = HashMap::new();
        for node in self.nodes() {
            self.explore_node(
                node,
                core,
                blases,
                &mut blas_instances_info,
                &mut primitives_blas_index,
                &mut materials,
                &mut scene_data,
            )?;
        }

        let blas_instances = blas_instances_info
            .into_iter()
            .enumerate()
            .map(
                |(blas_instance_index, (blas_index, transform))| vulkan_abstraction::BlasInstance {
                    blas_instance_index: blas_instance_index as u32,
                    blas: &blases[blas_index],
                    transform: to_vk_transform(transform),
                },
            )
            .collect::<Vec<_>>();

        let samplers: Result<Vec<_>, _> = scene_data
            .samplers
            .iter()
            .map(|sampler| {
                let default = gltf::texture::MinFilter::Linear;

                vulkan_abstraction::image::Sampler::new(
                    core.clone(),
                    vk::Filter::from_gltf(sampler.min_filter.unwrap_or(default)),
                    vk::Filter::from_gltf(sampler.mag_filter.unwrap_or(gltf::texture::MagFilter::Linear)),
                    vk::SamplerAddressMode::from_gltf(sampler.wrap_s_u),
                    vk::SamplerAddressMode::from_gltf(sampler.wrap_t_v),
                    vk::SamplerAddressMode::REPEAT,
                    vk::SamplerMipmapMode::from_gltf(sampler.min_filter.unwrap_or(default)),
                )
            })
            .collect();

        let images: Result<Vec<_>, _> = scene_data.images.into_iter().map(|image| to_vk_image(core, image)).collect();

        Ok((blas_instances, materials, scene_data.textures, samplers?, images?))
    }

    fn explore_node(
        &self,
        node: &vulkan_abstraction::gltf::Node,
        core: &Rc<vulkan_abstraction::Core>,
        blases: &mut Vec<vulkan_abstraction::BLAS>,
        blas_instances_info: &mut Vec<BlasInstanceInfo>,
        primitives_blas_index: &mut HashMap<vulkan_abstraction::gltf::PrimitiveUniqueKey, usize>,
        materials: &mut Vec<vulkan_abstraction::gltf::Material>,
        scene_data: &mut crate::SceneData,
    ) -> SrResult<()> {
        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                let primitive_unique_key = primitive.unique_key;

                let blas_index = match primitives_blas_index.get(&primitive_unique_key) {
                    Some(blas_index) => *blas_index,
                    None => {
                        let primitive_data = scene_data.primitive_data_map.remove(&primitive_unique_key).unwrap();

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
                blas_instances_info.push((blas_index, *node.transform()));
            }
        }

        if let Some(children) = node.children() {
            for child in children {
                self.explore_node(
                    child,
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

fn to_vk_image(
    core: &Rc<vulkan_abstraction::Core>,
    image: vulkan_abstraction::gltf::Image,
) -> SrResult<vulkan_abstraction::Image> {
    let format = vk::Format::from_gltf(image.format);

    let image = vulkan_abstraction::Image::new_from_data(
        Rc::clone(core),
        image.raw_data,
        vk::Extent3D {
            width: image.width as u32,
            height: image.height as u32,
            depth: 1,
        },
        format,
        vk::ImageTiling::OPTIMAL,
        gpu_allocator::MemoryLocation::GpuOnly,
        vk::ImageUsageFlags::SAMPLED,
        "gltf image",
    )?;

    Ok(image)
}

// Becuase of the oprhan rule of rust
// it is not possible to implement the trait from
// for the types gltf::image::Format and vk::Format
// so I created a custom trait
pub trait FromGltf<T> {
    fn from_gltf(value: T) -> Self;
}

impl FromGltf<gltf::image::Format> for vk::Format {
    fn from_gltf(value: gltf::image::Format) -> Self {
        match value {
            gltf::image::Format::R8 => vk::Format::R8_UNORM,
            gltf::image::Format::R8G8 => vk::Format::R8G8_UNORM,
            gltf::image::Format::R8G8B8 => vk::Format::R8G8B8_UNORM,
            gltf::image::Format::R8G8B8A8 => vk::Format::R8G8B8A8_UNORM,
            gltf::image::Format::R16 => vk::Format::R16_SFLOAT,
            gltf::image::Format::R16G16 => vk::Format::R16G16_SFLOAT,
            gltf::image::Format::R16G16B16 => vk::Format::R16G16B16_SFLOAT,
            gltf::image::Format::R16G16B16A16 => vk::Format::R16G16B16A16_SFLOAT,
            gltf::image::Format::R32G32B32FLOAT => vk::Format::R32G32B32_SFLOAT,
            gltf::image::Format::R32G32B32A32FLOAT => vk::Format::R32G32B32A32_SFLOAT,
        }
    }
}

impl FromGltf<gltf::texture::MinFilter> for vk::SamplerMipmapMode {
    fn from_gltf(value: gltf::texture::MinFilter) -> Self {
        match value {
            gltf::texture::MinFilter::Nearest => vk::SamplerMipmapMode::LINEAR,
            gltf::texture::MinFilter::Linear => vk::SamplerMipmapMode::LINEAR,
            gltf::texture::MinFilter::NearestMipmapNearest => vk::SamplerMipmapMode::NEAREST,
            gltf::texture::MinFilter::LinearMipmapNearest => vk::SamplerMipmapMode::NEAREST,
            gltf::texture::MinFilter::NearestMipmapLinear => vk::SamplerMipmapMode::LINEAR,
            gltf::texture::MinFilter::LinearMipmapLinear => vk::SamplerMipmapMode::LINEAR,
        }
    }
}

impl FromGltf<gltf::texture::MinFilter> for vk::Filter {
    fn from_gltf(value: gltf::texture::MinFilter) -> Self {
        match value {
            gltf::texture::MinFilter::Nearest => vk::Filter::NEAREST,
            gltf::texture::MinFilter::Linear => vk::Filter::LINEAR,
            gltf::texture::MinFilter::NearestMipmapNearest => vk::Filter::NEAREST,
            gltf::texture::MinFilter::LinearMipmapNearest => vk::Filter::LINEAR,
            gltf::texture::MinFilter::NearestMipmapLinear => vk::Filter::NEAREST,
            gltf::texture::MinFilter::LinearMipmapLinear => vk::Filter::LINEAR,
        }
    }
}

impl FromGltf<gltf::texture::MagFilter> for vk::Filter {
    fn from_gltf(value: gltf::texture::MagFilter) -> Self {
        match value {
            gltf::texture::MagFilter::Nearest => vk::Filter::NEAREST,
            gltf::texture::MagFilter::Linear => vk::Filter::LINEAR,
        }
    }
}

impl FromGltf<gltf::texture::WrappingMode> for vk::SamplerAddressMode {
    fn from_gltf(value: gltf::texture::WrappingMode) -> Self {
        match value {
            gltf::texture::WrappingMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            gltf::texture::WrappingMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
            gltf::texture::WrappingMode::Repeat => vk::SamplerAddressMode::REPEAT,
        }
    }
}
