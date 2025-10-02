use std::{collections::HashMap, rc::Rc};

use crate::{error::SrResult, vulkan_abstraction};

use nalgebra as na;

pub mod image;
pub mod material;
pub mod mesh;
pub mod node;
pub mod primitive;
pub mod texture;
pub mod vertex;

pub use image::*;
pub use material::*;
pub use mesh::*;
pub use node::*;
pub use primitive::*;
pub use texture::*;
pub use vertex::*;

pub type PrimitiveDataMap =
    HashMap<vulkan_abstraction::gltf::PrimitiveUniqueKey, vulkan_abstraction::gltf::PrimitiveData>;

pub struct Gltf {
    core: Rc<vulkan_abstraction::Core>,
    document: gltf::Document,
    buffers: Vec<gltf::buffer::Data>,
    images: Vec<gltf::image::Data>,
}

impl Gltf {
    pub fn new(core: Rc<vulkan_abstraction::Core>, path: &str) -> SrResult<Self> {
        let (document, buffers, images) = gltf::import(path)?;

        Ok(Self {
            core,
            document,
            buffers,
            images,
        })
    }

    pub fn create_scenes(
        &self,
    ) -> SrResult<(
        usize,
        Vec<crate::Scene>,
        Vec<vulkan_abstraction::gltf::Texture>,
        Vec<vulkan_abstraction::gltf::Image>,
        Vec<vulkan_abstraction::gltf::Sampler>,
        Vec<PrimitiveDataMap>,
    )> {
        // find the defualt scene index
        let default_scene_index = match self.document.default_scene() {
            Some(s) => s.index(),
            None => 0,
        };

        let samplers = self
            .document
            .samplers()
            .map(|sampler| vulkan_abstraction::gltf::Sampler {
                mag_filter: sampler.mag_filter().map(|filter| match filter {
                    gltf::texture::MagFilter::Nearest => {
                        vulkan_abstraction::gltf::MagFilter::Nearest
                    }
                    gltf::texture::MagFilter::Linear => vulkan_abstraction::gltf::MagFilter::Linear,
                }),
                min_filter: sampler.min_filter().map(|filter| match filter {
                    gltf::texture::MinFilter::Nearest => {
                        vulkan_abstraction::gltf::MinFilter::Nearest
                    }
                    gltf::texture::MinFilter::Linear => vulkan_abstraction::gltf::MinFilter::Linear,
                    gltf::texture::MinFilter::NearestMipmapNearest => {
                        vulkan_abstraction::gltf::MinFilter::NearestMipmapNearest
                    }
                    gltf::texture::MinFilter::LinearMipmapNearest => {
                        vulkan_abstraction::gltf::MinFilter::LinearMipmapNearest
                    }
                    gltf::texture::MinFilter::NearestMipmapLinear => {
                        vulkan_abstraction::gltf::MinFilter::NearestMipmapLinear
                    }
                    gltf::texture::MinFilter::LinearMipmapLinear => {
                        vulkan_abstraction::gltf::MinFilter::LinearMipmapLinear
                    }
                }),
                wrap_s_u: match sampler.wrap_s() {
                    gltf::texture::WrappingMode::ClampToEdge => {
                        vulkan_abstraction::gltf::WrappingMode::ClampToEdge
                    }
                    gltf::texture::WrappingMode::MirroredRepeat => {
                        vulkan_abstraction::gltf::WrappingMode::MirroredRepeat
                    }
                    gltf::texture::WrappingMode::Repeat => {
                        vulkan_abstraction::gltf::WrappingMode::Repeat
                    }
                },
                wrap_t_v: match sampler.wrap_t() {
                    gltf::texture::WrappingMode::ClampToEdge => {
                        vulkan_abstraction::gltf::WrappingMode::ClampToEdge
                    }
                    gltf::texture::WrappingMode::MirroredRepeat => {
                        vulkan_abstraction::gltf::WrappingMode::MirroredRepeat
                    }
                    gltf::texture::WrappingMode::Repeat => {
                        vulkan_abstraction::gltf::WrappingMode::Repeat
                    }
                },
            })
            .collect::<Vec<_>>();

        let textures = self
            .document
            .textures()
            .map(|texture| Texture {
                sampler: texture.sampler().index(),
                source: texture.source().index(),
            })
            .collect::<Vec<_>>();

        let images = self
            .images
            .iter()
            .map(|image| Image {
                format: image.format,
                height: image.height as usize,
                width: image.width as usize,
                raw_data: image.pixels.clone(), // TODO: consume gltf
            })
            .collect::<Vec<_>>();

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

        Ok((
            default_scene_index,
            scenes,
            textures,
            images,
            samplers,
            primitive_data_maps,
        ))
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
            for (i, primitive) in gltf_mesh
                .primitives()
                .filter(|p| Self::is_primitive_supported(p))
                .enumerate()
            {
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

                    let mut vertices: Vec<Vertex> = vec![];
                    // get vertices positions
                    reader.read_positions().unwrap().for_each(|position| {
                        vertices.push(Vertex {
                            position,
                            ..Default::default()
                        })
                    });

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
                            let indices = (0..vertices.len() as u32 / 3).collect::<Vec<_>>();

                            indices
                        };

                        let index_buffer = vulkan_abstraction::IndexBuffer::new_for_blas_from_data::<
                            u32,
                        >(
                            Rc::clone(&self.core), &indices
                        )?;

                        index_buffer
                    };

                    let (material, tex_coords) = {
                        let material = primitive.material();
                        let material_pbr = primitive.material().pbr_metallic_roughness();

                        let base_color_factor = material_pbr.base_color_factor();
                        let metallic_factor = material_pbr.metallic_factor();
                        let roughness_factor = material_pbr.roughness_factor();
                        let emissive_factor = material.emissive_factor();
                        let alpha_mode = material.alpha_mode();
                        let alpha_cutoff = material.alpha_cutoff().unwrap_or(0.5);
                        let double_sided = material.double_sided();

                        // The code is repeated because the type of the textures are not the same
                        // TODO: crate a macro
                        let (base_color_texture_index, base_color_tex_coord_index) =
                            match material_pbr.base_color_texture() {
                                Some(texture_info) => (
                                    Some(texture_info.texture().index()),
                                    texture_info.tex_coord(),
                                ),
                                None => (None, 0),
                            };
                        let (metallic_roughness_texture_index, metallic_roughness_tex_coord_index) =
                            match material_pbr.metallic_roughness_texture() {
                                Some(texture_info) => (
                                    Some(texture_info.texture().index()),
                                    texture_info.tex_coord(),
                                ),
                                None => (None, 0),
                            };
                        let (normal_texture_index, normal_tex_coord_index) =
                            match material.normal_texture() {
                                Some(texture_info) => (
                                    Some(texture_info.texture().index()),
                                    texture_info.tex_coord(),
                                ),
                                None => (None, 0),
                            };
                        let (occlusion_texture_index, occlusion_tex_coord_index) =
                            match material.occlusion_texture() {
                                Some(texture_info) => (
                                    Some(texture_info.texture().index()),
                                    texture_info.tex_coord(),
                                ),
                                None => (None, 0),
                            };
                        let (emissive_texture_index, emissive_tex_coord_index) =
                            match material.emissive_texture() {
                                Some(texture_info) => (
                                    Some(texture_info.texture().index()),
                                    texture_info.tex_coord(),
                                ),
                                None => (None, 0),
                            };

                        let pbr_mettalic_roughness_properties =
                            vulkan_abstraction::gltf::PbrMetallicRoughnessProperties {
                                base_color_factor,
                                metallic_factor,
                                roughness_factor,
                                base_color_texture_index,
                                metallic_roughness_texture_index,
                            };

                        let material = vulkan_abstraction::gltf::Material {
                            pbr_mettalic_roughness_properties,
                            normal_texture_index,
                            occlusion_texture_index,
                            emissive_factor,
                            emissive_texture_index,
                            alpha_mode,
                            alpha_cutoff,
                            double_sided,
                        };

                        let tex_coords = (
                            base_color_tex_coord_index,
                            metallic_roughness_tex_coord_index,
                            normal_tex_coord_index,
                            occlusion_tex_coord_index,
                            emissive_tex_coord_index,
                        );

                        (material, tex_coords)
                    };

                    // TODO: make macro
                    reader
                        .read_tex_coords(tex_coords.0)
                        .unwrap()
                        .into_f32()
                        .enumerate()
                        .for_each(|(j, coord)| vertices[j].base_color_tex_coord = coord);
                    reader
                        .read_tex_coords(tex_coords.1)
                        .unwrap()
                        .into_f32()
                        .enumerate()
                        .for_each(|(j, coord)| vertices[j].metallic_roughness_tex_coord = coord);
                    reader
                        .read_tex_coords(tex_coords.2)
                        .unwrap()
                        .into_f32()
                        .enumerate()
                        .for_each(|(j, coord)| vertices[j].normal_tex_coord = coord);
                    reader
                        .read_tex_coords(tex_coords.3)
                        .unwrap()
                        .into_f32()
                        .enumerate()
                        .for_each(|(j, coord)| vertices[j].occlusion_tex = coord);
                    reader
                        .read_tex_coords(tex_coords.4)
                        .unwrap()
                        .into_f32()
                        .enumerate()
                        .for_each(|(j, coord)| vertices[j].emissive_tex = coord);

                    let vertex_buffer = {
                        let vertex_buffer =
                            vulkan_abstraction::VertexBuffer::new_for_blas_from_data::<
                                vulkan_abstraction::gltf::Vertex,
                            >(Rc::clone(&self.core), &vertices)?;

                        vertex_buffer
                    };

                    let primitive_data = vulkan_abstraction::gltf::PrimitiveData {
                        vertex_buffer,
                        index_buffer,
                        material,
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

    fn is_primitive_supported(primitive: &gltf::Primitive) -> bool {
        match primitive.mode() {
            gltf::mesh::Mode::Triangles => true,
            m => {
                log::error!("Found unsupported primitive mode: {:?}", m);

                false
            }
        }
    }
}
