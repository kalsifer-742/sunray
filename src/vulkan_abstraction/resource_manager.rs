use std::collections::HashMap;
use std::hash::Hash;
use std::rc::Rc;

use ash::vk;

use crate::render_graph::graph::SamplerDesc;
use crate::vulkan_abstraction::{ArenaBuffer, Buffer, EntityGpuData, HostAccessibleBuffer, Material};
use crate::{CameraMatrices, error::SrResult, vulkan_abstraction};

const ARENA_CAPACITY: vk::DeviceSize = 4096 * 16;

//TODO handle growable

//TODO there is structural decision to make on what to save and how cause raster needs a different way to store the geometry data probabibly with a features set constable?
/// Owns the GPU-side scene resources. Split in two halves:
///
/// * **Stable assets**, keyed by the caller-provided `K` (e.g. one key per
///   BLAS / image): geometry + material data lives in arena buffers whose
///   slots survive across frames.
/// * **Per-frame state**, supplied each frame through [`Self::prepare_frame`]:
///   the camera matrices and the instance list `(key, transforms)`. The
///   instance buffer / TLAS / flat transform buffer / emissive indirection are
///   derived from those parameters every frame — nothing about instances is
///   retained here.
pub(crate) struct ResourceManager<K: Hash + Eq + Copy> {

    //TODO this is still to be moved 
    matrices_uniform_buffer: vulkan_abstraction::UniformBuffer<CameraMatrices>,
    /// Flat per-instance transforms in instance order; `EmissiveIndirectionEntry::entity_id`
    /// indexes into this buffer.
    transforms: vulkan_abstraction::StagingBuffer<vk::TransformMatrixKHR>,
    instances_buffer: vulkan_abstraction::StagingBuffer<vk::AccelerationStructureInstanceKHR>,
    
    
    tlas: vulkan_abstraction::TLAS,
    /// Dense `(emissive triangle slot, instance index)` table for NEE sampling.
    /// Recreated only when the entries actually change (the shader reads its
    /// length via `GetDimensions`, so the buffer must be exactly sized).
    emissive_indirection_gpu: vulkan_abstraction::GpuOnlyBuffer,
    emissive_indirection_cache: Vec<vulkan_abstraction::gltf::EmissiveIndirectionEntry>,

    // ── Stable per-asset GPU data, keyed by `K`.
    /// Per-BLAS mesh info (vertex/index BDA + material). The slot index is the
    /// `gl_InstanceCustomIndexEXT` every instance of that BLAS uses.
    meshes_info: vulkan_abstraction::ArenaGpuBuffer<EntityGpuData>,
    blas_emissive_triangles: vulkan_abstraction::ArenaGpuBuffer<vulkan_abstraction::gltf::EmissiveTriangle>,
    blases: HashMap<K, vulkan_abstraction::BLAS>,
    /// Key → slot in `meshes_info`.
    mesh_info_slots: HashMap<K, u32>,
    /// Key → slots of the BLAS's triangles in `blas_emissive_triangles`.
    emissive_triangle_slots: HashMap<K, Vec<u32>>,
    images: HashMap<K, vulkan_abstraction::Image>,
    /// Finite set of samplers, deduplicated by their description: glTF samplers
    /// don't need to be unique per texture. Never shrinks.
    samplers: HashMap<SamplerDesc, vulkan_abstraction::Sampler>,
    default_sampler: vulkan_abstraction::Sampler,

    //these are action to be done at the start or end of frame together with queued free slots for arena buffers
    //TODO this is to be moved to start of frame stuff with proper callbacks
    buffer_copies_queued: Vec<(vk::Buffer, vk::Buffer, vk::BufferCopy)>,

    core: Rc<vulkan_abstraction::Core>,
}

impl<K: Hash + Eq + Copy> ResourceManager<K> {
    pub fn new_empty(core: Rc<vulkan_abstraction::Core>) -> SrResult<Self> {
        let matrices_uniform_buffer = vulkan_abstraction::UniformBuffer::new(Rc::clone(&core), 1 as vk::DeviceSize)?;

        // SHADER_DEVICE_ADDRESS so the heap path can compute the buffer's BDA
        // when allocating a storage-buffer descriptor (`Buffer::storage_slot`
        // internally calls `vkGetBufferDeviceAddress`).
        let meshes_info = vulkan_abstraction::ArenaGpuBuffer::new(
            core.clone(),
            ARENA_CAPACITY,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "Meshes info GPU buffer",
        )?;

        let blas_emissive_triangles = vulkan_abstraction::ArenaGpuBuffer::new(
            core.clone(),
            ARENA_CAPACITY,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "blas emissive triangles",
        )?;

        let transforms = vulkan_abstraction::StagingBuffer::new(
            Rc::clone(&core),
            10000 as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "Per-frame instance transforms",
        )?;

        let mut instances_buffer = vulkan_abstraction::StagingBuffer::new(
            Rc::clone(&core),
            10000 as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            "Cpu side instances of blases",
        )?;
        
        let tlas = vulkan_abstraction::TLAS::new(Rc::clone(&core), &[], &mut instances_buffer)?;

        let default_sampler = vulkan_abstraction::Sampler::new(
            Rc::clone(&core),
            vk::Filter::LINEAR,
            vk::Filter::LINEAR,
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
            vk::SamplerMipmapMode::LINEAR,
        )?;

        Ok(Self {
            matrices_uniform_buffer,
            transforms,
            instances_buffer,
            tlas,
            emissive_indirection_gpu: vulkan_abstraction::Buffer::new_null(Rc::clone(&core)),
            emissive_indirection_cache: Vec::new(),

            meshes_info,
            blas_emissive_triangles,
            blases: HashMap::new(),
            mesh_info_slots: HashMap::new(),
            emissive_triangle_slots: HashMap::new(),
            images: HashMap::new(),
            samplers: HashMap::new(),
            default_sampler,

            buffer_copies_queued: vec![],
            core,
        })
    }

    pub fn start_of_frame(&mut self) -> SrResult<()> {
        //TODO this can be moved to a render pass at the start of each frame
        self.meshes_info.process_pending_frees();
        self.blas_emissive_triangles.process_pending_frees();

        if self.buffer_copies_queued.is_empty() {
            return Ok(());
        }

        let copies = std::mem::take(&mut self.buffer_copies_queued);

        let mut seen: HashMap<(vk::Buffer, vk::DeviceSize, vk::DeviceSize), usize> = HashMap::new();
        for (i, (_, dst, region)) in copies.iter().enumerate() {
            seen.insert((*dst, region.dst_offset, region.size), i);
        }
        let copies: Vec<_> = seen.values().map(|&i| copies[i]).collect();

        let device = self.core.device().inner();
        let graphics_queue = self.core.graphics_queue();
        let cmd_pool = self.core.graphics_cmd_pool();

        let cmd_buf = vulkan_abstraction::cmd_buffer::new_command_buffer(cmd_pool, device)?;
        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device.begin_command_buffer(cmd_buf, &begin_info)?;

            for (src, dst, region) in &copies {
                device.cmd_copy_buffer(cmd_buf, *src, *dst, std::slice::from_ref(region));
            }

            let unique_dsts: std::collections::HashSet<vk::Buffer> = copies.iter().map(|(_, dst, _)| *dst).collect();
            let barriers: Vec<vk::BufferMemoryBarrier2> = unique_dsts
                .into_iter()
                .map(|buf| {
                    vk::BufferMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR | vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                        .buffer(buf)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)
                })
                .collect();

            let dependency_info = vk::DependencyInfo::default().buffer_memory_barriers(&barriers);
            device.cmd_pipeline_barrier2(cmd_buf, &dependency_info);

            device.end_command_buffer(cmd_buf)?;
        }

        graphics_queue.submit_sync(cmd_buf)?;
        unsafe { device.free_command_buffers(cmd_pool.inner(), &[cmd_buf]) };

        Ok(())
    }

    // ─── Per-frame state ─────────────────────────────────────────────────────

    /// Upload everything that is decided per frame: the camera matrices and the
    /// instance list. `instances` pairs each BLAS key with the world transforms
    /// of its instances this frame; the instance buffer, the TLAS, the flat
    /// transform buffer, and the emissive indirection table are all derived
    /// from it here. The caller must guarantee no frame is in flight (the
    /// renderer's `device_wait_idle` covers this).
    pub fn prepare_frame(
        &mut self,
        matrices: CameraMatrices,
        instances: &[(K, Vec<vk::TransformMatrixKHR>)],
    ) -> SrResult<()> {
        self.set_matrices(matrices)?;

        let mut as_instances: Vec<vk::AccelerationStructureInstanceKHR> = Vec::new();
        let mut emissive_entries: Vec<vulkan_abstraction::gltf::EmissiveIndirectionEntry> = Vec::new();
        let no_emissive: Vec<u32> = Vec::new();

        {
            let transforms_mem = self.transforms.map_mut()?;

            for (key, instance_transforms) in instances {
                let Some(blas) = self.blases.get(key) else {
                    return Err(crate::error::SrError::new_custom(
                        "prepare_frame: instance references a BLAS key that was never loaded".to_string(),
                    ));
                };
                let mesh_info_slot = self.mesh_info_slots[key];
                let emissive_slots = self.emissive_triangle_slots.get(key).unwrap_or(&no_emissive);

                let blas_device_handle = unsafe {
                    self.core
                        .acceleration_structure_device()
                        .get_acceleration_structure_device_address(
                            &vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(blas.inner()),
                        )
                };

                for transform in instance_transforms {
                    let instance_index = as_instances.len();
                    

                    transforms_mem[instance_index] = *transform;

                    as_instances.push(vk::AccelerationStructureInstanceKHR {
                        transform: *transform,
                        instance_custom_index_and_mask: vk::Packed24_8::new(mesh_info_slot, 0xFF),
                        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                            0, // hit_group_offset = 0, same hit group for the whole scene
                            vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
                        ),
                        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                            device_handle: blas_device_handle,
                        },
                    });

                    for &tri_slot in emissive_slots {
                        emissive_entries.push(vulkan_abstraction::gltf::EmissiveIndirectionEntry {
                            blas_tri_index: tri_slot,
                            entity_id: instance_index as u32,
                        });
                    }
                }
            }
        }

        {
            let mapped = self.instances_buffer.map_mut()?;
            mapped[..as_instances.len()].copy_from_slice(&as_instances);
        }
        self.tlas
            .rebuild_from_buffer(as_instances.len() as u32, &self.instances_buffer)?;

        self.update_emissive_indirection(emissive_entries)?;

        Ok(())
    }

    fn set_matrices(
        &mut self,
        CameraMatrices {
            view_inverse,
            proj_inverse,
            view_proj,
            prev_view_proj,
        }: CameraMatrices,
    ) -> SrResult<()> {
        // nalgebra's Matrix4 is column-major in memory. HLSL/Slang's
        // `float4x4(v0, v1, v2, v3)` constructor reads each float4 as a ROW.
        // Transposing here means each on-disk float4 (which the shader reads as
        // a member of `Matrices`) is a ROW of the intended matrix, so the
        // shader's `float4x4(m.vi0, m.vi1, m.vi2, m.vi3)` reconstructs the
        // matrix correctly without any per-shader `transpose()` call.
        let mem = self.matrices_uniform_buffer.map_mut()?;
        mem[0] = CameraMatrices {
            view_inverse: view_inverse.transpose(),
            proj_inverse: proj_inverse.transpose(),
            view_proj: view_proj.transpose(),
            prev_view_proj: prev_view_proj.transpose(),
        };
        Ok(())
    }

    /// Recreate the dense emissive indirection buffer if the entries changed.
    /// The shader reads `num_lights` from the buffer's size (`GetDimensions`),
    /// so the buffer must be exactly sized — it can't be a persistent
    /// max-capacity allocation.
    fn update_emissive_indirection(
        &mut self,
        entries: Vec<vulkan_abstraction::gltf::EmissiveIndirectionEntry>,
    ) -> SrResult<()> {
        if entries == self.emissive_indirection_cache && !self.emissive_indirection_gpu.is_null() {
            return Ok(());
        }

        // A heap descriptor for a zero-sized buffer is invalid; keep a dummy
        // entry when the scene has no emissive geometry (the shader sees
        // num_lights through entry count of real scenes only).
        let dummy = [vulkan_abstraction::gltf::EmissiveIndirectionEntry {
            blas_tri_index: 0,
            entity_id: 0,
        }];
        let data: &[vulkan_abstraction::gltf::EmissiveIndirectionEntry] =
            if entries.is_empty() { &dummy } else { &entries };

        self.emissive_indirection_gpu = vulkan_abstraction::GpuOnlyBuffer::new_from_data(
            Rc::clone(&self.core),
            data,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "emissive indirection",
        )?;
        self.emissive_indirection_cache = entries;

        Ok(())
    }

    // ─── Asset management ────────────────────────────────────────────────────

    /// Register every asset of a loaded scene, assigning each BLAS and image a
    /// fresh key from `make_key`. Materials get their texture references
    /// resolved to descriptor-heap slots here (samplers are deduplicated into
    /// the manager's finite sampler set). Returns the BLAS keys, parallel to
    /// `blases`.
    pub fn add_scene_assets(
        &mut self,
        blases: Vec<crate::LoadedBlas>,
        textures: Vec<vulkan_abstraction::gltf::Texture>,
        sampler_descs: Vec<SamplerDesc>,
        images: Vec<vulkan_abstraction::Image>,
        make_key: &mut dyn FnMut() -> K,
    ) -> SrResult<Vec<K>> {
        let image_slots: Vec<u32> = images.iter().map(|image| image.sampled_slot()).collect();

        let mut sampler_slots = Vec::with_capacity(sampler_descs.len());
        for desc in &sampler_descs {
            sampler_slots.push(self.sampler_slot(desc)?);
        }
        let default_sampler_slot = self.default_sampler.slot();

        let resolve = |texture_index: Option<usize>| -> (u32, u32) {
            match texture_index {
                Some(i) => {
                    let texture = &textures[i];
                    let image_slot = image_slots[texture.source];
                    let sampler_slot = texture
                        .sampler
                        .map(|s| sampler_slots[s])
                        .unwrap_or(default_sampler_slot);
                    (image_slot, sampler_slot)
                }
                None => (Material::NULL_TEXTURE_INDEX, Material::NULL_TEXTURE_INDEX),
            }
        };

        let mut keys = Vec::with_capacity(blases.len());
        for loaded in blases {
            let key = make_key();
            let material = Material::new(&loaded.material, &resolve);
            self.add_blas(key, loaded.blas, material, &loaded.emissive_triangles)?;
            keys.push(key);
        }

        for image in images {
            self.images.insert(make_key(), image);
        }

        Ok(keys)
    }

    /// Register a BLAS under `key`: uploads its mesh info (slot becomes the
    /// instance custom index) and its local-space emissive triangles.
    pub fn add_blas(
        &mut self,
        key: K,
        blas: vulkan_abstraction::BLAS,
        material: Material,
        emissive_triangles: &[vulkan_abstraction::gltf::EmissiveTriangle],
    ) -> SrResult<()> {
        let gpu_data = EntityGpuData {
            vertex_buffer: blas.vertex_buffer().get_device_address(),
            index_buffer: blas.index_buffer().get_device_address(),
            material,
        };
        let (slot, copy_region) = self.meshes_info.allocate_and_update(&gpu_data)?;
        self.queue_copy(self.meshes_info.inner_staging(), self.meshes_info.inner(), copy_region);
        self.mesh_info_slots.insert(key, slot as u32);

        let mut tri_slots = Vec::with_capacity(emissive_triangles.len());
        for tri in emissive_triangles {
            let (tri_slot, tri_copy) = self.blas_emissive_triangles.allocate_and_update(tri)?;
            self.queue_copy(
                self.blas_emissive_triangles.inner_staging(),
                self.blas_emissive_triangles.inner(),
                tri_copy,
            );
            tri_slots.push(tri_slot as u32);
        }
        self.emissive_triangle_slots.insert(key, tri_slots);

        self.blases.insert(key, blas);
        Ok(())
    }

    /// Remove whatever asset `key` refers to (BLAS and/or image). Arena slots
    /// are deferred-freed; the BLAS / image objects are dropped immediately, so
    /// the caller must guarantee the GPU is idle.
    pub fn remove(&mut self, key: &K) {
        if let Some(slot) = self.mesh_info_slots.remove(key) {
            self.meshes_info.free_index(slot as usize);
        }
        if let Some(tri_slots) = self.emissive_triangle_slots.remove(key) {
            for slot in tri_slots {
                self.blas_emissive_triangles.free_index(slot as usize);
            }
        }
        self.blases.remove(key);
        self.images.remove(key);
    }

    /// Slot of the (deduplicated) sampler matching `desc`, creating it on first
    /// use. The sampler set only grows — samplers are never removed.
    fn sampler_slot(&mut self, desc: &SamplerDesc) -> SrResult<u32> {
        if let Some(sampler) = self.samplers.get(desc) {
            return Ok(sampler.slot());
        }
        let sampler = vulkan_abstraction::Sampler::new_from_desc(Rc::clone(&self.core), desc)?;
        let slot = sampler.slot();
        self.samplers.insert(desc.clone(), sampler);
        Ok(slot)
    }

    // ─── Accessors for the heap-mode push constant ──────────────────────────

    pub fn tlas(&self) -> &vulkan_abstraction::TLAS {
        &self.tlas
    }

    /// Buffer-device-address of the matrices uniform buffer. The Slang RT
    /// shaders read matrices via a `Matrices*` BDA pointer rather than a
    /// heap descriptor — see `shaders/rt_types.slang::RaytracingPC.matrices`.
    pub fn matrices_buffer_address(&self) -> vk::DeviceAddress {
        self.matrices_uniform_buffer.get_device_address()
    }

    // Each call lazily allocates a `StorageBuffer` descriptor slot on first use.
    pub fn meshes_info_storage_slot(&self) -> u32 {
        self.meshes_info.raw().storage_slot()
    }

    pub fn emissive_triangles_storage_slot(&self) -> u32 {
        self.blas_emissive_triangles.raw().storage_slot()
    }

    pub fn emissive_indirection_storage_slot(&self) -> u32 {
        self.emissive_indirection_gpu.raw().storage_slot()
    }

    pub fn entity_transforms_storage_slot(&self) -> u32 {
        self.transforms.raw().storage_slot()
    }

    // ─── Internal helpers ────────────────────────────────────────────────────

    fn queue_copy(&mut self, src: vk::Buffer, dst: vk::Buffer, region: vk::BufferCopy) {
        self.buffer_copies_queued.push((src, dst, region));
    }
}
