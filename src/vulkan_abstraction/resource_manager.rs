use std::collections::HashMap;
use std::hash::Hash;
use std::rc::Rc;

use ash::vk;

use crate::render_graph::graph::SamplerDesc;
use crate::vulkan_abstraction::{ArenaBuffer, Buffer, EntityGpuData, Material};
use crate::{error::SrResult, vulkan_abstraction};

const ARENA_CAPACITY: vk::DeviceSize = 4096 * 16;

//TODO handle growable

/// Deferred work executed at the start of a specific absolute frame (see
/// [`ResourceManager::start_of_frame`]).
type FrameCallback<K> = Box<dyn FnOnce(&mut ResourceManager<K>) -> SrResult<()>>;

/// The raw per-frame data resolved from the caller's `(key, transforms)`
/// instance list. The renderer uploads these into CpuToGpu buffers created on
/// the spot each frame (and deferred-freed through the end-of-frame
/// callbacks) — nothing per-frame is retained in the manager.
pub(crate) struct FrameInstanceData {
    pub as_instances: Vec<vk::AccelerationStructureInstanceKHR>,
    /// Flat per-instance transforms in instance order;
    /// `EmissiveIndirectionEntry::entity_id` indexes into this list.
    pub transforms: Vec<vk::TransformMatrixKHR>,
    /// Dense `(emissive triangle slot, instance index)` table for NEE sampling.
    pub emissive_entries: Vec<vulkan_abstraction::gltf::EmissiveIndirectionEntry>,
}

//TODO there is structural decision to make on what to save and how cause raster needs a different way to store the geometry data probabibly with a features set constable?
/// Owns the *stable* GPU-side scene resources, keyed by the caller-provided
/// `K` (one key per BLAS / image): geometry + material data lives in arena
/// buffers whose slots survive across frames, plus the TLAS (rebuilt per frame
/// from the renderer's local instance buffer). Per-frame data only passes
/// through [`Self::frame_instance_data`] — it is never stored here.
pub(crate) struct ResourceManager<K: Hash + Eq + Copy> {
    tlas: vulkan_abstraction::TLAS,

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

    /// Pending staging→GPU copy regions for the arena buffers; flushed by the
    /// callback `queue_copy` schedules for the upcoming frame.
    buffer_copies_queued: Vec<(vk::Buffer, vk::Buffer, vk::BufferCopy)>,
    /// Deferred work keyed by the absolute frame at whose start it must run
    /// (arena copy flushes, deferred slot frees). Drained by
    /// [`Self::start_of_frame`] — nothing runs unconditionally every frame.
    //TODO the copy flush should become a transfer pass recorded at the head of
    //     the render graph instead of its own synchronous submit, and the
    //     free callbacks should key off GPU completion (the frame timeline)
    //     rather than the CPU-side frame counter + the renderer's wait-idle.
    start_of_frame_callbacks: Vec<(u64, FrameCallback<K>)>,

    core: Rc<vulkan_abstraction::Core>,
}



// `K: 'static` because deferred frame work is stored as boxed `FnOnce(&mut Self)`.
impl<K: Hash + Eq + Copy + 'static> ResourceManager<K> {
    pub fn new_empty(core: Rc<vulkan_abstraction::Core>) -> SrResult<Self> {
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

        // Temporary one-element buffer for the initial empty TLAS build (the
        // build is synchronous, so dropping it right after is fine). Per-frame
        // instance buffers are created by the renderer each frame.
        let mut empty_instances_buffer = vulkan_abstraction::StagingBuffer::<vk::AccelerationStructureInstanceKHR>::new(
            Rc::clone(&core),
            1,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            "empty TLAS build instances",
        )?;

        let tlas = vulkan_abstraction::TLAS::new(Rc::clone(&core), &[], &mut empty_instances_buffer)?;

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
            tlas,

            meshes_info,
            blas_emissive_triangles,
            blases: HashMap::new(),
            mesh_info_slots: HashMap::new(),
            emissive_triangle_slots: HashMap::new(),
            images: HashMap::new(),
            samplers: HashMap::new(),
            default_sampler,

            buffer_copies_queued: vec![],
            start_of_frame_callbacks: vec![],
            core,
        })
    }

    // ─── Start-of-frame scheduling ───────────────────────────────────────────

    /// Absolute frame number the next rendered frame will carry.
    fn next_frame(&self) -> u64 {
        *self.core.absolute_frame_count.borrow() as u64 + 1
    }

    /// Schedule `callback` to run at the start of frame `frame` (or the first
    /// `start_of_frame` call at/after it).
    fn schedule_at_frame(&mut self, frame: u64, callback: impl FnOnce(&mut Self) -> SrResult<()> + 'static) {
        self.start_of_frame_callbacks.push((frame, Box::new(callback)));
    }

    /// Run the deferred work due at the start of `upcoming_frame` (the frame
    /// about to be rendered): arena copy flushes scheduled by asset loads and
    /// slot-free processing scheduled by `remove`. A callback may schedule
    /// further callbacks; ones due this same frame run before this returns.
    pub fn start_of_frame(&mut self, upcoming_frame: u64) -> SrResult<()> {
        let mut i = 0;
        while i < self.start_of_frame_callbacks.len() {
            if self.start_of_frame_callbacks[i].0 <= upcoming_frame {
                let (_, callback) = self.start_of_frame_callbacks.remove(i);
                callback(self)?;
            } else {
                i += 1;
            }
        }
        Ok(())
    }

    /// Flush the queued arena staging→GPU copies in one synchronous submit.
    /// Runs as the start-of-frame callback `queue_copy` schedules.
    //TODO this can be moved to a render pass at the start of each frame
    fn flush_queued_copies(&mut self) -> SrResult<()> {
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

    // ─── Per-frame data ──────────────────────────────────────────────────────

    /// Resolve the caller's per-frame `(key, transforms)` instance list into
    /// the raw arrays the frame needs: TLAS instances (custom index = the
    /// BLAS's stable mesh-info slot), the flat transform list (instance
    /// order), and the emissive indirection entries. Pure resolution — the
    /// renderer uploads the results into frame-local CpuToGpu buffers; nothing
    /// is stored here.
    pub fn frame_instance_data(&self, instances: &[(K, Vec<vk::TransformMatrixKHR>)]) -> SrResult<FrameInstanceData> {
        let mut as_instances: Vec<vk::AccelerationStructureInstanceKHR> = Vec::new();
        let mut transforms: Vec<vk::TransformMatrixKHR> = Vec::new();
        let mut emissive_entries: Vec<vulkan_abstraction::gltf::EmissiveIndirectionEntry> = Vec::new();
        let no_emissive: Vec<u32> = Vec::new();

        for (key, instance_transforms) in instances {
            let Some(blas) = self.blases.get(key) else {
                return Err(crate::error::SrError::new_custom(
                    "frame_instance_data: instance references a BLAS key that was never loaded".to_string(),
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
                let instance_index = transforms.len();

                transforms.push(*transform);

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

        Ok(FrameInstanceData {
            as_instances,
            transforms,
            emissive_entries,
        })
    }

    /// Rebuild the TLAS from instances already written into `instances_buffer`
    /// (the renderer's frame-local buffer). Synchronous.
    pub fn rebuild_tlas(&mut self, instance_count: u32, instances_buffer: &impl Buffer) -> SrResult<()> {
        self.tlas.rebuild_from_buffer(instance_count, instances_buffer)
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
    /// are deferred-freed (reclaimed by a start-of-frame callback scheduled for
    /// the frame at which no in-flight frame can still read them); the BLAS /
    /// image objects are dropped immediately, so the caller must guarantee the
    /// GPU is idle.
    pub fn remove(&mut self, key: &K) {
        let mut any_freed = false;
        if let Some(slot) = self.mesh_info_slots.remove(key) {
            self.meshes_info.free_index(slot as usize);
            any_freed = true;
        }
        if let Some(tri_slots) = self.emissive_triangle_slots.remove(key) {
            for slot in tri_slots {
                self.blas_emissive_triangles.free_index(slot as usize);
                any_freed = true;
            }
        }
        self.blases.remove(key);
        self.images.remove(key);

        if any_freed {
            // The arenas tag each free with the frame it happened on and only
            // reclaim once MAX_FRAMES_IN_FLIGHT frames have passed, so running
            // the callback at next_frame + MAX_FRAMES_IN_FLIGHT reclaims
            // exactly these slots (re-checked per slot — running early or late
            // is safe, just less precise).
            let due = self.next_frame() + crate::MAX_FRAMES_IN_FLIGHT as u64;
            self.schedule_at_frame(due, |rm| {
                rm.meshes_info.process_pending_frees();
                rm.blas_emissive_triangles.process_pending_frees();
                Ok(())
            });
        }
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

    // Each call lazily allocates a `StorageBuffer` descriptor slot on first use.
    pub fn meshes_info_storage_slot(&self) -> u32 {
        self.meshes_info.raw().storage_slot()
    }

    pub fn emissive_triangles_storage_slot(&self) -> u32 {
        self.blas_emissive_triangles.raw().storage_slot()
    }

    // ─── Internal helpers ────────────────────────────────────────────────────

    fn queue_copy(&mut self, src: vk::Buffer, dst: vk::Buffer, region: vk::BufferCopy) {
        // First copy since the last flush: schedule one flush callback for the
        // upcoming frame (the flush drains the whole queue, so later copies
        // queued before it runs piggyback on the same callback).
        if self.buffer_copies_queued.is_empty() {
            let due = self.next_frame();
            self.schedule_at_frame(due, Self::flush_queued_copies);
        }
        self.buffer_copies_queued.push((src, dst, region));
    }
}
