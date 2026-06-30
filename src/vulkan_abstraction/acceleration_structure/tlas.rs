use std::cell::Cell;
use std::rc::Rc;

use crate::vulkan_abstraction::Buffer;
use crate::vulkan_abstraction::descriptor_heap::DescriptorSlot;
use crate::{error::*, vulkan_abstraction};
use ash::vk;
// Resources:
// - https://github.com/adrien-ben/vulkan-examples-rs
// - https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/
// - https://github.com/SaschaWillems/Vulkan

// TODO: implement drop
pub struct TLAS {
    tlas: vulkan_abstraction::AccelerationStructure,
    //TODO always give slot
    slot: Cell<Option<DescriptorSlot>>,
}

impl TLAS {
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        blas_instances: &[vulkan_abstraction::BlasInstance],
        instances_buffer: &mut impl Buffer,
    ) -> SrResult<Self> {
        Self::insert_in_instances_buffer(Rc::clone(&core), blas_instances, instances_buffer)?;

        let geometry = Self::make_geometry(instances_buffer);

        let build_range_info = Self::make_build_range_info(blas_instances.len() as u32);

        let tlas = vulkan_abstraction::AccelerationStructure::new(
            core,
            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            &[build_range_info],
            &[geometry],
            true,
            false,
        )?;

        Ok(Self {
            tlas,
            slot: Cell::new(None),
        })
    }
    /// "the application must not use an update operation to do any of the following:
    /// - Change primitives or instances from active to inactive, or vice versa
    /// - Change the index or vertex formats of triangle geometry.
    /// - Change triangle geometry transform pointers from null to non-null or vice versa.
    /// - Change the number of geometries or instances in the structure.
    /// - Change the geometry flags for any geometry in the structure.
    /// - Change the number of vertices or primitives for any geometry in the structure."
    /// (from <https://docs.vulkan.org/spec/latest/chapters/accelstructures.html#acceleration-structure-update>)
    ///
    /// Basically from what I can tell only the following operations are allowed in a TLAS update:
    /// - Change one or more transform matrices
    /// - switch one BLAS instance for another, possibly to switch LODs
    #[allow(unused)]
    pub fn update(
        &mut self,
        blas_instances: &[vulkan_abstraction::BlasInstance],
        instances_buffer: &mut impl Buffer,
    ) -> SrResult<()> {
        if !self.tlas.allow_update {
            return SrResult::Err(SrError::new_custom("The structure is not updatable".to_string()));
        }
        Self::insert_in_instances_buffer(Rc::clone(self.tlas.core()), blas_instances, instances_buffer)?;

        let geometry = Self::make_geometry(instances_buffer);

        let build_range_info = Self::make_build_range_info(blas_instances.len() as u32);

        self.tlas.update(&[build_range_info], &[geometry])?;

        Ok(())
    }

    #[allow(unused)]
    pub fn rebuild(
        &mut self,
        blas_instances: &[vulkan_abstraction::BlasInstance],
        instances_buffer: &mut impl Buffer,
    ) -> SrResult<()> {
        Self::insert_in_instances_buffer(Rc::clone(self.tlas.core()), blas_instances, instances_buffer)?;

        let geometry = Self::make_geometry(instances_buffer);

        let build_range_info = Self::make_build_range_info(blas_instances.len() as u32);

        self.tlas.rebuild(&[build_range_info], &[geometry], false)?;

        Ok(())
    }

    /// Rebuild the TLAS from instances already written into `instances_buffer`
    /// (the caller maps the buffer and fills the first `instance_count` entries).
    /// Used by the per-frame path where the `ResourceManager` builds the raw
    /// `vk::AccelerationStructureInstanceKHR` list itself.
    pub fn rebuild_from_buffer(&mut self, instance_count: u32, instances_buffer: &impl Buffer) -> SrResult<()> {
        let geometry = Self::make_geometry(instances_buffer);

        let build_range_info = Self::make_build_range_info(instance_count);

        self.tlas.rebuild(&[build_range_info], &[geometry], false)?;

        Ok(())
    }

    fn insert_in_instances_buffer<'a>(
        core: Rc<vulkan_abstraction::Core>,
        blas_instances: &[vulkan_abstraction::BlasInstance],
        instances_buffer: &mut impl Buffer,
    ) -> SrResult<()> {
        let blas_instances: Vec<vk::AccelerationStructureInstanceKHR> = blas_instances
            .iter()
            .map(|blas_instance| {
                vk::AccelerationStructureInstanceKHR {
                    transform: blas_instance.transform,
                    instance_custom_index_and_mask: vk::Packed24_8::new(blas_instance.blas_instance_index, 0xFF), // mask = 0 (don't know what actually does, NV tutorial writes "Only be hit if rayMask & instance.mask != 0")
                    instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                        0, // hit_group_offset = 0, same hit group for the whole scene
                        vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8, // disable face culling for semplicity
                    ),
                    acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                        device_handle: unsafe {
                            core.acceleration_structure_device()
                                .get_acceleration_structure_device_address(
                                    &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                                        .acceleration_structure(blas_instance.blas.inner()),
                                )
                        },
                    },
                }
            })
            .collect();

        // buffer to hold the instances
        // let instances_buffer = vulkan_abstraction::Buffer::new_from_data(
        //     core,
        //     &blas_instances,
        //     gpu_allocator::MemoryLocation::CpuToGpu,
        //     vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        //     "TLAS instances buffer",
        // )?;

        let mapped_memory = instances_buffer.raw_mut().map_mut::<vk::AccelerationStructureInstanceKHR>()?;

        // 2. Scrivi i dati sovrascrivendo quelli vecchi
        for (i, instance) in blas_instances.iter().enumerate() {
            mapped_memory[i] = vk::AccelerationStructureInstanceKHR {
                transform: instance.transform,
                instance_custom_index_and_mask: instance.instance_custom_index_and_mask,
                instance_shader_binding_table_record_offset_and_flags: instance
                    .instance_shader_binding_table_record_offset_and_flags,
                acceleration_structure_reference: instance.acceleration_structure_reference,
            };
        }

        Ok(())
    }

    fn make_geometry(instances_buffer: &impl Buffer) -> vk::AccelerationStructureGeometryKHR<'_> {
        vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .flags(vk::GeometryFlagsKHR::empty())
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                    .array_of_pointers(false)
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: instances_buffer.get_device_address(),
                    }),
            })
    }

    fn make_build_range_info(primitive_count: u32) -> vk::AccelerationStructureBuildRangeInfoKHR {
        vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(primitive_count)
            .primitive_offset(0)
            .first_vertex(0)
            .transform_offset(0)
    }

    pub fn inner(&self) -> vk::AccelerationStructureKHR {
        self.tlas.inner()
    }

    /// `vkGetAccelerationStructureDeviceAddressKHR` of the underlying TLAS.
    /// Used by the heap-mode RT pipelines because Slang's
    /// `DescriptorHandle<RaytracingAccelerationStructure>` codegen is broken
    /// on `spvDescriptorHeapEXT` (Slang issue #10671) — the shader does the
    /// uint64→AS convert via inline SPIR-V instead.
    pub fn device_address(&self) -> vk::DeviceAddress {
        let core = self.tlas.core();
        unsafe {
            core.acceleration_structure_device()
                .get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(self.tlas.inner()),
                )
        }
    }

    /// Heap slot for `ACCELERATION_STRUCTURE_KHR`. Lazily allocates on first call.
    pub fn slot(&self) -> u32 {
        if let Some(s) = self.slot.get() {
            return s.shader_index();
        }
        let core = self.tlas.core();
        let address = unsafe {
            core.acceleration_structure_device()
                .get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(self.tlas.inner()),
                )
        };
        let mut heap = core.descriptor_heap_mut();
        let slot =
            heap.alloc_resource_slot(crate::vulkan_abstraction::descriptor_heap::ResourceDescriptorKind::AccelerationStructure);
        heap.write_acceleration_structure(slot, address)
            .expect("descriptor heap write_acceleration_structure failed");
        self.slot.set(Some(slot));
        slot.shader_index()
    }
}

impl Drop for TLAS {
    fn drop(&mut self) {
        
        if let Some(s) = self.slot.get() {
            self.tlas.core().descriptor_heap_mut().free(s);
        }
    }
}
