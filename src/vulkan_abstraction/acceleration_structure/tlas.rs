use crate::error::*;
use crate::vulkan_abstraction;
use crate::vulkan_abstraction::descriptor_heap::{DescriptorSlot, ResourceDescriptorKind};
use crate::vulkan_abstraction::{AccelerationStructure, AsBuildInputs, AsBuildJob, Buffer, Dynamic, RawBuffer};
use ash::vk;
use std::rc::Rc;
use std::sync::Arc;
use crate::vulkan_abstraction::acceleration_structure::BuildType;
// Resources:
// - https://github.com/adrien-ben/vulkan-examples-rs
// - https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/

pub struct Tlas {
    accel: AccelerationStructure,
    slot: DescriptorSlot,
    build_type: BuildType,
}

#[derive(Debug)]
pub struct TlasBuildDesc {
    instances_buffer: RawBuffer,
    instance_count: u32,
}




impl Tlas {
    /// Build a TLAS over the `instance_count` instances already written into
    /// `instances_buffer`
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        instances_buffer: &impl Buffer,
        instance_count: u32,
        build_type: BuildType,
    ) -> SrResult<Self> {
        let accel = AccelerationStructure::build_sync(
            Rc::clone(&core),
            Self::make_inputs(instances_buffer, instance_count, build_type),
        )?;

        let slot = {
            let mut heap = core.descriptor_heap_mut();
            let slot = heap.alloc_resource_slot(ResourceDescriptorKind::AccelerationStructure);
            heap.write_acceleration_structure(slot, accel.device_address())?;
            slot
        };

        Ok(Self { accel, slot, build_type })
    }

    /// Rebuild the TLAS from instances already written into `instances_buffer`
    /// (the renderer's frame-local buffer). Synchronous. Replaces the underlying
    /// structure and re-points the heap slot at the new structure's address.
    ///
    /// The old structure is dropped immediately; the renderer waits for device
    /// idle before this, so no in-flight frame still references it.
    pub fn rebuild_from_buffer(&mut self, instance_count: u32, instances_buffer: &impl Buffer) -> SrResult<()> {
        let accel = AccelerationStructure::build_sync(
            Rc::clone(self.accel.core()),
            Self::make_inputs(instances_buffer, instance_count, self.build_type),
        )?;

        self.accel = accel;
        // A rebuild yields a new handle/address — re-point the heap slot so it
        // never goes stale (harmless to the live RT path, which reads
        // `device_address()` fresh as a push constant, but correct for any
        // heap-descriptor consumer).
        self.write_slot()?;

        log::debug!("TOP_LEVEL acceleration structure rebuilt");
        Ok(())
    }

    /// In-place UPDATE of the TLAS from instances already written into
    /// `instances_buffer` — same instance count / layout, new contents
    /// (transforms, BLAS references). Requires the TLAS was built as
    /// [`BuildType::Updatable`]. Mirrors `Blas::update`: cheaper than a full
    /// rebuild and, since an UPDATE keeps the same handle/address, the heap slot
    /// stays valid so no re-point is needed. Synchronous.
    #[allow(unused)]
    pub fn update(&mut self, instance_count: u32, instances_buffer: &impl Buffer) -> SrResult<()> {
        if !Self::build_flags(self.build_type).contains(vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE) {
            return Err(SrError::new_custom("The structure is not updatable".to_string()));
        }

        self.accel
            .update_sync(Self::make_inputs(instances_buffer, instance_count, self.build_type))?;

        log::debug!("TOP_LEVEL acceleration structure updated in place");
        Ok(())
    }

    /// **Deferred** rebuild for the render graph. Eagerly create the new TLAS
    /// (its device address is valid immediately, so it can be baked into this
    /// frame's push constants) and return it together with the [`AsBuildJob`] the
    /// graph records into its command buffer. Does **not** mutate `self`: the new
    /// structure is not the live one until [`Self::commit_rebuild`] installs it.
    ///
    /// The caller keeps the returned structure and the scratch it feeds the job
    /// alive until the job's submission completes, then calls `commit_rebuild`
    /// (from an end-of-frame closure) to swap it in.
    #[allow(unused)]
    pub fn prepare_rebuild(
        &self,
        instance_count: u32,
        instances_buffer: &impl Buffer,
    ) -> SrResult<(AccelerationStructure, AsBuildJob)> {
        AccelerationStructure::build(
            Rc::clone(self.accel.core()),
            Self::make_inputs(instances_buffer, instance_count, self.build_type),
        )
    }

    /// Install a structure produced by [`Self::prepare_rebuild`] as the live TLAS
    /// once its build has completed on the GPU: swap it in, re-point the heap slot
    /// at its address, and return the **old** structure for the caller to drop
    /// (deferred past the fence that guarded the previous frame). The graph can't
    /// perform this CPU-side swap itself, so it runs in an end-of-frame closure.
    #[allow(unused)]
    pub fn commit_rebuild(&mut self, built: AccelerationStructure) -> SrResult<AccelerationStructure> {
        let old = std::mem::replace(&mut self.accel, built);
        // The new structure's address is valid at create time, so re-pointing the
        // slot here (after its build completed) never goes stale.
        self.write_slot()?;
        Ok(old)
    }

    /// Map a [`BuildType`] to its Vulkan build flags. `Updatable` reproduces
    /// the pre-rework `allow_update = true, fast_build = false` path.
    fn build_flags(build_type: BuildType) -> vk::BuildAccelerationStructureFlagsKHR {
        match build_type {
            BuildType::RapidlyChanging =>   vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE ,
            BuildType::SometimesChanges => {
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE
            }
            BuildType::Static => vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        }
    }

    /// (Re)write the current structure's device address into the heap slot.
    fn write_slot(&self) -> SrResult<()> {
        self.accel
            .core()
            .descriptor_heap_mut()
            .write_acceleration_structure(self.slot, self.accel.device_address())
    }

    /// Realize the owned build inputs for a TLAS over `instance_count` instances
    /// in `instances_buffer`. The geometry stores only the buffer's device
    /// address, so the `'static` geometry struct borrows nothing.
    fn make_inputs(instances_buffer: &impl Buffer, instance_count: u32, build_type: BuildType) -> AsBuildInputs {
        AsBuildInputs {
            ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            flags: Self::build_flags(build_type),
            geometries: vec![Self::make_geometry(instances_buffer)],
            ranges: vec![Self::make_build_range_info(instance_count)],
        }
    }

    fn make_geometry(instances_buffer: &impl Buffer) -> vk::AccelerationStructureGeometryKHR<'static> {
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
        self.accel.inner()
    }

    /// `vkGetAccelerationStructureDeviceAddressKHR` of the underlying TLAS
    /// (cached). Used by the heap-mode RT pipelines because Slang's
    /// `DescriptorHandle<RaytracingAccelerationStructure>` codegen is broken on
    /// `spvDescriptorHeapEXT` (Slang issue #10671) — the shader does the
    /// uint64→AS convert via inline SPIR-V instead.
    pub fn device_address(&self) -> vk::DeviceAddress {
        self.accel.device_address()
    }

    pub fn slot(&self) -> u32 {
        self.slot.shader_index()
    }
}

impl Drop for Tlas {
    fn drop(&mut self) {
        self.accel.core().descriptor_heap_mut().free(self.slot);
    }
}
