use std::rc::Rc;
use std::sync::Arc;
use crate::error::*;
use crate::vulkan_abstraction;
use crate::vulkan_abstraction::descriptor_heap::{DescriptorSlot, ResourceDescriptorKind};
use crate::vulkan_abstraction::{AccelerationStructure, AsBuildInputs, Buffer, RawBuffer};
use ash::vk;
// Resources:
// - https://github.com/adrien-ben/vulkan-examples-rs
// - https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/

pub struct Tlas {
    accel: AccelerationStructure,
    slot: DescriptorSlot,
}

#[derive(Clone)]
pub struct TlasBuildDesc{
    instances_buffer: Arc<RawBuffer>,
    instance_count: u32
}

impl Tlas {
    /// Build a TLAS over the `instance_count` instances already written into
    /// `instances_buffer`
    pub fn new(core: Rc<vulkan_abstraction::Core>, instances_buffer: &impl Buffer, instance_count: u32) -> SrResult<Self> {
        let geometry = Self::make_geometry(instances_buffer);
        let build_range_info = Self::make_build_range_info(instance_count);

        let accel = AccelerationStructure::build_sync(
            Rc::clone(&core),
            &AsBuildInputs {
                ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                flags: Self::build_flags(),
                geometries: &[geometry],
                ranges: &[build_range_info],
            },
        )?;

        let slot = {
            let mut heap = core.descriptor_heap_mut();
            let slot = heap.alloc_resource_slot(ResourceDescriptorKind::AccelerationStructure);
            heap.write_acceleration_structure(slot, accel.device_address())?;
            slot
        };

        Ok(Self { accel, slot })
    }

    /// Rebuild the TLAS from instances already written into `instances_buffer`
    /// (the renderer's frame-local buffer). Synchronous. Replaces the underlying
    /// structure and re-points the heap slot at the new structure's address.
    ///
    /// The old structure is dropped immediately; the renderer waits for device
    /// idle before this, so no in-flight frame still references it.
    pub fn rebuild_from_buffer(&mut self, instance_count: u32, instances_buffer: &impl Buffer) -> SrResult<()> {
        let geometry = Self::make_geometry(instances_buffer);
        let build_range_info = Self::make_build_range_info(instance_count);

        let accel = AccelerationStructure::build_sync(
            Rc::clone(self.accel.core()),
            &AsBuildInputs {
                ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                flags: Self::build_flags(),
                geometries: &[geometry],
                ranges: &[build_range_info],
            },
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

    /// External-command-buffer rebuild: record the build into `cmd_buf` instead
    /// of submitting. Returns the **old** structure and the build scratch buffer
    /// — both must be kept alive until `cmd_buf`'s submission completes, then
    /// dropped. Not yet wired into the per-frame path.
    #[allow(unused)]
    pub fn record_rebuild(
        &mut self,
        cmd_buf: vk::CommandBuffer,
        instance_count: u32,
        instances_buffer: &impl Buffer,
    ) -> SrResult<(AccelerationStructure, vulkan_abstraction::GpuOnlyBuffer)> {
        let geometry = Self::make_geometry(instances_buffer);
        let build_range_info = Self::make_build_range_info(instance_count);

        let (accel, scratch) = AccelerationStructure::record_build(
            Rc::clone(self.accel.core()),
            cmd_buf,
            &AsBuildInputs {
                ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                flags: Self::build_flags(),
                geometries: &[geometry],
                ranges: &[build_range_info],
            },
        )?;

        let old = std::mem::replace(&mut self.accel, accel);
        // The new structure's address is valid at create time, so the slot can
        // be re-pointed now even though the build hasn't executed yet.
        self.write_slot()?;
        Ok((old, scratch))
    }

    /// Build flags for the TLAS — identical to the pre-rework
    /// `allow_update = true, fast_build = false` path.
    fn build_flags() -> vk::BuildAccelerationStructureFlagsKHR {
        vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE
    }

    /// (Re)write the current structure's device address into the heap slot.
    fn write_slot(&self) -> SrResult<()> {
        self.accel
            .core()
            .descriptor_heap_mut()
            .write_acceleration_structure(self.slot, self.accel.device_address())
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
