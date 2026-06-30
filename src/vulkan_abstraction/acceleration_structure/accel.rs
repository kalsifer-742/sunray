use std::rc::Rc;

use crate::error::*;
use crate::vulkan_abstraction;
use crate::vulkan_abstraction::Buffer;
use ash::vk;

/// Plain build inputs for a single acceleration-structure build. Borrows the
/// caller's realized geometry/range slices; holds no policy and no ownership.
pub struct AsBuildInputs<'a> {
    pub ty: vk::AccelerationStructureTypeKHR,
    pub flags: vk::BuildAccelerationStructureFlagsKHR,
    pub geometries: &'a [vk::AccelerationStructureGeometryKHR<'a>],
    pub ranges: &'a [vk::AccelerationStructureBuildRangeInfoKHR],
}

//TODO there should be an option to give scratch buffers from outside


/// A bare GPU acceleration-structure **resource**: just the handle, its backing
/// buffer, and the device address. No build description and no rebuild policy —
/// those belong to the owning wrapper ([`vulkan_abstraction::Blas`] /
/// [`vulkan_abstraction::Tlas`]). A future cluster BLAS, which has no
/// `vk::AccelerationStructureKHR` handle (only an address), is expected to be a
/// *sibling* resource type exposing the same `device_address()`.
pub struct AccelerationStructure {
    core: Rc<vulkan_abstraction::Core>,
    handle: vk::AccelerationStructureKHR,
    #[allow(dead_code)]
    buffer: vulkan_abstraction::GpuOnlyBuffer,
    device_address: vk::DeviceAddress,
}

impl AccelerationStructure {
    /// Allocate the backing buffer, create the (unbuilt) handle, and record the
    /// build of `inputs` into `cmd_buf`. Returns the resource **and the scratch
    /// buffer**, which the caller must keep alive until `cmd_buf`'s submission
    /// completes. Does not submit — the caller owns timing and queue.
    pub fn record_build(
        core: Rc<vulkan_abstraction::Core>,
        cmd_buf: vk::CommandBuffer,
        inputs: &AsBuildInputs,
    ) -> SrResult<(Self, vulkan_abstraction::GpuOnlyBuffer)> {
        assert_eq!(inputs.geometries.len(), inputs.ranges.len());

        // Parameters used first to size the allocations; the real build info
        // below is derived from it with the dst handle + scratch address filled in.
        let incomplete_build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .geometries(inputs.geometries)
            .flags(inputs.flags)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .ty(inputs.ty);

        let size_info = unsafe {
            let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
            let primitive_counts = inputs.ranges.iter().map(|i| i.primitive_count).collect::<Vec<_>>();
            core.acceleration_structure_device().get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &incomplete_build_geometry_info,
                Some(primitive_counts.as_slice()),
                &mut size_info,
            );
            size_info
        };

        let (handle, buffer, device_address) = Self::create_backed(&core, inputs.ty, size_info.acceleration_structure_size)?;

        // Scratch buffer; discardable once the build has executed.
        let scratch_buffer = vulkan_abstraction::GpuOnlyBuffer::new_aligned::<u8>(
            Rc::clone(&core),
            size_info.build_scratch_size,
            core.device()
                .acceleration_structure_properties()
                .min_acceleration_structure_scratch_offset_alignment as u64,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
            "acceleration structure build scratch buffer",
        )?;

        let build_geometry_info = incomplete_build_geometry_info
            .dst_acceleration_structure(handle)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_buffer.get_device_address(),
            });

        unsafe {
            core.acceleration_structure_device().cmd_build_acceleration_structures(
                cmd_buf,
                &[build_geometry_info],
                &[Some(inputs.ranges)],
            );
        }

        Ok((
            Self {
                core,
                handle,
                buffer,
                device_address,
            },
            scratch_buffer,
        ))
    }

    /// Convenience: build synchronously on the graphics queue (one-shot command
    /// buffer, record + submit + wait + free). Behaviorally identical to the old
    /// `new_sync`; used by the classic per-frame path.
    pub fn build_sync(core: Rc<vulkan_abstraction::Core>, inputs: &AsBuildInputs) -> SrResult<Self> {
        let cmd_buf = vulkan_abstraction::cmd_buffer::new_command_buffer(core.graphics_cmd_pool(), core.device().inner())?;
        unsafe {
            core.device().inner().begin_command_buffer(
                cmd_buf,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }

        // `_scratch` must outlive the submit (it does — dropped at end of scope,
        // after `submit_sync` has waited for the build to finish).
        let (accel, _scratch) = Self::record_build(Rc::clone(&core), cmd_buf, inputs)?;

        unsafe { core.device().inner().end_command_buffer(cmd_buf)? }

        // NOTE: synchronous submit — bad for throughput when many builds happen
        // back to back. The `record_*` API exists precisely so callers can batch.
        core.graphics_queue().submit_sync(cmd_buf)?;

        unsafe {
            core.device()
                .inner()
                .free_command_buffers(core.graphics_cmd_pool().inner(), &[cmd_buf]);
        }

        Ok(accel)
    }

    /// Record an in-place UPDATE (src = dst = this handle) of `inputs` into
    /// `cmd_buf`. `inputs.flags` must contain `ALLOW_UPDATE` and the geometry
    /// layout must match the original build (see the Vulkan spec restrictions).
    /// Returns the scratch buffer to keep alive until `cmd_buf` completes.
    #[allow(dead_code)]
    pub fn record_update(
        &mut self,
        cmd_buf: vk::CommandBuffer,
        inputs: &AsBuildInputs,
    ) -> SrResult<vulkan_abstraction::GpuOnlyBuffer> {
        assert_eq!(inputs.geometries.len(), inputs.ranges.len());

        let incomplete_build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .geometries(inputs.geometries)
            .flags(inputs.flags)
            .mode(vk::BuildAccelerationStructureModeKHR::UPDATE)
            .ty(inputs.ty);

        let size_info = unsafe {
            let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
            let primitive_counts = inputs.ranges.iter().map(|i| i.primitive_count).collect::<Vec<_>>();
            self.core
                .acceleration_structure_device()
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &incomplete_build_geometry_info,
                    Some(primitive_counts.as_slice()),
                    &mut size_info,
                );
            size_info
        };

        let scratch_buffer = vulkan_abstraction::GpuOnlyBuffer::new_aligned::<u8>(
            Rc::clone(&self.core),
            size_info.build_scratch_size,
            self.core
                .device()
                .acceleration_structure_properties()
                .min_acceleration_structure_scratch_offset_alignment as u64,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
            "acceleration structure update scratch buffer",
        )?;

        let build_geometry_info = incomplete_build_geometry_info
            .src_acceleration_structure(self.handle)
            .dst_acceleration_structure(self.handle)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_buffer.get_device_address(),
            });

        unsafe {
            self.core.acceleration_structure_device().cmd_build_acceleration_structures(
                cmd_buf,
                &[build_geometry_info],
                &[Some(inputs.ranges)],
            );
        }

        Ok(scratch_buffer)
    }

    /// Convenience synchronous in-place UPDATE on the graphics queue.
    #[allow(dead_code)]
    pub fn update_sync(&mut self, inputs: &AsBuildInputs) -> SrResult<()> {
        let cmd_buf =
            vulkan_abstraction::cmd_buffer::new_command_buffer(self.core.graphics_cmd_pool(), self.core.device().inner())?;
        unsafe {
            self.core.device().inner().begin_command_buffer(
                cmd_buf,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }

        let _scratch = self.record_update(cmd_buf, inputs)?;

        unsafe { self.core.device().inner().end_command_buffer(cmd_buf)? }
        self.core.graphics_queue().submit_sync(cmd_buf)?;
        unsafe {
            self.core
                .device()
                .inner()
                .free_command_buffers(self.core.graphics_cmd_pool().inner(), &[cmd_buf]);
        }

        Ok(())
    }

    /// Record a COMPACT copy of this structure into a freshly-allocated,
    /// minimum-sized buffer, returning the new (compacted) resource. `cmd_buf`
    /// is only *recorded* — the caller submits it (any queue, any time) and must
    /// keep `self` alive until that submission completes (the copy reads it).
    ///
    /// `ty` must match this structure's type (the resource layer no longer
    /// stores it — the owning wrapper knows it). `compacted_size` must come from
    /// a prior compacted-size query (see [`vulkan_abstraction::CompactionQueryPool`]),
    /// and this structure must have been built with `ALLOW_COMPACTION`.
    pub fn record_compact_copy(
        &self,
        cmd_buf: vk::CommandBuffer,
        ty: vk::AccelerationStructureTypeKHR,
        compacted_size: vk::DeviceSize,
    ) -> SrResult<Self> {
        let (handle, buffer, device_address) = Self::create_backed(&self.core, ty, compacted_size)?;

        let copy_info = vk::CopyAccelerationStructureInfoKHR::default()
            .src(self.handle)
            .dst(handle)
            .mode(vk::CopyAccelerationStructureModeKHR::COMPACT);

        unsafe {
            self.core
                .acceleration_structure_device()
                .cmd_copy_acceleration_structure(cmd_buf, &copy_info);
        }

        Ok(Self {
            core: Rc::clone(&self.core),
            handle,
            buffer,
            device_address,
        })
    }

    /// Allocate a backing buffer of `size`, create an AS handle of type `ty` on
    /// it, and read back its device address. Shared by build and compaction.
    fn create_backed(
        core: &Rc<vulkan_abstraction::Core>,
        ty: vk::AccelerationStructureTypeKHR,
        size: vk::DeviceSize,
    ) -> SrResult<(
        vk::AccelerationStructureKHR,
        vulkan_abstraction::GpuOnlyBuffer,
        vk::DeviceAddress,
    )> {
        let buffer = vulkan_abstraction::GpuOnlyBuffer::new::<u8>(
            Rc::clone(core),
            size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            Self::buffer_name(ty),
        )?;

        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .ty(ty)
            .size(size)
            .buffer(buffer.inner())
            .offset(0)
            .create_flags(vk::AccelerationStructureCreateFlagsKHR::empty());

        let handle = unsafe {
            core.acceleration_structure_device()
                .create_acceleration_structure(&create_info, None)
        }?;

        // Valid as soon as the handle exists (on a bound buffer), independent of
        // whether the build has been submitted yet.
        let device_address = unsafe {
            core.acceleration_structure_device()
                .get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(handle),
                )
        };

        Ok((handle, buffer, device_address))
    }

    fn buffer_name(ty: vk::AccelerationStructureTypeKHR) -> &'static str {
        match ty {
            vk::AccelerationStructureTypeKHR::TOP_LEVEL => "TLAS buffer",
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL => "BLAS buffer",
            vk::AccelerationStructureTypeKHR::GENERIC => "generic acceleration structure buffer",
            _ => "(unknown AS type) acceleration structure buffer",
        }
    }

    /// `vkGetAccelerationStructureDeviceAddressKHR`, cached at creation.
    pub fn device_address(&self) -> vk::DeviceAddress {
        self.device_address
    }

    pub fn inner(&self) -> vk::AccelerationStructureKHR {
        self.handle
    }

    pub fn core(&self) -> &Rc<vulkan_abstraction::Core> {
        &self.core
    }
}

impl Drop for AccelerationStructure {
    fn drop(&mut self) {
        unsafe {
            self.core
                .acceleration_structure_device()
                .destroy_acceleration_structure(self.handle, None);
        }
    }
}
