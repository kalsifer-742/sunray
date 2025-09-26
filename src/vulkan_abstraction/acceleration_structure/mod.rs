pub mod blas;
pub mod tlas;

use std::{rc::Rc};

pub use blas::*;
pub use tlas::*;

use ash::vk;
use crate::{ error::*, vulkan_abstraction };

#[rustfmt::skip]
pub const IDENTITY_MATRIX : vk::TransformMatrixKHR = vk::TransformMatrixKHR {
    matrix: [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0
    ],
};

pub struct AccelerationStructure {
    core: Rc<vulkan_abstraction::Core>,
    handle: vk::AccelerationStructureKHR,
    #[allow(dead_code)]
    buffer: vulkan_abstraction::Buffer,
    allow_update: bool,
    level: vk::AccelerationStructureTypeKHR,
    number_of_geometries: usize,
}
impl AccelerationStructure {
    pub fn new(core: Rc<vulkan_abstraction::Core>, level: vk::AccelerationStructureTypeKHR, build_range_infos: &[vk::AccelerationStructureBuildRangeInfoKHR], geometries: &[vk::AccelerationStructureGeometryKHR], allow_update: bool) -> SrResult<Self> {
        assert_eq!(geometries.len(), build_range_infos.len());

        let allow_update_flag = if allow_update { vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE } else { vk::BuildAccelerationStructureFlagsKHR::empty() };

        // parameters on how to build the acceleration structure.
        // this temporary version is used to calculate how much memory to allocate for it,
        // and the final version which is used to really build the acceleration structure will be based on it,
        // with some additional args based on the allocations that were performed.
        let incomplete_build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .geometries(&geometries)
        // PREFER_FAST_TRACE -> prioritize trace performance over build time
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE | allow_update_flag)
        // BUILD as opposed to UPDATE
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .ty(level);

        // based on incomplete_build_info get the sizes of the acceleration structure buffer to allocate and
        // of the scratch buffer that will be used for building the acceleration structure (and can then be discarded)
        let acceleration_structure_size_info = unsafe {
            let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
            let primitive_counts = build_range_infos.iter().map(|i| i.primitive_count).collect::<Vec<_>>();

            core.acceleration_structure_device().get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &incomplete_build_geometry_info,
                &primitive_counts,
                &mut size_info,
            );

            size_info
        };

        // the vulkan buffer on which the acceleration structure will live
        let buffer = vulkan_abstraction::Buffer::new::<u8>(
            Rc::clone(&core),
            acceleration_structure_size_info.acceleration_structure_size as usize,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryAllocateFlags::DEVICE_ADDRESS,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;

        // information as to how to instantiate (but not "build") the acceleration structure in acceleration_structure_buffer.
        let acceleration_structure_create_info = vk::AccelerationStructureCreateInfoKHR::default()
        .ty(incomplete_build_geometry_info.ty)
        .size(acceleration_structure_size_info.acceleration_structure_size)
        .buffer(buffer.inner())
        .offset(0)
        .create_flags(vk::AccelerationStructureCreateFlagsKHR::empty());

        // the actual acceleration structure object which lives on the acceleration_structure_buffer, but has not been "built" yet
        let handle = unsafe {
            core.acceleration_structure_device().create_acceleration_structure(&acceleration_structure_create_info, None)
        }?;

        // the scratch buffer that will be used for building the acceleration structure (and can be dropped afterwards)
        let scratch_buffer = vulkan_abstraction::Buffer::new::<u8>(
            Rc::clone(&core),
            acceleration_structure_size_info.build_scratch_size as usize,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryAllocateFlags::DEVICE_ADDRESS,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;

        // info for building the acceleration structure
        let build_geometry_info = incomplete_build_geometry_info
        .dst_acceleration_structure(handle)
        .scratch_data(vk::DeviceOrHostAddressKHR {
            device_address: scratch_buffer.get_device_address(),
        });

        // one-shot command buffer which we will:
        // - fill with the commands to build the acceleration structure
        // - pass to the queue to be executed (thus building the acceleration structure)
        // - free
        let build_command_buffer = vulkan_abstraction::cmd_buffer::new_command_buffer(core.cmd_pool(), core.device().inner())?;

        //record build_command_buffer with the commands to build the acceleration structure
        unsafe {
            core.device().inner().begin_command_buffer(
                build_command_buffer,
                &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            core.acceleration_structure_device().cmd_build_acceleration_structures(
                build_command_buffer,
                &[build_geometry_info],
                &[build_range_infos],
            );

            core.device().inner().end_command_buffer(build_command_buffer)?
        }


        // build_command_buffer must not be in a pending state when
        // free_command_buffers is called on it
        // NOTE: this is actually quite bad for performance if there are many acceleration structure builds/updates being done one after the other
        core.queue().submit_sync(build_command_buffer)?;

        unsafe {
            core.device().inner().free_command_buffers(core.cmd_pool().inner(), &[build_command_buffer]);
        }


        Ok(Self {
            core,
            handle,
            buffer,
            allow_update,
            level,
            number_of_geometries: geometries.len(),
        })
    }

    /// "the application must not use an update operation to do any of the following:
    /// - Change primitives or instances from active to inactive, or vice versa
    /// - Change the index or vertex formats of triangle geometry.
    /// - Change triangle geometry transform pointers from null to non-null or vice versa.
    /// - Change the number of geometries or instances in the structure.
    /// - Change the geometry flags for any geometry in the structure.
    /// - Change the number of vertices or primitives for any geometry in the structure."
    /// (from https://docs.vulkan.org/spec/latest/chapters/accelstructures.html#acceleration-structure-update)
    ///
    /// Basically from what I can tell only the following operations are allowed in an update:
    /// For BLAS:
    /// - Change one or more transform matrices
    /// - Deform one or more vertices of one or more meshes
    /// - Possibly switch LODs, though this seems like a bad idea
    /// For TLAS:
    /// - Change one or more transform matrices
    /// - switch one BLAS instance for another, possibly to switch LODs
    ///
    /// It seems to be better to update matrices/LODs in the TLAS and keep BLASes mostly static, unless we need
    /// fancy behaviour like dynamic deformation of meshes, and even then when the deformation is significant or
    /// unpredictable it is better to rebuild the BLAS for optimal rt performance
    pub fn update(&mut self, build_range_infos: &[vk::AccelerationStructureBuildRangeInfoKHR], geometries: &[vk::AccelerationStructureGeometryKHR]) -> SrResult<()>{
        assert!(self.allow_update);
        assert_eq!(self.number_of_geometries, geometries.len());
        assert_eq!(self.number_of_geometries, build_range_infos.len());

        // parameters on how to build the acceleration structure.
        // this temporary version is used to calculate how much memory to allocate for it,
        // and the final version which is used to really build the acceleration structure will be based on it,
        // with some additional args based on the allocations that were performed.
        let incomplete_build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .geometries(&geometries)
            // PREFER_FAST_TRACE -> prioritize trace performance over build time
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE)
            // UPDATE as opposed to BUILD
            .mode(vk::BuildAccelerationStructureModeKHR::UPDATE)
            .ty(self.level);

        // based on incomplete_build_info get the sizes of the acceleration structure buffer to allocate and
        // of the scratch buffer that will be used for building the acceleration structure (and can then be discarded)
        let size_info = unsafe {
            let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
            let primitive_counts = build_range_infos.iter().map(|i| i.primitive_count).collect::<Vec<_>>();

            self.core.acceleration_structure_device().get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &incomplete_build_geometry_info,
                &primitive_counts,
                &mut size_info,
            );

            size_info
        };


        // the scratch buffer that will be used for building the acceleration structure (and can be dropped afterwards)
        let scratch_buffer = vulkan_abstraction::Buffer::new::<u8>(
            Rc::clone(&self.core),
            size_info.build_scratch_size as usize,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryAllocateFlags::DEVICE_ADDRESS,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;

        // info for building the acceleration structure
        let build_geometry_info = incomplete_build_geometry_info
        .src_acceleration_structure(self.handle)
        .dst_acceleration_structure(self.handle)
        .scratch_data(vk::DeviceOrHostAddressKHR {
            device_address: scratch_buffer.get_device_address(),
        });

        // one-shot command buffer which we will:
        // - fill with the commands to build the acceleration structure
        // - pass to the queue to be executed (thus building the acceleration structure)
        // - free
        let build_command_buffer = vulkan_abstraction::cmd_buffer::new_command_buffer(self.core.cmd_pool(), self.core.device().inner())?;

        //record build_command_buffer with the commands to build the acceleration structure
        unsafe {
            self.core.device().inner().begin_command_buffer(
                build_command_buffer,
                &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            self.core.acceleration_structure_device().cmd_build_acceleration_structures(
                build_command_buffer,
                &[build_geometry_info],
                &[build_range_infos],
            );

            self.core.device().inner().end_command_buffer(build_command_buffer)?
        }

        // build_command_buffer must not be in a pending state when
        // free_command_buffers is called on it
        // NOTE: this is actually quite bad for performance if there are many acceleration structure builds/updates being done one after the other
        self.core.queue().submit_sync(build_command_buffer)?;

        unsafe {
            self.core.device().inner().free_command_buffers(self.core.cmd_pool().inner(), &[build_command_buffer]);
        }

        log::debug!("{:?} acceleration structure updated", self.level);

        Ok(())
    }

    pub fn rebuild(&mut self, build_range_infos: &[vk::AccelerationStructureBuildRangeInfoKHR], geometries: &[vk::AccelerationStructureGeometryKHR]) -> SrResult<()> {
        *self = Self::new(Rc::clone(&self.core), self.level, build_range_infos, geometries, self.allow_update)?;

        log::debug!("{:?} acceleration structure rebuilt", self.level);
        Ok(())
    }

    pub fn inner(&self) -> vk::AccelerationStructureKHR { self.handle }
    pub fn core(&self) -> &Rc<vulkan_abstraction::Core> { &self.core }
}
impl Drop for AccelerationStructure {
    fn drop(&mut self) {
        unsafe { self.core.acceleration_structure_device().destroy_acceleration_structure(self.handle, None); }
    }
}
