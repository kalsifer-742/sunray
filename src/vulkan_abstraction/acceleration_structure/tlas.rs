use std::rc::Rc;

use ash::vk;

use super::BLAS;
use crate::{error::*, vulkan_abstraction};

// Resources:
// - https://github.com/adrien-ben/vulkan-examples-rs
// - https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/
// - https://github.com/SaschaWillems/Vulkan

// TODO: implement drop
pub struct TLAS {
    tlas: vulkan_abstraction::AccelerationStructure,
}

impl TLAS {
    pub fn new(core: Rc<vulkan_abstraction::Core>, blases: &[BLAS]) -> SrResult<Self> {
        let instances_buffer = Self::make_instances_buffer(Rc::clone(&core), blases)?;

        let geometry = Self::make_geometry(&instances_buffer);

        let build_range_info = Self::make_build_range_info(blases.len() as u32);

        let tlas = vulkan_abstraction::AccelerationStructure::new(
            core,
            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            &[build_range_info],
            &[geometry],
            true,
        )?;

        Ok(Self { tlas })
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
    /// Basically from what I can tell only the following operations are allowed in a TLAS update:
    /// - Change one or more transform matrices
    /// - switch one BLAS instance for another, possibly to switch LODs
    #[allow(unused)]
    pub fn update(&mut self, blases: &[BLAS]) -> SrResult<()> {
        let instances_buffer = Self::make_instances_buffer(Rc::clone(self.tlas.core()), blases)?;

        let geometry = Self::make_geometry(&instances_buffer);

        let build_range_info = Self::make_build_range_info(blases.len() as u32);

        self.tlas.update(&[build_range_info], &[geometry])?;

        Ok(())
    }

    #[allow(unused)]
    pub fn rebuild(&mut self, blases: &[BLAS]) -> SrResult<()> {
        let instances_buffer = Self::make_instances_buffer(Rc::clone(self.tlas.core()), blases)?;

        let geometry = Self::make_geometry(&instances_buffer);

        let build_range_info = Self::make_build_range_info(blases.len() as u32);

        self.tlas.rebuild(&[build_range_info], &[geometry])?;

        Ok(())
    }

    fn make_instances_buffer(
        core: Rc<vulkan_abstraction::Core>,
        blases: &[BLAS],
    ) -> SrResult<vulkan_abstraction::Buffer> {
        // this is the transformation for positioning individual BLASes
        // for now it's an Identity Matrix
        let transform_matrix = vulkan_abstraction::IDENTITY_MATRIX;

        let blas_instances: Vec<vk::AccelerationStructureInstanceKHR> = blases
            .iter()
            .map(|blas| {
                vk::AccelerationStructureInstanceKHR {
                    transform: transform_matrix,
                    instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xFF), // gl_InstanceCustomIndex = 0, mask = 0 (don't know what actually does, NV tutorial writes "Only be hit if rayMask & instance.mask != 0")
                    instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                        0, // hit_group_offset = 0, same hit group for the whole scene
                        vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8, // disable face culling for semplicity
                    ),
                    acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                        device_handle: unsafe {
                            core.acceleration_structure_device()
                                .get_acceleration_structure_device_address(
                                    &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                                        .acceleration_structure(blas.inner()),
                                )
                        },
                    },
                }
            })
            .collect();

        // buffer to hold the instances
        let instances_buffer = vulkan_abstraction::Buffer::new_from_data(
            core,
            &blas_instances,
            gpu_allocator::MemoryLocation::GpuOnly,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::TRANSFER_DST,
            "TLAS instances buffer"
        )?;

        Ok(instances_buffer)
    }

    fn make_geometry(
        instances_buffer: &vulkan_abstraction::Buffer,
    ) -> vk::AccelerationStructureGeometryKHR<'_> {
        vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
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
}
