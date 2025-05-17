use ash::{
    Device,
    vk::{
        AccelerationStructureDeviceAddressInfoKHR, AccelerationStructureInstanceKHR,
        AccelerationStructureKHR, Buffer, BufferCreateInfo, GeometryFlagsKHR,
        GeometryInstanceFlagsKHR, Packed24_8, PhysicalDeviceMemoryProperties, TaggedStructure,
        TransformMatrixKHR,
    },
};

use super::BLAS;
use crate::vulkan_abstraction::{self, instance_buffer::InstanceBuffer};

pub struct TLAS {
    tlas: AccelerationStructureKHR,
}

/*Reading other people code I understood that a commond way of handling the tlas is to inherit everyhting from the blas:
    - device address, ecc...
*/

impl TLAS {
    pub fn new(
        device: &Device,
        device_memory_props: &PhysicalDeviceMemoryProperties,
        cmd_pool: &vulkan_abstraction::CmdPool,
        blas: BLAS, //this should be an array
    ) -> Self {
        let transform_matrix = TransformMatrixKHR {
            matrix: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        };

        let accelertation_device_address_info =
            AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(blas.blas);

        let instance = AccelerationStructureInstanceKHR {
            transform: transform_matrix,
            instance_custom_index_and_mask: Packed24_8::new(0, 0),
            instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(
                0,
                GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE
                    .as_raw()
                    .try_into()
                    .unwrap(),
            ),
            acceleration_structure_reference: ash::vk::AccelerationStructureReferenceKHR {
                device_handle: unsafe {
                    blas.acceleration_structure_device
                        .get_acceleration_structure_device_address(
                            &accelertation_device_address_info,
                        )
                },
            },
        };

        let build_command_buffer = vulkan_abstraction::cmd_buffer::new(cmd_pool, device);

        //i should create a buffer for each instance
        let instance_buffer = InstanceBuffer::new::<u8>(device.clone(), 1, device_memory_props)?; //u8 as placeholder, also 1 is a placeholder size

        todo!()
    }
}
