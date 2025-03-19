use std::error::Error;
use ash::vk;
use crate::vkal;

#[derive(Clone, Copy)]
pub struct PhysicalDeviceInfo {
    pub physical_dev_idx: usize,

    pub best_queue_family_for_graphics: u32,
    pub number_of_queues: u32,
    pub format: vk::SurfaceFormatKHR,
    pub surface_capabilities: vk::SurfaceCapabilitiesKHR,
    pub presentation_mode: vk::PresentModeKHR,
}

impl PhysicalDeviceInfo {
    pub fn new(instance: &vkal::Instance, surface: &vkal::Surface) -> Result<Self, Box<dyn Error>> {
        let surface_instance = instance.surface_instance();
        let mut selected_physdev_idx = None;
        let mut selected_physdev_memsize = 0;
        let mut selected_physdev_qf = 0;
        let mut selected_physdev_qf_size = 0;

        let physdevs = unsafe { instance.enumerate_physical_devices() }?;
        for (pd_idx, physdev) in physdevs.into_iter().enumerate() {
            let qf_props = unsafe { instance.get_physical_device_queue_family_properties(physdev) };
            let mut biggest_qf_with_surface_and_gfx = None;
            let mut biggest_qf_with_surface_and_gfx_size = 0;
            for (qf_idx, qf_props) in qf_props.iter().enumerate() {
                let supports_surface = unsafe { surface_instance.get_physical_device_surface_support(physdev, qf_idx as u32, surface.inner()) }?;
                if supports_surface && qf_props.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    if biggest_qf_with_surface_and_gfx_size < qf_props.queue_count {
                        biggest_qf_with_surface_and_gfx_size = qf_props.queue_count;
                        biggest_qf_with_surface_and_gfx = Some(qf_idx);
                    }
                }
            }
            let selected_qf =
                if let Some(qf) = biggest_qf_with_surface_and_gfx { qf } else { continue; }; // unsuitable device
            let selected_qf_size = biggest_qf_with_surface_and_gfx_size;

            let memory_props = unsafe { instance.get_physical_device_memory_properties(physdev) };

            let max_device_local_heap_size =
                memory_props.memory_heaps_as_slice().iter()
                    .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
                    .map(|h| h.size)
                    .max().unwrap_or(0);

            if selected_physdev_memsize < max_device_local_heap_size {
                selected_physdev_memsize = max_device_local_heap_size;
                selected_physdev_idx = Some(pd_idx);
                selected_physdev_qf = selected_qf;
                selected_physdev_qf_size = selected_qf_size;
            }
        }

        let selected_physdev_idx = selected_physdev_idx.ok_or("No suitable physical device detected!")?;

        // print_physical_devices_info(instance, surface_instance, surface)?;

        println!("selected physical device {}, queue family {}", selected_physdev_idx, selected_physdev_qf);

        let selected_physdev = Self::get_physical_device_from_index(selected_physdev_idx, instance)?;

        let surface_formats = unsafe { surface_instance.get_physical_device_surface_formats(selected_physdev, surface.inner()) }?;
        let mut surface_format = &surface_formats[0];
        for sf in surface_formats.iter() {
            if sf.format == vk::Format::B8G8R8A8_SRGB && sf.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR {
                surface_format = sf;
            }
        }

        let surface_capabilities = unsafe { surface_instance.get_physical_device_surface_capabilities(selected_physdev, surface.inner()) }?;

        let present_modes = unsafe { surface_instance.get_physical_device_surface_present_modes(selected_physdev, surface.inner()) }?;
        let presentation_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else if present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
            vk::PresentModeKHR::IMMEDIATE
        } else {
            vk::PresentModeKHR::FIFO // fifo is guaranteed to exist
        };

        Ok(Self {
            physical_dev_idx: selected_physdev_idx, best_queue_family_for_graphics: selected_physdev_qf as u32,
            number_of_queues: selected_physdev_qf_size, format: surface_format.clone(), surface_capabilities, presentation_mode,
        })
    }

    pub fn get_physical_device(&self, instance: &ash::Instance) -> Result<vk::PhysicalDevice, Box<dyn Error>> {
        Self::get_physical_device_from_index(self.physical_dev_idx, instance)
    }

    fn get_physical_device_from_index(index: usize, instance: &ash::Instance) -> Result<vk::PhysicalDevice, Box<dyn Error>> {
        let physdevs = unsafe { instance.enumerate_physical_devices() }?;
        Ok(physdevs.get(index).ok_or("No such physical device")?.clone())
    }
}



// prints information about physical devices, useful as a reference on how to access information about them
#[allow(dead_code)]
fn print_physical_devices_info(instance: &ash::Instance, surface_instance: &ash::khr::surface::Instance, surface: vk::SurfaceKHR) -> Result<(), Box<dyn Error>> {
    let mut biggest_memory = None;

    let physdevs = unsafe { instance.enumerate_physical_devices() }?;
    for (pd_idx, physdev) in physdevs.into_iter().enumerate() {
        let props = unsafe { instance.get_physical_device_properties(physdev) };

        println!("device {pd_idx}, name: {:?}", props.device_name_as_c_str()?);
        println!("    device id: {}", props.device_id);
        println!("    api version: {}.{}.{}", vk::api_version_major(props.api_version), vk::api_version_minor(props.api_version), vk::api_version_patch(props.api_version));

        let q_families_props = unsafe { instance.get_physical_device_queue_family_properties(physdev) };
        println!("    Queue families count: {}", q_families_props.len());


        for (qf_idx, q_family_props) in q_families_props.iter().enumerate() {
            let supports_surface = unsafe { surface_instance.get_physical_device_surface_support(physdev, qf_idx as u32, surface) }?;

            println!("        Family {qf_idx}: {} queues, supports surface? {supports_surface}, flags: {:?}", q_family_props.queue_count, q_family_props.queue_flags);
        }

        let surface_formats = unsafe { surface_instance.get_physical_device_surface_formats(physdev, surface) }?;
        println!("    Supported formats:");
        for (i, surface_format) in surface_formats.iter().enumerate() {
            println!("        {i}: {surface_format:?}");
        }

        let surface_caps = unsafe { surface_instance.get_physical_device_surface_capabilities(physdev, surface) }?;
        println!("    Current extent: {:?}", surface_caps.current_extent);
        println!("    Supports: {:?}", surface_caps.supported_usage_flags);

        let present_modes = unsafe { surface_instance.get_physical_device_surface_present_modes(physdev, surface) }?;
        println!("    Presentation modes: {:?}", present_modes);

        let memory_props = unsafe { instance.get_physical_device_memory_properties(physdev) };
        println!("    Memory types: {}", memory_props.memory_type_count);
        for (i, mem_type) in memory_props.memory_types_as_slice().iter().enumerate() {
            println!("        {i}: {mem_type:?}");
        }

        println!("    Memory heaps: {}", memory_props.memory_heap_count);
        for (i, mem_heap) in memory_props.memory_heaps_as_slice().iter().enumerate() {
            println!("        {i}: {mem_heap:?} {mem_heap:x?}");
        }

        let mut max_device_local_heap_size = 0;
        for mem_heap in memory_props.memory_heaps_as_slice().iter() {
            if mem_heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                max_device_local_heap_size = max_device_local_heap_size.max(mem_heap.size);
            }
        }
        if let Some(biggest) = biggest_memory {
            if biggest < max_device_local_heap_size {
                biggest_memory = Some(max_device_local_heap_size);
            }
        } else {
            biggest_memory = Some(max_device_local_heap_size);
        }
        println!("   Max device local heap size: {} = 0x{0:x} Bytes", max_device_local_heap_size);
    }
    Ok(())
}