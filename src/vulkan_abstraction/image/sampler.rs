use std::cell::Cell;
use std::rc::Rc;

use ash::vk;

use crate::vulkan_abstraction::descriptor_heap::DescriptorSlot;
use crate::{error::SrResult, vulkan_abstraction};

/// Plain-data copy of the sampler parameters. Under the descriptor-heap model the sampler
/// is materialised by `vkWriteSamplerDescriptorsEXT` from a `SamplerCreateInfo` directly,
/// so there is no `vkCreateSampler` / `vkDestroySampler` round-trip and no `vk::Sampler`
/// handle to track — these params are the entire sampler.
#[derive(Copy, Clone)]
struct SamplerParams {
    min_filter: vk::Filter,
    mag_filter: vk::Filter,
    address_mode_u: vk::SamplerAddressMode,
    address_mode_v: vk::SamplerAddressMode,
    address_mode_w: vk::SamplerAddressMode,
    mipmap_mode: vk::SamplerMipmapMode,
    max_anisotropy: f32,
}

pub struct Sampler {
    core: Rc<vulkan_abstraction::Core>,
    params: SamplerParams,
    /// Lazily-allocated heap slot for this sampler.
    slot: Cell<Option<DescriptorSlot>>,
    /// TEMPORARY shim: legacy descriptor-set writes still want a `vk::Sampler` handle. Lazily
    /// created on first `inner()` call so heap-only callers don't pay for the round-trip.
    /// Remove once every pass binds via heap.
    legacy_handle: Cell<vk::Sampler>,
}

impl Sampler {
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        min_filter: vk::Filter,
        mag_filter: vk::Filter,
        address_mode_u: vk::SamplerAddressMode,
        address_mode_v: vk::SamplerAddressMode,
        address_mode_w: vk::SamplerAddressMode,
        mipmap_mode: vk::SamplerMipmapMode,
    ) -> SrResult<Self> {
        let params = SamplerParams {
            min_filter,
            mag_filter,
            address_mode_u,
            address_mode_v,
            address_mode_w,
            mipmap_mode,
            max_anisotropy: core.device().properties().limits.max_sampler_anisotropy,
        };

        Ok(Self {
            core,
            params,
            slot: Cell::new(None),
            legacy_handle: Cell::new(vk::Sampler::null()),
        })
    }

    /// TEMPORARY shim: returns a real `vk::Sampler` for legacy descriptor-set writers.
    /// Remove once every pass uses the heap.
    pub fn inner(&self) -> vk::Sampler {
        let cur = self.legacy_handle.get();
        if cur != vk::Sampler::null() {
            return cur;
        }
        let info = self.params.to_create_info();
        let handle = unsafe { self.core.device().inner().create_sampler(&info, None) }
            .expect("vkCreateSampler failed in legacy Sampler::inner shim");
        self.legacy_handle.set(handle);
        handle
    }

    /// Heap slot for this sampler in the sampler heap. Allocated and written on first call.
    pub fn slot(&self) -> u32 {
        if let Some(s) = self.slot.get() {
            return s.shader_index();
        }
        let mut heap = self.core.descriptor_heap_mut();
        let slot = heap.alloc_sampler_slot();
        heap.write_sampler(slot, &self.params.to_create_info())
            .expect("descriptor heap write_sampler failed");
        self.slot.set(Some(slot));
        slot.shader_index()
    }
}

impl SamplerParams {
    fn to_create_info(&self) -> vk::SamplerCreateInfo<'static> {
        vk::SamplerCreateInfo::default()
            .flags(vk::SamplerCreateFlags::empty())
            .min_filter(self.min_filter)
            .mag_filter(self.mag_filter)
            .address_mode_u(self.address_mode_u)
            .address_mode_v(self.address_mode_v)
            .address_mode_w(self.address_mode_w)
            .anisotropy_enable(true)
            .max_anisotropy(self.max_anisotropy)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(self.mipmap_mode)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(0.0)
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        if let Some(s) = self.slot.get() {
            self.core.descriptor_heap_mut().free(s);
        }
        let legacy = self.legacy_handle.get();
        if legacy != vk::Sampler::null() {
            unsafe { self.core.device().inner().destroy_sampler(legacy, None) };
        }
    }
}
