use std::rc::Rc;

use ash::vk;

use crate::{error::SrResult, vulkan_abstraction};

pub struct Sampler {
    core: Rc<vulkan_abstraction::Core>,
    sampler: vk::Sampler,
}

impl Sampler {
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        min_filter: vk::Filter,
        mag_filter: vk::Filter,
        address_mode_u: vk::SamplerAddressMode,
        address_mode_v: vk::SamplerAddressMode,
        address_mode_w: vk::SamplerAddressMode,
    ) -> SrResult<Self> {
        let create_info = vk::SamplerCreateInfo::default()
            .flags(vk::SamplerCreateFlags::empty())
            // linear filtering both for magnification and minification
            .min_filter(min_filter)
            .mag_filter(mag_filter)
            // repeat (tile) the texture on all axes
            .address_mode_u(address_mode_u)
            .address_mode_v(address_mode_v)
            .address_mode_w(address_mode_w)
            // use supported anisotropy
            // TODO: does this make sense for raytracing?
            .anisotropy_enable(true)
            .max_anisotropy(core.device().properties().limits.max_sampler_anisotropy)
            // use normalized ([0,1] range) coordinates
            .unnormalized_coordinates(false)
            // no need for a comparison function ("mainly used for percentage-closer filtering on shadow maps")
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            // mipmapping
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(0.0);

        let sampler = unsafe { core.device().inner().create_sampler(&create_info, None) }?;

        Ok(Self { core, sampler })
    }

    pub fn inner(&self) -> vk::Sampler {
        self.sampler
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        let device = self.core.device().inner();

        unsafe {
            device.destroy_sampler(self.sampler, None);
        }
    }
}
