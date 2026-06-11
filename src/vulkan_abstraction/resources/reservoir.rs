/// Represents a single light candidate reservoir for Spatiotemporal Reservoir Resampling (ReSTIR).
/// 48 bytes, packed to match std430: vec3 light_pos followed by scalar w_sum packs into 16 bytes
/// without internal padding; same for light_normal/M.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Reservoir {
    /// The exact 3D world position on the light source that was sampled.
    pub light_pos: [f32; 3],
    /// The sum of all candidate weights evaluated so far.
    pub w_sum: f32,

    /// The 3D world normal of the light source at the sampled position.
    pub light_normal: [f32; 3],
    /// The number of light candidates that have been processed to get this winner.
    pub m: f32,

    /// The index of the winning light candidate in the emissive triangles array.
    pub light_idx: u32,
    /// The final unbiased probabilistic weight of this reservoir, used to scale the final shadow ray.
    pub w: f32,
    /// Octahedral-packed hit surface normal at the shading point (for temporal validation).
    pub hit_normal_packed: u32,
    /// Virtual ray distance to the shading point (for temporal validation).
    pub depth: f32,
}

/// Represents a single surface sample reservoir for indirect illumination (ReSTIR GI, Ouyang 2021).
/// 48 bytes, same std430 packing strategy as `Reservoir`: two vec3+float pairs fill the first
/// 32 bytes, and sample_normal is octahedral-packed into a u32 so the whole struct fits in 48 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct ReservoirGI {
    /// The world-space position of the indirect hit point (x2) that was sampled.
    pub sample_pos: [f32; 3],
    /// The sum of all candidate weights evaluated so far.
    pub w_sum: f32,

    /// The incoming radiance arriving at x1 along the direction (x2 -> x1) from the sampled path
    /// (emission at x2 plus any next-event estimation performed at x2).
    pub sample_radiance: [f32; 3],
    /// The number of candidate samples that have been processed to get this winner.
    pub m: f32,

    /// Octahedral-packed surface normal at the indirect hit point x2 (needed for the Jacobian
    /// when reusing this sample from a neighboring pixel's shading point).
    pub sample_normal_packed: u32,
    /// The final unbiased probabilistic weight of this reservoir, used to scale the indirect contribution.
    pub w: f32,
    /// Octahedral-packed surface normal at the shading point x1 (for temporal validation).
    pub hit_normal_packed: u32,
    /// Virtual ray distance to the shading point x1 (for temporal validation).
    pub depth: f32,
}
