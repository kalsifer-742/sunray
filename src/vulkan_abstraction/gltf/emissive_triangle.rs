/// This struct is used for Next-Event-Estimation when tracing rays.
/// Instead of only a single random ray the renderer traces an additional ray called shadow ray directed to an emissive triangle.
/// This converges the rendering equation much faster especially at the current 1 spp.
///
/// Triangles are stored in local space (per-BLAS). The shader applies the entity transform at sample time.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EmissiveTriangle {
    pub v0: [f32; 4],       // x, y, z = Local Position 0 | w = Padding (area computed by shader after transform)
    pub v1: [f32; 4],       // x, y, z = Local Position 1 | w = Padding
    pub v2: [f32; 4],       // x, y, z = Local Position 2 | w = Padding
    pub emission: [f32; 4], // r, g, b = Emission Color * Strength | a = Padding
}

/// Entry in the dense emissive indirection buffer used for NEE sampling.
/// The shader picks a random entry, fetches the local-space triangle from the BLAS emissive buffer,
/// and applies the entity's transform to get world-space coordinates.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EmissiveIndirectionEntry {
    pub blas_tri_index: u32,
    pub entity_id: u32,
}
