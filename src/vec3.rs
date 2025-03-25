use std::ops::Deref;
use nalgebra as na;

/*
 * this type is useful since rust's padding rules are different from glsl's, and this is particularly
 * evident with vec3: between 2 instances glsl adds 4 bytes of padding, which rust simply does not do.
 * using [f32;3] can work to circumvent this, but I trust the standard glsl struct layout to know better
 * than me what layout performs best on the GPU.
 * The other alternatives would be:
 * - using other fields to add padding manually: ugly but explicit
 * - using Vector4 instead of Vector3: low effort but not explicit at all
 * unfortunately there is no way to force the alignment of single struct fields aside from wrapping
 * them into another struct like this and using #[repr(align(N))]
 */
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct Vec3 {
    v: na::Vector3<f32>
}
impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Vec3 { Self { v: na::Vector3::new(x, y, z) } }
}
impl Deref for Vec3 {
    type Target = na::Vector3<f32>;
    fn deref(&self) -> &Self::Target { &self.v }
}
