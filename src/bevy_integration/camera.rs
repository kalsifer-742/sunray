//! ECS-idiomatic camera: a [`SunrayCamera`] component on an entity whose
//! `Transform` positions/orients the view. The plugin extracts the transform
//! into [`super::state::ExtractedCamera`] each frame and feeds it to
//! `Renderer::set_camera` in the render world.

use bevy_ecs::prelude::*;
use bevy_transform::components::Transform;

/// Put this on an entity together with a `Transform` to make it the view
/// camera. The entity's translation is the eye; it looks down its local `-Z`
/// (Bevy's forward), matching `Camera3d` conventions.
///
/// Only the first matching entity is used (single-view renderer).
#[derive(Component, Clone, Copy, Debug)]
pub struct SunrayCamera {
    /// Vertical field of view, in degrees.
    pub fov_y_degrees: f32,
}

impl Default for SunrayCamera {
    fn default() -> Self {
        Self { fov_y_degrees: 45.0 }
    }
}



/// Resolve a `Transform` + [`SunrayCamera`] into eye/target/fov, ready to store
/// in the (Send) extracted-camera resource.
pub(crate) fn eye_target_fov(transform: &Transform, cam: &SunrayCamera) -> ([f32; 3], [f32; 3], f32) {
    let eye = transform.translation;
    // `Transform::forward()` is the local -Z axis in world space.
    let target = eye + *transform.forward();
    ([eye.x, eye.y, eye.z], [target.x, target.y, target.z], cam.fov_y_degrees)
}
