//! ECS-driven scene instances.
//!
//! Spawn entities carrying [`SunrayInstance`] + a `Transform` (Bevy fills in
//! `GlobalTransform`) and the extract system rebuilds the renderer's per-frame
//! instance list from them every frame — adding, despawning, or moving such
//! entities Just Works, because nothing about instances is retained on the
//! renderer side (its contract is a caller-owned per-frame list).

use ash::vk;
use bevy_ecs::prelude::*;
use bevy_transform::components::GlobalTransform;

/// One ray-traced instance of a BLAS from the currently loaded scene.
///
/// `blas_index` indexes the scene's BLAS list — the order of the
/// `(key, transforms)` pairs `load_gltf` returned (one entry per unique mesh).
/// The entity's world transform places the instance.
///
/// While at least one `SunrayInstance` entity exists, the entity-driven list
/// **replaces** the glTF scene's baked instances for that frame; despawn them
/// all to fall back to the baked scene.
#[derive(Component, Clone, Copy, Debug)]
pub struct SunrayInstance {
    /// Index into the loaded scene's BLAS list.
    pub blas_index: usize,
}

/// Convert a Bevy world transform into the row-major 3x4 matrix Vulkan's
/// acceleration structures expect. glam matrices are column-major, so
/// `m[col][row]` lays out each row of the KHR matrix.
pub(crate) fn transform_matrix_khr(transform: &GlobalTransform) -> vk::TransformMatrixKHR {
    let m = transform.to_matrix().to_cols_array_2d();
    vk::TransformMatrixKHR {
        matrix: [
            m[0][0], m[1][0], m[2][0], m[3][0], //
            m[0][1], m[1][1], m[2][1], m[3][1], //
            m[0][2], m[1][2], m[2][2], m[3][2],
        ],
    }
}
