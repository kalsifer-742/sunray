pub mod accel;
pub mod blas;
pub mod compaction;
pub mod tlas;

pub use accel::*;
pub use blas::*;
pub use compaction::*;
pub use tlas::*;

/// Render-graph resource description for an imported acceleration structure.
/// (The graph's AS path is forward-looking scaffolding — see
/// `render_graph::transient_resources`.)
#[derive(Clone, Debug)]
pub enum ASDesc {
    Blas(BlasDesc),
    Tlas(TlasBuildDesc)
}
