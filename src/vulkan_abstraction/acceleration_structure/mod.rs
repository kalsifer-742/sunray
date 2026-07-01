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
#[derive(Debug)]
pub enum ASDesc {
    Blas(BlasDesc),
    Tlas(TlasBuildDesc)
}

#[derive(Debug,Copy, Clone, PartialEq, Eq, Hash)]
pub enum BuildType {
    RapidlyChanging,
    SometimesChanges,
    Static,
}

#[derive(Copy, Clone, Debug)]
pub enum  OpType {
    SlowBuild,
    FastBuild,
    Update
}
