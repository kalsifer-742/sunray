pub mod accel;
pub mod blas;
pub mod compaction;
pub mod tlas;

pub use accel::*;
pub use blas::*;
pub use compaction::*;
pub use tlas::*;

/// Render-graph resource description for an imported acceleration structure.
/// Used only as the phantom `Desc` carried by a `Handle<AccelerationStructure>`
/// for typing — the graph imports the AS by `Arc`, so this description is never
/// consulted for imports (transient AS creation is unimplemented). Kept plain
/// (`Clone`) so a handle is cheap to clone.
#[derive(Debug, Clone)]
pub enum ASDesc {
    Blas(BlasDesc),
    Tlas(TlasBuildDesc),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum BuildType {
    RapidlyChanging,
    SometimesChanges,
    Static,
}

/// A single build operation to record for an acceleration structure this frame.
/// Chosen by [`AsState::next_op`] and folded back in by [`AsState::mark_built`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OpType {
    /// Full build tuned for trace performance (`PREFER_FAST_TRACE`) — the initial
    /// build and the "settle" rebuild after a structure goes quiet.
    SlowBuild,
    /// Full rebuild tuned for build speed (`PREFER_FAST_BUILD`) — used mid-churn
    /// once too many in-place updates have accumulated and quality has drifted.
    FastBuild,
    /// Cheap in-place refit (same geometry layout, new contents).
    Update,
}

/// Heuristic counters for an acceleration structure in the [`AsState::Changing`]
/// state: how many consecutive frames it has been stable, and how many in-place
/// updates have accumulated since the last full (re)build. Together they decide
/// when to stop updating and rebuild, and when to settle back to `Optimal`.
#[derive(Copy, Clone, Debug)]
pub struct Dynamic {
    frames_without_changes: u32,
    number_of_updates_since_last_rebuild: u32,
}

impl Dynamic {
    fn new() -> Self {
        Self {
            frames_without_changes: 0,
            number_of_updates_since_last_rebuild: 0,
        }
    }
}

/// Rebuild-vs-update heuristic state for one acceleration structure, shared by
/// [`Blas`] and [`Tlas`]. `Optimal` is a quiet, quality-built structure; the
/// first change moves it into `Changing`, where the embedded [`Dynamic`] counters
/// drive the choice between a cheap update and a full rebuild.
#[derive(Copy, Clone, Debug)]
pub enum AsState {
    Optimal,
    Changing(Dynamic),
}

impl AsState {
    /// After this many in-place updates without a full rebuild, force a
    /// (fast) rebuild so the structure's trace quality doesn't degrade.
    const MAX_UPDATES_BEFORE_REBUILD: u32 = 8;
    /// After this many consecutive unchanged frames while `Changing`, do one
    /// quality (slow) rebuild and settle back to `Optimal`.
    const FRAMES_TO_SETTLE: u32 = 16;

    /// The steady state a freshly built AS of `build_type` starts in: rapidly
    /// changing geometry begins its churn window immediately, everything else is
    /// treated as `Optimal` until it first changes.
    pub fn initial(build_type: BuildType) -> Self {
        match build_type {
            BuildType::RapidlyChanging => AsState::Changing(Dynamic::new()),
            BuildType::SometimesChanges | BuildType::Static => AsState::Optimal,
        }
    }

    /// Decide the operation to record this frame given whether the build inputs
    /// changed. `None` means "nothing to do" — either steady, or still coasting
    /// toward a settle. Advisory: a non-updatable structure maps a returned
    /// [`OpType::Update`] to a rebuild at the call site.
    pub fn next_op(&self, inputs_changed: bool) -> Option<OpType> {
        match self {
            AsState::Optimal => inputs_changed.then_some(OpType::Update),
            AsState::Changing(dynamic) => {
                if inputs_changed {
                    if dynamic.number_of_updates_since_last_rebuild >= Self::MAX_UPDATES_BEFORE_REBUILD {
                        Some(OpType::FastBuild)
                    } else {
                        Some(OpType::Update)
                    }
                } else if dynamic.frames_without_changes + 1 >= Self::FRAMES_TO_SETTLE {
                    Some(OpType::SlowBuild)
                } else {
                    None
                }
            }
        }
    }

    /// Fold the just-completed operation back into the heuristic. Called once per
    /// frame from the end-of-frame closure (after the recorded op has finished on
    /// the GPU); `None` is an idle frame with no op, which advances the
    /// frames-without-changes counter toward a settle.
    ///
    /// - [`OpType::Update`]     → one more update since the last rebuild, churn reset.
    /// - [`OpType::FastBuild`]  → counters reset, stays `Changing` (still churning).
    /// - [`OpType::SlowBuild`]  → counters reset, settles to `Optimal`.
    /// - `None` (idle frame)    → advances `frames_without_changes` while `Changing`.
    pub fn mark_built(&mut self, completed: Option<OpType>) {
        match completed {
            Some(OpType::Update) => match self {
                AsState::Changing(dynamic) => {
                    dynamic.number_of_updates_since_last_rebuild += 1;
                    dynamic.frames_without_changes = 0;
                }
                // An update out of a quiet structure opens a churn window.
                AsState::Optimal => {
                    *self = AsState::Changing(Dynamic {
                        frames_without_changes: 0,
                        number_of_updates_since_last_rebuild: 1,
                    });
                }
            },
            // Fast rebuild mid-churn: reset counters but stay dynamic.
            Some(OpType::FastBuild) => *self = AsState::Changing(Dynamic::new()),
            // Quality rebuild: the structure has settled.
            Some(OpType::SlowBuild) => *self = AsState::Optimal,
            // Idle frame: coast toward a settle.
            None => {
                if let AsState::Changing(dynamic) = self {
                    dynamic.frames_without_changes += 1;
                }
            }
        }
    }
}
