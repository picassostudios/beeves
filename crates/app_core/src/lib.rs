//! Headless core for the Gaussian-Bezier design tool.
//!
//! The canonical, editable source model is a [`GaussianBezierStroke`]: an editable
//! Bezier skeleton, a brush model, a cloud of curve-attached Gaussian splats, and a
//! residual deformation layer. GPU buffers (built in a later phase) are disposable
//! render caches derived from this model — never the source of truth.
//!
//! Two synchronization directions tie the views together:
//! * **Forward** ([`stroke::update_splat_world_cache`]): skeleton edit -> re-evaluate
//!   every splat's world center/covariance from its curve-local `(t, u, v)` coords.
//! * **Reverse** ([`solver::apply_splat_edits_bidirectional`]): direct splat edits are
//!   classified for *coherence*; coherent edits update the skeleton, incoherent edits
//!   are absorbed as residual deformation.

pub mod bezier;
pub mod blend;
pub mod brush;
pub mod commands;
pub mod document;
pub mod fitting;
pub mod gaussian_blend;
pub mod ids;
pub mod math;
pub mod selection;
pub mod serialization;
pub mod solver;
pub mod splat;
pub mod stroke;

// Convenient flat re-exports for downstream crates (wasm bridge, tests).
pub use bezier::{
    ArcLengthTable, ArcSample, BezierSkeleton, ControlPointRef, CubicBezier, Frame, HandleKind,
};
pub use blend::{blend_splats, smudge_splats, BlendCarry};
pub use brush::{BrushModel, CurveProfile};
pub use commands::{Command, History};
pub use document::{BlendMode, CanvasSettings, ColorSpace, Document, Layer};
pub use fitting::{
    fit_polyline_adaptive, fit_polyline_to_skeleton, simplify_rdp, AdaptiveFitParams,
    IncrementalAdaptiveFit,
};
pub use ids::{LayerId, SplatId, StrokeId};
pub use math::{covariance_from_sigmas, solve_spd, Rng};
pub use selection::{hit_test_splat, SelectionState, SpatialGrid, SplatHit};
pub use solver::{
    apply_splat_edits_bidirectional, CenterlineTarget, CurveFitOptions, EditCoherence, FitOutcome,
    SplatEdit,
};
pub use splat::{GaussianSplat, GpuSplat, SplatRole};
pub use stroke::{GaussianBezierStroke, StrokeDirtyFlags, SyncPolicy};

pub use glam::Vec2;
