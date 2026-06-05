//! Reverse synchronization: splat edits -> Bezier skeleton (or residuals).
//!
//! When the user sculpts splats directly, we must decide what they *meant*:
//! * a coherent, structural move of the centerline -> update the skeleton;
//! * a scattered or edge-only move -> absorb as residual deformation.
//!
//! The decision is a confidence score blended from five metrics (section 9.1 of the
//! spec). High confidence updates the skeleton; medium blends a partial update with
//! residuals; low keeps the skeleton untouched and stores residuals only.

use std::collections::BTreeMap;

use crate::bezier::{bernstein, BezierSkeleton};
use crate::ids::{SplatId, StrokeId};
use crate::math::{smoothstep, solve_spd, Vec2};
use crate::stroke::{update_splat_world_cache, GaussianBezierStroke};

/// A single observed splat displacement.
#[derive(Clone, Copy, Debug)]
pub struct SplatEdit {
    pub stroke: StrokeId,
    pub splat_id: SplatId,
    pub old_center: Vec2,
    pub new_center: Vec2,
}

impl SplatEdit {
    pub fn delta(&self) -> Vec2 {
        self.new_center - self.old_center
    }
}

/// A target sample for the centerline fit: "at arc coord `t`, the centerline should
/// pass near `position`," weighted by how structurally meaningful the splat is.
#[derive(Clone, Copy, Debug)]
pub struct CenterlineTarget {
    pub t: f32,
    pub position: Vec2,
    pub weight: f32,
}

/// The five coherence metrics plus the derived confidence.
#[derive(Clone, Copy, Debug, Default)]
pub struct EditCoherence {
    pub same_stroke_ratio: f32,
    pub t_interval_coverage: f32,
    pub displacement_smoothness: f32,
    pub core_weight_ratio: f32,
    pub fit_error: f32,
    pub fit_quality: f32,
    pub confidence: f32,
}

/// Options for the least-squares curve fit.
#[derive(Clone, Copy, Debug)]
pub struct CurveFitOptions {
    /// Tikhonov weight pulling control points toward their previous values, scaled by
    /// the total target weight (so its effect is roughly scale-invariant).
    pub regularization: f32,
    /// Keep segment endpoints (anchors) fixed and fit only the interior handles.
    pub preserve_endpoints: bool,
    /// Safety clamp on how far any control point may move in one fit.
    pub max_handle_movement: f32,
}

impl Default for CurveFitOptions {
    fn default() -> Self {
        Self {
            regularization: 0.02,
            preserve_endpoints: true,
            max_handle_movement: 5000.0,
        }
    }
}

/// What the bidirectional dispatcher did with a stroke's edits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitOutcome {
    /// Skeleton updated structurally.
    Structural,
    /// Partial skeleton update plus residuals.
    Partial,
    /// Skeleton untouched; edits stored as residuals.
    Residual,
    /// Reverse sync disabled for this stroke.
    SyncDisabled,
}

/// Result of classifying one stroke's edits, including the trial-fitted skeleton.
pub struct CoherenceResult {
    pub coherence: EditCoherence,
    pub trial_skeleton: BezierSkeleton,
    pub preserve_endpoints: bool,
}

// --- Weights of the confidence blend (section 9.1) ---
const W_SAME_STROKE: f32 = 0.25;
const W_COVERAGE: f32 = 0.20;
const W_SMOOTHNESS: f32 = 0.20;
const W_CORE: f32 = 0.20;
const W_FIT: f32 = 0.15;

/// Weight a splat contributes to the centerline fit. Floor keeps even edge-only edits
/// from producing a singular system.
fn target_weight(role_structural: f32) -> f32 {
    0.05 + role_structural
}

/// Estimate the centerline point each edited splat implies, by removing the splat's
/// known local offset (`u*N + v*T`), residuals, against the *old* frame.
pub fn estimate_centerline_targets(
    stroke: &GaussianBezierStroke,
    edits: &[SplatEdit],
) -> Vec<CenterlineTarget> {
    let mut targets = Vec::with_capacity(edits.len());
    for edit in edits {
        let Some(splat) = stroke.find_splat(edit.splat_id) else {
            continue;
        };
        let frame = stroke.skeleton.frame_at_arc_t(splat.t);
        let local = frame.tangent * (splat.v + splat.residual_local.x)
            + frame.normal * (splat.u + splat.residual_local.y)
            + splat.residual_world;
        targets.push(CenterlineTarget {
            t: splat.t,
            position: edit.new_center - local,
            weight: target_weight(splat.role.structural_weight()),
        });
    }
    targets
}

/// Fit (a subset of) the skeleton's control points to the centerline targets in a
/// weighted, regularized least-squares sense. Returns a new skeleton with rebuilt
/// arc-length table. Endpoints are pinned per-segment when `preserve_endpoints` is set,
/// preserving C0 continuity at joins.
pub fn fit_bezier_to_targets(
    skeleton: &BezierSkeleton,
    targets: &[CenterlineTarget],
    options: &CurveFitOptions,
) -> BezierSkeleton {
    // Bucket targets by the segment they fall on (via the old parameterization).
    let mut by_segment: BTreeMap<usize, Vec<(f32, Vec2, f32)>> = BTreeMap::new();
    for tgt in targets {
        let (seg, s) = skeleton.curve_param_at_arc_t(tgt.t);
        by_segment
            .entry(seg)
            .or_default()
            .push((s, tgt.position, tgt.weight));
    }

    let mut segments = skeleton.segments.clone();
    for (seg_idx, samples) in by_segment {
        if seg_idx >= segments.len() {
            continue;
        }
        if let Some(fitted) = fit_single_segment(&segments[seg_idx], &samples, options) {
            segments[seg_idx] = fitted;
        }
    }

    BezierSkeleton::from_segments(segments, skeleton.closed)
}

/// Fit one cubic segment's control points to `(s, position, weight)` samples.
fn fit_single_segment(
    seg: &crate::bezier::CubicBezier,
    samples: &[(f32, Vec2, f32)],
    options: &CurveFitOptions,
) -> Option<crate::bezier::CubicBezier> {
    if samples.is_empty() {
        return None;
    }
    let unknowns: Vec<usize> = if options.preserve_endpoints {
        vec![1, 2]
    } else {
        vec![0, 1, 2, 3]
    };
    let fixed: Vec<usize> = (0..4).filter(|i| !unknowns.contains(i)).collect();
    let n = unknowns.len();

    let mut m = vec![vec![0.0f64; n]; n];
    let mut rx = vec![0.0f64; n];
    let mut ry = vec![0.0f64; n];

    for &(s, pos, w) in samples {
        let b = bernstein(s);
        let w = w as f64;
        // Move fixed control-point contributions to the RHS.
        let mut resid_x = pos.x as f64;
        let mut resid_y = pos.y as f64;
        for &fi in &fixed {
            resid_x -= b[fi] as f64 * seg.control(fi).x as f64;
            resid_y -= b[fi] as f64 * seg.control(fi).y as f64;
        }
        for a in 0..n {
            let ba = b[unknowns[a]] as f64;
            for c in 0..n {
                m[a][c] += w * ba * b[unknowns[c]] as f64;
            }
            rx[a] += w * ba * resid_x;
            ry[a] += w * ba * resid_y;
        }
    }

    // Relative Tikhonov regularization: pull each control point toward its old value by
    // a fixed *relative* fraction of its own data weight, so sparsely-constrained points
    // (e.g. endpoints, touched by few targets) follow the data just as faithfully as the
    // densely-constrained interior handles. A small absolute floor keeps any fully
    // unconstrained parameter well-posed (it then simply stays at its old value).
    const FLOOR: f64 = 1e-3;
    for a in 0..n {
        let lambda_a = (options.regularization as f64 * m[a][a]).max(FLOOR);
        m[a][a] += lambda_a;
        rx[a] += lambda_a * seg.control(unknowns[a]).x as f64;
        ry[a] += lambda_a * seg.control(unknowns[a]).y as f64;
    }

    let sol_x = solve_spd(&m, &rx)?;
    let sol_y = solve_spd(&m, &ry)?;

    let mut out = *seg;
    for (a, &idx) in unknowns.iter().enumerate() {
        let mut p = Vec2::new(sol_x[a] as f32, sol_y[a] as f32);
        // Clamp pathological movement.
        let old = seg.control(idx);
        let mv = p - old;
        if mv.length() > options.max_handle_movement {
            p = old + mv.normalize() * options.max_handle_movement;
        }
        out.set_control(idx, p);
    }
    Some(out)
}

/// Classify a stroke's edits: compute the five metrics, a trial skeleton fit, and the
/// blended confidence.
pub fn measure_edit_coherence(
    stroke: &GaussianBezierStroke,
    edits: &[SplatEdit],
    targets: &[CenterlineTarget],
    total_edits: usize,
) -> CoherenceResult {
    let m = edits.len().max(1);

    // (1) Fraction of all edited splats that belong to this stroke.
    let same_stroke_ratio = (edits.len() as f32 / total_edits.max(1) as f32).clamp(0.0, 1.0);

    // Pull per-edit (t, displacement, structural-weight) from the stroke.
    let mut samples: Vec<(f32, Vec2, f32)> = Vec::with_capacity(edits.len());
    for edit in edits {
        if let Some(splat) = stroke.find_splat(edit.splat_id) {
            samples.push((splat.t, edit.delta(), splat.role.structural_weight()));
        }
    }
    let t_min = samples.iter().map(|s| s.0).fold(f32::INFINITY, f32::min);
    let t_max = samples.iter().map(|s| s.0).fold(f32::NEG_INFINITY, f32::max);

    // (2) Edit *density* within the touched t-span: of all splats whose t lies in the
    //     span, what fraction were edited. A contiguous brush-drag selects ~everything
    //     in the band (density ~1); scattered/edge-only edits skip most (density low).
    let candidates = stroke
        .splats
        .iter()
        .filter(|s| s.t >= t_min - 1e-4 && s.t <= t_max + 1e-4)
        .count()
        .max(1);
    let t_interval_coverage = (m as f32 / candidates as f32).clamp(0.0, 1.0);

    // (3) Smoothness of displacement as a function of t.
    let displacement_smoothness = displacement_smoothness(&mut samples.clone());

    // (4) How much of the edit weight is on structural (core) splats.
    let core_weight_ratio = if samples.is_empty() {
        0.0
    } else {
        samples.iter().map(|s| s.2).sum::<f32>() / samples.len() as f32
    };

    // Trial fit. Free the endpoints only when the edit spans essentially the whole
    // curve (a whole-stroke translate); otherwise pin anchors and bend the handles.
    let covers_whole = t_min < 0.1 && t_max > 0.9;
    let preserve_endpoints = !covers_whole;
    let options = CurveFitOptions {
        preserve_endpoints,
        ..Default::default()
    };
    let trial_skeleton = fit_bezier_to_targets(&stroke.skeleton, targets, &options);

    // (5) Fit quality: RMS distance from the fitted centerline (evaluated the way
    //     forward sync will) to each target.
    let mut sq = 0.0f32;
    let mut count = 0.0f32;
    for tgt in targets {
        let p = trial_skeleton.frame_at_arc_t(tgt.t).position;
        sq += (p - tgt.position).length_squared();
        count += 1.0;
    }
    let rms = if count > 0.0 { (sq / count).sqrt() } else { 0.0 };
    let scale = (stroke.sync.max_structural_error * 2.0).max(1.0);
    let fit_quality = (-rms / scale).exp();

    let confidence = W_SAME_STROKE * same_stroke_ratio
        + W_COVERAGE * t_interval_coverage
        + W_SMOOTHNESS * displacement_smoothness
        + W_CORE * core_weight_ratio
        + W_FIT * fit_quality;

    CoherenceResult {
        coherence: EditCoherence {
            same_stroke_ratio,
            t_interval_coverage,
            displacement_smoothness,
            core_weight_ratio,
            fit_error: rms,
            fit_quality,
            confidence,
        },
        trial_skeleton,
        preserve_endpoints,
    }
}

/// 1 = displacement varies perfectly smoothly with t; ~0 = neighbor displacements are
/// uncorrelated (scatter). Computed from mean neighbor-to-neighbor displacement change
/// normalized by mean displacement magnitude.
fn displacement_smoothness(samples: &mut [(f32, Vec2, f32)]) -> f32 {
    const K: f32 = 2.5;
    if samples.len() < 2 {
        return 1.0;
    }
    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mean_mag =
        samples.iter().map(|s| s.1.length()).sum::<f32>() / samples.len() as f32;
    if mean_mag < 1e-4 {
        return 1.0; // no real movement -> trivially smooth
    }
    let mut diff = 0.0f32;
    for w in samples.windows(2) {
        diff += (w[1].1 - w[0].1).length();
    }
    let roughness = diff / ((samples.len() - 1) as f32 * mean_mag);
    1.0 / (1.0 + K * roughness)
}

/// Absorb the unexplained portion of each edit into the splat's local residual so the
/// splat lands exactly at its edited position. Shared by the structural and
/// residual-only paths.
fn absorb_into_residuals(stroke: &mut GaussianBezierStroke, edits: &[SplatEdit]) {
    for edit in edits {
        let Some((t, center)) = stroke.find_splat(edit.splat_id).map(|s| (s.t, s.center)) else {
            continue;
        };
        let frame = stroke.skeleton.frame_at_arc_t(t);
        let remaining = edit.new_center - center;
        let add = Vec2::new(remaining.dot(frame.tangent), remaining.dot(frame.normal));
        if let Some(s) = stroke.find_splat_mut(edit.splat_id) {
            s.residual_local += add;
        }
    }
    update_splat_world_cache(stroke);
}

/// Store edits as residual deformation, leaving the skeleton untouched.
pub fn apply_as_residual_edits(stroke: &mut GaussianBezierStroke, edits: &[SplatEdit]) {
    absorb_into_residuals(stroke, edits);
}

/// After a (full or partial) skeleton update, pin each edited splat exactly at its
/// target by storing the remaining delta as a residual.
pub fn preserve_remaining_deltas_as_residuals(
    stroke: &mut GaussianBezierStroke,
    edits: &[SplatEdit],
) {
    absorb_into_residuals(stroke, edits);
}

/// Linearly blend two skeletons' control points (same topology) by `f in [0,1]`.
fn blend_skeletons(old: &BezierSkeleton, new: &BezierSkeleton, f: f32) -> BezierSkeleton {
    let mut segs = old.segments.clone();
    for (i, seg) in segs.iter_mut().enumerate() {
        if let Some(ns) = new.segments.get(i) {
            for c in 0..4 {
                seg.set_control(c, seg.control(c).lerp(ns.control(c), f));
            }
        }
    }
    BezierSkeleton::from_segments(segs, old.closed)
}

/// The full reverse-sync dispatch. Groups edits by stroke, classifies each group, and
/// applies a structural / partial / residual update accordingly. Returns the outcome
/// per stroke (useful for tests and UI feedback).
pub fn apply_splat_edits_bidirectional(
    doc: &mut crate::document::Document,
    edits: &[SplatEdit],
) -> Vec<(StrokeId, FitOutcome)> {
    // Group by parent stroke.
    let mut groups: BTreeMap<StrokeId, Vec<SplatEdit>> = BTreeMap::new();
    for edit in edits {
        groups.entry(edit.stroke).or_default().push(*edit);
    }
    let total_edits = edits.len();

    let mut outcomes = Vec::new();
    for (sid, group) in groups {
        let Some(stroke) = doc.stroke_mut(sid) else {
            continue;
        };
        let (high, low) = (stroke.sync.high_confidence, stroke.sync.low_confidence);
        if !stroke.sync.splat_to_curve {
            apply_as_residual_edits(stroke, &group);
            outcomes.push((sid, FitOutcome::SyncDisabled));
            continue;
        }

        let targets = estimate_centerline_targets(stroke, &group);
        let result = measure_edit_coherence(stroke, &group, &targets, total_edits);
        let conf = result.coherence.confidence;

        let outcome = if conf >= high {
            stroke.skeleton = result.trial_skeleton;
            update_splat_world_cache(stroke);
            preserve_remaining_deltas_as_residuals(stroke, &group);
            FitOutcome::Structural
        } else if conf >= low {
            // Apply a fraction of the structural update, scaled within the medium band,
            // then absorb the remainder as residuals.
            let f = ((conf - low) / (high - low)).clamp(0.0, 1.0) * 0.5;
            stroke.skeleton = blend_skeletons(&stroke.skeleton, &result.trial_skeleton, f);
            update_splat_world_cache(stroke);
            preserve_remaining_deltas_as_residuals(stroke, &group);
            FitOutcome::Partial
        } else {
            apply_as_residual_edits(stroke, &group);
            FitOutcome::Residual
        };
        outcomes.push((sid, outcome));
    }
    outcomes
}

/// Sculpt helper: move every splat within `radius` of `center` by `delta`, falling off
/// smoothly to the brush edge. Returns the resulting edits (to feed into
/// [`apply_splat_edits_bidirectional`]).
pub fn sculpt_move_splats(
    stroke: &GaussianBezierStroke,
    center: Vec2,
    delta: Vec2,
    radius: f32,
) -> Vec<SplatEdit> {
    let mut edits = Vec::new();
    for splat in &stroke.splats {
        let dist = (splat.center - center).length();
        if dist < radius {
            let w = smoothstep(1.0, 0.0, dist / radius);
            edits.push(SplatEdit {
                stroke: stroke.id,
                splat_id: splat.id,
                old_center: splat.center,
                new_center: splat.center + delta * w,
            });
        }
    }
    edits
}
