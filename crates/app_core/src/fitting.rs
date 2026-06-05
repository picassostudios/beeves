//! Freehand-stroke -> Bezier skeleton fitting — the geometry behind the brush tool.
//!
//! A raw pointer/pen path arrives as a dense, noisy polyline. To turn it into the
//! canonical editable [`BezierSkeleton`] we:
//!   1. lightly clean the input (drop duplicate/near-coincident samples, one pass of
//!      neighbor averaging to take the high-frequency jitter off);
//!   2. run Ramer-Douglas-Peucker to pick a small set of **anchor** points that still
//!      track the path within `tolerance`, preserving sharp corners;
//!   3. fit one cubic segment between each pair of consecutive anchors, placing the
//!      interior handles from Catmull-Rom tangent estimates so neighboring segments
//!      meet smoothly (the chain is C0 at the anchors, and visually C1 at smooth ones).
//!
//! Everything here is deterministic and depends only on `glam` + existing `app_core`
//! modules, mirroring the style of [`crate::solver`]'s fitter.

use crate::bezier::{BezierSkeleton, CubicBezier};
use crate::brush::BrushModel;
use crate::document::Document;
use crate::ids::{LayerId, StrokeId};
use crate::math::Vec2;

/// Two samples closer than this (px) are treated as the same point during cleanup.
const DEDUPE_EPS: f32 = 1e-3;

/// Ramer-Douglas-Peucker polyline simplification.
///
/// Returns a subsequence of `points` (always including the first and last) such that
/// every dropped point lies within `epsilon` of the retained polyline. The classic
/// recursive split-on-farthest-point algorithm, written iteratively with an explicit
/// stack so deep paths can't blow the call stack.
pub fn simplify_rdp(points: &[Vec2], epsilon: f32) -> Vec<Vec2> {
    let n = points.len();
    if n <= 2 {
        return points.to_vec();
    }
    let eps = epsilon.max(0.0);

    // `keep[i]` marks anchors that survive simplification.
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;

    // Stack of (start, end) index ranges still to subdivide.
    let mut stack: Vec<(usize, usize)> = vec![(0, n - 1)];
    while let Some((start, end)) = stack.pop() {
        if end <= start + 1 {
            continue;
        }
        let a = points[start];
        let b = points[end];
        let mut max_dist = -1.0f32;
        let mut max_idx = start;
        for (i, p) in points.iter().enumerate().take(end).skip(start + 1) {
            let d = perpendicular_distance(*p, a, b);
            if d > max_dist {
                max_dist = d;
                max_idx = i;
            }
        }
        if max_dist > eps {
            keep[max_idx] = true;
            stack.push((start, max_idx));
            stack.push((max_idx, end));
        }
    }

    points
        .iter()
        .zip(keep)
        .filter_map(|(p, k)| k.then_some(*p))
        .collect()
}

/// Perpendicular distance from `p` to the segment `a-b` (degenerates to point distance
/// when `a == b`).
fn perpendicular_distance(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq < 1e-12 {
        return (p - a).length();
    }
    // Reject onto the infinite line through a-b. RDP measures from the line (the segment
    // endpoints are themselves kept anchors), so no clamping of the projection.
    let ap = p - a;
    let cross = ab.x * ap.y - ab.y * ap.x;
    cross.abs() / len_sq.sqrt()
}

/// Fit a freehand polyline to a [`BezierSkeleton`] of cubic segments.
///
/// `tolerance` is the RDP threshold in px: smaller = more segments / tighter tracking.
/// The result always has at least one segment, even for 2- or 3-point inputs.
pub fn fit_polyline_to_skeleton(points: &[Vec2], tolerance: f32) -> BezierSkeleton {
    let cleaned = clean_input(points);

    // Degenerate inputs: synthesize a tiny straight skeleton so downstream code (splat
    // generation, arc-length table) always has a non-empty, finite curve to work with.
    if cleaned.len() < 2 {
        let p = cleaned.first().copied().unwrap_or(Vec2::ZERO);
        let q = p + Vec2::new(1.0, 0.0);
        return BezierSkeleton::single(straight_cubic(p, q));
    }

    let tol = if tolerance > 0.0 { tolerance } else { 1.0 };
    let anchors = simplify_rdp(&cleaned, tol);
    // RDP keeps both endpoints, so `anchors.len() >= 2` whenever `cleaned.len() >= 2`.
    debug_assert!(anchors.len() >= 2);

    let tangents = estimate_tangents(&anchors);

    let mut segments: Vec<CubicBezier> = Vec::with_capacity(anchors.len() - 1);
    for i in 0..anchors.len() - 1 {
        let a = anchors[i];
        let b = anchors[i + 1];
        let chord = (b - a).length();
        // Place handles a third of the chord out along each endpoint's tangent — the
        // standard Catmull-Rom -> Bezier handle length. Falls back to a straight segment
        // when the chord is degenerate.
        let h = chord / 3.0;
        let p1 = a + tangents[i] * h;
        let p2 = b - tangents[i + 1] * h;
        segments.push(CubicBezier::new(a, p1, p2, b));
    }

    BezierSkeleton::from_segments(segments, false)
}

/// Convenience: a perfectly straight cubic from `a` to `b` (handles on the chord).
fn straight_cubic(a: Vec2, b: Vec2) -> CubicBezier {
    let p1 = a + (b - a) / 3.0;
    let p2 = a + (b - a) * (2.0 / 3.0);
    CubicBezier::new(a, p1, p2, b)
}

/// Drop near-coincident consecutive samples, then run a single light smoothing pass
/// (weighted neighbor average) over the interior to knock down pointer jitter without
/// rounding off real corners — RDP downstream is responsible for the actual shape.
fn clean_input(points: &[Vec2]) -> Vec<Vec2> {
    // 1. Dedupe consecutive duplicates.
    let mut deduped: Vec<Vec2> = Vec::with_capacity(points.len());
    for &p in points {
        if !p.is_finite() {
            continue;
        }
        match deduped.last() {
            Some(&last) if (p - last).length() < DEDUPE_EPS => {}
            _ => deduped.push(p),
        }
    }
    if deduped.len() < 3 {
        return deduped;
    }

    // 2. One pass of [0.25, 0.5, 0.25] smoothing on interior points; keep endpoints
    //    pinned so the fitted curve starts/ends exactly where the user did.
    let mut smoothed = deduped.clone();
    for i in 1..deduped.len() - 1 {
        smoothed[i] = deduped[i - 1] * 0.25 + deduped[i] * 0.5 + deduped[i + 1] * 0.25;
    }
    smoothed
}

/// Catmull-Rom tangent estimate at each anchor: the (normalized) direction of the
/// chord spanning its neighbors. Endpoints use the one-sided chord. Degenerate spans
/// fall back to +x so handles are always finite.
fn estimate_tangents(anchors: &[Vec2]) -> Vec<Vec2> {
    let n = anchors.len();
    let mut tangents = Vec::with_capacity(n);
    for i in 0..n {
        let prev = anchors[i.saturating_sub(1)];
        let next = anchors[(i + 1).min(n - 1)];
        let dir = next - prev;
        tangents.push(if dir.length_squared() > 1e-12 {
            dir.normalize()
        } else {
            Vec2::X
        });
    }
    tangents
}

impl Document {
    /// Fit a freehand `points` polyline to a Bezier skeleton, then create a stroke from
    /// it on `layer` with `brush`. `tolerance` is the RDP fit threshold in px. Returns
    /// the new stroke's id (see [`Document::add_stroke`]).
    pub fn add_freehand_stroke(
        &mut self,
        layer: LayerId,
        points: &[Vec2],
        brush: BrushModel,
        tolerance: f32,
    ) -> StrokeId {
        let skeleton = fit_polyline_to_skeleton(points, tolerance);
        self.add_stroke(layer, skeleton, brush)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rdp_keeps_endpoints_and_drops_collinear() {
        let pts = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(3.0, 0.0),
        ];
        let out = simplify_rdp(&pts, 0.01);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], pts[0]);
        assert_eq!(out[1], *pts.last().unwrap());
    }

    #[test]
    fn rdp_keeps_a_corner() {
        let pts = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(5.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 5.0),
            Vec2::new(10.0, 10.0),
        ];
        let out = simplify_rdp(&pts, 0.5);
        // Endpoints + the corner at (10,0).
        assert!(out.contains(&Vec2::new(10.0, 0.0)));
        assert!(out.len() >= 3);
    }

    #[test]
    fn fit_handles_two_point_input() {
        let pts = vec![Vec2::new(0.0, 0.0), Vec2::new(100.0, 0.0)];
        let sk = fit_polyline_to_skeleton(&pts, 1.0);
        assert!(!sk.segments.is_empty());
        assert!((sk.frame_at_arc_t(0.0).position - pts[0]).length() < 1.0);
        assert!((sk.frame_at_arc_t(1.0).position - pts[1]).length() < 1.0);
    }

    #[test]
    fn fit_handles_single_point_input() {
        let sk = fit_polyline_to_skeleton(&[Vec2::new(7.0, 7.0)], 1.0);
        assert_eq!(sk.segments.len(), 1);
        assert!(sk.total_length() > 0.0);
    }
}
