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

use crate::bezier::{bernstein, BezierSkeleton, CubicBezier};
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

// =====================================================================================
// Adaptive-window, curvature-driven fitting
// =====================================================================================
//
// An alternative to the RDP path above, used by the vector-draw tool. The idea: a single
// cubic Bezier represents one gentle arc well, so we grow a *window* of input samples as
// long as the stroke stays gentle, then close it and emit one cubic. The window length is
// governed by how much the stroke *turns*:
//
//   * straight / gently-curving runs accumulate almost no turning  -> the window stretches
//     far -> few anchors;
//   * tight curves accumulate turning quickly                      -> the window closes
//     soon -> anchors land exactly where the drawing bends.
//
// On top of that, true corners (a sharp local turn) are detected first and always become
// anchors with independent tangents, so sharp features survive. Each window is fit with a
// weighted least-squares cubic (Schneider, *Graphics Gems* 1990) using centripetal
// parameterization and one Newton reparameterization pass, and the window is shrunk if
// needed so the fit always stays within `tolerance`. The net effect is minimal anchors for
// a given fidelity — fewer than RDP + the fixed `chord/3` handle heuristic.

/// Tunables for [`fit_polyline_adaptive`].
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveFitParams {
    /// Max allowed deviation (px) of the fitted curve from the input samples. Smaller =
    /// tighter tracking / more anchors.
    pub tolerance: f32,
    /// Per-window turning budget Φ_max (radians). A window closes once the stroke has bent
    /// this much, so window length is inversely proportional to local curvature. Keeping it
    /// below ~π also keeps each cubic in the well-conditioned "one arc" regime. ~1.6–2.1.
    pub max_turn: f32,
    /// Corner threshold Φ_corner (radians). A vertex whose local (wide-support) turn exceeds
    /// this is a hard corner: it always becomes an anchor and its tangents are left
    /// independent so the corner stays sharp. ~1.0–1.3 (≈ 60–75°).
    pub corner_turn: f32,
    /// Arc-length step (px) the cleaned input is resampled to before curvature analysis, so
    /// the turning estimates don't depend on raw pointer-sampling density (which varies with
    /// draw speed).
    pub resample_step: f32,
}

impl Default for AdaptiveFitParams {
    fn default() -> Self {
        Self {
            tolerance: 1.5,
            max_turn: 1.9,    // ≈ 109°
            corner_turn: 1.1, // ≈ 63°
            resample_step: 3.0,
        }
    }
}

/// Fit a freehand polyline to a [`BezierSkeleton`] with a curvature-adaptive window (see the
/// module section above). The returned skeleton has its per-anchor `corner` flags set, so
/// the direct-edit tool treats detected corners as hard (non-mirrored) joins.
pub fn fit_polyline_adaptive(points: &[Vec2], params: &AdaptiveFitParams) -> BezierSkeleton {
    let cleaned = clean_input(points);
    if cleaned.len() < 2 {
        let p = cleaned.first().copied().unwrap_or(Vec2::ZERO);
        return BezierSkeleton::single(straight_cubic(p, p + Vec2::new(1.0, 0.0)));
    }

    // Resample to ~uniform arc length so curvature is speed-independent.
    let pts = resample_uniform(&cleaned, params.resample_step.max(0.5));
    if pts.len() < 3 {
        return BezierSkeleton::single(fit_cubic_lsq(&pts).0);
    }

    let tol = params.tolerance.max(1e-3);
    let max_turn = params.max_turn.max(0.05);
    let corner_turn = params.corner_turn.max(0.05);

    let turn = turning_angles(&pts);
    let corners = detect_corners(&pts, corner_turn);
    let breakpoints = adaptive_breakpoints(&pts, &turn, &corners, max_turn, tol);

    // One cubic per [bp, next-bp] window.
    let mut segments: Vec<CubicBezier> = Vec::with_capacity(breakpoints.len().saturating_sub(1));
    for w in breakpoints.windows(2) {
        segments.push(fit_cubic_lsq(&pts[w[0]..=w[1]]).0);
    }
    if segments.is_empty() {
        segments.push(fit_cubic_lsq(&pts).0);
    }

    // Make smooth (non-corner) interior joins visually C1, then assemble and flag corners.
    smooth_interior_joins(&mut segments, &breakpoints, &corners);
    let mut sk = BezierSkeleton::from_segments(segments, false);
    for (j, &idx) in breakpoints.iter().enumerate() {
        if let Some(meta) = sk.anchors.get_mut(j) {
            meta.corner = corners.get(idx).copied().unwrap_or(false);
        }
    }
    sk
}

impl Document {
    /// Fit `points` with the curvature-adaptive fitter, then create a stroke on `layer`.
    /// The vector-draw counterpart to [`Document::add_freehand_stroke`].
    pub fn add_vector_stroke(
        &mut self,
        layer: LayerId,
        points: &[Vec2],
        brush: BrushModel,
        params: &AdaptiveFitParams,
    ) -> StrokeId {
        let skeleton = fit_polyline_adaptive(points, params);
        self.add_stroke(layer, skeleton, brush)
    }
}

/// Resample a polyline to roughly-uniform arc-length spacing `step`, always keeping the
/// first and last vertex. Decouples the downstream curvature estimate from the raw pointer
/// sampling density.
fn resample_uniform(points: &[Vec2], step: f32) -> Vec<Vec2> {
    if points.len() < 2 {
        return points.to_vec();
    }
    let last = points[points.len() - 1];
    let mut out = vec![points[0]];
    let mut prev = points[0];
    let mut acc = 0.0f32; // arc length carried since the last emitted sample
    for &p in &points[1..] {
        let mut seg = p - prev;
        let mut seg_len = seg.length();
        while acc + seg_len >= step && seg_len > 1e-9 {
            let t = (step - acc) / seg_len;
            let np = prev + seg * t.clamp(0.0, 1.0);
            out.push(np);
            prev = np;
            seg = p - prev;
            seg_len = seg.length();
            acc = 0.0;
        }
        acc += seg_len;
        prev = p;
    }
    if out.last().map_or(true, |&l| (l - last).length() > 1e-4) {
        out.push(last);
    }
    out
}

/// Exterior (turning) angle at each vertex in `[0, π]`: the angle between the incoming and
/// outgoing chord. Endpoints are 0. Only the magnitude of bending is kept — that's what the
/// window budget and corner test consume.
fn turning_angles(pts: &[Vec2]) -> Vec<f32> {
    let n = pts.len();
    let mut turn = vec![0.0f32; n];
    for i in 1..n - 1 {
        let a = pts[i] - pts[i - 1];
        let b = pts[i + 1] - pts[i];
        if a.length_squared() > 1e-12 && b.length_squared() > 1e-12 {
            let cross = a.x * b.y - a.y * b.x;
            turn[i] = cross.atan2(a.dot(b)).abs();
        }
    }
    turn
}

/// Detect corner vertices: a ±K-sample chord angle (robust to single-step jitter) that both
/// exceeds `corner_turn` and is a local maximum within its window (non-maximum suppression,
/// so one physical corner yields exactly one anchor). Endpoints are never flagged — they are
/// already anchors.
fn detect_corners(pts: &[Vec2], corner_turn: f32) -> Vec<bool> {
    let n = pts.len();
    let mut is_corner = vec![false; n];
    if n < 3 {
        return is_corner;
    }
    const K: usize = 2;
    let mut wide = vec![0.0f32; n];
    for i in 1..n - 1 {
        let lo = i.saturating_sub(K);
        let hi = (i + K).min(n - 1);
        let a = pts[i] - pts[lo];
        let b = pts[hi] - pts[i];
        if a.length_squared() > 1e-12 && b.length_squared() > 1e-12 {
            wide[i] = (a.x * b.y - a.y * b.x).atan2(a.dot(b)).abs();
        }
    }
    for i in 1..n - 1 {
        if wide[i] < corner_turn {
            continue;
        }
        let lo = i.saturating_sub(K);
        let hi = (i + K).min(n - 1);
        if (lo..=hi).all(|j| wide[j] <= wide[i] + 1e-6) {
            is_corner[i] = true;
        }
    }
    is_corner
}

/// Ordered breakpoint (anchor) indices along the resampled polyline: the two endpoints,
/// every corner, and the adaptive interior splits. Within each smooth span a window grows
/// while accumulated turning stays under `max_turn`, then is shrunk until a single cubic fits
/// within `tol`.
fn adaptive_breakpoints(
    pts: &[Vec2],
    turn: &[f32],
    corners: &[bool],
    max_turn: f32,
    tol: f32,
) -> Vec<usize> {
    let n = pts.len();
    // Hard anchors that always split the path: endpoints + corners.
    let mut hard = vec![0usize];
    hard.extend((1..n - 1).filter(|&i| corners[i]));
    hard.push(n - 1);

    let mut bp = vec![0usize];
    for span in hard.windows(2) {
        let (lo, hi) = (span[0], span[1]);
        let mut start = lo;
        while start < hi {
            // 1) Grow by the turning budget (cheap; no fitting). Invariant: `acc` is the
            //    total turning of the window's interior vertices.
            let mut end = start + 1;
            let mut acc = 0.0f32;
            while end < hi && acc + turn[end] <= max_turn {
                acc += turn[end];
                end += 1;
            }
            // 2) Shrink until one cubic fits within tolerance. Progress is guaranteed: a
            //    two-point window is a straight chord with ~zero error.
            while end > start + 1 && fit_cubic_lsq(&pts[start..=end]).1 > tol {
                end -= 1;
            }
            bp.push(end);
            start = end;
        }
    }
    bp
}

/// After fitting, rotate the two handles at each smooth (non-corner) interior anchor to be
/// collinear through the anchor, keeping each handle's length — a visually C1 join. Corner
/// anchors are left untouched so they stay sharp.
fn smooth_interior_joins(segments: &mut [CubicBezier], breakpoints: &[usize], corners: &[bool]) {
    for j in 1..segments.len() {
        let idx = breakpoints[j];
        if corners.get(idx).copied().unwrap_or(false) {
            continue;
        }
        let anchor = segments[j].p0; // == segments[j - 1].p3
        let out_dir = segments[j].p1 - anchor; // forward
        let in_dir = segments[j - 1].p2 - anchor; // backward
        let len_out = out_dir.length();
        let len_in = in_dir.length();
        // Average direction: `out` points forward, `in` points backward, so subtract.
        let t = out_dir.normalize_or_zero() - in_dir.normalize_or_zero();
        if t.length_squared() < 1e-12 {
            continue;
        }
        let t = t.normalize();
        segments[j].p1 = anchor + t * len_out;
        segments[j - 1].p2 = anchor - t * len_in;
    }
}

/// Least-squares fit of one cubic to a window of points, with fixed endpoints and
/// data-estimated end-tangent directions (Schneider). Returns the cubic and its max
/// at-parameter deviation. Centripetal parameterization plus one Newton reparameterization
/// pass keep the error low so each window can stretch as far as the tolerance allows.
fn fit_cubic_lsq(pts: &[Vec2]) -> (CubicBezier, f32) {
    let n = pts.len();
    // Self-defensive against degenerate windows so callers don't have to rely on upstream
    // invariants: <2 points has no chord to fit, so synthesize a tiny straight cubic.
    if n < 2 {
        let p = pts.first().copied().unwrap_or(Vec2::ZERO);
        return (straight_cubic(p, p + Vec2::new(1.0, 0.0)), 0.0);
    }
    let (p0, p3) = (pts[0], pts[n - 1]);
    if n == 2 {
        return (straight_cubic(p0, p3), 0.0);
    }
    let t_hat1 = start_tangent(pts);
    let t_hat2 = end_tangent(pts);

    let mut u = centripetal_params(pts);
    let mut curve = solve_handles(pts, &u, p0, p3, t_hat1, t_hat2);
    reparameterize(pts, &curve, &mut u);
    curve = solve_handles(pts, &u, p0, p3, t_hat1, t_hat2);

    (curve, max_error(pts, &curve, &u))
}

/// Unit tangent leaving the first point, averaged over a few early samples to damp jitter.
fn start_tangent(pts: &[Vec2]) -> Vec2 {
    let n = pts.len();
    let k = (n / 4).clamp(1, 3).min(n - 1);
    safe_dir(pts[k] - pts[0], Vec2::X)
}

/// Unit tangent arriving at the last point, pointing *backward* (toward the interior) — the
/// convention `p2 = p3 + t_hat2 * alpha2`.
fn end_tangent(pts: &[Vec2]) -> Vec2 {
    let n = pts.len();
    let k = (n / 4).clamp(1, 3).min(n - 1);
    safe_dir(pts[n - 1 - k] - pts[n - 1], -Vec2::X)
}

fn safe_dir(v: Vec2, fallback: Vec2) -> Vec2 {
    if v.length_squared() > 1e-12 {
        v.normalize()
    } else {
        fallback
    }
}

/// Centripetal (chord^0.5) parameterization in `[0,1]`. Reduces the overshoot/cusps that
/// uniform and pure chord-length parameterizations produce on tight turns.
fn centripetal_params(pts: &[Vec2]) -> Vec<f32> {
    let n = pts.len();
    let mut u = vec![0.0f32; n];
    for i in 1..n {
        u[i] = u[i - 1] + (pts[i] - pts[i - 1]).length().sqrt().max(1e-6);
    }
    let total = u[n - 1].max(1e-6);
    for x in &mut u {
        *x /= total;
    }
    u
}

/// Solve the two handle magnitudes (α1, α2) that minimize Σ|B(u_i) − pts_i|² with the
/// endpoints and tangent directions fixed (Schneider's `generateBezier`). Falls back to
/// `chord/3` handles when the normal equations are degenerate or yield a non-positive length.
fn solve_handles(
    pts: &[Vec2],
    u: &[f32],
    p0: Vec2,
    p3: Vec2,
    t_hat1: Vec2,
    t_hat2: Vec2,
) -> CubicBezier {
    let (mut c00, mut c01, mut c11, mut x0, mut x1) = (0.0f32, 0.0, 0.0, 0.0, 0.0);
    for (i, &p) in pts.iter().enumerate() {
        let b = bernstein(u[i]);
        let a1 = t_hat1 * b[1];
        let a2 = t_hat2 * b[2];
        c00 += a1.dot(a1);
        c01 += a1.dot(a2);
        c11 += a2.dot(a2);
        // residual = point − (part fixed by the endpoints): p0·(B0+B1) + p3·(B2+B3).
        let d = p - (p0 * (b[0] + b[1]) + p3 * (b[2] + b[3]));
        x0 += a1.dot(d);
        x1 += a2.dot(d);
    }
    let det = c00 * c11 - c01 * c01;
    let chord = (p3 - p0).length();
    let (alpha1, alpha2) = if det.abs() > 1e-9 {
        ((x0 * c11 - x1 * c01) / det, (c00 * x1 - c01 * x0) / det)
    } else {
        (0.0, 0.0)
    };
    let (alpha1, alpha2) = if alpha1.is_finite()
        && alpha2.is_finite()
        && alpha1 > 1e-3 * chord
        && alpha2 > 1e-3 * chord
    {
        (alpha1, alpha2)
    } else {
        (chord / 3.0, chord / 3.0)
    };
    CubicBezier::new(p0, p0 + t_hat1 * alpha1, p3 + t_hat2 * alpha2, p3)
}

/// Refine each sample's parameter toward the nearest point on `curve` with one Newton step
/// of the root-find `(B(u) − p)·B'(u) = 0`.
fn reparameterize(pts: &[Vec2], curve: &CubicBezier, u: &mut [f32]) {
    for (i, &p) in pts.iter().enumerate() {
        let s = u[i];
        let q = curve.point(s);
        let d1 = curve.velocity(s);
        let d2 = second_derivative(curve, s);
        let diff = q - p;
        let denom = d1.dot(d1) + diff.dot(d2);
        if denom.abs() > 1e-9 {
            u[i] = (s - diff.dot(d1) / denom).clamp(0.0, 1.0);
        }
    }
}

/// Second derivative d²B/ds² of a cubic Bezier at `s`.
fn second_derivative(c: &CubicBezier, s: f32) -> Vec2 {
    let a = c.p2 - c.p1 * 2.0 + c.p0;
    let b = c.p3 - c.p2 * 2.0 + c.p1;
    (a * (1.0 - s) + b * s) * 6.0
}

/// Max distance between each input sample and the curve evaluated at that sample's
/// (reparameterized) parameter.
fn max_error(pts: &[Vec2], curve: &CubicBezier, u: &[f32]) -> f32 {
    pts.iter()
        .enumerate()
        .map(|(i, &p)| (curve.point(u[i]) - p).length())
        .fold(0.0f32, f32::max)
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
