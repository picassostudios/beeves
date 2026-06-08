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
    /// Softness/strictness of the fit in `[0,1]` — how strongly anchors are biased toward the
    /// *curvature extrema* of the stroke (the apex of each bend: a sine wave's peaks and
    /// troughs) rather than spread evenly by the turning budget.
    ///
    /// * `0.0` (strict): pure curvature-adaptive tracking — anchors land wherever the turning
    ///   budget / tolerance demand. Hugs the input closely, more anchors.
    /// * `1.0` (soft): anchors are pinned at the salient curvature extrema (plus corners and
    ///   endpoints) and the turning budget / tolerance are relaxed so *no* extra anchors are
    ///   inserted between them — a sine wave fits as one cubic per peak-to-trough arc.
    ///
    /// Intermediate values both pin the extrema and progressively relax the budget. At `0.0`
    /// the extrema pass is skipped entirely, so the fit is byte-identical to the original
    /// curvature-adaptive behaviour.
    pub smoothness: f32,
}

impl Default for AdaptiveFitParams {
    fn default() -> Self {
        Self {
            tolerance: 1.5,
            max_turn: 1.9,    // ≈ 109°
            corner_turn: 1.1, // ≈ 63°
            resample_step: 3.0,
            smoothness: 0.0,
        }
    }
}

impl AdaptiveFitParams {
    /// Per-window turning budget after applying `smoothness`. Relaxed upward as the fit softens
    /// so that — once the curvature extrema are pinned as anchors — the budget stops inserting
    /// *extra* anchors between them. At `smoothness == 0` this is exactly `max_turn`; at `1.0`
    /// it is large enough that no bounded inter-extremum span ever trips it.
    fn effective_max_turn(&self) -> f32 {
        let s = self.smoothness.clamp(0.0, 1.0);
        self.max_turn.max(0.05) + s * 6.0
    }

    /// Fit tolerance after applying `smoothness`. Opened up substantially as the fit softens:
    /// "soft" means a loose interpolation, so the per-window error budget must comfortably exceed
    /// typical hand jitter (a few px), otherwise that jitter forces a tolerance-driven split on
    /// every wobble and the stroke over-segments through a curvature change. At `smoothness == 0`
    /// this is exactly `tolerance`; at `1.0` it is 6× — loose enough to ride over the noise.
    fn effective_tolerance(&self) -> f32 {
        let s = self.smoothness.clamp(0.0, 1.0);
        self.tolerance.max(1e-3) * (1.0 + 5.0 * s)
    }

    /// Per-vertex curvature-extremum mask for `pts`. Empty (all-false) when `smoothness == 0`
    /// so the strict path is unchanged; otherwise the salient local maxima of curvature (see
    /// [`detect_curvature_extrema`]). `smoothness` controls *selectivity*: softer fits demand a
    /// deeper curvature dip around each apex and a wider minimum spacing between anchors, so one
    /// bend yields exactly one anchor instead of a cluster.
    fn curvature_extrema(&self, pts: &[Vec2]) -> Vec<bool> {
        let s = self.smoothness.clamp(0.0, 1.0);
        if s <= 0.0 {
            return vec![false; pts.len()];
        }
        // Minimum topographic prominence (radians of wide-support turning) for an apex to count
        // as a real bend rather than a residual noise hump. The noise floor on a hand-drawn curve
        // sits around ~0.3 at this support; genuine bends rise well above it. Raising the bar with
        // softness keeps only the boldest bends when the slider is pushed toward "soft".
        let min_prominence = 0.35 + 0.20 * s;
        // Minimum separation between anchors, in resampled samples. Widens with softness so a
        // broad bend collapses to a single anchor rather than a row of them.
        let min_spacing = (4.0 + 12.0 * s).round() as usize;
        detect_curvature_extrema(pts, min_prominence, min_spacing)
    }

    /// Per-vertex turning that feeds the window budget in [`adaptive_breakpoints`]. At
    /// `smoothness == 0` it is the raw [`turning_angles`] (the original behaviour). When
    /// softening it is measured on position-smoothed samples: raw per-vertex turning on a
    /// hand-drawn stroke is jitter-dominated and sums up fast, which would make the budget close
    /// a window in the *middle* of a smooth bend on a noisy stroke (the spurious cluster of extra
    /// anchors the user is seeing). De-noising it lets the budget reflect the real turning, so
    /// within a bend only the pinned curvature extrema and the tolerance can introduce a split.
    fn budget_turning(&self, pts: &[Vec2]) -> Vec<f32> {
        if self.smoothness <= 0.0 {
            turning_angles(pts)
        } else {
            turning_angles(&smooth_positions(pts, 4))
        }
    }
}

/// `passes` rounds of [0.25, 0.5, 0.25] neighbour averaging over the interior, endpoints pinned.
/// A cheap low-pass on the polyline used to take hand jitter off the curvature/turning estimates
/// without moving the stroke's endpoints.
fn smooth_positions(pts: &[Vec2], passes: usize) -> Vec<Vec2> {
    let n = pts.len();
    let mut sm = pts.to_vec();
    if n < 3 {
        return sm;
    }
    for _ in 0..passes {
        let raw = sm.clone();
        for i in 1..n - 1 {
            sm[i] = raw[i - 1] * 0.25 + raw[i] * 0.5 + raw[i + 1] * 0.25;
        }
    }
    sm
}

/// Absolute curvature floor (radians of wide-support turning) below which an apex is treated as
/// pointer jitter on a near-straight run, not a real bend. Keeps noise from spawning anchors.
const EXTREMA_FLOOR: f32 = 0.1;

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

    let tol = params.effective_tolerance();
    let max_turn = params.effective_max_turn();
    let corner_turn = params.corner_turn.max(0.05);

    let turn = params.budget_turning(&pts);
    let corners = detect_corners(&pts, corner_turn);
    let extrema = params.curvature_extrema(&pts);
    let breakpoints =
        adaptive_breakpoints(&pts, &turn, &corners, &extrema, max_turn, tol, usize::MAX);

    // One cubic per [bp, next-bp] window, C1 across smooth interior joins by construction.
    let mut segments = fit_windows_c1(&pts, &breakpoints, &corners, None, None);
    if segments.is_empty() {
        segments.push(fit_cubic_lsq(&pts).0);
    }

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

/// Smoothed curvature magnitude at each interior vertex: a ±K wide-support chord angle (robust
/// to single-step jitter) followed by several [0.25, 0.5, 0.25] passes. Because the input is
/// resampled to ~uniform arc length this is proportional to |curvature|, so its local maxima
/// are the apexes of the stroke's bends.
///
/// The heavy smoothing is deliberate: a hand-drawn curve carries enough jitter that the raw
/// curvature signal has a local maximum every few samples. Without it, *one* visual bend would
/// spawn a whole cluster of anchors. Smoothing merges that jitter into a single broad hump per
/// real bend, so a sustained curve reads as one plateau (no spurious peaks) and only genuine
/// turns survive as maxima.
fn curvature_magnitude(pts: &[Vec2]) -> Vec<f32> {
    let n = pts.len();
    let mut c = vec![0.0f32; n];
    if n < 3 {
        return c;
    }
    // Heavily low-pass the *positions* first. At a fine resample step a couple of px of hand
    // jitter produces per-sample turning that dwarfs the true curvature of a gentle bend, so a
    // curvature estimate on the raw samples is noise-dominated. Jitter is high-frequency (it
    // alternates sample to sample) while a real bend spans many samples, so several binomial
    // passes erase the noise without touching the shape of genuine turns. (Endpoints stay
    // pinned; only the curvature *estimate* uses these — the fit itself runs on the original
    // samples, so accuracy is unaffected.)
    let sm = smooth_positions(pts, 4);
    // Measure turning between two *wide* chords (±K samples). The wide support is the key to
    // telling a real bend from jitter: a genuine turn keeps bending the same way across the whole
    // span, so the wide angle adds up; residual noise alternates direction and averages out. At a
    // ~3px resample step this spans ~50px each side — the scale of a feature the eye reads as "a
    // bend" rather than "a wobble".
    const K: usize = 18;
    for i in 1..n - 1 {
        let lo = i.saturating_sub(K);
        let hi = (i + K).min(n - 1);
        let a = sm[i] - sm[lo];
        let b = sm[hi] - sm[i];
        if a.length_squared() > 1e-12 && b.length_squared() > 1e-12 {
            c[i] = (a.x * b.y - a.y * b.x).atan2(a.dot(b)).abs();
        }
    }
    // A final couple of passes over the curvature profile itself, to merge any residual ripple.
    for _ in 0..2 {
        let raw = c.clone();
        for i in 1..n - 1 {
            c[i] = raw[i - 1] * 0.25 + raw[i] * 0.5 + raw[i + 1] * 0.25;
        }
    }
    // Pin the (unmeasured) endpoints to their neighbours so the prominence walk in
    // `detect_curvature_extrema` doesn't see an artificial zero-valley at the array ends and
    // declare every near-end bump maximally prominent.
    if n >= 2 {
        c[0] = c[1];
        c[n - 1] = c[n - 2];
    }
    c
}

/// Detect curvature extrema — the apex of each distinct bend (e.g. the peaks and troughs of a
/// sine wave). A vertex is a *candidate* when its smoothed curvature is (a) a local maximum,
/// (b) above [`EXTREMA_FLOOR`] (so pointer jitter on a near-straight run is ignored), and (c)
/// topographically *prominent*: walking outward to the next higher curvature peak (or an
/// endpoint), the curvature must rise above its higher-side valley by at least `min_prominence`
/// radians. Because curvature is measured over a wide support, a real bend (which keeps turning
/// the same way across the span) towers over the residual ripple left by hand jitter, so the
/// prominence cleanly separates the two — and a constant-radius arc, whose curvature is a flat
/// plateau, yields no prominent apex at all.
///
/// Candidates are then thinned by prominence-ranked non-max suppression: the most prominent apex
/// wins, and any candidate within `min_spacing` samples of an already-kept one is dropped. That
/// is what guarantees *one anchor per bend* — a wobbly hand-drawn turn produces several nearby
/// candidates, but only its strongest survives. Endpoints and their immediate neighbours are
/// never flagged (the endpoints are already anchors).
fn detect_curvature_extrema(pts: &[Vec2], min_prominence: f32, min_spacing: usize) -> Vec<bool> {
    let n = pts.len();
    let mut mask = vec![false; n];
    if n < 5 {
        return mask;
    }
    let c = curvature_magnitude(pts);
    const K: usize = 2;
    // (idx, prominence) for every apex passing the local-max + floor + prominence gates.
    let mut cands: Vec<(usize, f32)> = Vec::new();
    for i in 2..n - 2 {
        if c[i] < EXTREMA_FLOOR {
            continue;
        }
        // (a) local maximum within ±K.
        let lo = i.saturating_sub(K);
        let hi = (i + K).min(n - 1);
        if !(lo..=hi).all(|j| c[j] <= c[i] + 1e-6) {
            continue;
        }
        // (c) prominence: the deepest valley reached before curvature climbs past this apex
        //     again, on each side. The controlling col is the *higher* of the two valleys.
        let mut left_min = c[i];
        let mut j = i;
        while j > 0 {
            j -= 1;
            if c[j] > c[i] + 1e-6 {
                break;
            }
            left_min = left_min.min(c[j]);
        }
        let mut right_min = c[i];
        let mut j = i;
        while j < n - 1 {
            j += 1;
            if c[j] > c[i] + 1e-6 {
                break;
            }
            right_min = right_min.min(c[j]);
        }
        let col = left_min.max(right_min);
        if c[i] - col >= min_prominence {
            cands.push((i, c[i] - col));
        }
    }

    if cands.is_empty() {
        return mask;
    }

    // Collapse each connected curvature *hump* to a single anchor at its centre. A rounded bend
    // (constant curvature across the turn) smooths into a flat-topped plateau, so the gate above
    // marks a whole run of equally-prominent maxima across it; emitting each would place two
    // anchors straddling the apex instead of one on it. `cands` is in ascending index order, so
    // we walk it and merge neighbours that sit on the same hump — defined as the curvature between
    // them never dipping by `min_prominence` (a shallow saddle). A genuinely separate bend (deep
    // saddle, e.g. a sine's peak and the next trough, where curvature falls to ~0 between them) is
    // left as its own hump. Each cluster collapses to the midpoint of its span — the apex.
    let mut clusters: Vec<(usize, f32)> = Vec::new();
    let mut i = 0;
    while i < cands.len() {
        let start = i;
        let mut best = cands[i].1;
        while i + 1 < cands.len() {
            let (a, b) = (cands[i].0, cands[i + 1].0);
            let saddle = c[a..=b].iter().copied().fold(f32::INFINITY, f32::min);
            if c[a].min(c[b]) - saddle < min_prominence {
                i += 1; // same hump — keep extending the cluster
                best = best.max(cands[i].1);
            } else {
                break; // a real valley separates them — distinct bends
            }
        }
        clusters.push(((cands[start].0 + cands[i].0) / 2, best));
        i += 1;
    }

    // Prominence-ranked NMS across the (already apex-centred) hump clusters: keep the boldest,
    // drop any whose centre is within `min_spacing` of a kept one.
    clusters.sort_by(|a, b| b.1.total_cmp(&a.1));
    let spacing = min_spacing.max(1);
    let mut kept: Vec<usize> = Vec::new();
    for (idx, _) in clusters {
        if kept.iter().all(|&k| idx.abs_diff(k) >= spacing) {
            kept.push(idx);
            mask[idx] = true;
        }
    }
    mask
}

/// Ordered breakpoint (anchor) indices along the resampled polyline: the two endpoints,
/// every corner, every pinned curvature `extrema`, and the adaptive interior splits. Within
/// each remaining span a window grows while accumulated turning stays under `max_turn`, then is
/// shrunk until a single cubic fits within `tol`. `max_window` caps a window at that many
/// samples even on a perfectly straight run (where neither the turning budget nor the tolerance
/// would ever close it) — used by the incremental fitter to keep the live open window bounded;
/// pass `usize::MAX` to disable.
///
/// Corners and curvature extrema are both *hard* splits, but they differ downstream: corners
/// keep independent tangents (stay sharp), while extrema are smooth joins (the caller leaves
/// their `corner` flag false, so `smooth_interior_joins` makes them C1). Pinning extrema biases
/// the anchors toward the apex of each bend; relaxing `max_turn`/`tol` (via the params'
/// `smoothness`) then stops the budget inserting extra anchors between them.
fn adaptive_breakpoints(
    pts: &[Vec2],
    turn: &[f32],
    corners: &[bool],
    extrema: &[bool],
    max_turn: f32,
    tol: f32,
    max_window: usize,
) -> Vec<usize> {
    let n = pts.len();
    // Hard anchors that always split the path: endpoints + corners + curvature extrema. The
    // single ascending filter keeps the indices sorted and de-duplicated even when a vertex is
    // flagged as both a corner and an extremum.
    let mut hard = vec![0usize];
    hard.extend((1..n - 1).filter(|&i| corners[i] || extrema[i]));
    hard.push(n - 1);

    let mut bp = vec![0usize];
    for span in hard.windows(2) {
        let (lo, hi) = (span[0], span[1]);
        let mut start = lo;
        while start < hi {
            // 1) Grow by the turning budget (cheap; no fitting). Invariant: `acc` is the
            //    total turning of the window's interior vertices. The `end - start` guard
            //    force-splits an otherwise-unbounded straight run at `max_window` samples.
            let mut end = start + 1;
            let mut acc = 0.0f32;
            while end < hi && acc + turn[end] <= max_turn && end - start < max_window {
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

/// Least-squares fit of one cubic to a window of points, with fixed endpoints and
/// data-estimated end-tangent directions (Schneider). Returns the cubic and its max
/// at-parameter deviation. Centripetal parameterization plus one Newton reparameterization
/// pass keep the error low so each window can stretch as far as the tolerance allows.
fn fit_cubic_lsq(pts: &[Vec2]) -> (CubicBezier, f32) {
    let n = pts.len();
    if n < 3 {
        // Degenerate window: no interior to estimate tangents from. The with-tangents form
        // handles the <2 / ==2 cases; pass any directions (unused for the straight fallback).
        return fit_cubic_lsq_with_tangents(pts, Vec2::X, -Vec2::X);
    }
    fit_cubic_lsq_with_tangents(pts, start_tangent(pts), end_tangent(pts))
}

/// As [`fit_cubic_lsq`], but with the two end-tangent *directions* supplied by the caller
/// (`t_hat1` leaves `p0` forward; `t_hat2` leaves `p3` backward, per `solve_handles`). The
/// least-squares solve still chooses the handle *lengths* that best fit the samples. Used to
/// stitch neighbouring windows with a shared tangent at smooth joins — C1 by construction,
/// without the curve-bowing that rotating already-fitted handles caused.
fn fit_cubic_lsq_with_tangents(pts: &[Vec2], t_hat1: Vec2, t_hat2: Vec2) -> (CubicBezier, f32) {
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

    let mut u = centripetal_params(pts);
    let mut curve = solve_handles(pts, &u, p0, p3, t_hat1, t_hat2);
    reparameterize(pts, &curve, &mut u);
    curve = solve_handles(pts, &u, p0, p3, t_hat1, t_hat2);

    (curve, max_error(pts, &curve, &u))
}

/// Forward unit tangent of the polyline centred at `idx`, from a short symmetric chord. Used as
/// the *shared* direction at a smooth interior anchor so the two cubics meeting there are C1.
fn tangent_dir_at(pts: &[Vec2], idx: usize) -> Vec2 {
    let n = pts.len();
    if n < 2 {
        return Vec2::X;
    }
    let k = 3.min(idx).min(n - 1 - idx).max(1);
    safe_dir(pts[(idx + k).min(n - 1)] - pts[idx.saturating_sub(k)], Vec2::X)
}

/// Fit one cubic per `[bps[i], bps[i+1]]` window, made C1 at every smooth interior join: the two
/// cubits sharing a smooth anchor are fit with a single shared tangent *direction* there, so the
/// join is tangent-continuous while each window's handle *lengths* are still chosen by the
/// least-squares solve. Corners (per the `corner` mask) and the stroke endpoints use
/// window-local tangents so corners stay sharp. `start_override` / `end_override`, when set,
/// force the forward tangent direction at the first window's start / last window's end — the
/// incremental fitter uses them to stay C1 with already-committed geometry.
fn fit_windows_c1(
    pts: &[Vec2],
    bps: &[usize],
    corner: &[bool],
    start_override: Option<Vec2>,
    end_override: Option<Vec2>,
) -> Vec<CubicBezier> {
    let m = bps.len().saturating_sub(1);
    let mut segs = Vec::with_capacity(m);
    for i in 0..m {
        let (lo, hi) = (bps[i], bps[i + 1]);
        let win = &pts[lo..=hi];
        let is_corner = |idx: usize| corner.get(idx).copied().unwrap_or(false);

        // Start tangent (forward). First window: an override (committed boundary) wins, else the
        // stroke start uses a window-local tangent. A corner uses the window-local tangent (sharp);
        // a smooth interior anchor uses the shared centred direction (C1 with the previous window).
        let t1 = if i == 0 {
            start_override.unwrap_or_else(|| start_tangent(win))
        } else if is_corner(lo) {
            start_tangent(win)
        } else {
            tangent_dir_at(pts, lo)
        };

        // End tangent (backward, per `solve_handles`). Symmetric to the start.
        let t2 = if i == m - 1 {
            end_override.map(|d| -d).unwrap_or_else(|| end_tangent(win))
        } else if is_corner(hi) {
            end_tangent(win)
        } else {
            -tangent_dir_at(pts, hi)
        };

        segs.push(fit_cubic_lsq_with_tangents(win, t1, t2).0);
    }
    segs
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

// =====================================================================================
// Incremental adaptive fitting (live drawing)
// =====================================================================================
//
// [`fit_polyline_adaptive`] re-fits the *entire* polyline every call. Driven once per
// pointer move that is O(n) per move and O(n^2) over a gesture — the longer the stroke, the
// slower each move. [`IncrementalAdaptiveFit`] removes the quadratic by exploiting a property
// the batch fitter already has but throws away: the fit's data dependencies are *local*.
//
// Within a span between hard anchors the windows are grown greedily from the start, so once a
// window closes (the next one begins) its breakpoint is fixed by points local to it —
// appending samples at the cursor never moves it. Smooth joins are made C1 by fitting each
// window with a shared tangent direction at the anchor (see `fit_windows_c1`), computed from
// points local to that anchor, so a committed cubic is independent of everything far ahead. We
// therefore **commit** (freeze) every window except the last one near the cursor and only re-fit
// that short open tail. A `max_window` cap forces a commit on long straight/corner-free runs that
// would otherwise keep one window open forever.
//
// Commit protocol, so frozen geometry never has to change:
//   * the *last* window stays open (it ends at the moving cursor and re-fits every push);
//   * a window is frozen only once a later window exists, so its forward tangent is already
//     anchored by local points;
//   * at the boundary between committed and open the open side is fit with a `start_override`
//     tangent that leaves along the frozen cubic's outgoing direction, so the committed cubic is
//     never touched while the join stays C1. Corner boundaries are left independent.

/// Live, incremental counterpart to [`fit_polyline_adaptive`]. Feed raw pointer points one at
/// a time with [`Self::push_point`]; read the current fit with [`Self::skeleton`]. Per-point
/// cost is bounded by the open window length rather than the whole stroke.
pub struct IncrementalAdaptiveFit {
    params: AdaptiveFitParams,
    /// Uniform arc-length resample of the raw input so far (append-only).
    pts: Vec<Vec2>,
    /// Streaming resample cursor: last walked point (an emitted sample during the inner loop,
    /// then the last raw point) and arc length carried since the last *emitted* sample.
    walk: Vec2,
    carry: f32,
    started: bool,
    /// Latest raw point — used as a provisional tip so the open window reaches the cursor.
    cursor: Vec2,
    /// Frozen cubic per committed window.
    committed: Vec<CubicBezier>,
    /// Corner flag at the START anchor of each committed window (`committed_corner[0]` is the
    /// stroke start: always `false`).
    committed_corner: Vec<bool>,
    /// Open (not-yet-frozen) windows near the cursor, rebuilt every push.
    open: Vec<CubicBezier>,
    /// Corner flag at the START anchor of each open window (`open_corner[0]` is the
    /// committed/open boundary).
    open_corner: Vec<bool>,
    /// `pts` index of the boundary anchor (start of the first open window).
    committed_at: usize,
    /// Corner flag at `committed_at`; `open_corner[0]` mirrors it.
    boundary_corner: bool,
}

impl IncrementalAdaptiveFit {
    /// Cap (in resampled samples) on the open window, so per-push cost stays bounded even on a
    /// dead-straight drag that neither the turning budget nor the tolerance would ever split.
    /// At the zoom-aware resample step this is a few hundred px — far longer than any naturally
    /// curving window, so it only ever bites pathological straight runs.
    const MAX_WINDOW: usize = 128;

    /// Start a fit at the gesture's first point.
    pub fn new(params: AdaptiveFitParams, first: Vec2) -> Self {
        Self {
            params,
            pts: vec![first],
            walk: first,
            carry: 0.0,
            started: true,
            cursor: first,
            committed: Vec::new(),
            committed_corner: Vec::new(),
            open: Vec::new(),
            open_corner: Vec::new(),
            committed_at: 0,
            boundary_corner: false,
        }
    }

    fn step(&self) -> f32 {
        self.params.resample_step.max(0.5)
    }

    /// Feed the next raw pointer point. Streams it into the uniform resample, then re-fits and
    /// (where possible) freezes the open tail.
    pub fn push_point(&mut self, raw: Vec2) {
        self.cursor = raw;
        if !self.started {
            self.started = true;
            self.pts.push(raw);
            self.walk = raw;
            self.carry = 0.0;
        } else {
            self.resample_to(raw);
        }
        self.refit_open();
    }

    /// Streaming form of [`resample_uniform`]: emit uniform-arc-length samples up to `raw`,
    /// carrying the leftover length so spacing is continuous across calls.
    fn resample_to(&mut self, raw: Vec2) {
        let step = self.step();
        let mut seg = raw - self.walk;
        let mut seg_len = seg.length();
        while self.carry + seg_len >= step && seg_len > 1e-9 {
            let t = ((step - self.carry) / seg_len).clamp(0.0, 1.0);
            let np = self.walk + seg * t;
            self.pts.push(np);
            self.walk = np;
            seg = raw - self.walk;
            seg_len = seg.length();
            self.carry = 0.0;
        }
        self.carry += seg_len;
        self.walk = raw;
    }

    /// Re-fit the open region `pts[committed_at..]` (plus the provisional cursor tip), then
    /// freeze every window except the last.
    fn refit_open(&mut self) {
        let tol = self.params.effective_tolerance();
        let max_turn = self.params.effective_max_turn();
        let corner_turn = self.params.corner_turn.max(0.05);

        // Open input: the uncommitted resampled samples plus the live cursor as a provisional
        // endpoint, so the last window tracks the cursor instead of lagging by up to one step.
        let mut open_in: Vec<Vec2> = self.pts[self.committed_at..].to_vec();
        if open_in
            .last()
            .map_or(true, |&p| (p - self.cursor).length() > 1e-4)
        {
            open_in.push(self.cursor);
        }

        if open_in.len() < 2 {
            // Nothing to fit yet (single point). Leave the open region empty.
            self.open.clear();
            self.open_corner.clear();
            return;
        }

        let turn = self.params.budget_turning(&open_in);
        let corner = detect_corners(&open_in, corner_turn);
        let extrema = self.params.curvature_extrema(&open_in);
        let bps = adaptive_breakpoints(
            &open_in,
            &turn,
            &corner,
            &extrema,
            max_turn,
            tol,
            Self::MAX_WINDOW,
        );

        // One cubic per window, C1 across smooth interior joins by construction. At the committed
        // boundary, force the open side's start tangent to leave along the frozen cubic's outgoing
        // direction, so the join stays C1 without touching the committed cubic.
        let start_override = if self.boundary_corner {
            None
        } else {
            self.committed
                .last()
                .map(|last| safe_dir(open_in[0] - last.p2, Vec2::X))
        };
        let mut segs = fit_windows_c1(&open_in, &bps, &corner, start_override, None);
        if segs.is_empty() {
            self.open.clear();
            self.open_corner.clear();
            return;
        }

        // Per-window start-anchor corner flags (index 0 is the committed/open boundary).
        let mut seg_corner: Vec<bool> = Vec::with_capacity(segs.len());
        seg_corner.push(self.boundary_corner);
        for w in bps.windows(2).skip(1) {
            seg_corner.push(corner.get(w[0]).copied().unwrap_or(false));
        }

        // Freeze every window except the last (it still ends at the cursor). A window's end
        // breakpoint maps back to a real `pts` index (only the very last point — the cursor — is
        // provisional, and it is never an interior breakpoint).
        let base = self.committed_at;
        let freeze = segs.len() - 1;
        for i in 0..freeze {
            self.committed.push(segs[i]);
            self.committed_corner.push(seg_corner[i]);
            self.committed_at = base + bps[i + 1];
            self.boundary_corner = corner.get(bps[i + 1]).copied().unwrap_or(false);
        }

        self.open = segs.split_off(freeze);
        self.open_corner = vec![self.boundary_corner];
    }

    /// Assemble the current fit (committed + open windows) into a skeleton with per-anchor
    /// corner flags set, matching [`fit_polyline_adaptive`]'s output contract.
    pub fn skeleton(&self) -> BezierSkeleton {
        let mut segs = self.committed.clone();
        segs.extend_from_slice(&self.open);
        if segs.is_empty() {
            let p = self.pts.first().copied().unwrap_or(Vec2::ZERO);
            return BezierSkeleton::single(straight_cubic(p, p + Vec2::new(1.0, 0.0)));
        }

        let mut flags = self.committed_corner.clone();
        flags.extend_from_slice(&self.open_corner);
        flags.push(false); // end endpoint

        let mut sk = BezierSkeleton::from_segments(segs, false);
        for (j, &f) in flags.iter().enumerate() {
            if let Some(meta) = sk.anchors.get_mut(j) {
                meta.corner = f;
            }
        }
        sk
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

    // --- Incremental adaptive fitter ------------------------------------------------

    /// Max distance from any input point to the fitted skeleton. Each segment is sampled at
    /// ~0.5px along its chord so the discrete-sample spacing never dominates the true geometric
    /// deviation we are trying to measure.
    fn max_dev_to_skeleton(sk: &BezierSkeleton, pts: &[Vec2]) -> f32 {
        let mut samples = Vec::new();
        for seg in &sk.segments {
            let chord = (seg.p3 - seg.p0).length();
            let n = ((chord * 2.0).ceil() as usize).clamp(16, 4096);
            for k in 0..=n {
                samples.push(seg.point(k as f32 / n as f32));
            }
        }
        pts.iter()
            .map(|&p| {
                samples
                    .iter()
                    .map(|&s| (s - p).length())
                    .fold(f32::INFINITY, f32::min)
            })
            .fold(0.0f32, f32::max)
    }

    /// A circular arc — high, sustained curvature so the fitter closes (and commits) several
    /// windows over the course of the gesture.
    fn arc(n: usize, sweep: f32) -> Vec<Vec2> {
        (0..n)
            .map(|i| {
                let a = i as f32 / (n - 1) as f32 * sweep;
                Vec2::new(80.0 * a.cos(), 80.0 * a.sin())
            })
            .collect()
    }

    fn feed(points: &[Vec2], params: &AdaptiveFitParams) -> IncrementalAdaptiveFit {
        let mut fit = IncrementalAdaptiveFit::new(*params, points[0]);
        for &p in &points[1..] {
            fit.push_point(p);
        }
        fit
    }

    /// A sine wave of `periods` full cycles spanning `len` px in x with amplitude `amp`. Its
    /// curvature peaks at each crest/trough and falls to zero at the zero-crossings — the canonical
    /// case for the curvature-extrema anchor bias.
    fn sine_wave(periods: f32, n: usize, amp: f32, len: f32) -> Vec<Vec2> {
        (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1) as f32;
                let x = t * len;
                let y = amp * (t * periods * std::f32::consts::TAU).sin();
                Vec2::new(x, y)
            })
            .collect()
    }

    /// The x of the k-th sine extremum (crest or trough) for `periods` cycles over `len`: the
    /// quarter-phase offsets where `sin` reaches ±1.
    fn sine_extremum_x(k: usize, periods: f32, len: f32) -> f32 {
        // sin peaks at phase (k + 0.5)·π for k = 0, 1, 2, ... → t = (k + 0.5) / (2·periods).
        (k as f32 + 0.5) / (2.0 * periods) * len
    }

    #[test]
    fn soft_fit_pins_anchors_at_sine_peaks_and_troughs() {
        // Two full periods (300px each), steep enough that each crest/trough is a clear curvature
        // extremum: four interior extrema in all.
        let periods = 2.0;
        let len = 600.0;
        let amp = 60.0;
        let pts = sine_wave(periods, 400, amp, len);

        let strict = fit_polyline_adaptive(
            &pts,
            &AdaptiveFitParams {
                smoothness: 0.0,
                ..AdaptiveFitParams::default()
            },
        );
        let soft = fit_polyline_adaptive(
            &pts,
            &AdaptiveFitParams {
                smoothness: 1.0,
                ..AdaptiveFitParams::default()
            },
        );

        // Softening never adds anchors: it pins the extrema and relaxes the budget that would
        // otherwise scatter extra splits along each arc.
        assert!(
            soft.anchor_count() <= strict.anchor_count(),
            "soft fit ({}) should be no denser than strict ({})",
            soft.anchor_count(),
            strict.anchor_count()
        );

        let extrema_x: Vec<f32> = (0..4).map(|k| sine_extremum_x(k, periods, len)).collect();

        // (1) Every crest/trough has a soft-fit anchor sitting on it (within a few px of the apex).
        for (k, &ex) in extrema_x.iter().enumerate() {
            let near =
                (0..soft.anchor_count()).any(|j| (soft.anchor_position(j).x - ex).abs() < 12.0);
            assert!(near, "expected a soft-fit anchor near sine extremum #{k} at x≈{ex}");
        }

        // (2) ...and *nowhere else*: every interior soft anchor lies on a crest/trough, so the fit
        //     is exactly the user's spec — an anchor at the start, at each trough and peak, and
        //     between them nothing. (The two endpoints are excluded from this interior check.)
        for j in 1..soft.anchor_count() - 1 {
            let ax = soft.anchor_position(j).x;
            let on_extremum = extrema_x.iter().any(|&ex| (ax - ex).abs() < 12.0);
            assert!(
                on_extremum,
                "interior soft anchor #{j} at x≈{ax} is not on any sine extremum"
            );
        }

        // (3) And the curve still hugs the input — pinning anchors at the apexes does not sacrifice fit.
        assert!(
            max_dev_to_skeleton(&soft, &pts) < 6.0,
            "soft fit deviates too far: {}",
            max_dev_to_skeleton(&soft, &pts)
        );
    }

    /// Deterministic pseudo-random in `[0,1)` (the classic GLSL hash), so noise tests are
    /// reproducible without an RNG dependency.
    fn hash01(i: usize, salt: usize) -> f32 {
        let x = ((i as f32 + salt as f32 * 0.123) * 12.9898).sin() * 43758.547;
        x - x.floor()
    }

    /// A circular arc (constant true curvature) perturbed by per-sample noise of amplitude
    /// `noise` px — a stand-in for a hand-drawn curve, where every sample wiggles.
    fn noisy_arc(n: usize, sweep: f32, radius: f32, noise: f32) -> Vec<Vec2> {
        (0..n)
            .map(|i| {
                let a = i as f32 / (n - 1) as f32 * sweep;
                let base = Vec2::new(radius * a.cos(), radius * a.sin());
                let dx = (hash01(i, 1) - 0.5) * 2.0 * noise;
                let dy = (hash01(i, 2) - 0.5) * 2.0 * noise;
                base + Vec2::new(dx, dy)
            })
            .collect()
    }

    #[test]
    fn soft_fit_gives_one_anchor_per_bend_not_a_cluster() {
        // The reported regression: drawing through a curvature change sprinkled a cluster of
        // anchors. A noisy circular arc is the worst case — constant true curvature (no genuine
        // apex anywhere), so a correct soft fit must NOT pin a row of extrema along it; jitter
        // alone used to spawn one every few samples.
        let pts = noisy_arc(220, 1.4 * std::f32::consts::PI, 130.0, 2.5);
        let soft = fit_polyline_adaptive(
            &pts,
            &AdaptiveFitParams {
                smoothness: 1.0,
                ..AdaptiveFitParams::default()
            },
        );
        // A ~250° arc needs only a handful of cubics; anything approaching the input density
        // (hundreds of samples) means the jitter is leaking through as false extrema.
        assert!(
            soft.anchor_count() <= 8,
            "noisy arc over-segmented into {} anchors (clustering regression)",
            soft.anchor_count()
        );
        // And it still tracks the arc through the noise (a soft fit rides over jitter, so the
        // bound is loose — this only guards against the fit wandering off the arc entirely).
        assert!(
            max_dev_to_skeleton(&soft, &pts) < 12.0,
            "soft fit of noisy arc deviates too far: {}",
            max_dev_to_skeleton(&soft, &pts)
        );
    }

    #[test]
    fn soft_fit_survives_a_noisy_sine() {
        // The user's actual case: a curvy stroke with hand jitter. Each crest/trough should get
        // ONE anchor, not a knot of them.
        let periods = 2.0;
        let len = 600.0;
        let amp = 60.0;
        let clean = sine_wave(periods, 400, amp, len);
        let pts: Vec<Vec2> = clean
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                p + Vec2::new(
                    (hash01(i, 3) - 0.5) * 2.0 * 1.5,
                    (hash01(i, 4) - 0.5) * 2.0 * 1.5,
                )
            })
            .collect();
        let soft = fit_polyline_adaptive(
            &pts,
            &AdaptiveFitParams {
                smoothness: 1.0,
                ..AdaptiveFitParams::default()
            },
        );
        // Four interior extrema + two endpoints, give or take one stray from the noise — not the
        // dozens a clustering bug would produce.
        assert!(
            soft.anchor_count() <= 8,
            "noisy sine over-segmented into {} anchors",
            soft.anchor_count()
        );
        // Each crest/trough still has an anchor near it.
        for k in 0..4 {
            let ex = sine_extremum_x(k, periods, len);
            let near =
                (0..soft.anchor_count()).any(|j| (soft.anchor_position(j).x - ex).abs() < 18.0);
            assert!(near, "expected an anchor near noisy-sine extremum #{k} at x≈{ex}");
        }
    }

    #[test]
    fn hook_at_default_smoothness_is_sparse() {
        // The screenshot case: a single hook/U-turn drawn with hand jitter. The whole bend should
        // collapse to roughly one anchor at the apex plus the two endpoints — a small handful,
        // not the dozen the budget produced when it was integrating jitter. Tested at the *default*
        // tool smoothness (0.5), since that is what the user draws with out of the box.
        let pts = noisy_arc(200, std::f32::consts::PI, 130.0, 2.5); // a 180° hook
        let fit = fit_polyline_adaptive(
            &pts,
            &AdaptiveFitParams {
                smoothness: 0.5,
                ..AdaptiveFitParams::default()
            },
        );
        assert!(
            fit.anchor_count() <= 5,
            "hook over-segmented into {} anchors at default smoothness",
            fit.anchor_count()
        );
        // It still follows the hook (loose, since this is a softened fit).
        assert!(
            max_dev_to_skeleton(&fit, &pts) < 10.0,
            "hook fit wanders off: {}",
            max_dev_to_skeleton(&fit, &pts)
        );
    }

    #[test]
    fn rounded_bend_gets_one_centred_apex_anchor() {
        // A rounded corner: a horizontal lead-in, a quarter-circle turn, then a vertical lead-out.
        // The curvature is a flat-topped plateau across the arc, which used to surface *two*
        // anchors at the plateau's shoulders, straddling the apex. The hump should now collapse to
        // a single anchor at the apex (the arc's 45° midpoint).
        let r = 90.0;
        let mut pts = Vec::new();
        // Lead-in: horizontal, approaching the arc start at (0, r).
        for i in 0..=40 {
            pts.push(Vec2::new(-160.0 + i as f32 * 4.0, r));
        }
        // Quarter arc, centre (0,0): from angle 90° down to 0°, apex at 45°.
        let arc_apex = {
            let mut apex = Vec2::ZERO;
            for i in 0..=40 {
                let a = std::f32::consts::FRAC_PI_2 * (1.0 - i as f32 / 40.0);
                let p = Vec2::new(r * a.cos(), r * a.sin());
                if i == 20 {
                    apex = p;
                }
                pts.push(p);
            }
            apex
        };
        // Lead-out: vertical, descending from the arc end at (r, 0).
        for i in 1..=40 {
            pts.push(Vec2::new(r, -(i as f32 * 4.0)));
        }

        let fit = fit_polyline_adaptive(
            &pts,
            &AdaptiveFitParams {
                smoothness: 0.6,
                ..AdaptiveFitParams::default()
            },
        );
        // Exactly one anchor should sit in the apex neighbourhood — not a straddling pair.
        let near_apex: Vec<Vec2> = (0..fit.anchor_count())
            .map(|j| fit.anchor_position(j))
            .filter(|p| (*p - arc_apex).length() < r * 0.6)
            .collect();
        assert_eq!(
            near_apex.len(),
            1,
            "expected a single apex anchor, got {} near the bend: {:?}",
            near_apex.len(),
            near_apex
        );
        // And it should be close to the true apex, not off on a shoulder.
        assert!(
            (near_apex[0] - arc_apex).length() < 22.0,
            "apex anchor landed {:.1}px from the apex",
            (near_apex[0] - arc_apex).length()
        );
    }

    /// A smooth S-curve, no hard corners.
    fn s_curve() -> Vec<Vec2> {
        (0..=120)
            .map(|i| {
                let x = i as f32 * 2.0;
                let y = 60.0 * (x * 0.012).sin();
                Vec2::new(x, y)
            })
            .collect()
    }

    #[test]
    fn incremental_tracks_the_input_like_the_batch_fitter() {
        let pts = s_curve();
        let params = AdaptiveFitParams::default();
        let inc = feed(&pts, &params).skeleton();
        let batch = fit_polyline_adaptive(&pts, &params);

        // Both should hug the input within a couple of px of the tolerance.
        assert!(
            max_dev_to_skeleton(&inc, &pts) < params.tolerance + 3.0,
            "incremental deviation too high: {}",
            max_dev_to_skeleton(&inc, &pts)
        );
        // Endpoints land on the gesture's ends.
        assert!((inc.frame_at_arc_t(0.0).position - pts[0]).length() < 2.0);
        assert!((inc.frame_at_arc_t(1.0).position - *pts.last().unwrap()).length() < 2.0);
        // Anchor counts are in the same ballpark (incremental may add a couple from the cap).
        let (ia, ba) = (inc.anchor_count(), batch.anchor_count());
        assert!(
            ia.abs_diff(ba) <= 3,
            "incremental anchors {ia} vs batch {ba} diverge too much"
        );
    }

    #[test]
    fn committed_geometry_is_stable_as_more_points_arrive() {
        // Feeding a prefix then more points must not move the early (committed) segments. A
        // 1.5-turn arc curves enough that windows close (and commit) well before the prefix ends.
        let pts = arc(240, 3.0 * std::f32::consts::PI);
        let params = AdaptiveFitParams::default();

        let mut fit = IncrementalAdaptiveFit::new(params, pts[0]);
        for &p in &pts[1..140] {
            fit.push_point(p);
        }
        let committed_prefix = fit.committed.clone();
        assert!(
            !committed_prefix.is_empty(),
            "expected some committed windows partway through a long curved gesture"
        );

        for &p in &pts[140..] {
            fit.push_point(p);
        }
        // The previously-committed windows are a byte-identical prefix of the final committed set.
        for (old, new) in committed_prefix.iter().zip(&fit.committed) {
            assert_eq!(old.p0, new.p0);
            assert_eq!(old.p1, new.p1);
            assert_eq!(old.p2, new.p2);
            assert_eq!(old.p3, new.p3);
        }
    }

    #[test]
    fn incremental_preserves_a_sharp_corner() {
        // An L: horizontal then vertical, with a hard 90° corner at (100, 0).
        let mut pts = Vec::new();
        for i in 0..=50 {
            pts.push(Vec2::new(i as f32 * 2.0, 0.0));
        }
        for i in 1..=50 {
            pts.push(Vec2::new(100.0, i as f32 * 2.0));
        }
        let sk = feed(&pts, &AdaptiveFitParams::default()).skeleton();
        // The corner near (100,0) survives as a flagged anchor.
        let corner = (0..sk.anchor_count()).any(|j| {
            sk.anchors.get(j).map(|m| m.corner).unwrap_or(false)
                && (sk.anchor_position(j) - Vec2::new(100.0, 0.0)).length() < 6.0
        });
        assert!(corner, "expected a corner anchor near the L's elbow");
    }

    #[test]
    fn straight_run_stays_bounded_via_the_window_cap() {
        // A long dead-straight drag has no corners and never exceeds tolerance, so without the
        // cap it would keep one window open across the whole stroke. The cap must split it, so
        // committed windows accumulate (proving the open region stays bounded).
        let pts: Vec<Vec2> = (0..=2000).map(|i| Vec2::new(i as f32, 0.0)).collect();
        let params = AdaptiveFitParams::default();
        let fit = feed(&pts, &params);
        assert!(
            !fit.committed.is_empty(),
            "the window cap should have forced commits on a long straight run"
        );
        // And the fit is still essentially the straight line.
        assert!(max_dev_to_skeleton(&fit.skeleton(), &pts) < 2.0);
    }

    #[test]
    fn incremental_handles_degenerate_inputs() {
        let params = AdaptiveFitParams::default();
        // One point: a usable (if tiny) skeleton, never a panic.
        let one = IncrementalAdaptiveFit::new(params, Vec2::new(5.0, 5.0)).skeleton();
        assert_eq!(one.segments.len(), 1);
        // A stationary pointer (repeated identical points) stays degenerate-safe.
        let mut still = IncrementalAdaptiveFit::new(params, Vec2::new(5.0, 5.0));
        for _ in 0..10 {
            still.push_point(Vec2::new(5.0, 5.0));
        }
        assert!(!still.skeleton().segments.is_empty());
    }
}
