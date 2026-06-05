//! Acceptance tests for freehand-stroke -> Bezier skeleton fitting (the brush tool).
//!
//!   1. dense samples from a KNOWN cubic, fit back, stays within a few px;
//!   2. a straight line -> low error, few segments;
//!   3. a right-angle polyline -> RDP keeps the corner (>= 2 segments);
//!   4. `Document::add_freehand_stroke` produces a stroke with splats.

use app_core::bezier::{BezierSkeleton, CubicBezier};
use app_core::brush::BrushModel;
use app_core::document::Document;
use app_core::math::Vec2;
use app_core::{fit_polyline_adaptive, fit_polyline_to_skeleton, simplify_rdp, AdaptiveFitParams};

/// Densely sample a polyline along a single cubic at `n` even native-parameter steps.
fn sample_cubic(curve: &CubicBezier, n: usize) -> Vec<Vec2> {
    (0..=n).map(|i| curve.point(i as f32 / n as f32)).collect()
}

/// For each input point, the distance to the nearest point on a finely-sampled fit.
fn max_tracking_error(fit: &BezierSkeleton, originals: &[Vec2]) -> f32 {
    // Pre-sample the fitted skeleton densely in arc-length space.
    let samples: Vec<Vec2> = (0..=2000)
        .map(|i| fit.frame_at_arc_t(i as f32 / 2000.0).position)
        .collect();
    let mut worst = 0.0f32;
    for &p in originals {
        let nearest = samples
            .iter()
            .map(|s| (*s - p).length())
            .fold(f32::INFINITY, f32::min);
        worst = worst.max(nearest);
    }
    worst
}

// --- 1. Fit back a known cubic ----------------------------------------------------

#[test]
fn fits_known_cubic_within_a_few_px() {
    let curve = CubicBezier::new(
        Vec2::new(100.0, 200.0),
        Vec2::new(180.0, 40.0),
        Vec2::new(320.0, 360.0),
        Vec2::new(420.0, 200.0),
    );
    let pts = sample_cubic(&curve, 200);

    let fit = fit_polyline_to_skeleton(&pts, 1.5);

    // Endpoints are preserved exactly (RDP keeps first/last; light smoothing pins them).
    assert!((fit.frame_at_arc_t(0.0).position - curve.p0).length() < 1.5);
    assert!((fit.frame_at_arc_t(1.0).position - curve.p3).length() < 1.5);

    // The whole fitted curve tracks every original sample closely.
    let err = max_tracking_error(&fit, &pts);
    assert!(err < 4.0, "fit should track the source cubic; max error = {err}");

    assert!(!fit.segments.is_empty());
}

// --- 2. Straight line -> low error, few segments ----------------------------------

#[test]
fn fits_straight_line_with_few_segments() {
    let a = Vec2::new(0.0, 50.0);
    let b = Vec2::new(500.0, 50.0);
    let pts: Vec<Vec2> = (0..=100)
        .map(|i| a.lerp(b, i as f32 / 100.0))
        .collect();

    let fit = fit_polyline_to_skeleton(&pts, 1.0);

    // A straight line collapses to a single cubic segment.
    assert_eq!(fit.segments.len(), 1, "a straight line should need one segment");

    let err = max_tracking_error(&fit, &pts);
    assert!(err < 1.0, "straight-line fit error should be tiny; got {err}");
}

// --- 3. Right-angle polyline -> RDP keeps the corner ------------------------------

#[test]
fn right_angle_polyline_keeps_corner() {
    // An "L": go right along the x-axis, then up. The corner is at (100, 0).
    let mut pts: Vec<Vec2> = Vec::new();
    for i in 0..=20 {
        pts.push(Vec2::new(i as f32 * 5.0, 0.0)); // (0,0) -> (100,0)
    }
    for i in 1..=20 {
        pts.push(Vec2::new(100.0, i as f32 * 5.0)); // (100,0) -> (100,100)
    }

    // RDP alone keeps the corner.
    let anchors = simplify_rdp(&pts, 1.0);
    assert!(
        anchors.iter().any(|p| (*p - Vec2::new(100.0, 0.0)).length() < 1e-3),
        "RDP must retain the right-angle corner"
    );

    let fit = fit_polyline_to_skeleton(&pts, 1.0);
    assert!(
        fit.segments.len() >= 2,
        "right-angle path needs >= 2 segments, got {}",
        fit.segments.len()
    );

    // The fit still passes near the corner.
    let err = max_tracking_error(&fit, &pts);
    assert!(err < 8.0, "L-shape fit error too large: {err}");
}

// --- 4. add_freehand_stroke produces a stroke with splats -------------------------

#[test]
fn add_freehand_stroke_produces_splats() {
    let mut doc = Document::new();
    let layer = doc.add_layer("Brush layer");

    // A gentle freehand arc.
    let pts: Vec<Vec2> = (0..=80)
        .map(|i| {
            let t = i as f32 / 80.0;
            Vec2::new(50.0 + t * 400.0, 200.0 - (t * std::f32::consts::PI).sin() * 120.0)
        })
        .collect();

    let sid = doc.add_freehand_stroke(layer, &pts, BrushModel::default(), 2.0);

    let stroke = doc.stroke(sid).expect("stroke exists");
    assert!(!stroke.skeleton.segments.is_empty());
    assert!(stroke.splats.len() > 10, "freehand stroke should have splats");
    assert!(stroke.splats.iter().all(|s| s.center.is_finite()));

    // Registered on the layer.
    assert_eq!(doc.layers[0].stroke_ids, vec![sid]);
    assert!(doc.splat_count() > 0);
}

// --- 5. Curvature-adaptive fitter -------------------------------------------------

/// Anchor count of the RDP path (for the minimal-anchors comparison below).
fn rdp_segments(pts: &[Vec2], tol: f32) -> usize {
    simplify_rdp(pts, tol).len() - 1
}

#[test]
fn adaptive_collapses_a_straight_line_to_one_segment() {
    let a = Vec2::new(0.0, 50.0);
    let b = Vec2::new(500.0, 50.0);
    let pts: Vec<Vec2> = (0..=100).map(|i| a.lerp(b, i as f32 / 100.0)).collect();

    let sk = fit_polyline_adaptive(&pts, &AdaptiveFitParams::default());
    assert_eq!(sk.segments.len(), 1, "a straight line needs exactly one window");
    assert!(max_tracking_error(&sk, &pts) < 1.0);
    // Endpoints are never flagged as corners.
    assert!(sk.anchors.iter().all(|m| !m.corner));
}

#[test]
fn adaptive_tracks_a_known_cubic_with_minimal_anchors() {
    let curve = CubicBezier::new(
        Vec2::new(100.0, 200.0),
        Vec2::new(180.0, 40.0),
        Vec2::new(320.0, 360.0),
        Vec2::new(420.0, 200.0),
    );
    let pts = sample_cubic(&curve, 200);

    let params = AdaptiveFitParams::default();
    let sk = fit_polyline_adaptive(&pts, &params);

    // Tracks the source closely, with endpoints preserved.
    assert!((sk.frame_at_arc_t(0.0).position - curve.p0).length() < 2.0);
    assert!((sk.frame_at_arc_t(1.0).position - curve.p3).length() < 2.0);
    let err = max_tracking_error(&sk, &pts);
    assert!(err < params.tolerance + 2.5, "adaptive fit error = {err}");

    // The least-squares window fit needs no more anchors than RDP for the same tolerance —
    // the "minimal anchors" property — and only a handful in absolute terms.
    let adaptive = sk.segments.len();
    let rdp = rdp_segments(&pts, params.tolerance);
    assert!(
        adaptive <= rdp,
        "adaptive used more windows ({adaptive}) than RDP anchors ({rdp})"
    );
    assert!(adaptive <= 3, "a single smooth cubic should need very few windows, got {adaptive}");
}

#[test]
fn adaptive_preserves_a_sharp_corner_as_an_independent_anchor() {
    // An "L": right along x, then up. The corner is at (100, 0).
    let mut pts: Vec<Vec2> = Vec::new();
    for i in 0..=40 {
        pts.push(Vec2::new(i as f32 * 2.5, 0.0)); // (0,0) -> (100,0)
    }
    for i in 1..=40 {
        pts.push(Vec2::new(100.0, i as f32 * 2.5)); // (100,0) -> (100,100)
    }

    let sk = fit_polyline_adaptive(&pts, &AdaptiveFitParams::default());

    assert!(sk.segments.len() >= 2, "an L needs >= 2 windows");
    // A corner anchor sits at the bend and is flagged (so the edit tool keeps it sharp).
    let corner_idx = (0..sk.anchor_count())
        .find(|&j| (sk.anchor_position(j) - Vec2::new(100.0, 0.0)).length() < 6.0);
    let corner_idx = corner_idx.expect("an anchor should land on the bend");
    assert!(sk.anchors[corner_idx].corner, "the bend anchor must be flagged a corner");
    // And the fit still passes near the corner.
    assert!(max_tracking_error(&sk, &pts) < 8.0);
}

#[test]
fn adaptive_window_is_long_when_straight_and_short_when_curved() {
    // A long straight run, then a semicircle. The straight run should be covered by a single
    // long window (its terminating anchor lands near the bend); the semicircle turns a full
    // π — more than one window's turning budget — so it must be split into several short
    // windows. That contrast is exactly the curvature-adaptive behaviour.
    let mut pts: Vec<Vec2> = Vec::new();
    for i in 0..=80 {
        pts.push(Vec2::new(i as f32 * 2.5, 0.0)); // (0,0) -> (200,0), straight
    }
    let r = 40.0;
    let center = Vec2::new(200.0, r);
    for i in 1..=120 {
        // Sweep π radians (a half turn), so the budget forces multiple windows in the arc.
        let ang = -std::f32::consts::FRAC_PI_2 + (i as f32 / 120.0) * std::f32::consts::PI;
        pts.push(center + Vec2::new(ang.cos() * r, ang.sin() * r));
    }

    let sk = fit_polyline_adaptive(&pts, &AdaptiveFitParams::default());

    // The straight run is one window: the second anchor is far down the straight part.
    assert!(
        sk.anchor_position(1).x >= 150.0,
        "straight run should be a single long window; 2nd anchor x = {}",
        sk.anchor_position(1).x
    );
    // The curve forces extra anchors -> more total windows than a pure straight line.
    assert!(sk.segments.len() >= 3, "tight curve should add windows, got {}", sk.segments.len());
    assert!(max_tracking_error(&sk, &pts) < 4.0);
}

#[test]
fn adaptive_handles_degenerate_inputs() {
    // Single point and two points must still yield a usable (non-empty) skeleton.
    let one = fit_polyline_adaptive(&[Vec2::new(5.0, 5.0)], &AdaptiveFitParams::default());
    assert_eq!(one.segments.len(), 1);
    assert!(one.total_length() > 0.0);

    let two = fit_polyline_adaptive(
        &[Vec2::new(0.0, 0.0), Vec2::new(100.0, 0.0)],
        &AdaptiveFitParams::default(),
    );
    assert!(!two.segments.is_empty());
    assert!((two.frame_at_arc_t(1.0).position - Vec2::new(100.0, 0.0)).length() < 1.0);
}
