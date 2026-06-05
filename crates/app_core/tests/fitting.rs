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
use app_core::{fit_polyline_to_skeleton, simplify_rdp};

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
