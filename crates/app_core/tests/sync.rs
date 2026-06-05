//! End-to-end acceptance tests for the bidirectional splat <-> curve synchronization.
//!
//! These mirror the "Core acceptance tests" (spec section 20) and the Phase-5
//! bidirectional behaviors:
//!   A. curve-to-splat forward sync (handles move -> splats follow, residuals survive)
//!   B. coherent central drag        -> structural skeleton update
//!   C. incoherent scatter           -> residual-only, skeleton untouched
//!   D. width-like edge push         -> centerline does not shift
//!   E. whole-stroke translate       -> control points translate

use app_core::bezier::{BezierSkeleton, ControlPointRef, CubicBezier};
use app_core::brush::BrushModel;
use app_core::document::Document;
use app_core::ids::StrokeId;
use app_core::math::Vec2;
use app_core::solver::{
    apply_splat_edits_bidirectional, sculpt_move_splats, FitOutcome, SplatEdit,
};
use app_core::splat::SplatRole;
use app_core::Rng;

/// A long, gently-curved stroke with a few hundred splats — enough sample density for
/// the coherence statistics to be meaningful.
fn make_doc() -> (Document, StrokeId) {
    let mut doc = Document::new();
    let layer = doc.add_layer("Layer 1");
    let curve = CubicBezier::new(
        Vec2::new(100.0, 300.0),
        Vec2::new(250.0, 250.0),
        Vec2::new(450.0, 250.0),
        Vec2::new(600.0, 300.0),
    );
    let sid = doc.add_stroke(layer, BezierSkeleton::single(curve), BrushModel::default());
    (doc, sid)
}

fn control_points(doc: &Document, sid: StrokeId) -> [Vec2; 4] {
    let seg = doc.stroke(sid).unwrap().skeleton.segments[0];
    [seg.p0, seg.p1, seg.p2, seg.p3]
}

fn max_control_movement(a: &[Vec2; 4], b: &[Vec2; 4]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x - *y).length())
        .fold(0.0, f32::max)
}

// --- Test A: curve-to-splat forward sync -----------------------------------------

#[test]
fn test_a_curve_to_splat_with_residual_survival() {
    let (mut doc, sid) = make_doc();

    // Give one splat a deliberate local residual ("paint detail").
    let (probe_id, residual) = {
        let stroke = doc.stroke_mut(sid).unwrap();
        let idx = stroke.splats.len() / 2;
        let s = &mut stroke.splats[idx];
        s.residual_local = Vec2::new(4.0, 7.0);
        (s.id, s.residual_local)
    };
    doc.stroke_mut(sid).unwrap().update_world_cache();

    let before_centers: Vec<Vec2> =
        doc.stroke(sid).unwrap().splats.iter().map(|s| s.center).collect();

    // Move control point P1 (bend a handle), then resync forward.
    {
        let stroke = doc.stroke_mut(sid).unwrap();
        stroke.skeleton.segments[0].p1 += Vec2::new(0.0, -120.0);
        stroke.skeleton.rebuild_arc_length_table();
        stroke.update_world_cache();
    }

    let stroke = doc.stroke(sid).unwrap();

    // (1) Splats actually moved.
    let moved = stroke
        .splats
        .iter()
        .zip(&before_centers)
        .filter(|(s, c)| (s.center - **c).length() > 1.0)
        .count();
    assert!(moved > stroke.splats.len() / 2, "most splats should follow the curve");

    // (2) The residual splat still sits at its frame + residual offset (residual
    //     survived the curve edit, riding the local frame).
    let s = stroke.find_splat(probe_id).unwrap();
    let frame = stroke.skeleton.frame_at_arc_t(s.t);
    let expected = frame.position
        + frame.tangent * (s.v + residual.x)
        + frame.normal * (s.u + residual.y);
    assert!((s.center - expected).length() < 1e-3, "residual must ride the frame");
    assert_eq!(s.residual_local, residual, "residual value preserved");

    // (3) Continuity: no NaNs, arc table monotonic.
    assert!(stroke.splats.iter().all(|s| s.center.is_finite()));
    for w in stroke.skeleton.arc_length_table.samples.windows(2) {
        assert!(w[1].length >= w[0].length);
    }
}

// --- Test B: coherent central drag -> structural update --------------------------

#[test]
fn test_b_coherent_central_drag_updates_skeleton() {
    let (mut doc, sid) = make_doc();
    let before = control_points(&doc, sid);

    // Drag a central band straight up. Large-ish radius covers the central cross
    // section but not the endpoints.
    let mid = doc.stroke(sid).unwrap().skeleton.frame_at_arc_t(0.5).position;
    let drag = Vec2::new(0.0, -60.0);
    let edits = sculpt_move_splats(doc.stroke(sid).unwrap(), mid, drag, 170.0);
    let targets: Vec<(u32, Vec2)> =
        edits.iter().map(|e| (e.splat_id, e.new_center)).collect();
    assert!(edits.len() > 20, "central drag should select a meaningful band");

    let outcomes = apply_splat_edits_bidirectional(&mut doc, &edits);
    assert_eq!(outcomes[0].1, FitOutcome::Structural, "central drag should be structural");

    // Skeleton bent: at least one handle moved substantially toward the drag.
    let after = control_points(&doc, sid);
    assert!(max_control_movement(&before, &after) > 15.0, "skeleton should bend");
    // The interior handles moved in the drag (upward) direction.
    assert!(after[1].y < before[1].y || after[2].y < before[2].y);

    // Selected splats remain at their dragged positions (residual pins the remainder).
    let stroke = doc.stroke(sid).unwrap();
    let mut residual_sum = 0.0;
    for (id, target) in &targets {
        let s = stroke.find_splat(*id).unwrap();
        assert!((s.center - *target).length() < 1e-2, "edited splat pinned to target");
        residual_sum += s.residual_local.length();
    }
    // The skeleton explained most of the motion: mean residual is small vs the 60px drag.
    let mean_residual = residual_sum / targets.len() as f32;
    assert!(mean_residual < 30.0, "skeleton should explain most of the drag, got {mean_residual}");
}

// --- Test C: incoherent scatter -> residual only ---------------------------------

#[test]
fn test_c_incoherent_scatter_keeps_skeleton() {
    let (mut doc, sid) = make_doc();
    let before = control_points(&doc, sid);

    // Pick ~30 edge splats spread along the stroke and perturb them randomly.
    let mut rng = Rng::new(1234);
    let edge_ids: Vec<u32> = doc
        .stroke(sid)
        .unwrap()
        .splats
        .iter()
        .filter(|s| s.role == SplatRole::Edge)
        .map(|s| s.id)
        .collect();
    let step = (edge_ids.len() / 30).max(1);
    let mut edits = Vec::new();
    {
        let stroke = doc.stroke(sid).unwrap();
        for id in edge_ids.iter().step_by(step) {
            let s = stroke.find_splat(*id).unwrap();
            let jitter = Vec2::new(rng.signed() * 20.0, rng.signed() * 20.0);
            edits.push(SplatEdit {
                stroke: sid,
                splat_id: *id,
                old_center: s.center,
                new_center: s.center + jitter,
            });
        }
    }

    let outcomes = apply_splat_edits_bidirectional(&mut doc, &edits);
    assert_eq!(outcomes[0].1, FitOutcome::Residual, "scatter should be residual-only");

    // Skeleton is untouched.
    let after = control_points(&doc, sid);
    assert!(max_control_movement(&before, &after) < 1e-3, "skeleton must not move");

    // Residuals were updated on the perturbed splats.
    let stroke = doc.stroke(sid).unwrap();
    let updated = edits
        .iter()
        .filter(|e| stroke.find_splat(e.splat_id).unwrap().residual_local.length() > 1.0)
        .count();
    assert!(updated > edits.len() / 2, "residuals should absorb the scatter");
}

// --- Test D: width-like edge push -> centerline stays put -------------------------

#[test]
fn test_d_edge_push_does_not_shift_centerline() {
    let (mut doc, sid) = make_doc();
    let before = control_points(&doc, sid);

    // Push every edge splat outward along its local normal (a symmetric "widen").
    let mut edits = Vec::new();
    {
        let stroke = doc.stroke(sid).unwrap();
        for s in &stroke.splats {
            if s.role == SplatRole::Edge && s.u.abs() > 1e-3 {
                let normal = stroke.skeleton.frame_at_arc_t(s.t).normal;
                let outward = normal * s.u.signum() * 6.0;
                edits.push(SplatEdit {
                    stroke: sid,
                    splat_id: s.id,
                    old_center: s.center,
                    new_center: s.center + outward,
                });
            }
        }
    }
    assert!(edits.len() > 20);

    let outcomes = apply_splat_edits_bidirectional(&mut doc, &edits);
    // A symmetric width change must not be read as a structural centerline move.
    assert_ne!(outcomes[0].1, FitOutcome::Structural);

    let after = control_points(&doc, sid);
    assert!(
        max_control_movement(&before, &after) < 2.0,
        "centerline must not shift for a symmetric width edit"
    );

    // The widening was preserved as residual deformation.
    let stroke = doc.stroke(sid).unwrap();
    let updated = edits
        .iter()
        .filter(|e| stroke.find_splat(e.splat_id).unwrap().residual_local.length() > 1.0)
        .count();
    assert!(updated > 0, "edge push should be stored as residuals");
}

// --- Test E: whole-stroke translate -> control points translate ------------------

#[test]
fn test_e_whole_stroke_translate_moves_anchors() {
    let (mut doc, sid) = make_doc();
    let before = control_points(&doc, sid);

    // Move every splat by the same vector (a rigid drag of the whole stroke).
    let delta = Vec2::new(40.0, 25.0);
    let mut edits = Vec::new();
    {
        let stroke = doc.stroke(sid).unwrap();
        for s in &stroke.splats {
            edits.push(SplatEdit {
                stroke: sid,
                splat_id: s.id,
                old_center: s.center,
                new_center: s.center + delta,
            });
        }
    }

    let outcomes = apply_splat_edits_bidirectional(&mut doc, &edits);
    assert_eq!(outcomes[0].1, FitOutcome::Structural);

    // Every control point (including the anchors) translated by ~delta.
    let after = control_points(&doc, sid);
    for i in 0..4 {
        assert!(
            (after[i] - (before[i] + delta)).length() < 3.0,
            "control point {i} should translate by delta"
        );
    }
}

// --- Test F: direct-edit (node) tool -> drag handles/anchors, splats follow -------

/// Exercises the exact `app_core` calls the WasmApp Edit (direct-selection) tool makes:
/// locate a control point via `control_points`, drag it with `move_control_point`, then
/// forward-sync the splats with `update_world_cache`. Asserts the curve reshapes, the
/// splats follow, and a hand-painted residual survives on its local frame.
#[test]
fn test_f_direct_handle_edit_moves_splats_and_keeps_residual() {
    let (mut doc, sid) = make_doc();

    // Hand-paint a residual on a probe splat, then bake it into the world cache.
    let (probe_id, residual) = {
        let stroke = doc.stroke_mut(sid).unwrap();
        let idx = stroke.splats.len() / 2;
        let s = &mut stroke.splats[idx];
        s.residual_local = Vec2::new(3.0, -5.0);
        (s.id, s.residual_local)
    };
    doc.stroke_mut(sid).unwrap().update_world_cache();
    let before: Vec<Vec2> = doc.stroke(sid).unwrap().splats.iter().map(|s| s.center).collect();

    // The Edit tool's pointer_move path: drag anchor 0's out-handle, mirror on (smooth).
    let target = {
        let sk = &doc.stroke(sid).unwrap().skeleton;
        // Confirm the handle is discoverable the way `pick_control_point` finds it.
        assert!(
            sk.control_points()
                .iter()
                .any(|(r, _)| *r == ControlPointRef::out_handle(0)),
            "out-handle of anchor 0 must be an editable control point"
        );
        sk.anchor_position(0) + Vec2::new(40.0, -90.0)
    };
    {
        let stroke = doc.stroke_mut(sid).unwrap();
        stroke
            .skeleton
            .move_control_point(ControlPointRef::out_handle(0), target, true);
        stroke.update_world_cache();
    }

    let stroke = doc.stroke(sid).unwrap();

    // (1) The handle landed exactly where it was dragged.
    let moved_handle = stroke.skeleton.control_point(ControlPointRef::out_handle(0)).unwrap();
    assert!((moved_handle - target).length() < 1e-3, "handle should track the cursor");

    // (2) Splats followed the reshaped curve.
    let moved = stroke
        .splats
        .iter()
        .zip(&before)
        .filter(|(s, c)| (s.center - **c).length() > 1.0)
        .count();
    assert!(moved > stroke.splats.len() / 3, "splats should follow the handle drag");

    // (3) The residual still rides its local frame (paint detail preserved).
    let s = stroke.find_splat(probe_id).unwrap();
    let frame = stroke.skeleton.frame_at_arc_t(s.t);
    let expected = frame.position
        + frame.tangent * (s.v + residual.x)
        + frame.normal * (s.u + residual.y);
    assert!((s.center - expected).length() < 1e-3, "residual must ride the frame");
    assert_eq!(s.residual_local, residual, "residual value preserved");
    assert!(stroke.splats.iter().all(|s| s.center.is_finite()));
}

/// Dragging an anchor with the Edit tool moves the whole node rigidly: the anchor and its
/// tangent handle translate together, like Illustrator's Direct-Selection anchor drag.
#[test]
fn test_f_direct_anchor_drag_translates_node_rigidly() {
    let (mut doc, sid) = make_doc();

    let (before_anchor, before_handle) = {
        let sk = &doc.stroke(sid).unwrap().skeleton;
        (
            sk.anchor_position(0),
            sk.control_point(ControlPointRef::out_handle(0)).unwrap(),
        )
    };
    let delta = Vec2::new(30.0, 20.0);
    {
        let stroke = doc.stroke_mut(sid).unwrap();
        stroke
            .skeleton
            .move_control_point(ControlPointRef::anchor(0), before_anchor + delta, true);
        stroke.update_world_cache();
    }

    let sk = &doc.stroke(sid).unwrap().skeleton;
    assert!((sk.anchor_position(0) - (before_anchor + delta)).length() < 1e-3);
    // The handle rode along, preserving its offset from the anchor.
    let after_handle = sk.control_point(ControlPointRef::out_handle(0)).unwrap();
    assert!((after_handle - (before_handle + delta)).length() < 1e-3, "handle rides the anchor");
}

// --- Save/load survives a structural edit ----------------------------------------

#[test]
fn test_persistence_round_trip_after_edits() {
    let (mut doc, sid) = make_doc();

    // Apply a structural edit and a residual edit, then save/load.
    let mid = doc.stroke(sid).unwrap().skeleton.frame_at_arc_t(0.5).position;
    let edits = sculpt_move_splats(doc.stroke(sid).unwrap(), mid, Vec2::new(0.0, -40.0), 170.0);
    apply_splat_edits_bidirectional(&mut doc, &edits);

    let json = app_core::serialization::save_json(&doc).unwrap();
    let loaded = app_core::serialization::load_json(&json).unwrap();

    let a = doc.stroke(sid).unwrap();
    let b = loaded.strokes.values().next().unwrap();
    assert_eq!(a.splats.len(), b.splats.len());
    // Skeleton and residuals survived; rebuilt world caches match.
    for (sa, sb) in a.splats.iter().zip(&b.splats) {
        assert!((sa.center - sb.center).length() < 1e-2, "world center mismatch after reload");
        assert!((sa.residual_local - sb.residual_local).length() < 1e-4);
    }
}
