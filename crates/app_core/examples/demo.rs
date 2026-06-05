//! Runnable demonstration of the bidirectional editing loop.
//!
//!   cargo run -p app_core --example demo
//!
//! Builds a stroke, prints the splat cloud size, then exercises three reverse-sync
//! scenarios (coherent drag, incoherent scatter, symmetric widen) while printing the
//! coherence metrics and the chosen outcome. Finally writes `examples/demo.gspf.json`.

use std::path::PathBuf;

use app_core::bezier::{BezierSkeleton, CubicBezier};
use app_core::brush::BrushModel;
use app_core::document::Document;
use app_core::ids::StrokeId;
use app_core::math::Vec2;
use app_core::solver::{
    apply_splat_edits_bidirectional, estimate_centerline_targets, measure_edit_coherence,
    sculpt_move_splats, SplatEdit,
};
use app_core::splat::SplatRole;
use app_core::Rng;

fn banner(title: &str) {
    println!("\n=== {title} ===");
}

fn classify_and_apply(doc: &mut Document, sid: StrokeId, edits: &[SplatEdit], label: &str) {
    // Peek at the coherence metrics before applying (for display only).
    let stroke = doc.stroke(sid).unwrap();
    let targets = estimate_centerline_targets(stroke, edits);
    let result = measure_edit_coherence(stroke, edits, &targets, edits.len());
    let c = result.coherence;
    println!(
        "{label}: edits={:<4} same={:.2} coverage={:.2} smooth={:.2} core={:.2} fit_q={:.2} -> confidence={:.3}",
        edits.len(),
        c.same_stroke_ratio,
        c.t_interval_coverage,
        c.displacement_smoothness,
        c.core_weight_ratio,
        c.fit_quality,
        c.confidence,
    );

    let before = doc.stroke(sid).unwrap().skeleton.segments[0];
    let outcomes = apply_splat_edits_bidirectional(doc, edits);
    let after = doc.stroke(sid).unwrap().skeleton.segments[0];
    let handle_move = (after.p1 - before.p1).length().max((after.p2 - before.p2).length());
    println!(
        "  -> outcome={:?}, max handle move = {:.1}px",
        outcomes[0].1, handle_move
    );
}

fn main() {
    // 1. Build a document with one gently-curved stroke.
    let mut doc = Document::new();
    let layer = doc.add_layer("Layer 1");
    let curve = CubicBezier::new(
        Vec2::new(100.0, 300.0),
        Vec2::new(250.0, 250.0),
        Vec2::new(450.0, 250.0),
        Vec2::new(600.0, 300.0),
    );
    let sid = doc.add_stroke(layer, BezierSkeleton::single(curve), BrushModel::default());

    banner("Draw");
    println!(
        "Stroke drawn: {} splats over a {:.0}px skeleton.",
        doc.stroke(sid).unwrap().splats.len(),
        doc.stroke(sid).unwrap().skeleton.total_length(),
    );

    // 2. Coherent central drag -> should update the skeleton structurally.
    banner("Reverse sync");
    let mid = doc.stroke(sid).unwrap().skeleton.frame_at_arc_t(0.5).position;
    let drag = sculpt_move_splats(doc.stroke(sid).unwrap(), mid, Vec2::new(0.0, -60.0), 170.0);
    classify_and_apply(&mut doc, sid, &drag, "coherent central drag");

    // 3. Incoherent scatter of edge splats -> should stay residual.
    let mut rng = Rng::new(7);
    let scatter: Vec<SplatEdit> = {
        let stroke = doc.stroke(sid).unwrap();
        stroke
            .splats
            .iter()
            .filter(|s| s.role == SplatRole::Edge)
            .step_by(7)
            .map(|s| SplatEdit {
                stroke: sid,
                splat_id: s.id,
                old_center: s.center,
                new_center: s.center + Vec2::new(rng.signed() * 20.0, rng.signed() * 20.0),
            })
            .collect()
    };
    classify_and_apply(&mut doc, sid, &scatter, "incoherent scatter   ");

    // 4. Symmetric widen (edge splats pushed along the normal) -> centerline holds.
    let widen: Vec<SplatEdit> = {
        let stroke = doc.stroke(sid).unwrap();
        stroke
            .splats
            .iter()
            .filter(|s| s.role == SplatRole::Edge && s.u.abs() > 1e-3)
            .map(|s| {
                let n = stroke.skeleton.frame_at_arc_t(s.t).normal;
                SplatEdit {
                    stroke: sid,
                    splat_id: s.id,
                    old_center: s.center,
                    new_center: s.center + n * s.u.signum() * 6.0,
                }
            })
            .collect()
    };
    classify_and_apply(&mut doc, sid, &widen, "symmetric widen      ");

    // 5. Save the result next to this example.
    banner("Persist");
    let json = app_core::serialization::save_json(&doc).unwrap();
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("../../examples");
    let _ = std::fs::create_dir_all(&path);
    path.push("demo.gspf.json");
    std::fs::write(&path, &json).expect("write demo file");
    println!(
        "Wrote {} ({} bytes, {} splats).",
        path.display(),
        json.len(),
        doc.splat_count()
    );
}
