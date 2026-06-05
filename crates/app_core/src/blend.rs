//! The colour-blend (smudge) tool: bring neighbouring gaussians' **colours** into each other.
//!
//! Blending here means colour diffusion, *not* geometry. Each gaussian keeps its exact
//! position, covariance (size/orientation), and curve-local coordinates — only its colour is
//! pulled toward its neighbours', so adjacent splats' colours bleed together. Nothing here
//! ever moves a splat, resizes it, or touches the skeleton, so by construction every gaussian
//! stays exactly on the curve it lives on; the ellipses never collapse or merge into larger
//! blobs.
//!
//! Because it edits only the per-splat colour (not the world cache derived from geometry), it
//! flags each touched stroke for GPU re-upload directly (`dirty_flags.gpu_upload`).
//!
//! Two entry points share the machinery:
//! * [`blend_splats`] — stateless: pull every splat under the brush toward the brush-region
//!   average colour (used by the headless API and tests).
//! * [`smudge_splats`] — the interactive brush. Carries a colour between dabs ([`BlendCarry`])
//!   and deposits it as the cursor drags, then picks up new colour — so colour transports and
//!   mixes along a stroke like a real smudge. The neighbourhood spans *all* strokes under the
//!   brush, so two overlapping strokes' colours blend into each other.

use std::collections::BTreeSet;

use crate::document::Document;
use crate::ids::{SplatId, StrokeId};
use crate::math::{smoothstep, Vec2};

/// A splat gathered under the brush (read-only snapshot), so a dab is a *simultaneous* update:
/// every target colour is computed from the pre-dab state, then all splats recolour at once.
#[derive(Clone, Copy)]
struct Sample {
    stroke: StrokeId,
    splat: SplatId,
    color: [f32; 4],
    /// Brush falloff weight in `[0,1]` (1 at the cursor, 0 at the brush edge).
    brush_w: f32,
    /// Locked splats still *influence* the average (they are visually present) but never recolour.
    editable: bool,
}

/// The colour a smudge brush carries between dabs ("paint on the finger"). Reset at the start
/// of each stroke; loaded on first contact, then continuously deposited and refreshed.
#[derive(Clone, Copy, Debug, Default)]
pub struct BlendCarry {
    color: [f32; 4],
    /// False until the first pickup, so the first dab loads paint rather than smearing blank.
    loaded: bool,
}

impl BlendCarry {
    /// Forget any carried paint (call when a new smudge stroke begins).
    pub fn reset(&mut self) {
        *self = BlendCarry::default();
    }
}

#[inline]
fn lerp(a: f32, b: f32, f: f32) -> f32 {
    a + (b - a) * f
}

#[inline]
fn lerp4(a: [f32; 4], b: [f32; 4], f: f32) -> [f32; 4] {
    [
        lerp(a[0], b[0], f),
        lerp(a[1], b[1], f),
        lerp(a[2], b[2], f),
        lerp(a[3], b[3], f),
    ]
}

/// Gather every splat within `radius` of `center`, across all strokes (read-only snapshot).
fn gather(doc: &Document, center: Vec2, radius: f32) -> Vec<Sample> {
    let mut samples = Vec::new();
    if radius <= 0.0 {
        return samples;
    }
    for stroke in doc.strokes.values() {
        for splat in &stroke.splats {
            let dist = (splat.center - center).length();
            if dist >= radius {
                continue;
            }
            let brush_w = smoothstep(1.0, 0.0, dist / radius);
            if brush_w <= 0.0 {
                continue;
            }
            samples.push(Sample {
                stroke: stroke.id,
                splat: splat.id,
                color: splat.color,
                brush_w,
                editable: !splat.locked,
            });
        }
    }
    samples
}

/// Brush-falloff-weighted average colour over the gathered samples (centre-of-brush splats
/// dominate). Returns `None` for an empty set.
fn region_average_color(samples: &[Sample]) -> Option<[f32; 4]> {
    let mut wsum = 0.0f32;
    let mut color = [0.0f32; 4];
    for s in samples {
        let w = s.brush_w;
        wsum += w;
        for (k, c) in color.iter_mut().enumerate() {
            *c += s.color[k] * w;
        }
    }
    if wsum <= 1e-6 {
        return None;
    }
    let inv = 1.0 / wsum;
    Some([color[0] * inv, color[1] * inv, color[2] * inv, color[3] * inv])
}

/// Pull every editable sample's colour a fraction `f = strength * brush_w` toward `paint`,
/// then flag each touched stroke for GPU re-upload. Geometry is never read or written, so the
/// gaussians' positions and sizes are untouched. Returns the number of splats recoloured.
fn deposit_color(doc: &mut Document, samples: &[Sample], paint: [f32; 4], strength: f32) -> usize {
    if strength <= 0.0 {
        return 0;
    }
    let mut affected: BTreeSet<StrokeId> = BTreeSet::new();
    let mut recoloured = 0usize;
    for s in samples {
        if !s.editable {
            continue;
        }
        let f = (strength * s.brush_w).clamp(0.0, 1.0);
        if f <= 0.0 {
            continue;
        }
        let Some(stroke) = doc.stroke_mut(s.stroke) else {
            continue;
        };
        let Some(splat) = stroke.find_splat_mut(s.splat) else {
            continue;
        };
        splat.color = lerp4(splat.color, paint, f);
        affected.insert(s.stroke);
        recoloured += 1;
    }
    // Colour is a per-splat appearance field with no derived world cache, so just mark the
    // stroke's GPU instances stale — the renderer re-packs the new colours on the next frame.
    for sid in affected {
        if let Some(stroke) = doc.stroke_mut(sid) {
            stroke.dirty_flags.gpu_upload = true;
        }
    }
    recoloured
}

/// Stateless colour homogenise: pull every splat within `radius` of `center` toward the
/// brush-region average colour, a fraction `strength` per call. Geometry is never touched.
/// Returns the number of splats recoloured. Used by the headless API and tests; the
/// interactive brush uses [`smudge_splats`].
pub fn blend_splats(doc: &mut Document, center: Vec2, radius: f32, strength: f32) -> usize {
    if strength <= 0.0 {
        return 0;
    }
    let samples = gather(doc, center, radius);
    // "Blending colours together" needs at least two splats to mean anything.
    if samples.len() < 2 {
        return 0;
    }
    let Some(paint) = region_average_color(&samples) else {
        return 0;
    };
    deposit_color(doc, &samples, paint, strength)
}

/// Interactive smudge dab. Deposits the carried colour onto splats under the brush (pulling
/// their colour toward it), then refreshes the carried colour toward the new brush-region
/// average — so colour transports and mixes along the drag like a real smudge brush. On the
/// first contact of a stroke (`carry` not yet loaded) it only picks up colour. Geometry is
/// never touched. Returns the number of splats recoloured.
pub fn smudge_splats(
    doc: &mut Document,
    center: Vec2,
    radius: f32,
    strength: f32,
    carry: &mut BlendCarry,
) -> usize {
    let samples = gather(doc, center, radius);
    let Some(region) = region_average_color(&samples) else {
        return 0;
    };

    // First contact: load the brush with the region's colour, smear nothing yet.
    if !carry.loaded {
        carry.color = region;
        carry.loaded = true;
        return 0;
    }

    let recoloured = deposit_color(doc, &samples, carry.color, strength);

    // Pick up: drift the carried colour toward the new region so the brush gradually takes on
    // colours it passes over (this is what lets a smudge blend along a gradient). A higher
    // strength refreshes faster (more homogenise); a lower one carries colour further.
    let pickup = strength.clamp(0.0, 1.0);
    carry.color = lerp4(carry.color, region, pickup);

    recoloured
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bezier::{BezierSkeleton, CubicBezier};
    use crate::brush::BrushModel;
    use crate::ids::LayerId;

    /// A straight horizontal stroke from (0,0) to (300,0).
    fn horizontal_doc() -> (Document, StrokeId, LayerId) {
        let mut doc = Document::new();
        let layer = doc.add_layer("L");
        let curve = CubicBezier::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(100.0, 0.0),
            Vec2::new(200.0, 0.0),
            Vec2::new(300.0, 0.0),
        );
        let sid = doc.add_stroke(layer, BezierSkeleton::single(curve), BrushModel::default());
        (doc, sid, layer)
    }

    /// Paint the stroke half red (x < 150) / half blue, so there is colour to blend.
    fn paint_two_tone(doc: &mut Document, sid: StrokeId) {
        for s in &mut doc.stroke_mut(sid).unwrap().splats {
            s.color = if s.center.x < 150.0 {
                [1.0, 0.0, 0.0, 1.0]
            } else {
                [0.0, 0.0, 1.0, 1.0]
            };
        }
    }

    fn nearest_splat(doc: &Document, sid: StrokeId, p: Vec2) -> SplatId {
        doc.stroke(sid)
            .unwrap()
            .splats
            .iter()
            .min_by(|a, b| {
                (a.center - p)
                    .length_squared()
                    .partial_cmp(&(b.center - p).length_squared())
                    .unwrap()
            })
            .unwrap()
            .id
    }

    /// Snapshot the geometry (position + covariance shape + curve-local coords) of every splat.
    fn geometry(doc: &Document, sid: StrokeId) -> Vec<(Vec2, f32, f32, f32, f32, f32)> {
        doc.stroke(sid)
            .unwrap()
            .splats
            .iter()
            .map(|s| (s.center, s.sigma_tangent, s.sigma_normal, s.t, s.u, s.v))
            .collect()
    }

    #[test]
    fn blend_changes_only_colour_not_geometry() {
        // The core correction: blending diffuses COLOUR and must leave every gaussian's
        // position and size exactly as they were — no collapsing into larger ellipses.
        let (mut doc, sid, _) = horizontal_doc();
        paint_two_tone(&mut doc, sid);
        let geom_before = geometry(&doc, sid);
        let colors_before: Vec<[f32; 4]> =
            doc.stroke(sid).unwrap().splats.iter().map(|s| s.color).collect();

        let moved = blend_splats(&mut doc, Vec2::new(150.0, 0.0), 60.0, 0.8);
        assert!(moved > 0);

        assert_eq!(geometry(&doc, sid), geom_before, "geometry must be untouched");
        let colors_after: Vec<[f32; 4]> =
            doc.stroke(sid).unwrap().splats.iter().map(|s| s.color).collect();
        assert_ne!(colors_after, colors_before, "but colours must change");
    }

    #[test]
    fn blend_leaves_the_skeleton_untouched() {
        let (mut doc, sid, _) = horizontal_doc();
        paint_two_tone(&mut doc, sid);
        let sk_before: Vec<Vec2> = doc
            .stroke(sid)
            .unwrap()
            .skeleton
            .segments
            .iter()
            .flat_map(|s| [s.p0, s.p1, s.p2, s.p3])
            .collect();
        blend_splats(&mut doc, Vec2::new(150.0, 0.0), 60.0, 0.8);
        let sk_after: Vec<Vec2> = doc
            .stroke(sid)
            .unwrap()
            .skeleton
            .segments
            .iter()
            .flat_map(|s| [s.p0, s.p1, s.p2, s.p3])
            .collect();
        assert_eq!(sk_before, sk_after);
    }

    #[test]
    fn blend_marks_touched_strokes_for_gpu_reupload() {
        // Regression guard: a colour-only edit must still flag the stroke dirty, or the change
        // never reaches the GPU (the bug where "blend does nothing" on screen).
        let (mut doc, sid, _) = horizontal_doc();
        paint_two_tone(&mut doc, sid);
        doc.stroke_mut(sid).unwrap().dirty_flags.gpu_upload = false;
        blend_splats(&mut doc, Vec2::new(150.0, 0.0), 60.0, 0.8);
        assert!(
            doc.stroke(sid).unwrap().dirty_flags.gpu_upload,
            "blend must flag the stroke for re-upload so the recolour renders"
        );
    }

    #[test]
    fn blend_mixes_colours_of_neighbouring_splats() {
        let (mut doc, sid, _) = horizontal_doc();
        let a = nearest_splat(&doc, sid, Vec2::new(140.0, 0.0));
        let b = nearest_splat(&doc, sid, Vec2::new(160.0, 0.0));
        assert_ne!(a, b);
        {
            let stroke = doc.stroke_mut(sid).unwrap();
            stroke.find_splat_mut(a).unwrap().color = [1.0, 0.0, 0.0, 1.0];
            stroke.find_splat_mut(b).unwrap().color = [0.0, 0.0, 1.0, 1.0];
        }
        let gap_before = {
            let s = doc.stroke(sid).unwrap();
            (s.find_splat(a).unwrap().color[0] - s.find_splat(b).unwrap().color[0]).abs()
        };
        blend_splats(&mut doc, Vec2::new(150.0, 0.0), 60.0, 0.8);
        let gap_after = {
            let s = doc.stroke(sid).unwrap();
            (s.find_splat(a).unwrap().color[0] - s.find_splat(b).unwrap().color[0]).abs()
        };
        assert!(
            gap_after < gap_before,
            "colour gap should shrink: before={gap_before}, after={gap_after}"
        );
    }

    #[test]
    fn blend_strongly_homogenises_colour_in_one_pass() {
        // A brushed band painted half red / half blue should collapse toward a shared blend in
        // a single pass — proving the effect is *perceptible*, not infinitesimal.
        let (mut doc, sid, _) = horizontal_doc();
        paint_two_tone(&mut doc, sid);
        let a = nearest_splat(&doc, sid, Vec2::new(140.0, 0.0)); // red side
        let b = nearest_splat(&doc, sid, Vec2::new(160.0, 0.0)); // blue side
        blend_splats(&mut doc, Vec2::new(150.0, 0.0), 60.0, 0.8);
        let s = doc.stroke(sid).unwrap();
        assert!(s.find_splat(a).unwrap().color[2] > 0.3, "red splat should gain strong blue");
        assert!(s.find_splat(b).unwrap().color[0] > 0.3, "blue splat should gain strong red");
    }

    #[test]
    fn zero_strength_or_empty_brush_is_a_no_op() {
        let (mut doc, sid, _) = horizontal_doc();
        paint_two_tone(&mut doc, sid);
        let before: Vec<[f32; 4]> =
            doc.stroke(sid).unwrap().splats.iter().map(|s| s.color).collect();
        assert_eq!(blend_splats(&mut doc, Vec2::new(150.0, 0.0), 60.0, 0.0), 0);
        assert_eq!(blend_splats(&mut doc, Vec2::new(5000.0, 5000.0), 60.0, 0.8), 0);
        let after: Vec<[f32; 4]> =
            doc.stroke(sid).unwrap().splats.iter().map(|s| s.color).collect();
        assert_eq!(before, after);
    }

    #[test]
    fn locked_splats_keep_their_colour_but_still_influence() {
        let (mut doc, sid, _) = horizontal_doc();
        paint_two_tone(&mut doc, sid);
        let locked = nearest_splat(&doc, sid, Vec2::new(150.0, 0.0));
        {
            let s = doc.stroke_mut(sid).unwrap().find_splat_mut(locked).unwrap();
            s.locked = true;
            s.color = [0.0, 1.0, 0.0, 1.0]; // distinct green
        }
        blend_splats(&mut doc, Vec2::new(150.0, 0.0), 60.0, 0.9);
        let s = doc.stroke(sid).unwrap().find_splat(locked).unwrap();
        assert_eq!(s.color, [0.0, 1.0, 0.0, 1.0], "locked splat keeps its colour");
    }

    #[test]
    fn blend_mixes_colour_across_two_overlapping_strokes() {
        let mut doc = Document::new();
        let layer = doc.add_layer("L");
        let horiz = CubicBezier::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(100.0, 0.0),
            Vec2::new(200.0, 0.0),
            Vec2::new(300.0, 0.0),
        );
        let red = BrushModel {
            base_color: [1.0, 0.0, 0.0, 1.0],
            ..BrushModel::default()
        };
        let s_red = doc.add_stroke(layer, BezierSkeleton::single(horiz), red);

        let vert = CubicBezier::new(
            Vec2::new(150.0, -150.0),
            Vec2::new(150.0, -50.0),
            Vec2::new(150.0, 50.0),
            Vec2::new(150.0, 150.0),
        );
        let blue = BrushModel {
            base_color: [0.0, 0.0, 1.0, 1.0],
            ..BrushModel::default()
        };
        let s_blue = doc.add_stroke(layer, BezierSkeleton::single(vert), blue);

        let red_geom = geometry(&doc, s_red);
        let blue_geom = geometry(&doc, s_blue);
        blend_splats(&mut doc, Vec2::new(150.0, 0.0), 70.0, 0.8);

        let red_blue = doc
            .stroke(s_red)
            .unwrap()
            .splats
            .iter()
            .filter(|s| (s.center - Vec2::new(150.0, 0.0)).length() < 70.0)
            .map(|s| s.color[2])
            .fold(0.0f32, f32::max);
        assert!(red_blue > 0.0, "red stroke should pick up some blue near the crossing");
        let blue_red = doc
            .stroke(s_blue)
            .unwrap()
            .splats
            .iter()
            .filter(|s| (s.center - Vec2::new(150.0, 0.0)).length() < 70.0)
            .map(|s| s.color[0])
            .fold(0.0f32, f32::max);
        assert!(blue_red > 0.0, "blue stroke should pick up some red near the crossing");
        // Both strokes' geometry must be untouched (only colour bled across).
        assert_eq!(geometry(&doc, s_red), red_geom, "red geometry unchanged");
        assert_eq!(geometry(&doc, s_blue), blue_geom, "blue geometry unchanged");
    }

    #[test]
    fn smudge_transports_colour_from_pickup_point_to_a_new_region() {
        // Load the brush over a red patch, then dab it onto a blue region: the blue splats
        // there must pick up red — colour transported along the drag, geometry untouched.
        let (mut doc, sid, _) = horizontal_doc();
        {
            let stroke = doc.stroke_mut(sid).unwrap();
            for s in &mut stroke.splats {
                s.color = if s.center.x < 100.0 {
                    [1.0, 0.0, 0.0, 1.0]
                } else {
                    [0.0, 0.0, 1.0, 1.0]
                };
            }
        }
        let geom_before = geometry(&doc, sid);
        let mut carry = BlendCarry::default();
        let loaded = smudge_splats(&mut doc, Vec2::new(40.0, 0.0), 40.0, 0.6, &mut carry);
        assert_eq!(loaded, 0, "first contact only picks up paint");
        assert!(carry.loaded);

        let target = nearest_splat(&doc, sid, Vec2::new(250.0, 0.0));
        let red_before = doc.stroke(sid).unwrap().find_splat(target).unwrap().color[0];
        smudge_splats(&mut doc, Vec2::new(250.0, 0.0), 40.0, 0.6, &mut carry);
        let red_after = doc.stroke(sid).unwrap().find_splat(target).unwrap().color[0];
        assert!(
            red_after > red_before + 0.1,
            "blue splat should gain red from the carried paint: {red_before} -> {red_after}"
        );
        assert_eq!(geometry(&doc, sid), geom_before, "smudge must not move or resize splats");
    }

    #[test]
    fn smudge_reset_forgets_carried_paint() {
        let (mut doc, _sid, _) = horizontal_doc();
        let mut carry = BlendCarry::default();
        smudge_splats(&mut doc, Vec2::new(40.0, 0.0), 40.0, 0.6, &mut carry);
        assert!(carry.loaded);
        carry.reset();
        assert!(!carry.loaded, "reset must clear the loaded flag");
    }
}
