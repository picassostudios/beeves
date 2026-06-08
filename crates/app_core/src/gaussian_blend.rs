//! The gaussian-blend tool: a stroke whose splats take their colour from the art beneath them.
//!
//! A *gaussian blend* stroke is an ordinary Gaussian splat cloud laid along a fitted Bézier
//! skeleton (so it renders through the normal splat path and edits like any other stroke), but it
//! carries no colour of its own. Instead, every frame, each of its splats is recoloured from a
//! gaussian-weighted average of the splats of the strokes *beneath* it in document z-order — the
//! vector strokes (which keep a hit-test splat cloud) and the gaussian strokes below it. Where
//! little lies underneath, the splat's opacity fades toward zero so the blend does not paint over
//! empty canvas. The result is a soft, live "pick up what's under me" mark that mixes with the art
//! below using the ordinary premultiplied splat blending — distinct from:
//!
//!   * [`crate::blend`] — the smudge tool, which *destructively* rewrites neighbouring colours;
//!   * `renderer::vector_blend` — a render-time directional smear of the flat vector layer.
//!
//! Only colour and per-splat alpha are written here; positions, covariances, curve-local coords
//! and the skeleton are never touched (the geometry is owned by the stroke's normal regeneration).
//! Touched strokes are flagged `dirty_flags.gpu_upload` so the new colours re-pack next frame.
//!
//! Z-order note: a blend never samples another blend (blends are derived, not source content), so
//! the pass is order-independent among blends and has no feedback. It samples only non-blend
//! strokes positioned earlier in document order (layers back-to-front, strokes within a layer in
//! order).

use std::collections::HashMap;

use crate::document::Document;
use crate::ids::StrokeId;
use crate::math::Vec2;

/// Sampling radius as a multiple of the blend stroke's brush radius (half-width). Wider than the
/// stroke so each splat averages a generous neighbourhood of the art beneath it rather than
/// point-sampling — that neighbourhood average is what makes the mark read as a soft blend. (The
/// gaussian-blend tool shrinks the brush radius for smaller splats, so this factor is large enough
/// that the *colour* neighbourhood stays wide even though the splats themselves are small.)
const SAMPLE_RADIUS_FACTOR: f32 = 3.0;

/// One source splat flattened out of the document for sampling: its world centre, straight RGBA
/// colour, per-splat alpha, and the z-order index of its parent stroke.
struct Source {
    z: u32,
    center: Vec2,
    color: [f32; 4],
    alpha: f32,
}

/// Re-derive the colours of every `gaussian_blend` stroke in `doc` from the splats beneath it,
/// and flag those strokes for GPU re-upload. Returns true if any blend stroke was recoloured (so
/// callers can skip work when there are none). Cheap to call when there are no blend strokes.
///
/// Cost is `O(blend_splats × source_splats)` with an AABB early-out per source; fine for clean
/// line art, and gated on the presence of blend strokes so the common case is free.
pub fn resample_document_blends(doc: &mut Document) -> bool {
    // z-order index per stroke: layers back-to-front, strokes within a layer in order.
    let mut z_of: HashMap<StrokeId, u32> = HashMap::new();
    let mut order: u32 = 0;
    for layer in &doc.layers {
        for &sid in &layer.stroke_ids {
            z_of.insert(sid, order);
            order += 1;
        }
    }

    // Flatten every non-blend stroke's splats into a source list with their z-order index.
    let mut sources: Vec<Source> = Vec::new();
    let mut has_blend = false;
    for (sid, stroke) in doc.strokes.iter() {
        if stroke.gaussian_blend {
            has_blend = true;
            continue;
        }
        let Some(&z) = z_of.get(&sid) else {
            continue; // stroke not attached to any layer => undefined z, skip as a source
        };
        for s in &stroke.splats {
            sources.push(Source {
                z,
                center: s.center,
                color: s.color,
                alpha: s.alpha,
            });
        }
    }
    if !has_blend {
        return false;
    }

    // Compute new (color, alpha) for each blend stroke's splats from the sources below it.
    // Gather updates first (immutable borrow of `doc.strokes`), then apply (mutable borrow).
    let mut updates: Vec<(StrokeId, Vec<[f32; 5]>)> = Vec::new();
    for (sid, stroke) in doc.strokes.iter() {
        if !stroke.gaussian_blend {
            continue;
        }
        let Some(&zb) = z_of.get(&sid) else {
            continue;
        };
        let radius = (stroke.brush.radius * SAMPLE_RADIUS_FACTOR).max(1.0);
        let r2 = radius * radius;
        // radius ≈ 3σ so the gaussian has decayed to ~exp(-4.5) at the rim.
        let inv_two_sigma2 = 1.0 / (2.0 * (radius / 3.0).powi(2));
        let opacity = (stroke.brush.opacity * stroke.blend_strength).clamp(0.0, 1.0);

        let mut per_splat: Vec<[f32; 5]> = Vec::with_capacity(stroke.splats.len());
        for sp in &stroke.splats {
            let c = sp.center;
            let mut wsum = 0.0f32; // Σ w               (gaussian weight)
            let mut osum = 0.0f32; // Σ w·op            (coverage numerator)
            let mut col = [0.0f32; 3]; // Σ w·op·rgb
            for src in &sources {
                if src.z >= zb {
                    continue; // only strokes strictly below this blend in z-order
                }
                let d = src.center - c;
                if d.x.abs() > radius || d.y.abs() > radius {
                    continue; // cheap AABB reject before the distance test
                }
                let d2 = d.length_squared();
                if d2 > r2 {
                    continue;
                }
                let w = (-d2 * inv_two_sigma2).exp();
                let op = (src.alpha * src.color[3]).clamp(0.0, 1.0);
                wsum += w;
                let wo = w * op;
                osum += wo;
                col[0] += wo * src.color[0];
                col[1] += wo * src.color[1];
                col[2] += wo * src.color[2];
            }
            // Opacity-weighted mean colour; coverage in [0,1] drives the splat's alpha so the mark
            // is solid where the art beneath is solid and fades out over bare canvas.
            let (rgb, coverage) = if osum > 1e-6 {
                (
                    [col[0] / osum, col[1] / osum, col[2] / osum],
                    (osum / wsum.max(1e-6)).clamp(0.0, 1.0),
                )
            } else {
                ([0.0, 0.0, 0.0], 0.0)
            };
            let alpha = (opacity * coverage).clamp(0.0, 1.0);
            per_splat.push([rgb[0], rgb[1], rgb[2], 1.0, alpha]);
        }
        updates.push((sid, per_splat));
    }

    let mut changed = false;
    for (sid, per_splat) in updates {
        let Some(stroke) = doc.stroke_mut(sid) else {
            continue;
        };
        for (sp, v) in stroke.splats.iter_mut().zip(per_splat.iter()) {
            sp.color = [v[0], v[1], v[2], v[3]];
            sp.alpha = v[4];
        }
        stroke.dirty_flags.gpu_upload = true;
        changed = true;
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bezier::{BezierSkeleton, CubicBezier};
    use crate::brush::BrushModel;
    use crate::math::Vec2;

    fn straight(a: Vec2, b: Vec2) -> CubicBezier {
        CubicBezier::new(a, a + (b - a) / 3.0, a + (b - a) * (2.0 / 3.0), b)
    }

    /// Build a solid red gaussian stroke and a gaussian-blend stroke drawn over it; after a
    /// resample the blend's splats should take on the red colour with non-zero opacity.
    #[test]
    fn blend_picks_up_color_from_the_stroke_beneath() {
        let mut doc = Document::new();
        let layer = doc.add_layer("L");
        let brush = BrushModel {
            radius: 12.0,
            base_color: [1.0, 0.0, 0.0, 1.0],
            ..BrushModel::default()
        };
        // Underlying red stroke.
        doc.add_stroke(
            layer,
            BezierSkeleton::single(straight(Vec2::new(0.0, 0.0), Vec2::new(200.0, 0.0))),
            brush.clone(),
        );
        // Blend stroke on top, overlapping the red one.
        let blend = doc.add_stroke(
            layer,
            BezierSkeleton::single(straight(Vec2::new(0.0, 0.0), Vec2::new(200.0, 0.0))),
            brush,
        );
        doc.stroke_mut(blend).unwrap().gaussian_blend = true;

        assert!(resample_document_blends(&mut doc));

        let s = doc.stroke(blend).unwrap();
        let mut max_alpha = 0.0f32;
        for sp in &s.splats {
            // Picked-up colour is red (or empty/black where no coverage), never some other hue.
            assert!(sp.color[0] >= sp.color[1] - 1e-4 && sp.color[0] >= sp.color[2] - 1e-4);
            max_alpha = max_alpha.max(sp.alpha);
        }
        assert!(max_alpha > 0.1, "the blend should be visible over the red stroke");
        // A red core splat should read essentially pure red.
        let core = s
            .splats
            .iter()
            .max_by(|a, b| a.alpha.total_cmp(&b.alpha))
            .unwrap();
        assert!(core.color[0] > 0.8 && core.color[1] < 0.2 && core.color[2] < 0.2);
    }

    /// With nothing beneath it, a blend stroke fades to fully transparent (it must not paint a
    /// black/garbage average of an empty neighbourhood).
    #[test]
    fn blend_over_empty_canvas_is_transparent() {
        let mut doc = Document::new();
        let layer = doc.add_layer("L");
        let blend = doc.add_stroke(
            layer,
            BezierSkeleton::single(straight(Vec2::new(0.0, 0.0), Vec2::new(100.0, 0.0))),
            BrushModel { radius: 10.0, ..BrushModel::default() },
        );
        doc.stroke_mut(blend).unwrap().gaussian_blend = true;

        assert!(resample_document_blends(&mut doc));
        let s = doc.stroke(blend).unwrap();
        for sp in &s.splats {
            assert!(sp.alpha <= 1e-4, "no art beneath => transparent splat");
        }
    }

    /// A blend never samples another blend, and only samples strokes below it in z-order: a red
    /// stroke placed *above* the blend must not colour it.
    #[test]
    fn blend_ignores_strokes_above_it() {
        let mut doc = Document::new();
        let layer = doc.add_layer("L");
        let brush = BrushModel { radius: 12.0, ..BrushModel::default() };
        // Blend first (lower z) ...
        let blend = doc.add_stroke(
            layer,
            BezierSkeleton::single(straight(Vec2::new(0.0, 0.0), Vec2::new(200.0, 0.0))),
            brush.clone(),
        );
        doc.stroke_mut(blend).unwrap().gaussian_blend = true;
        // ... then a red stroke above it.
        doc.add_stroke(
            layer,
            BezierSkeleton::single(straight(Vec2::new(0.0, 0.0), Vec2::new(200.0, 0.0))),
            BrushModel { base_color: [1.0, 0.0, 0.0, 1.0], ..brush },
        );

        resample_document_blends(&mut doc);
        let s = doc.stroke(blend).unwrap();
        for sp in &s.splats {
            assert!(sp.alpha <= 1e-4, "stroke above must not feed the blend below it");
        }
    }
}
