//! The canonical editable primitive: a Bezier skeleton plus curve-attached splats.

use serde::{Deserialize, Serialize};

use crate::bezier::BezierSkeleton;
use crate::brush::BrushModel;
use crate::ids::{SplatId, StrokeId};
use crate::math::{covariance_from_sigmas, Rng};
use crate::splat::{GaussianSplat, SplatRole};

/// Tangential Gaussian sigma as a multiple of station spacing. >= ~1.5 makes adjacent
/// core splats overlap into a continuous ridge (no visible beading when zoomed in),
/// at the cost of slightly rounder stroke ends.
const TANGENT_SIGMA_FACTOR: f32 = 1.6;

/// Normal Gaussian sigma of interior (fill) cross-section rows, as a fraction of the
/// stroke half-width. Wide enough that adjacent rows overlap and the fill is gap-free.
const SIGMA_NORMAL_FILL_FACTOR: f32 = 0.15;

/// Normal Gaussian sigma of the outer **boundary ring**, as a fraction of the half-width.
/// Thinner than the fill so the silhouette is sharply localized at ±width; paired with the
/// brush's high `edge_hardness` (see `update_splat_world_cache`) this gives a crisp
/// perimeter while the interior stays soft.
const SIGMA_NORMAL_EDGE_FACTOR: f32 = 0.10;

/// Serde default for [`GaussianBezierStroke::blend_strength`] (full-strength smear), so strokes
/// saved before the field existed load with a sensible value.
fn default_blend_strength() -> f32 {
    1.0
}

/// Policy controlling how the two views synchronize.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SyncPolicy {
    /// Forward direction enabled: curve edits update splats.
    pub curve_to_splat: bool,
    /// Reverse direction enabled: splat edits may update the curve.
    pub splat_to_curve: bool,
    /// Max acceptable residual (px) when accepting a structural skeleton update.
    pub max_structural_error: f32,
    /// Confidence at/above which a splat edit is treated as fully structural.
    pub high_confidence: f32,
    /// Confidence below which a splat edit is treated as residual-only.
    pub low_confidence: f32,
}

impl Default for SyncPolicy {
    fn default() -> Self {
        Self {
            curve_to_splat: true,
            splat_to_curve: true,
            max_structural_error: 3.0,
            high_confidence: 0.7,
            low_confidence: 0.4,
        }
    }
}

/// Tracks which derived data is stale and needs recomputation/upload.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct StrokeDirtyFlags {
    pub arc_length: bool,
    pub world_cache: bool,
    pub gpu_upload: bool,
}

impl StrokeDirtyFlags {
    pub fn mark_all(&mut self) {
        self.arc_length = true;
        self.world_cache = true;
        self.gpu_upload = true;
    }
}

/// A stroke = editable skeleton + brush + splats attached by curve-local coordinates +
/// residual deformation + sync policy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GaussianBezierStroke {
    pub id: StrokeId,
    pub skeleton: BezierSkeleton,
    pub brush: BrushModel,
    pub splats: Vec<GaussianSplat>,
    #[serde(default)]
    pub sync: SyncPolicy,
    /// When true the stroke is drawn as a conventional vector — a tessellated, antialiased
    /// stroked Bézier outline — instead of its Gaussian splat cloud. The splats are still
    /// generated and kept (so hit-testing, selection, and direct-edit all keep working); the
    /// renderer just skips them for this stroke and draws the path. Set by the vector-draw tool.
    #[serde(default)]
    pub render_as_vector: bool,
    /// When true the stroke is a *vector blend* path: it carries no colour of its own. Instead
    /// the renderer draws it in a dedicated pass (`renderer::vector_blend`) that samples the
    /// vector-stroke layer beneath the ribbon and directionally smears those colours along the
    /// path tangent — a live, non-destructive smudge whose region is described by a vector. This
    /// implies `render_as_vector` for the splat passes (both are set together), so the stroke's
    /// splats are still generated and kept for hit-testing / direct-edit but never drawn as a
    /// splat cloud or a solid ribbon. Set by the vector-blend tool.
    #[serde(default)]
    pub vector_blend: bool,
    /// Strength of the vector-blend smear in `[0,1]`: the opacity with which the smeared result
    /// is composited over the layer beneath the ribbon (0 = invisible, 1 = full smear). Only
    /// meaningful when `vector_blend` is set.
    #[serde(default = "default_blend_strength")]
    pub blend_strength: f32,
    /// When true the stroke is a *gaussian blend*: a normal Gaussian splat cloud (so it renders
    /// through the ordinary splat path, `render_as_vector = false`), but its splats carry no
    /// colour of their own. Each frame their colours are re-derived from the splats of the strokes
    /// *beneath* this one in document z-order — a gaussian-weighted average of whatever lies under
    /// each splat (see `crate::gaussian_blend`). The geometry stays fully parametric/editable (the
    /// skeleton + the curve-local splats); only the colours are a live, recomputed view of the art
    /// below. `blend_strength` scales the result's opacity. Distinct from `vector_blend` (a
    /// render-time directional smear of the vector layer) and from the `crate::blend` smudge tool
    /// (which destructively rewrites colours). Set by the gaussian-blend tool.
    #[serde(default)]
    pub gaussian_blend: bool,
    #[serde(skip)]
    pub dirty_flags: StrokeDirtyFlags,
    /// Monotonic allocator for splat ids within this stroke.
    #[serde(default)]
    next_splat_id: SplatId,
}

impl GaussianBezierStroke {
    /// Build a stroke from a skeleton and brush. The caller assigns `id` (the stroke's
    /// SlotMap key) before generating splats so the splats point back at their parent.
    pub fn new(id: StrokeId, skeleton: BezierSkeleton, brush: BrushModel) -> Self {
        let mut stroke = Self {
            id,
            skeleton,
            brush,
            splats: Vec::new(),
            sync: SyncPolicy::default(),
            render_as_vector: false,
            vector_blend: false,
            blend_strength: default_blend_strength(),
            gaussian_blend: false,
            dirty_flags: StrokeDirtyFlags::default(),
            next_splat_id: 0,
        };
        stroke.regenerate_splats();
        stroke
    }

    fn alloc_splat_id(&mut self) -> SplatId {
        let id = self.next_splat_id;
        self.next_splat_id += 1;
        id
    }

    pub fn find_splat(&self, id: SplatId) -> Option<&GaussianSplat> {
        self.splats.iter().find(|s| s.id == id)
    }

    pub fn find_splat_mut(&mut self, id: SplatId) -> Option<&mut GaussianSplat> {
        self.splats.iter_mut().find(|s| s.id == id)
    }

    /// Recolour the whole stroke to `rgb` (linear RGB in `[0,1]`). Each splat's per-splat hue
    /// variation — texture jitter and any blended colour — is preserved by shifting it from
    /// the old base colour to the new one, rather than flattening every splat to one value;
    /// alpha is left unchanged. Updates the brush base colour so future regeneration matches,
    /// and flags the GPU instances stale. This is a pure colour edit — geometry (positions,
    /// covariance, curve-local coords) and the skeleton are untouched.
    pub fn set_base_color(&mut self, rgb: [f32; 3]) {
        let old = self.brush.base_color;
        let (dr, dg, db) = (rgb[0] - old[0], rgb[1] - old[1], rgb[2] - old[2]);
        for s in &mut self.splats {
            s.color[0] = (s.color[0] + dr).clamp(0.0, 1.0);
            s.color[1] = (s.color[1] + dg).clamp(0.0, 1.0);
            s.color[2] = (s.color[2] + db).clamp(0.0, 1.0);
        }
        self.brush.base_color = [rgb[0], rgb[1], rgb[2], old[3]];
        self.dirty_flags.gpu_upload = true;
    }

    /// Regenerate the entire splat cloud from the brush model. Destroys any existing
    /// splats (and their residuals) — call only when (re)creating a stroke.
    pub fn regenerate_splats(&mut self) {
        self.splats.clear();
        self.next_splat_id = 0;

        let length = self.skeleton.total_length();
        let spacing = self.brush.spacing.max(1.0);
        let stations = ((length / spacing).ceil() as usize).max(2);
        let mut rng = Rng::new(self.brush.seed);

        for i in 0..stations {
            let t = i as f32 / (stations - 1) as f32;
            let width = self.brush.radius * self.brush.width_profile.eval(t);
            let cross = sample_cross_section(width, self.brush.texture_strength, &mut rng);
            for sample in cross {
                let id = self.alloc_splat_id();
                let mut splat = GaussianSplat::new(id, self.id);
                splat.t = t;
                splat.u = sample.u;
                splat.v = rng.signed() * spacing * 0.15; // small tangential jitter
                // Tangential sigma is set to `TANGENT_SIGMA_FACTOR` (>1) times the
                // inter-station step so adjacent core splats overlap heavily and sum into a
                // smooth ridge along the curve — otherwise their composited sum ripples
                // ("beading"/scalloping) which becomes visible when zoomed in. Across the
                // width, `sigma_normal` comes from the cross-section sample: wide for the
                // gap-free fill, thin for the crisp boundary ring (see `sample_cross_section`).
                splat.sigma_tangent = spacing * TANGENT_SIGMA_FACTOR;
                splat.sigma_normal = sample.sigma_normal;
                splat.rotation_jitter = rng.signed() * 0.05;
                splat.color = jitter_color(self.brush.base_color, self.brush.texture_strength, &mut rng);
                splat.alpha = (self.brush.opacity * self.brush.opacity_profile.eval(t)).clamp(0.0, 1.0);
                splat.role = sample.role;
                self.splats.push(splat);
            }
        }

        self.update_world_cache();
        self.dirty_flags.mark_all();
    }

    /// Forward sync: re-evaluate every splat's world center and covariance from its
    /// curve-local coords + residuals against the current skeleton.
    pub fn update_world_cache(&mut self) {
        update_splat_world_cache(self);
    }
}

/// Re-evaluate world-space caches for all splats in a stroke (forward sync).
///
/// `mu = P(t) + u*N(t) + v*T(t) + residual_local(in frame) + residual_world`.
pub fn update_splat_world_cache(stroke: &mut GaussianBezierStroke) {
    // Resolved once outside the loop (the borrow checker won't let us read `stroke.brush`
    // while iterating `stroke.splats` mutably). This is also the load path: a deserialized
    // stroke has no per-splat `hardness` (it is `#[serde(skip)]`), so deriving it here from
    // role keeps generation and load in agreement.
    let body_hardness = stroke.brush.hardness;
    let edge_hardness = stroke.brush.edge_hardness;
    for splat in &mut stroke.splats {
        let frame = stroke.skeleton.frame_at_arc_t(splat.t);
        let local = frame.tangent * (splat.v + splat.residual_local.x)
            + frame.normal * (splat.u + splat.residual_local.y);
        splat.center = frame.position + local + splat.residual_world;

        let theta = frame.angle + splat.rotation_jitter;
        splat.covariance = covariance_from_sigmas(theta, splat.sigma_tangent, splat.sigma_normal);
        // No inverse is cached: it's derived on demand (GPU in-shader, CPU via
        // `GaussianSplat::response_q`), so re-deriving the world cache never pays a
        // per-splat matrix inverse — the hot path when adding/editing many splats.
        splat.radius_px = 3.0 * splat.sigma_tangent.max(splat.sigma_normal);
        // Perimeter splats (the Edge boundary ring) render crisp; interior splats (Core /
        // Texture) keep the brush's softer `hardness`, so painterly fuzz stays inside the
        // silhouette and never blurs the line's outer edge.
        splat.hardness = match splat.role {
            SplatRole::Edge => edge_hardness,
            _ => body_hardness,
        };
    }
    stroke.dirty_flags.world_cache = false;
    stroke.dirty_flags.gpu_upload = true;
}

/// Number of evenly-spaced rows sampled across the stroke width (centerline to both
/// edges). More rows = more, finer splats across the width; the step between rows
/// (`2 / (ROWS - 1)` of the half-width) is matched to `sigma_normal` so the stroke stays
/// gap-free. Rows within `CORE_FRAC` of the centerline are tagged `Core` (read as a
/// structural edit by reverse-sync); outer rows are `Edge` (width/detail).
const CROSS_ROWS: usize = 11;
const CORE_FRAC: f32 = 0.4;

/// One sampled cross-section station: a normal offset `u`, its functional `role`, and the
/// normal Gaussian sigma to give it (thin for the boundary ring, wide for the fill).
struct CrossSample {
    u: f32,
    role: SplatRole,
    sigma_normal: f32,
}

/// Cross-section sampling across the stroke width — a `Core` band around the centerline,
/// `Edge` fill rows out to the boundary, and a thin outermost **boundary ring** that (with
/// the brush's high `edge_hardness`) gives the line a crisp perimeter. Optional texture
/// jitter is placed *inside* the boundary so painterly fuzz never softens the silhouette.
fn sample_cross_section(width: f32, texture: f32, rng: &mut Rng) -> Vec<CrossSample> {
    let fill_sigma = (width * SIGMA_NORMAL_FILL_FACTOR).max(0.5);
    let edge_sigma = (width * SIGMA_NORMAL_EDGE_FACTOR).max(0.4);

    let mut out: Vec<CrossSample> = Vec::with_capacity(CROSS_ROWS + 4);
    for i in 0..CROSS_ROWS {
        // frac runs -1 -> 1 across the full width.
        let frac = -1.0 + 2.0 * (i as f32) / (CROSS_ROWS as f32 - 1.0);
        let role = if frac.abs() <= CORE_FRAC {
            SplatRole::Core
        } else {
            SplatRole::Edge
        };
        // The two outermost rows are the boundary ring: a thinner normal sigma localizes the
        // silhouette right at ±width instead of letting a wide Gaussian bleed past it.
        let is_boundary = i == 0 || i == CROSS_ROWS - 1;
        let sigma_normal = if is_boundary { edge_sigma } else { fill_sigma };
        out.push(CrossSample { u: frac * width, role, sigma_normal });
    }

    // Sprinkle a couple of texture splats just *inside* the edge (≤ 0.9·width). Keeping fuzz
    // inboard of the boundary ring is what lets the interior be painterly while the
    // perimeter stays crisp.
    if texture > 0.0 {
        let n_tex = (texture * 4.0).round() as usize;
        for _ in 0..n_tex {
            let side = if rng.next_f32() < 0.5 { -1.0 } else { 1.0 };
            let u = side * width * rng.range(0.55, 0.9);
            out.push(CrossSample { u, role: SplatRole::Texture, sigma_normal: fill_sigma });
        }
    }
    out
}

fn jitter_color(base: [f32; 4], strength: f32, rng: &mut Rng) -> [f32; 4] {
    let j = strength * 0.1;
    [
        (base[0] + rng.signed() * j).clamp(0.0, 1.0),
        (base[1] + rng.signed() * j).clamp(0.0, 1.0),
        (base[2] + rng.signed() * j).clamp(0.0, 1.0),
        base[3],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bezier::CubicBezier;
    use crate::math::Vec2;

    fn test_stroke() -> GaussianBezierStroke {
        let curve = CubicBezier::new(
            Vec2::new(100.0, 200.0),
            Vec2::new(180.0, 120.0),
            Vec2::new(300.0, 280.0),
            Vec2::new(420.0, 200.0),
        );
        GaussianBezierStroke::new(StrokeId::default(), BezierSkeleton::single(curve), BrushModel::default())
    }

    #[test]
    fn generation_produces_splats_with_local_coords() {
        let stroke = test_stroke();
        assert!(stroke.splats.len() > 20);
        // Every splat carries identity: parent, t in range, finite center.
        for s in &stroke.splats {
            assert_eq!(s.parent_stroke, stroke.id);
            assert!((0.0..=1.0).contains(&s.t));
            assert!(s.center.is_finite());
            assert!(s.radius_px > 0.0);
        }
        // Both core and edge roles are present.
        assert!(stroke.splats.iter().any(|s| s.role == SplatRole::Core));
        assert!(stroke.splats.iter().any(|s| s.role == SplatRole::Edge));
    }

    #[test]
    fn set_base_color_recolours_without_touching_geometry() {
        let mut stroke = test_stroke();
        // Snapshot geometry so we can prove a recolour leaves it bit-identical.
        let geom: Vec<(Vec2, f32, f32, f32, f32)> = stroke
            .splats
            .iter()
            .map(|s| (s.center, s.sigma_tangent, s.sigma_normal, s.t, s.u))
            .collect();
        stroke.dirty_flags.gpu_upload = false;

        stroke.set_base_color([1.0, 0.0, 0.0]);

        // Brush base colour updated (alpha preserved).
        assert_eq!(stroke.brush.base_color, [1.0, 0.0, 0.0, 1.0]);
        // Every splat is now ~red (within the small texture jitter), alpha untouched.
        for s in &stroke.splats {
            assert!(s.color[0] > 0.8, "red channel raised");
            assert!(s.color[2] < 0.2, "blue channel dropped");
            assert_eq!(s.color[3], 1.0, "colour alpha preserved");
        }
        // Geometry is untouched, and the stroke is flagged for GPU re-upload.
        let after: Vec<(Vec2, f32, f32, f32, f32)> = stroke
            .splats
            .iter()
            .map(|s| (s.center, s.sigma_tangent, s.sigma_normal, s.t, s.u))
            .collect();
        assert_eq!(after, geom, "recolour must not move or resize splats");
        assert!(stroke.dirty_flags.gpu_upload, "recolour flags GPU re-upload");
    }

    #[test]
    fn world_center_respects_offset_from_centerline() {
        let stroke = test_stroke();
        // A splat with u>0 should sit off the centerline by ~|u| in the normal dir.
        for s in &stroke.splats {
            let frame = stroke.skeleton.frame_at_arc_t(s.t);
            let predicted = frame.position
                + frame.normal * (s.u + s.residual_local.y)
                + frame.tangent * (s.v + s.residual_local.x);
            assert!((s.center - predicted).length() < 1e-3);
        }
    }

    #[test]
    fn core_stations_overlap_to_avoid_beading() {
        let stroke = test_stroke();

        // One representative core station per arc position `t`: pick the core splat
        // closest to the centerline (smallest |u|) so we measure true centerline spacing,
        // not the off-axis jitter of multiple core rows at the same `t`.
        use std::collections::BTreeMap;
        let mut by_t: BTreeMap<u64, &GaussianSplat> = BTreeMap::new();
        for s in stroke.splats.iter().filter(|s| s.role == SplatRole::Core) {
            // Quantize t to a stable key; centerline rows share the same `t`.
            let key = (s.t * 1e6).round() as u64;
            match by_t.get(&key) {
                Some(prev) if prev.u.abs() <= s.u.abs() => {}
                _ => {
                    by_t.insert(key, s);
                }
            }
        }

        let stations: Vec<&GaussianSplat> = by_t.values().copied().collect();
        assert!(stations.len() >= 2, "need at least two core stations to measure a gap");

        for pair in stations.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let gap = (b.center - a.center).length();
            // Overlap condition: consecutive core stations sit comfortably within one
            // tangential sigma of each other, so their Gaussians sum into a smooth ridge
            // rather than a string of beads.
            assert!(
                gap < 1.2 * a.sigma_tangent,
                "core stations too far apart: gap {gap} >= 1.2 * sigma_tangent {} (beading risk)",
                a.sigma_tangent
            );
        }
    }

    #[test]
    fn edge_ring_is_crisp_and_interior_is_soft() {
        let stroke = test_stroke();
        let brush = &stroke.brush;
        assert!(
            brush.edge_hardness > brush.hardness,
            "test relies on the boundary ring being harder than the interior"
        );

        // Every Edge (boundary) splat renders at the brush's crisp edge_hardness; every
        // interior splat (Core / Texture) renders at the softer interior hardness.
        for s in &stroke.splats {
            let expected = match s.role {
                SplatRole::Edge => brush.edge_hardness,
                _ => brush.hardness,
            };
            assert!(
                (s.hardness - expected).abs() < 1e-6,
                "role {:?} should map to hardness {expected}, got {}",
                s.role,
                s.hardness
            );
        }

        // The thinnest splats across the width are Edge ring splats: the boundary is
        // localized more tightly than the fill, which is what sharpens the silhouette.
        let min_sigma = stroke
            .splats
            .iter()
            .min_by(|a, b| a.sigma_normal.partial_cmp(&b.sigma_normal).unwrap())
            .unwrap();
        assert_eq!(min_sigma.role, SplatRole::Edge, "thinnest row is the boundary ring");
    }

    #[test]
    fn texture_fuzz_stays_inside_the_perimeter() {
        // A textured brush must keep its painterly splats inboard of ±width so they never
        // soften the crisp outer edge.
        let mut brush = BrushModel::default();
        brush.texture_strength = 1.0; // force texture splats to be generated
        let curve = CubicBezier::new(
            Vec2::new(100.0, 200.0),
            Vec2::new(180.0, 120.0),
            Vec2::new(300.0, 280.0),
            Vec2::new(420.0, 200.0),
        );
        let stroke =
            GaussianBezierStroke::new(StrokeId::default(), BezierSkeleton::single(curve), brush);
        let tex: Vec<_> = stroke
            .splats
            .iter()
            .filter(|s| s.role == SplatRole::Texture)
            .collect();
        assert!(!tex.is_empty(), "texture_strength > 0 should produce texture splats");
        for s in tex {
            assert!(
                s.u.abs() <= stroke.brush.radius,
                "texture splat at u={} escaped the half-width {}",
                s.u,
                stroke.brush.radius
            );
        }
    }

    #[test]
    fn hardness_is_recomputed_on_cache_rebuild() {
        // `hardness` is a #[serde(skip)] cache, so the load path (which calls
        // update_world_cache, not regenerate) must restore it. Simulate a freshly loaded
        // stroke by clearing the cache and rebuilding.
        let mut stroke = test_stroke();
        for s in &mut stroke.splats {
            s.hardness = 0.0;
        }
        stroke.update_world_cache();
        assert!(stroke.splats.iter().any(|s| s.role == SplatRole::Edge));
        for s in &stroke.splats {
            let expected = match s.role {
                SplatRole::Edge => stroke.brush.edge_hardness,
                _ => stroke.brush.hardness,
            };
            assert!((s.hardness - expected).abs() < 1e-6, "hardness not restored for {:?}", s.role);
        }
    }

    #[test]
    fn deterministic_regeneration() {
        let a = test_stroke();
        let b = test_stroke();
        assert_eq!(a.splats.len(), b.splats.len());
        for (sa, sb) in a.splats.iter().zip(&b.splats) {
            assert!((sa.center - sb.center).length() < 1e-6);
            assert_eq!(sa.role, sb.role);
        }
    }
}
