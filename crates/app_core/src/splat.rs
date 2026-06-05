//! The Gaussian splat: a curve-attached anisotropic 2D Gaussian.
//!
//! A splat is never an anonymous point in a cloud — it always carries its parent
//! stroke, its curve-local coordinates `(t, u, v)`, a residual deformation, and a
//! `role`. That metadata is exactly what makes reverse editing (splat -> curve)
//! tractable.

use serde::{Deserialize, Serialize};

use crate::ids::{SplatId, StrokeId};
use crate::math::{Mat2, Vec2};

/// Functional role of a splat within its stroke. Drives the reverse-sync heuristics:
/// moving `Core` splats reads as a structural curve edit; moving `Edge`/`Texture`
/// splats reads as width/detail.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplatRole {
    Core,
    Edge,
    Texture,
    Highlight,
    Shadow,
    ResidualDetail,
}

impl SplatRole {
    /// How strongly a splat of this role signals a *structural* (centerline) edit when
    /// moved. Used by the coherence classifier's `core_weight_ratio`.
    pub fn structural_weight(self) -> f32 {
        match self {
            SplatRole::Core => 1.0,
            SplatRole::Edge => 0.25,
            SplatRole::Texture => 0.1,
            SplatRole::Highlight => 0.1,
            SplatRole::Shadow => 0.1,
            SplatRole::ResidualDetail => 0.0,
        }
    }
}

/// CPU-side splat. World-space `center`/`covariance`/`radius_px` are **caches** derived
/// from the skeleton + local coords + residuals; they are recomputed by
/// `stroke::update_splat_world_cache` and are not serialized. The inverse covariance is
/// deliberately **not** cached: the GPU derives it in-shader, and CPU hit-testing computes
/// it on the fly via [`GaussianSplat::response_q`], so an edit never pays a per-splat
/// matrix inverse.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GaussianSplat {
    pub id: SplatId,
    pub parent_stroke: StrokeId,

    // --- Curve-local attachment ---
    /// Normalized arc-length position along the skeleton, `[0,1]`.
    pub t: f32,
    /// Normal offset from the centerline.
    pub u: f32,
    /// Tangent offset (optional jitter along the curve).
    pub v: f32,

    // --- Residual deformation (survives curve edits when stored in local frame) ---
    pub residual_local: Vec2,
    pub residual_world: Vec2,

    // --- Shape in the local frame ---
    pub sigma_tangent: f32,
    pub sigma_normal: f32,
    pub rotation_jitter: f32,

    // --- Appearance ---
    pub color: [f32; 4],
    pub alpha: f32,

    // --- Editing metadata ---
    pub role: SplatRole,
    #[serde(default)]
    pub selected: bool,
    #[serde(default)]
    pub locked: bool,

    // --- Cached world-space render data (not serialized) ---
    #[serde(skip)]
    pub center: Vec2,
    #[serde(skip)]
    pub covariance: Mat2,
    #[serde(skip)]
    pub radius_px: f32,
    /// Per-splat edge hardness in `[0,1]`. A cached appearance value derived from the parent
    /// brush + this splat's `role` (crisp boundary ring vs soft interior), so the perimeter
    /// of a stroke can be crisp while its interior stays fuzzy. Recomputed by
    /// `stroke::update_splat_world_cache`, hence not serialized.
    #[serde(skip)]
    pub hardness: f32,
}

impl GaussianSplat {
    /// A bare splat attached to `stroke` with id `id`; geometry/appearance filled in by
    /// the brush generator.
    pub fn new(id: SplatId, stroke: StrokeId) -> Self {
        Self {
            id,
            parent_stroke: stroke,
            t: 0.0,
            u: 0.0,
            v: 0.0,
            residual_local: Vec2::ZERO,
            residual_world: Vec2::ZERO,
            sigma_tangent: 1.0,
            sigma_normal: 1.0,
            rotation_jitter: 0.0,
            color: [0.0, 0.0, 0.0, 1.0],
            alpha: 1.0,
            role: SplatRole::Core,
            selected: false,
            locked: false,
            center: Vec2::ZERO,
            covariance: Mat2::IDENTITY,
            radius_px: 1.0,
            hardness: 0.0,
        }
    }

    /// Squared Mahalanobis distance of the world-space offset `d` under this splat's
    /// covariance, i.e. `dᵀ Σ⁻¹ d`. Derived directly from the forward covariance
    /// `Σ = [a b; b c]` (a 2×2 inverse is three multiplies) so no inverse needs to be
    /// cached or kept in sync. `q < 9` is the standard 3σ footprint test.
    pub fn response_q(&self, d: Vec2) -> f32 {
        let a = self.covariance.x_axis.x;
        let b = self.covariance.x_axis.y;
        let c = self.covariance.y_axis.y;
        let det = (a * c - b * b).max(1e-12);
        (c * d.x * d.x - 2.0 * b * d.x * d.y + a * d.y * d.y) / det
    }
}

/// Pack an `[r, g, b, a]` color in `[0,1]` into a little-endian `RGBA8` word: R in the
/// low byte, A in the high byte. Mirrors WGSL `unpack4x8unorm`, whose `.x` reads the low
/// byte — so `unpack4x8unorm(color).xyz` is `(r, g, b)` and `.w` is `a`.
pub fn pack_rgba8(c: [f32; 4]) -> u32 {
    let to_u8 = |v: f32| ((v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32) & 0xff;
    to_u8(c[0]) | (to_u8(c[1]) << 8) | (to_u8(c[2]) << 16) | (to_u8(c[3]) << 24)
}

/// Compact, GPU-friendly instance record (44-byte stride). The covariance is symmetric, so
/// only three entries are stored (`[a b; b c]`); the **inverse** is intentionally *not*
/// stored — the visible fragment shader derives it after a screen-space low-pass, and the
/// picking shader inverts the 2×2 directly (three multiplies). Color is packed to a single
/// `RGBA8` word rather than four floats. Built in a later (rendering) phase; kept here so
/// the packing logic lives with the model and can be unit-tested.
///
/// Field order/sizes are mirrored exactly by the `Splat` struct in `splat.wgsl` /
/// `picking.wgsl`. All fields are 4-byte scalars (no `vec` types) so the std-layout
/// alignment is a uniform 4 bytes and the 44-byte stride matches `#[repr(C)]` on both sides.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSplat {
    pub center: [f32; 2],
    /// Forward (non-inverse) covariance `[a b; b c]` in world units, symmetric.
    pub cov_a: f32,
    pub cov_b: f32,
    pub cov_c: f32,
    /// Premultiplied-agnostic color packed as little-endian `RGBA8` (see [`pack_rgba8`]).
    pub color: u32,
    pub alpha: f32,
    pub radius: f32,
    pub stroke_id: u32,
    pub flags: u32,
    /// Edge hardness in `[0,1]` (from the parent brush). 0 = soft Gaussian falloff,
    /// 1 = crisp antialiased edge. Consumed only by the fragment shader.
    pub hardness: f32,
}

impl GpuSplat {
    pub const FLAG_SELECTED: u32 = 1 << 0;
    pub const FLAG_LOCKED: u32 = 1 << 1;

    /// Pack a CPU splat into its GPU instance form. `stroke_index` is a stable per-stroke
    /// key (the generational `StrokeId` is not GPU-friendly). Edge hardness is read from the
    /// splat's own cached `hardness` (derived from the brush + role in
    /// `stroke::update_splat_world_cache`), so the crisp boundary ring and the soft interior
    /// of one stroke can carry different hardness within a single draw.
    pub fn from_splat(s: &GaussianSplat, stroke_index: u32) -> Self {
        let cov = s.covariance;
        let mut flags = 0;
        if s.selected {
            flags |= Self::FLAG_SELECTED;
        }
        if s.locked {
            flags |= Self::FLAG_LOCKED;
        }
        GpuSplat {
            center: [s.center.x, s.center.y],
            // glam stores column-major: x_axis = first column = (m00, m10). Forward
            // covariance, symmetric: [a b; b c] = [m00 m10; m10 m11].
            cov_a: cov.x_axis.x,
            cov_b: cov.x_axis.y,
            cov_c: cov.y_axis.y,
            color: pack_rgba8(s.color),
            alpha: s.alpha,
            radius: s.radius_px,
            stroke_id: stroke_index,
            flags,
            hardness: s.hardness.clamp(0.0, 1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::covariance_from_sigmas;

    #[test]
    fn gpu_pack_keeps_forward_covariance_and_color() {
        let mut s = GaussianSplat::new(0, StrokeId::default());
        s.covariance = covariance_from_sigmas(0.5, 5.0, 2.0);
        s.center = Vec2::new(10.0, 20.0);
        s.radius_px = 15.0;
        s.color = [1.0, 0.0, 0.0, 1.0];
        s.selected = true;
        s.hardness = 0.6;
        let g = GpuSplat::from_splat(&s, 3);
        assert_eq!(g.center, [10.0, 20.0]);
        assert_eq!(g.stroke_id, 3);
        assert!((g.hardness - 0.6).abs() < 1e-6, "per-splat hardness is packed through");
        assert!(g.flags & GpuSplat::FLAG_SELECTED != 0);
        // Forward covariance is a valid (positive-definite, symmetric) ellipse.
        assert!(g.cov_a * g.cov_c - g.cov_b * g.cov_b > 0.0);
        // RGBA8 packing: red in the low byte, alpha in the high byte, green zeroed.
        assert_eq!(g.color & 0xff, 255);
        assert_eq!((g.color >> 8) & 0xff, 0);
        assert_eq!((g.color >> 24) & 0xff, 255);
    }

    #[test]
    fn response_q_matches_explicit_inverse() {
        // response_q must equal dᵀ Σ⁻¹ d computed via the cached-free path against glam's
        // own matrix inverse, for a rotated anisotropic covariance.
        let mut s = GaussianSplat::new(0, StrokeId::default());
        s.covariance = covariance_from_sigmas(0.6, 7.0, 2.0);
        let inv = s.covariance.inverse();
        for d in [Vec2::new(3.0, -1.0), Vec2::new(0.0, 4.0), Vec2::new(-5.0, 2.5)] {
            let expected = d.dot(inv * d);
            assert!((s.response_q(d) - expected).abs() < 1e-3, "q mismatch at {d:?}");
        }
    }
}
