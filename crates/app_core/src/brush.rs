//! Brush model and 1D profile curves used to drive splat generation.

use serde::{Deserialize, Serialize};

/// A 1D profile evaluated along the stroke's normalized arc length `t in [0,1]`. Used
/// for width and opacity variation. Kept deliberately simple for the MVP.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CurveProfile {
    /// Constant value everywhere.
    Constant { value: f32 },
    /// Linear ramp from `start` (t=0) to `end` (t=1).
    Linear { start: f32, end: f32 },
    /// Smooth taper to zero at both ends (sin-shaped), peaking at the middle.
    Taper { peak: f32 },
    /// Piecewise-linear through `(t, value)` keyframes, sorted by `t`.
    Points { points: Vec<(f32, f32)> },
}

impl CurveProfile {
    pub fn eval(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            CurveProfile::Constant { value } => *value,
            CurveProfile::Linear { start, end } => start + (end - start) * t,
            CurveProfile::Taper { peak } => peak * (std::f32::consts::PI * t).sin(),
            CurveProfile::Points { points } => eval_points(points, t),
        }
    }
}

fn eval_points(points: &[(f32, f32)], t: f32) -> f32 {
    if points.is_empty() {
        return 1.0;
    }
    if t <= points[0].0 {
        return points[0].1;
    }
    if t >= points[points.len() - 1].0 {
        return points[points.len() - 1].1;
    }
    for w in points.windows(2) {
        let (t0, v0) = w[0];
        let (t1, v1) = w[1];
        if t >= t0 && t <= t1 {
            let f = if (t1 - t0).abs() < 1e-6 {
                0.0
            } else {
                (t - t0) / (t1 - t0)
            };
            return v0 + (v1 - v0) * f;
        }
    }
    points[points.len() - 1].1
}

impl Default for CurveProfile {
    fn default() -> Self {
        CurveProfile::Constant { value: 1.0 }
    }
}

/// Default boundary-ring hardness for brushes (and for documents saved before
/// `edge_hardness` existed): crisp, so perimeters read sharp out of the box.
fn default_edge_hardness() -> f32 {
    0.95
}

/// Parameters controlling how a stroke skeleton is realized as a splat cloud.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrushModel {
    pub base_color: [f32; 4],
    /// Half-width of the stroke at full width-profile.
    pub radius: f32,
    pub opacity: f32,
    /// Arc-length distance between splat stations along the curve. Smaller = more,
    /// finer splats (it also scales the per-splat tangential size, so coverage holds).
    pub spacing: f32,
    /// Interior edge hardness: 0 = very soft Gaussian falloff, 1 = crisp antialiased edge.
    /// Applied to the stroke's *interior* splats (core band + texture) so the inside of a
    /// line can stay soft/fuzzy. Consumed by the fragment shader (see `splat.wgsl`).
    pub hardness: f32,
    /// Hardness of the stroke's outer boundary ring. Kept high so the **perimeter** of a
    /// line is crisp even when `hardness` (the interior) is soft. Defaults near 1.0; lower
    /// it for a stroke whose silhouette should also be soft.
    #[serde(default = "default_edge_hardness")]
    pub edge_hardness: f32,
    /// Amount of random texture jitter (color + position).
    pub texture_strength: f32,
    /// Seed for deterministic jitter.
    pub seed: u64,
    #[serde(default)]
    pub width_profile: CurveProfile,
    #[serde(default)]
    pub opacity_profile: CurveProfile,
}

impl Default for BrushModel {
    fn default() -> Self {
        Self {
            base_color: [0.1, 0.2, 0.9, 1.0],
            radius: 24.0,
            opacity: 0.8,
            spacing: 4.0,
            hardness: 0.6,
            edge_hardness: default_edge_hardness(),
            texture_strength: 0.15,
            seed: 0xC0FFEE,
            width_profile: CurveProfile::default(),
            opacity_profile: CurveProfile::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_evaluate_in_range() {
        assert_eq!(CurveProfile::Constant { value: 0.7 }.eval(0.3), 0.7);
        let lin = CurveProfile::Linear { start: 0.0, end: 1.0 };
        assert!((lin.eval(0.5) - 0.5).abs() < 1e-6);
        let taper = CurveProfile::Taper { peak: 1.0 };
        assert!(taper.eval(0.0).abs() < 1e-6);
        assert!((taper.eval(0.5) - 1.0).abs() < 1e-6);
        let pts = CurveProfile::Points {
            points: vec![(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)],
        };
        assert!((pts.eval(0.25) - 0.5).abs() < 1e-6);
        assert!((pts.eval(0.75) - 0.5).abs() < 1e-6);
    }
}
