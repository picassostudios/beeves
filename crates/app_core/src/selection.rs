//! Selection state, CPU hit-testing, and a coarse spatial grid for acceleration.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::document::Document;
use crate::ids::{SplatId, StrokeId};
use crate::math::Vec2;

/// Current selection. Splats are addressed by `(stroke, splat)` since splat ids are
/// only unique within their parent stroke.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SelectionState {
    pub strokes: Vec<StrokeId>,
    pub splats: Vec<(StrokeId, SplatId)>,
}

impl SelectionState {
    pub fn clear(&mut self) {
        self.strokes.clear();
        self.splats.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.strokes.is_empty() && self.splats.is_empty()
    }
}

/// Result of a point hit-test against the splat field.
#[derive(Clone, Copy, Debug)]
pub struct SplatHit {
    pub stroke: StrokeId,
    pub splat: SplatId,
    /// Gaussian response at the query point (alpha * exp(-q/2)); higher = stronger.
    pub score: f32,
}

/// Find the strongest-contributing splat at world point `p`, considering only points
/// inside each splat's 3-sigma footprint (`q < 9`).
pub fn hit_test_splat(doc: &Document, p: Vec2) -> Option<SplatHit> {
    let mut best: Option<SplatHit> = None;
    for stroke in doc.strokes.values() {
        for splat in &stroke.splats {
            let d = p - splat.center;
            let q = splat.response_q(d);
            if q < 9.0 {
                let score = splat.alpha * (-0.5 * q).exp();
                if best.is_none_or(|b| score > b.score) {
                    best = Some(SplatHit {
                        stroke: stroke.id,
                        splat: splat.id,
                        score,
                    });
                }
            }
        }
    }
    best
}

/// Reference to a splat by `(stroke, splat)`.
pub type SplatRef = (StrokeId, SplatId);

/// A uniform-grid spatial index over splat centers, for accelerating picking and
/// sculpt-radius queries on large documents. Rebuilt on demand (cheap for the MVP).
#[derive(Clone, Debug)]
pub struct SpatialGrid {
    pub cell_size: f32,
    cells: HashMap<(i32, i32), Vec<SplatRef>>,
}

impl SpatialGrid {
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size: cell_size.max(1.0),
            cells: HashMap::new(),
        }
    }

    fn cell_of(&self, p: Vec2) -> (i32, i32) {
        (
            (p.x / self.cell_size).floor() as i32,
            (p.y / self.cell_size).floor() as i32,
        )
    }

    /// Build the index from all splats in the document.
    pub fn build(doc: &Document, cell_size: f32) -> Self {
        let mut grid = Self::new(cell_size);
        for stroke in doc.strokes.values() {
            for splat in &stroke.splats {
                let cell = grid.cell_of(splat.center);
                grid.cells.entry(cell).or_default().push((stroke.id, splat.id));
            }
        }
        grid
    }

    /// Candidate splats within `radius` of `p` (a superset; caller refines by distance).
    pub fn query_radius(&self, p: Vec2, radius: f32) -> Vec<SplatRef> {
        let r_cells = (radius / self.cell_size).ceil() as i32;
        let (cx, cy) = self.cell_of(p);
        let mut out = Vec::new();
        for dy in -r_cells..=r_cells {
            for dx in -r_cells..=r_cells {
                if let Some(refs) = self.cells.get(&(cx + dx, cy + dy)) {
                    out.extend_from_slice(refs);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bezier::{BezierSkeleton, CubicBezier};
    use crate::brush::BrushModel;

    fn doc_with_stroke() -> Document {
        let mut doc = Document::new();
        let layer = doc.add_layer("L");
        let curve = CubicBezier::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(40.0, 0.0),
            Vec2::new(80.0, 0.0),
            Vec2::new(120.0, 0.0),
        );
        doc.add_stroke(layer, BezierSkeleton::single(curve), BrushModel::default());
        doc
    }

    #[test]
    fn hit_test_finds_splat_near_curve() {
        let doc = doc_with_stroke();
        // A point right on the centerline near the middle should hit something.
        let hit = hit_test_splat(&doc, Vec2::new(60.0, 0.0));
        assert!(hit.is_some());
        // A point far away hits nothing.
        assert!(hit_test_splat(&doc, Vec2::new(10_000.0, 10_000.0)).is_none());
    }

    #[test]
    fn spatial_grid_returns_superset_of_radius_query() {
        let doc = doc_with_stroke();
        let grid = SpatialGrid::build(&doc, 16.0);
        let near = grid.query_radius(Vec2::new(60.0, 0.0), 40.0);
        assert!(!near.is_empty());
    }
}
