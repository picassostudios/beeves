//! Top-level document model: canvas, layers, and the stroke store.

use serde::{Deserialize, Serialize};
use slotmap::SlotMap;

use crate::bezier::BezierSkeleton;
use crate::brush::BrushModel;
use crate::ids::{LayerId, StrokeId};
use crate::selection::SelectionState;
use crate::stroke::GaussianBezierStroke;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColorSpace {
    Srgb,
    LinearSrgb,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlendMode {
    #[default]
    Normal,
    Multiply,
    Screen,
    Add,
    Subtract,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CanvasSettings {
    pub width: f32,
    pub height: f32,
    pub background: [f32; 4],
    pub color_space: ColorSpace,
}

impl Default for CanvasSettings {
    fn default() -> Self {
        Self {
            width: 1920.0,
            height: 1080.0,
            background: [1.0, 1.0, 1.0, 1.0],
            color_space: ColorSpace::Srgb,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Layer {
    pub id: LayerId,
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub blend_mode: BlendMode,
    pub stroke_ids: Vec<StrokeId>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Document {
    pub version: u32,
    pub canvas: CanvasSettings,
    pub layers: Vec<Layer>,
    pub strokes: SlotMap<StrokeId, GaussianBezierStroke>,
    #[serde(default)]
    pub selection: SelectionState,
    #[serde(skip)]
    layer_ids: SlotMap<LayerId, ()>,
}

impl Default for Document {
    fn default() -> Self {
        Self::new()
    }
}

impl Document {
    pub fn new() -> Self {
        Self {
            version: 1,
            canvas: CanvasSettings::default(),
            layers: Vec::new(),
            strokes: SlotMap::with_key(),
            selection: SelectionState::default(),
            layer_ids: SlotMap::with_key(),
        }
    }

    /// Create a new empty layer and return its id.
    pub fn add_layer(&mut self, name: impl Into<String>) -> LayerId {
        let id = self.layer_ids.insert(());
        self.layers.push(Layer {
            id,
            name: name.into(),
            visible: true,
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
            stroke_ids: Vec::new(),
        });
        id
    }

    fn layer_mut(&mut self, id: LayerId) -> Option<&mut Layer> {
        self.layers.iter_mut().find(|l| l.id == id)
    }

    /// Create a stroke from a skeleton + brush, attach it to `layer`, and return its id.
    pub fn add_stroke(
        &mut self,
        layer: LayerId,
        skeleton: BezierSkeleton,
        brush: BrushModel,
    ) -> StrokeId {
        let id = self
            .strokes
            .insert_with_key(|key| GaussianBezierStroke::new(key, skeleton, brush));
        if let Some(layer) = self.layer_mut(layer) {
            layer.stroke_ids.push(id);
        }
        id
    }

    /// Remove a stroke from the store and detach it from every layer. Returns the
    /// removed stroke if it existed. Used by interactive tools to drop a transient
    /// preview (e.g. an abandoned pen path).
    pub fn remove_stroke(&mut self, id: StrokeId) -> Option<GaussianBezierStroke> {
        let removed = self.strokes.remove(id);
        if removed.is_some() {
            for layer in &mut self.layers {
                layer.stroke_ids.retain(|&sid| sid != id);
            }
            self.selection.clear();
        }
        removed
    }

    pub fn stroke(&self, id: StrokeId) -> Option<&GaussianBezierStroke> {
        self.strokes.get(id)
    }

    pub fn stroke_mut(&mut self, id: StrokeId) -> Option<&mut GaussianBezierStroke> {
        self.strokes.get_mut(id)
    }

    /// Total splat count across all strokes (used for perf accounting).
    pub fn splat_count(&self) -> usize {
        self.strokes.values().map(|s| s.splats.len()).sum()
    }

    /// Rebuild all derived caches after a deserialization (arc-length tables + world
    /// caches are skipped on the wire and must be recomputed).
    pub fn rebuild_all_caches(&mut self) {
        for stroke in self.strokes.values_mut() {
            stroke.skeleton.rebuild_arc_length_table();
            stroke.update_world_cache();
        }
        // Re-register layer ids so freshly-created layers don't collide with loaded ones.
        self.layer_ids = SlotMap::with_key();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bezier::CubicBezier;
    use crate::math::Vec2;

    #[test]
    fn build_document_with_layer_and_stroke() {
        let mut doc = Document::new();
        let layer = doc.add_layer("Layer 1");
        let curve = CubicBezier::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(40.0, 80.0),
            Vec2::new(120.0, 80.0),
            Vec2::new(160.0, 0.0),
        );
        let sid = doc.add_stroke(layer, BezierSkeleton::single(curve), BrushModel::default());
        assert!(doc.stroke(sid).is_some());
        assert_eq!(doc.layers[0].stroke_ids, vec![sid]);
        assert!(doc.splat_count() > 0);
    }
}
