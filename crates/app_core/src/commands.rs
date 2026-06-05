//! Undo/redo via full per-stroke snapshots.
//!
//! The MVP stores whole-stroke before/after snapshots (skeleton + splats + residuals).
//! That is memory-heavy but trivially correct; a delta-based scheme can replace it once
//! the edit set stabilizes. Snapshots cover *mutation* of existing strokes, which is
//! what every editing tool produces.

use crate::document::Document;
use crate::ids::StrokeId;
use crate::stroke::GaussianBezierStroke;

/// A captured copy of one stroke at a point in time.
#[derive(Clone, Debug)]
pub struct StrokeSnapshot {
    pub stroke: StrokeId,
    pub data: GaussianBezierStroke,
}

/// A reversible edit.
#[derive(Clone, Debug)]
pub enum Command {
    /// In-place mutation of an existing stroke.
    Mutate {
        before: StrokeSnapshot,
        after: StrokeSnapshot,
    },
}

/// Undo/redo stacks.
#[derive(Debug, Default)]
pub struct History {
    undo: Vec<Command>,
    redo: Vec<Command>,
    limit: usize,
}

impl History {
    pub fn new() -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
            limit: 256,
        }
    }

    /// Capture the current state of `stroke` so it can be committed after mutation.
    pub fn snapshot(doc: &Document, stroke: StrokeId) -> Option<StrokeSnapshot> {
        doc.stroke(stroke).map(|s| StrokeSnapshot {
            stroke,
            data: s.clone(),
        })
    }

    /// Record a completed mutation: `before` was captured before the change; the
    /// current state of the stroke in `doc` is the "after". Clears the redo stack.
    pub fn commit(&mut self, doc: &Document, before: StrokeSnapshot) {
        let Some(after) = Self::snapshot(doc, before.stroke) else {
            return;
        };
        self.undo.push(Command::Mutate { before, after });
        if self.undo.len() > self.limit {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    fn restore(doc: &mut Document, snap: &StrokeSnapshot) {
        if let Some(stroke) = doc.stroke_mut(snap.stroke) {
            *stroke = snap.data.clone();
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Undo the most recent command. Returns whether anything was undone.
    pub fn undo(&mut self, doc: &mut Document) -> bool {
        let Some(cmd) = self.undo.pop() else {
            return false;
        };
        match &cmd {
            Command::Mutate { before, .. } => Self::restore(doc, before),
        }
        self.redo.push(cmd);
        true
    }

    /// Redo the most recently undone command.
    pub fn redo(&mut self, doc: &mut Document) -> bool {
        let Some(cmd) = self.redo.pop() else {
            return false;
        };
        match &cmd {
            Command::Mutate { after, .. } => Self::restore(doc, after),
        }
        self.undo.push(cmd);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bezier::{BezierSkeleton, CubicBezier};
    use crate::brush::BrushModel;
    use crate::math::Vec2;

    #[test]
    fn undo_redo_round_trips_a_mutation() {
        let mut doc = Document::new();
        let layer = doc.add_layer("L");
        let curve = CubicBezier::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(30.0, 60.0),
            Vec2::new(90.0, 60.0),
            Vec2::new(120.0, 0.0),
        );
        let sid = doc.add_stroke(layer, BezierSkeleton::single(curve), BrushModel::default());
        let mut history = History::new();

        let before = History::snapshot(&doc, sid).unwrap();
        // Mutate: bend a handle and resync.
        {
            let stroke = doc.stroke_mut(sid).unwrap();
            stroke.skeleton.segments[0].p1.y += 100.0;
            stroke.skeleton.rebuild_arc_length_table();
            stroke.update_world_cache();
        }
        let changed_y = doc.stroke(sid).unwrap().skeleton.segments[0].p1.y;
        history.commit(&doc, before);

        assert!(history.undo(&mut doc));
        assert!((doc.stroke(sid).unwrap().skeleton.segments[0].p1.y - 60.0).abs() < 1e-3);

        assert!(history.redo(&mut doc));
        assert!((doc.stroke(sid).unwrap().skeleton.segments[0].p1.y - changed_y).abs() < 1e-3);
    }
}
