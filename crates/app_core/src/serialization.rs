//! JSON (`.gspf`) save/load.
//!
//! Cached, derivable data (arc-length tables, world-space splat centers/covariances) is
//! marked `#[serde(skip)]` on the model and rebuilt on load, keeping files compact and
//! making the editable model the single source of truth.

use crate::document::Document;

/// Serialize a document to pretty-printed JSON.
pub fn save_json(doc: &Document) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(doc)
}

/// Serialize a document to compact JSON.
pub fn save_json_compact(doc: &Document) -> Result<String, serde_json::Error> {
    serde_json::to_string(doc)
}

/// Deserialize a document and rebuild all derived caches.
pub fn load_json(json: &str) -> Result<Document, serde_json::Error> {
    let mut doc: Document = serde_json::from_str(json)?;
    doc.rebuild_all_caches();
    Ok(doc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bezier::{BezierSkeleton, CubicBezier};
    use crate::brush::BrushModel;
    use crate::math::Vec2;

    fn demo_doc() -> Document {
        let mut doc = Document::new();
        let layer = doc.add_layer("Layer 1");
        let curve = CubicBezier::new(
            Vec2::new(100.0, 200.0),
            Vec2::new(180.0, 120.0),
            Vec2::new(300.0, 280.0),
            Vec2::new(420.0, 200.0),
        );
        doc.add_stroke(layer, BezierSkeleton::single(curve), BrushModel::default());
        doc
    }

    #[test]
    fn round_trip_preserves_model_and_rebuilds_caches() {
        let original = demo_doc();
        let json = save_json(&original).unwrap();
        let loaded = load_json(&json).unwrap();

        assert_eq!(original.layers.len(), loaded.layers.len());
        assert_eq!(original.splat_count(), loaded.splat_count());

        // Caches (skipped on the wire) must be rebuilt: arc-length table populated and
        // world centers recomputed to match the original.
        let so = original.strokes.values().next().unwrap();
        let sl = loaded.strokes.values().next().unwrap();
        assert!(sl.skeleton.total_length() > 0.0);
        assert_eq!(so.splats.len(), sl.splats.len());
        for (a, b) in so.splats.iter().zip(&sl.splats) {
            assert!((a.center - b.center).length() < 1e-3, "world cache mismatch");
            assert_eq!(a.role, b.role);
            assert!((a.t - b.t).abs() < 1e-6);
        }
    }

    #[test]
    fn skipped_fields_keep_json_compact() {
        let doc = demo_doc();
        let json = save_json(&doc).unwrap();
        // Cached, derivable fields should not appear in the serialized form.
        assert!(!json.contains("arc_length_table"));
        assert!(!json.contains("inv_covariance"));
        assert!(!json.contains("covariance"));
    }
}
