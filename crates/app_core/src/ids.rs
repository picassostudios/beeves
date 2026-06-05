//! Identifier types.
//!
//! Strokes and layers live in `SlotMap`s, so they use generational keys (stable across
//! insert/remove). Splats live in a per-stroke `Vec`, so a plain monotonic `u32` id is
//! enough — it just needs to be stable for the lifetime of a sculpt/edit interaction.

use slotmap::new_key_type;

new_key_type! {
    /// Generational key for a stroke inside `Document::strokes`.
    pub struct StrokeId;
    /// Generational key for a layer.
    pub struct LayerId;
}

/// Identifier for a splat within its parent stroke's `splats` vector.
pub type SplatId = u32;
