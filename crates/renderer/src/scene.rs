//! Incremental, GPU-resident layout of the splat field.
//!
//! This is the CPU half of the render fast-path. It owns a persistent "mirror" of every
//! splat in `GpuSplat` form, grouped into a **stable per-stroke slice** so that:
//!
//!   * adding a stroke onto an N-splat document is O(new splats) — it appends, it does not
//!     rebuild (#3);
//!   * editing a stroke re-packs and re-uploads only that stroke's slice (#3);
//!   * a camera-only frame touches no splat data at all — `reconcile` returns no dirty
//!     ranges and only `visible_ranges` (an O(strokes) AABB sweep) runs (#1/#2);
//!   * removed strokes free their slice for reuse, and a freed tail shrinks the mirror.
//!
//! The packing is parallelized across a stroke's splats on native (#6). All of this is
//! deviceless and unit-tested below; the wgpu glue that applies the dirty ranges and draws
//! the visible ranges lives in [`crate::SplatRenderer`].

use std::collections::HashMap;

use glam::Vec2;

use app_core::document::Document;
use app_core::ids::StrokeId;
use app_core::splat::{GaussianSplat, GpuSplat};

/// One stroke's reserved region in the mirror. `capacity >= count`; the `count..capacity`
/// tail (if any) holds stale entries that are simply never drawn, which lets a stroke whose
/// splat count shrinks (e.g. the live brush preview) stay in place without reallocating.
#[derive(Clone, Copy, Debug)]
struct Slot {
    offset: usize,
    capacity: usize,
    count: usize,
    /// World-space AABB (already padded by each splat's radius) for coarse view culling.
    min: Vec2,
    max: Vec2,
    /// Generation marker: a slot not touched in the current `reconcile` pass was removed.
    seen: u64,
    /// Stable per-stroke id packed into `GpuSplat::stroke_id`. Assigned once and kept for the
    /// life of the stroke so adding/removing other strokes never forces a re-pack here.
    stable_id: u32,
}

/// Persistent CPU mirror + per-stroke slot table. Reused across frames; never rebuilt.
#[derive(Default)]
pub struct SceneLayout {
    mirror: Vec<GpuSplat>,
    slots: HashMap<StrokeId, Slot>,
    /// Reclaimed `(offset, capacity)` regions available for reuse by later allocations.
    free: Vec<(usize, usize)>,
    gen: u64,
    next_stable_id: u32,
    /// Reused per-stroke pack buffer so packing never allocates per frame (#1).
    scratch: Vec<GpuSplat>,
}

impl SceneLayout {
    /// Reconcile the mirror against the document: pack only new or `gpu_upload`-dirty
    /// strokes, free removed ones, and clear the dirty flags we consumed. Returns the
    /// coalesced `(offset, len)` ranges (in splats) that must be re-uploaded to the GPU.
    pub fn reconcile(&mut self, doc: &mut Document) -> Vec<(usize, usize)> {
        self.gen = self.gen.wrapping_add(1);
        let gen = self.gen;
        let mut dirty: Vec<(usize, usize)> = Vec::new();

        for stroke in doc.strokes.values_mut() {
            let id = stroke.id;
            // Vector-rendered strokes are drawn as tessellated paths (see
            // `SplatRenderer::render_vector_paths`), not splats. Skip them here so they never
            // enter the resident splat mirror. Not marking the slot `seen` lets the removal
            // sweep reclaim a slot if a stroke is ever toggled to vector after the fact.
            if stroke.render_as_vector {
                continue;
            }
            let is_new = !self.slots.contains_key(&id);
            if !is_new && !stroke.dirty_flags.gpu_upload {
                // Unchanged: just mark it alive so the removal sweep keeps it.
                if let Some(slot) = self.slots.get_mut(&id) {
                    slot.seen = gen;
                }
                continue;
            }

            let stable_id = match self.slots.get(&id) {
                Some(slot) => slot.stable_id,
                None => {
                    let v = self.next_stable_id;
                    self.next_stable_id = self.next_stable_id.wrapping_add(1);
                    v
                }
            };
            // Pack into the reused scratch buffer (parallel on native, #6), then place.
            // Edge hardness is per-splat now (carried on `GaussianSplat`), so packing no
            // longer threads the brush hardness through.
            pack_into(&mut self.scratch, &stroke.splats, stable_id);
            let count = self.scratch.len();
            let (min, max) = aabb_of(&stroke.splats);

            let (offset, capacity) = self.alloc(id, count);
            self.mirror[offset..offset + count].copy_from_slice(&self.scratch);
            dirty.push((offset, count));

            self.slots.insert(
                id,
                Slot { offset, capacity, count, min, max, seen: gen, stable_id },
            );
            stroke.dirty_flags.gpu_upload = false;
        }

        // Removal sweep: any slot not seen this generation belongs to a deleted stroke.
        let dead: Vec<StrokeId> = self
            .slots
            .iter()
            .filter(|(_, s)| s.seen != gen)
            .map(|(k, _)| *k)
            .collect();
        for k in dead {
            if let Some(s) = self.slots.remove(&k) {
                self.free.push((s.offset, s.capacity));
            }
        }
        self.coalesce_free();
        self.reclaim_tail();

        coalesce(dirty)
    }

    /// Place `count` splats for stroke `id` and return `(offset, capacity)`.
    ///
    /// The ordering matters for the live brush/pen preview, which is regenerated with a
    /// *larger* splat count on every pointer move:
    ///   1. fits in its current slot -> reuse in place (also the plain edit case);
    ///   2. outgrew its slot but sits at the mirror tail -> extend the tail in place. This is
    ///      the preview's hot path; handling it here is what stops a long stroke from
    ///      orphaning its old region as a hole every frame (which ballooned the mirror and
    ///      forced a full resident re-upload — the cause of the tail flicker on long strokes);
    ///   3. outgrew its slot mid-buffer -> vacate the old region and reallocate with slack so
    ///      even a non-tail growing stroke reallocates logarithmically, not every frame.
    fn alloc(&mut self, id: StrokeId, count: usize) -> (usize, usize) {
        let cap_needed = count.max(1);
        if let Some(slot) = self.slots.get(&id).copied() {
            if slot.capacity >= count {
                return (slot.offset, slot.capacity);
            }
            if slot.offset + slot.capacity == self.mirror.len() {
                // Tail slot: grow it in place — no hole left behind.
                self.mirror.resize(slot.offset + cap_needed, GpuSplat::default());
                return (slot.offset, cap_needed);
            }
            // Mid-buffer growth: vacate the old region, reallocate with slack.
            self.free.push((slot.offset, slot.capacity));
            return self.allocate_fresh(grow_capacity(cap_needed));
        }
        // Brand-new stroke: pack it tightly (no slack) so the layout stays compact.
        self.allocate_fresh(cap_needed)
    }

    /// Carve `cap` splats from a freed hole (first fit, splitting any remainder), else extend
    /// the mirror tail. Returns `(offset, cap)`.
    fn allocate_fresh(&mut self, cap: usize) -> (usize, usize) {
        if let Some(pos) = self.free.iter().position(|&(_, c)| c >= cap) {
            let (off, c) = self.free.swap_remove(pos);
            if c > cap {
                self.free.push((off + cap, c - cap));
            }
            return (off, cap);
        }
        let off = self.mirror.len();
        self.mirror.resize(off + cap, GpuSplat::default());
        (off, cap)
    }

    /// Merge adjacent freed regions so holes don't fragment over a long session and a merged
    /// tail hole can be reclaimed.
    fn coalesce_free(&mut self) {
        if self.free.len() <= 1 {
            return;
        }
        self.free.sort_unstable_by_key(|&(off, _)| off);
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(self.free.len());
        for (off, cap) in self.free.drain(..) {
            if let Some(last) = merged.last_mut() {
                if last.0 + last.1 == off {
                    last.1 += cap;
                    continue;
                }
            }
            merged.push((off, cap));
        }
        self.free = merged;
    }

    /// Shrink the mirror when its tail is a freed region (e.g. after the brush preview
    /// stroke at the end is discarded), so the resident buffer does not grow unboundedly.
    fn reclaim_tail(&mut self) {
        loop {
            let end = self.mirror.len();
            match self.free.iter().position(|&(off, cap)| off + cap == end) {
                Some(pos) => {
                    let (off, _) = self.free.swap_remove(pos);
                    self.mirror.truncate(off);
                }
                None => break,
            }
        }
    }

    /// Visible draw ranges as `(start_instance, count)` pairs, contiguous slots merged. A
    /// slot is visible if its padded AABB intersects `[view_min, view_max]`. O(strokes) and
    /// allocation-light — this is the only per-frame work on a camera-only frame.
    pub fn visible_ranges(&self, view_min: Vec2, view_max: Vec2) -> Vec<(u32, u32)> {
        let mut vis: Vec<(usize, usize)> = self
            .slots
            .values()
            .filter(|s| s.count > 0 && aabb_intersects(s.min, s.max, view_min, view_max))
            .map(|s| (s.offset, s.count))
            .collect();
        vis.sort_unstable_by_key(|&(o, _)| o);

        let mut ranges: Vec<(u32, u32)> = Vec::new();
        for (o, c) in vis {
            if let Some(last) = ranges.last_mut() {
                if (last.0 + last.1) as usize == o {
                    last.1 += c as u32;
                    continue;
                }
            }
            ranges.push((o as u32, c as u32));
        }
        ranges
    }

    /// The full mirror slice (resident-buffer source of truth).
    pub fn mirror(&self) -> &[GpuSplat] {
        &self.mirror
    }

    /// Number of splat slots backing the mirror (including reserved tails / holes).
    pub fn len(&self) -> usize {
        self.mirror.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mirror.is_empty()
    }

    /// Drop all layout state. Used when the document is wholesale replaced (e.g. on load) so
    /// stale slots from the previous document don't linger.
    pub fn clear(&mut self) {
        self.mirror.clear();
        self.slots.clear();
        self.free.clear();
        self.scratch.clear();
        // `gen`/`next_stable_id` keep advancing — they only need to stay monotonic.
    }
}

/// Pack a stroke's splats into `out` (reused), preserving order. Parallel on native (#6).
#[cfg(not(target_arch = "wasm32"))]
fn pack_into(out: &mut Vec<GpuSplat>, splats: &[GaussianSplat], stable_id: u32) {
    use rayon::prelude::*;
    splats
        .par_iter()
        .map(|s| GpuSplat::from_splat(s, stable_id))
        .collect_into_vec(out);
}

/// Serial fallback for wasm (no thread pool without cross-origin isolation + wasm threads).
#[cfg(target_arch = "wasm32")]
fn pack_into(out: &mut Vec<GpuSplat>, splats: &[GaussianSplat], stable_id: u32) {
    out.clear();
    out.extend(splats.iter().map(|s| GpuSplat::from_splat(s, stable_id)));
}

/// Padded world-space AABB over a stroke's splat footprints. The `+2` matches the EWA
/// screen-space low-pass pad used by the old per-splat cull, so nothing visible is dropped.
fn aabb_of(splats: &[GaussianSplat]) -> (Vec2, Vec2) {
    if splats.is_empty() {
        return (Vec2::ZERO, Vec2::ZERO);
    }
    let mut min = Vec2::splat(f32::INFINITY);
    let mut max = Vec2::splat(f32::NEG_INFINITY);
    for s in splats {
        let r = Vec2::splat(s.radius_px + 2.0);
        min = min.min(s.center - r);
        max = max.max(s.center + r);
    }
    (min, max)
}

fn aabb_intersects(amin: Vec2, amax: Vec2, bmin: Vec2, bmax: Vec2) -> bool {
    amax.x >= bmin.x && amin.x <= bmax.x && amax.y >= bmin.y && amin.y <= bmax.y
}

/// Capacity to reserve when reallocating a *growing* stroke away from the tail: 1.5× with a
/// small floor, so repeated mid-buffer growth reallocates logarithmically, not every frame.
fn grow_capacity(count: usize) -> usize {
    (count + count / 2).max(count + 16)
}

/// Merge sorted/overlapping/adjacent `(offset, len)` ranges to minimize `write_buffer` calls.
fn coalesce(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    if ranges.len() <= 1 {
        return ranges;
    }
    ranges.sort_unstable_by_key(|&(o, _)| o);
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (o, l) in ranges {
        if let Some(last) = out.last_mut() {
            if o <= last.0 + last.1 {
                let end = (o + l).max(last.0 + last.1);
                last.1 = end - last.0;
                continue;
            }
        }
        out.push((o, l));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_core::brush::BrushModel;
    use app_core::document::Document;
    use app_core::{BezierSkeleton, CubicBezier};

    fn stroke_skeleton(x0: f32) -> BezierSkeleton {
        stroke_skeleton_len(x0, 160.0)
    }

    /// A roughly `len`-long stroke from `x0`. Longer `len` => more stations => more splats,
    /// which is how we simulate a brush preview growing as the user draws.
    fn stroke_skeleton_len(x0: f32, len: f32) -> BezierSkeleton {
        BezierSkeleton::single(CubicBezier::new(
            Vec2::new(x0, 0.0),
            Vec2::new(x0 + len * 0.25, 80.0),
            Vec2::new(x0 + len * 0.75, 80.0),
            Vec2::new(x0 + len, 0.0),
        ))
    }

    fn doc_with_one_stroke() -> (Document, StrokeId) {
        let mut doc = Document::new();
        let layer = doc.add_layer("L");
        let sid = doc.add_stroke(layer, stroke_skeleton(0.0), BrushModel::default());
        (doc, sid)
    }

    const HUGE_MIN: Vec2 = Vec2::new(-1.0e6, -1.0e6);
    const HUGE_MAX: Vec2 = Vec2::new(1.0e6, 1.0e6);

    #[test]
    fn first_reconcile_packs_full_stroke() {
        let (mut doc, sid) = doc_with_one_stroke();
        let n = doc.stroke(sid).unwrap().splats.len();
        assert!(n > 0);

        let mut scene = SceneLayout::default();
        let dirty = scene.reconcile(&mut doc);

        assert_eq!(dirty, vec![(0, n)], "new stroke uploads its whole slice once");
        assert_eq!(scene.len(), n, "mirror sized exactly to the one stroke");
        assert_eq!(scene.visible_ranges(HUGE_MIN, HUGE_MAX), vec![(0, n as u32)]);
    }

    #[test]
    fn camera_only_frame_uploads_nothing() {
        let (mut doc, _sid) = doc_with_one_stroke();
        let mut scene = SceneLayout::default();
        let _ = scene.reconcile(&mut doc);

        // No edits -> the second reconcile must find nothing dirty (the flag was consumed).
        let dirty = scene.reconcile(&mut doc);
        assert!(dirty.is_empty(), "unchanged document re-uploads nothing");
    }

    #[test]
    fn edit_in_place_uploads_only_that_stroke() {
        let (mut doc, sid) = doc_with_one_stroke();
        let n = doc.stroke(sid).unwrap().splats.len();
        let mut scene = SceneLayout::default();
        let _ = scene.reconcile(&mut doc);

        // An in-place edit (same splat count) re-marks the stroke dirty.
        doc.stroke_mut(sid).unwrap().dirty_flags.gpu_upload = true;
        let dirty = scene.reconcile(&mut doc);
        assert_eq!(dirty, vec![(0, n)], "edit re-uploads exactly the stroke's slice");
        assert_eq!(scene.len(), n, "no growth on a same-size edit");
    }

    #[test]
    fn adding_a_stroke_appends_without_touching_the_first() {
        let (mut doc, sid_a) = doc_with_one_stroke();
        let n_a = doc.stroke(sid_a).unwrap().splats.len();
        let mut scene = SceneLayout::default();
        let _ = scene.reconcile(&mut doc);

        let layer = doc.layers[0].id;
        let sid_b = doc.add_stroke(layer, stroke_skeleton(1000.0), BrushModel::default());
        let n_b = doc.stroke(sid_b).unwrap().splats.len();

        let dirty = scene.reconcile(&mut doc);
        // Only the new stroke is dirty, and it is appended after the first (offset == n_a).
        assert_eq!(dirty, vec![(n_a, n_b)], "add is O(new): only the appended slice uploads");
        assert_eq!(scene.len(), n_a + n_b);
    }

    #[test]
    fn removing_a_stroke_frees_its_slice_and_stops_drawing_it() {
        let mut doc = Document::new();
        let layer = doc.add_layer("L");
        let sid_a = doc.add_stroke(layer, stroke_skeleton(0.0), BrushModel::default());
        let sid_b = doc.add_stroke(layer, stroke_skeleton(1000.0), BrushModel::default());
        let n_a = doc.stroke(sid_a).unwrap().splats.len();
        let n_b = doc.stroke(sid_b).unwrap().splats.len();

        let mut scene = SceneLayout::default();
        let _ = scene.reconcile(&mut doc);
        assert_eq!(scene.visible_ranges(HUGE_MIN, HUGE_MAX), vec![(0, (n_a + n_b) as u32)]);

        // Remove the FIRST stroke (a mid-buffer hole). Its slice must drop out of drawing.
        doc.remove_stroke(sid_a);
        let dirty = scene.reconcile(&mut doc);
        assert!(dirty.is_empty(), "a pure removal re-uploads nothing");
        assert_eq!(
            scene.visible_ranges(HUGE_MIN, HUGE_MAX),
            vec![(n_a as u32, n_b as u32)],
            "only the surviving stroke (still at its stable offset) draws"
        );
    }

    #[test]
    fn new_stroke_reuses_a_freed_hole() {
        let mut doc = Document::new();
        let layer = doc.add_layer("L");
        let sid_a = doc.add_stroke(layer, stroke_skeleton(0.0), BrushModel::default());
        let _sid_b = doc.add_stroke(layer, stroke_skeleton(1000.0), BrushModel::default());
        let n_a = doc.stroke(sid_a).unwrap().splats.len();

        let mut scene = SceneLayout::default();
        let _ = scene.reconcile(&mut doc);
        let len_two = scene.len();

        // Free the first stroke's hole, then add a same-shaped stroke: it should slot into
        // the hole rather than grow the mirror.
        doc.remove_stroke(sid_a);
        let _ = scene.reconcile(&mut doc);
        let sid_c = doc.add_stroke(layer, stroke_skeleton(0.0), BrushModel::default());
        let n_c = doc.stroke(sid_c).unwrap().splats.len();
        assert_eq!(n_c, n_a, "same brush + skeleton => same splat count");

        let dirty = scene.reconcile(&mut doc);
        assert_eq!(dirty, vec![(0, n_c)], "reused the freed hole at offset 0");
        assert_eq!(scene.len(), len_two, "no growth: the hole absorbed the new stroke");
    }

    #[test]
    fn offscreen_strokes_are_culled_from_draw() {
        let mut doc = Document::new();
        let layer = doc.add_layer("L");
        // Two strokes far apart in x.
        doc.add_stroke(layer, stroke_skeleton(0.0), BrushModel::default());
        doc.add_stroke(layer, stroke_skeleton(10_000.0), BrushModel::default());

        let mut scene = SceneLayout::default();
        let _ = scene.reconcile(&mut doc);

        // A view box around only the first stroke must exclude the far one.
        let ranges = scene.visible_ranges(Vec2::new(-50.0, -50.0), Vec2::new(300.0, 200.0));
        assert_eq!(ranges.len(), 1, "only the near stroke is visible");
        assert_eq!(ranges[0].0, 0, "and it is the first slice");

        // A view far from everything draws nothing.
        assert!(scene
            .visible_ranges(Vec2::new(1.0e5, 1.0e5), Vec2::new(1.1e5, 1.1e5))
            .is_empty());
    }

    #[test]
    fn growing_preview_stroke_stays_compact_and_fully_drawn() {
        // Regression: the live brush preview regenerates a *growing* stroke and re-marks it
        // dirty on every pointer move. The mirror must track the stroke's *current* count —
        // not the sum of counts over all frames — by growing the tail slot in place. Before
        // the fix this orphaned a hole each frame, ballooning the mirror and forcing a full
        // resident re-upload (the long-stroke glitch / tail flicker).
        let mut doc = Document::new();
        let layer = doc.add_layer("L");
        let sid = doc.add_stroke(layer, stroke_skeleton(0.0), BrushModel::default());

        let mut scene = SceneLayout::default();
        let _ = scene.reconcile(&mut doc);

        let mut last = 0usize;
        for i in 1..=50 {
            // Extend the stroke (longer skeleton => more stations => more splats) and mark it
            // dirty exactly as `regenerate_splats` does mid-drag.
            let skel = stroke_skeleton_len(0.0, 160.0 + i as f32 * 120.0);
            {
                let stroke = doc.stroke_mut(sid).unwrap();
                stroke.skeleton = skel;
                stroke.regenerate_splats();
            }
            let count = doc.stroke(sid).unwrap().splats.len();
            assert!(count >= last, "frame {i}: stroke should only grow");
            last = count;

            let dirty = scene.reconcile(&mut doc);
            // Only this stroke uploads, only its current slice, at a stable offset 0.
            assert_eq!(dirty, vec![(0, count)], "frame {i}: uploads exactly the stroke");
            // The mirror stays compact — equal to the current count, no hole accumulation.
            assert_eq!(scene.len(), count, "frame {i}: mirror compact (no per-frame bloat)");
            // The whole stroke is still drawn as one contiguous range.
            assert_eq!(
                scene.visible_ranges(HUGE_MIN, HUGE_MAX),
                vec![(0, count as u32)],
                "frame {i}: entire stroke visible"
            );
        }
    }
}
