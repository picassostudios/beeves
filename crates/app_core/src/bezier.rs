//! Bezier skeleton geometry.
//!
//! User-facing splat positions use a normalized **arc-length** coordinate `t in [0,1]`
//! so the splat distribution stays stable as the curve bends. Internally each cubic
//! segment is parameterized by its native `s in [0,1]`. The [`ArcLengthTable`] maps
//! between the two and is rebuilt whenever the skeleton changes.

use serde::{Deserialize, Serialize};

use crate::math::Vec2;

/// A single cubic Bezier segment.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CubicBezier {
    pub p0: Vec2,
    pub p1: Vec2,
    pub p2: Vec2,
    pub p3: Vec2,
}

impl CubicBezier {
    pub fn new(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2) -> Self {
        Self { p0, p1, p2, p3 }
    }

    /// Position at native parameter `s in [0,1]`.
    pub fn point(&self, s: f32) -> Vec2 {
        let mt = 1.0 - s;
        let b0 = mt * mt * mt;
        let b1 = 3.0 * mt * mt * s;
        let b2 = 3.0 * mt * s * s;
        let b3 = s * s * s;
        self.p0 * b0 + self.p1 * b1 + self.p2 * b2 + self.p3 * b3
    }

    /// First derivative `dP/ds` (unnormalized tangent / velocity).
    pub fn velocity(&self, s: f32) -> Vec2 {
        let mt = 1.0 - s;
        (self.p1 - self.p0) * (3.0 * mt * mt)
            + (self.p2 - self.p1) * (6.0 * mt * s)
            + (self.p3 - self.p2) * (3.0 * s * s)
    }

    /// Unit tangent at `s`. Falls back to +x if the velocity is degenerate.
    pub fn tangent(&self, s: f32) -> Vec2 {
        let v = self.velocity(s);
        if v.length_squared() > 1e-12 {
            v.normalize()
        } else {
            Vec2::X
        }
    }

    /// The `i`-th control point (`0..=3`), used by the fitter.
    pub fn control(&self, i: usize) -> Vec2 {
        match i {
            0 => self.p0,
            1 => self.p1,
            2 => self.p2,
            3 => self.p3,
            _ => panic!("control index out of range"),
        }
    }

    pub fn set_control(&mut self, i: usize, v: Vec2) {
        match i {
            0 => self.p0 = v,
            1 => self.p1 = v,
            2 => self.p2 = v,
            3 => self.p3 = v,
            _ => panic!("control index out of range"),
        }
    }
}

/// The cubic Bernstein basis at parameter `s`: `[B0, B1, B2, B3]`.
pub fn bernstein(s: f32) -> [f32; 4] {
    let mt = 1.0 - s;
    [mt * mt * mt, 3.0 * mt * mt * s, 3.0 * mt * s * s, s * s * s]
}

/// An orthonormal local frame on the curve at some arc-length position.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub position: Vec2,
    /// Unit tangent (direction of increasing `t`).
    pub tangent: Vec2,
    /// Unit normal (`tangent` rotated +90 degrees).
    pub normal: Vec2,
    /// Tangent angle in radians, for covariance rotation.
    pub angle: f32,
}

/// Per-anchor metadata (corner vs smooth). Minimal for the MVP.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct AnchorMeta {
    pub corner: bool,
}

/// Which of an anchor's three editable control points a direct-edit gesture addresses.
///
/// Anchors are the on-curve join points; each anchor owns up to two off-curve tangent
/// handles. The *out* handle (`p1` of the segment leaving the anchor) shapes the curve as
/// it departs; the *in* handle (`p2` of the segment arriving at the anchor) shapes it as
/// it arrives. The first anchor of an open path has no in-handle and the last has no
/// out-handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandleKind {
    /// The on-curve anchor point itself.
    Anchor,
    /// Tangent handle leaving the anchor toward the next segment (`p1`).
    OutHandle,
    /// Tangent handle arriving at the anchor from the previous segment (`p2`).
    InHandle,
}

/// A reference to one editable control point: an anchor index plus which of its points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControlPointRef {
    /// Anchor (join) index in `0..anchor_count()`.
    pub anchor: usize,
    pub kind: HandleKind,
}

impl ControlPointRef {
    pub fn anchor(anchor: usize) -> Self {
        Self {
            anchor,
            kind: HandleKind::Anchor,
        }
    }
    pub fn out_handle(anchor: usize) -> Self {
        Self {
            anchor,
            kind: HandleKind::OutHandle,
        }
    }
    pub fn in_handle(anchor: usize) -> Self {
        Self {
            anchor,
            kind: HandleKind::InHandle,
        }
    }
}

/// One sample of the arc-length parameterization.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct ArcSample {
    /// Normalized arc-length position in `[0,1]` across the whole skeleton.
    pub global_t: f32,
    pub segment_index: u32,
    /// Native parameter within the segment.
    pub local_s: f32,
    /// Cumulative arc length up to this sample.
    pub length: f32,
    pub position: Vec2,
    pub tangent: Vec2,
}

/// Lookup table mapping normalized arc length to `(segment, local_s)` and back.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ArcLengthTable {
    pub samples: Vec<ArcSample>,
    pub total_length: f32,
}

/// A chain of cubic segments forming one stroke skeleton.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BezierSkeleton {
    pub segments: Vec<CubicBezier>,
    #[serde(default)]
    pub anchors: Vec<AnchorMeta>,
    #[serde(default)]
    pub closed: bool,
    /// Cached parameterization; rebuilt on load and after any edit. Not serialized.
    #[serde(skip)]
    pub arc_length_table: ArcLengthTable,
}

/// Samples per segment used when building the arc-length table. Higher = smoother
/// arc-length mapping at the cost of a larger table.
const SAMPLES_PER_SEGMENT: usize = 64;

impl BezierSkeleton {
    pub fn from_segments(segments: Vec<CubicBezier>, closed: bool) -> Self {
        let anchors = vec![AnchorMeta::default(); segments.len() + 1];
        let mut sk = Self {
            segments,
            anchors,
            closed,
            arc_length_table: ArcLengthTable::default(),
        };
        sk.rebuild_arc_length_table();
        sk
    }

    /// Convenience for a single cubic segment.
    pub fn single(curve: CubicBezier) -> Self {
        Self::from_segments(vec![curve], false)
    }

    pub fn total_length(&self) -> f32 {
        self.arc_length_table.total_length
    }

    /// Recompute the arc-length table from current control points. Call after any edit
    /// to the segments.
    pub fn rebuild_arc_length_table(&mut self) {
        let mut samples: Vec<ArcSample> = Vec::new();
        let mut cumulative = 0.0f32;
        let mut prev_pos: Option<Vec2> = None;

        for (seg_idx, seg) in self.segments.iter().enumerate() {
            // Avoid duplicating the shared join sample between consecutive segments.
            let start_k = if seg_idx == 0 { 0 } else { 1 };
            for k in start_k..=SAMPLES_PER_SEGMENT {
                let s = k as f32 / SAMPLES_PER_SEGMENT as f32;
                let pos = seg.point(s);
                if let Some(prev) = prev_pos {
                    cumulative += (pos - prev).length();
                }
                prev_pos = Some(pos);
                samples.push(ArcSample {
                    global_t: 0.0, // filled in below once total length is known
                    segment_index: seg_idx as u32,
                    local_s: s,
                    length: cumulative,
                    position: pos,
                    tangent: seg.tangent(s),
                });
            }
        }

        let total = cumulative.max(1e-6);
        for sample in &mut samples {
            sample.global_t = sample.length / total;
        }

        self.arc_length_table = ArcLengthTable {
            samples,
            total_length: cumulative,
        };
    }

    /// Find the index of the last table sample whose length is `<= target_len`.
    fn bracket_by_length(&self, target_len: f32) -> usize {
        let samples = &self.arc_length_table.samples;
        // Binary search on the monotonic `length` field.
        let mut lo = 0usize;
        let mut hi = samples.len() - 1;
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if samples[mid].length <= target_len {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        lo
    }

    /// Map a normalized arc-length coordinate `t in [0,1]` to a `(segment, local_s)`
    /// native parameter on the curve.
    pub fn curve_param_at_arc_t(&self, t: f32) -> (usize, f32) {
        let samples = &self.arc_length_table.samples;
        if samples.len() < 2 {
            return (0, t.clamp(0.0, 1.0));
        }
        let t = t.clamp(0.0, 1.0);
        let target_len = t * self.arc_length_table.total_length;
        let i = self.bracket_by_length(target_len).min(samples.len() - 2);
        let a = &samples[i];
        let b = &samples[i + 1];
        let span = (b.length - a.length).max(1e-6);
        let f = ((target_len - a.length) / span).clamp(0.0, 1.0);

        if a.segment_index == b.segment_index {
            let s = a.local_s + (b.local_s - a.local_s) * f;
            (a.segment_index as usize, s)
        } else {
            // Crossing a segment join: snap to whichever side is closer.
            if f < 0.5 {
                (a.segment_index as usize, a.local_s)
            } else {
                (b.segment_index as usize, b.local_s)
            }
        }
    }

    /// Evaluate the orthonormal frame (position/tangent/normal/angle) at arc coord `t`.
    pub fn frame_at_arc_t(&self, t: f32) -> Frame {
        let (seg_idx, s) = self.curve_param_at_arc_t(t);
        let seg = &self.segments[seg_idx.min(self.segments.len() - 1)];
        let position = seg.point(s);
        let tangent = seg.tangent(s);
        let normal = tangent.perp(); // (-y, x): tangent rotated +90 degrees
        let angle = tangent.y.atan2(tangent.x);
        Frame {
            position,
            tangent,
            normal,
            angle,
        }
    }

    /// Translate every control point by `delta` (rigid move of the whole skeleton).
    pub fn translate(&mut self, delta: Vec2) {
        for seg in &mut self.segments {
            seg.p0 += delta;
            seg.p1 += delta;
            seg.p2 += delta;
            seg.p3 += delta;
        }
        self.rebuild_arc_length_table();
    }

    // --- Direct (node) editing: anchors and tangent handles --------------------------
    //
    // The skeleton is a chain of cubic segments sharing endpoints, so anchor `j` is the
    // join between the segment arriving at it (its `p2`/`p3`) and the segment leaving it
    // (its `p0`/`p1`). For an open path of `m` segments there are `m + 1` anchors; a closed
    // path wraps, so it has exactly `m` anchors. The helpers below resolve an anchor index
    // to the segment(s) that own its control points, which is everything the direct-edit
    // tool needs to read or move a single handle.

    /// Number of distinct anchors (on-curve join points).
    pub fn anchor_count(&self) -> usize {
        if self.segments.is_empty() {
            0
        } else if self.closed {
            self.segments.len()
        } else {
            self.segments.len() + 1
        }
    }

    /// Segment whose `p0`/`p1` belong to anchor `j` (the segment *leaving* the anchor).
    fn seg_out(&self, j: usize) -> Option<usize> {
        let m = self.segments.len();
        if m == 0 {
            None
        } else if self.closed {
            Some(j % m)
        } else if j < m {
            Some(j)
        } else {
            None
        }
    }

    /// Segment whose `p2`/`p3` belong to anchor `j` (the segment *arriving* at the anchor).
    fn seg_in(&self, j: usize) -> Option<usize> {
        let m = self.segments.len();
        if m == 0 {
            None
        } else if self.closed {
            Some((j + m - 1) % m)
        } else if (1..=m).contains(&j) {
            Some(j - 1)
        } else {
            None
        }
    }

    /// World position of anchor `j`'s on-curve point.
    pub fn anchor_position(&self, j: usize) -> Vec2 {
        if let Some(si) = self.seg_out(j) {
            self.segments[si].p0
        } else if let Some(si) = self.seg_in(j) {
            self.segments[si].p3
        } else {
            Vec2::ZERO
        }
    }

    /// Resolve a control-point reference to its current world position. Returns `None` for
    /// a handle that does not exist (e.g. the in-handle of an open path's first anchor).
    pub fn control_point(&self, r: ControlPointRef) -> Option<Vec2> {
        match r.kind {
            HandleKind::Anchor => {
                (r.anchor < self.anchor_count()).then(|| self.anchor_position(r.anchor))
            }
            HandleKind::OutHandle => self.seg_out(r.anchor).map(|si| self.segments[si].p1),
            HandleKind::InHandle => self.seg_in(r.anchor).map(|si| self.segments[si].p2),
        }
    }

    /// Every editable control point with its world position, for overlay rendering and
    /// hit-testing. Handles come first so a nearest-match pick grabs an overlapping handle
    /// rather than the anchor sitting beneath it.
    pub fn control_points(&self) -> Vec<(ControlPointRef, Vec2)> {
        let mut out = Vec::with_capacity(self.anchor_count() * 3);
        for j in 0..self.anchor_count() {
            for r in [ControlPointRef::out_handle(j), ControlPointRef::in_handle(j)] {
                if let Some(p) = self.control_point(r) {
                    out.push((r, p));
                }
            }
        }
        for j in 0..self.anchor_count() {
            out.push((ControlPointRef::anchor(j), self.anchor_position(j)));
        }
        out
    }

    /// Whether anchor `j` is a hard corner (handles move independently) rather than a
    /// smooth point (handles stay collinear).
    fn is_corner(&self, j: usize) -> bool {
        self.anchors.get(j).map(|a| a.corner).unwrap_or(false)
    }

    /// Move a control point to `new_pos` (world coords), then rebuild the arc-length table.
    ///
    /// * **Anchor** — drags the node rigidly: the anchor and both of its tangent handles
    ///   translate together, so the local curve shape is preserved (Illustrator's
    ///   Direct-Selection anchor drag).
    /// * **Handle** — moves just that handle. On a smooth (non-corner) anchor the opposite
    ///   handle is rotated to stay collinear through the anchor, keeping its own length, so
    ///   the curve flows smoothly through the point. Pass `mirror = false` (e.g. an
    ///   Alt-drag) to break the tangent and move the handle independently.
    pub fn move_control_point(&mut self, r: ControlPointRef, new_pos: Vec2, mirror: bool) {
        match r.kind {
            HandleKind::Anchor => {
                let delta = new_pos - self.anchor_position(r.anchor);
                if let Some(si) = self.seg_out(r.anchor) {
                    self.segments[si].p0 += delta;
                    self.segments[si].p1 += delta;
                }
                if let Some(si) = self.seg_in(r.anchor) {
                    self.segments[si].p3 += delta;
                    self.segments[si].p2 += delta;
                }
            }
            HandleKind::OutHandle => {
                if let Some(si) = self.seg_out(r.anchor) {
                    self.segments[si].p1 = new_pos;
                }
                self.mirror_opposite_handle(r.anchor, r.kind, new_pos, mirror);
            }
            HandleKind::InHandle => {
                if let Some(si) = self.seg_in(r.anchor) {
                    self.segments[si].p2 = new_pos;
                }
                self.mirror_opposite_handle(r.anchor, r.kind, new_pos, mirror);
            }
        }
        self.rebuild_arc_length_table();
    }

    /// After a handle of a smooth anchor is dragged to `new_pos`, rotate the *opposite*
    /// handle to stay collinear through the anchor while preserving its own length. No-op
    /// for corner anchors, when `mirror` is false, or when the moved handle collapses onto
    /// the anchor (no well-defined direction). The opposite handle is `p2` of the
    /// arriving segment (for a moved out-handle) or `p1` of the leaving segment (for a
    /// moved in-handle).
    fn mirror_opposite_handle(
        &mut self,
        anchor: usize,
        moved: HandleKind,
        new_pos: Vec2,
        mirror: bool,
    ) {
        if !mirror || self.is_corner(anchor) {
            return;
        }
        let a = self.anchor_position(anchor);
        let dir = new_pos - a;
        if dir.length_squared() < 1e-12 {
            return;
        }
        let dir = dir.normalize();
        let opp_seg = match moved {
            HandleKind::OutHandle => self.seg_in(anchor),
            HandleKind::InHandle => self.seg_out(anchor),
            HandleKind::Anchor => return,
        };
        let Some(si) = opp_seg else { return };
        match moved {
            HandleKind::OutHandle => {
                let len = (self.segments[si].p2 - a).length();
                self.segments[si].p2 = a - dir * len;
            }
            HandleKind::InHandle => {
                let len = (self.segments[si].p1 - a).length();
                self.segments[si].p1 = a - dir * len;
            }
            HandleKind::Anchor => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_curve() -> CubicBezier {
        CubicBezier::new(
            Vec2::new(100.0, 200.0),
            Vec2::new(180.0, 120.0),
            Vec2::new(300.0, 280.0),
            Vec2::new(420.0, 200.0),
        )
    }

    #[test]
    fn arc_length_table_is_monotonic() {
        let sk = BezierSkeleton::single(sample_curve());
        let table = &sk.arc_length_table;
        assert!(table.total_length > 0.0);
        for w in table.samples.windows(2) {
            assert!(w[1].length >= w[0].length);
            assert!(w[1].global_t >= w[0].global_t);
        }
        assert!((table.samples.last().unwrap().global_t - 1.0).abs() < 1e-5);
    }

    #[test]
    fn frame_endpoints_match_control_points() {
        let curve = sample_curve();
        let sk = BezierSkeleton::single(curve);
        let f0 = sk.frame_at_arc_t(0.0);
        let f1 = sk.frame_at_arc_t(1.0);
        assert!((f0.position - curve.p0).length() < 0.5);
        assert!((f1.position - curve.p3).length() < 0.5);
    }

    #[test]
    fn normal_is_perpendicular_to_tangent() {
        let sk = BezierSkeleton::single(sample_curve());
        for i in 0..=10 {
            let f = sk.frame_at_arc_t(i as f32 / 10.0);
            assert!(f.tangent.dot(f.normal).abs() < 1e-4);
            assert!((f.tangent.length() - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn single_segment_exposes_two_anchors_and_two_handles() {
        let sk = BezierSkeleton::single(sample_curve());
        assert_eq!(sk.anchor_count(), 2);
        // Anchor 0 has an out-handle but no in-handle; anchor 1 the reverse.
        assert!(sk.control_point(ControlPointRef::out_handle(0)).is_some());
        assert!(sk.control_point(ControlPointRef::in_handle(0)).is_none());
        assert!(sk.control_point(ControlPointRef::in_handle(1)).is_some());
        assert!(sk.control_point(ControlPointRef::out_handle(1)).is_none());
        // Anchors coincide with the curve endpoints.
        assert!((sk.anchor_position(0) - sample_curve().p0).length() < 1e-5);
        assert!((sk.anchor_position(1) - sample_curve().p3).length() < 1e-5);
        // control_points() lists the two existing handles plus the two anchors.
        assert_eq!(sk.control_points().len(), 4);
    }

    #[test]
    fn dragging_anchor_carries_its_handles() {
        let mut sk = BezierSkeleton::single(sample_curve());
        let before_out = sk.control_point(ControlPointRef::out_handle(0)).unwrap();
        let a0 = sk.anchor_position(0);
        let delta = Vec2::new(25.0, -10.0);
        sk.move_control_point(ControlPointRef::anchor(0), a0 + delta, true);
        // Anchor moved by delta and the out-handle rode along (offset preserved).
        assert!((sk.anchor_position(0) - (a0 + delta)).length() < 1e-4);
        let after_out = sk.control_point(ControlPointRef::out_handle(0)).unwrap();
        assert!((after_out - (before_out + delta)).length() < 1e-4);
    }

    #[test]
    fn dragging_handle_sets_it_and_mirrors_neighbor_at_smooth_join() {
        // Two segments meeting at a smooth interior anchor (index 1).
        let s0 = CubicBezier::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(30.0, 0.0),
            Vec2::new(70.0, 0.0),
            Vec2::new(100.0, 0.0),
        );
        let s1 = CubicBezier::new(
            Vec2::new(100.0, 0.0),
            Vec2::new(130.0, 0.0),
            Vec2::new(170.0, 0.0),
            Vec2::new(200.0, 0.0),
        );
        let mut sk = BezierSkeleton::from_segments(vec![s0, s1], false);
        assert_eq!(sk.anchor_count(), 3);
        let anchor1 = sk.anchor_position(1);
        let in_len_before = (sk.control_point(ControlPointRef::in_handle(1)).unwrap() - anchor1)
            .length();

        // Drag anchor 1's out-handle straight up.
        let target = anchor1 + Vec2::new(0.0, 40.0);
        sk.move_control_point(ControlPointRef::out_handle(1), target, true);

        let out = sk.control_point(ControlPointRef::out_handle(1)).unwrap();
        let inn = sk.control_point(ControlPointRef::in_handle(1)).unwrap();
        assert!((out - target).length() < 1e-4, "out-handle set exactly");
        // In-handle is collinear-opposite through the anchor, with its length preserved.
        let out_dir = (out - anchor1).normalize();
        let in_dir = (inn - anchor1).normalize();
        assert!((out_dir + in_dir).length() < 1e-3, "handles collinear-opposite");
        let in_len_after = (inn - anchor1).length();
        assert!((in_len_after - in_len_before).abs() < 1e-3, "opposite length preserved");
    }

    #[test]
    fn alt_drag_breaks_the_tangent() {
        let s0 = CubicBezier::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(30.0, 0.0),
            Vec2::new(70.0, 0.0),
            Vec2::new(100.0, 0.0),
        );
        let s1 = CubicBezier::new(
            Vec2::new(100.0, 0.0),
            Vec2::new(130.0, 0.0),
            Vec2::new(170.0, 0.0),
            Vec2::new(200.0, 0.0),
        );
        let mut sk = BezierSkeleton::from_segments(vec![s0, s1], false);
        let in_before = sk.control_point(ControlPointRef::in_handle(1)).unwrap();
        let anchor1 = sk.anchor_position(1);
        // mirror=false: the opposite handle must stay put.
        sk.move_control_point(
            ControlPointRef::out_handle(1),
            anchor1 + Vec2::new(0.0, 40.0),
            false,
        );
        let in_after = sk.control_point(ControlPointRef::in_handle(1)).unwrap();
        assert!((in_after - in_before).length() < 1e-6, "in-handle untouched by alt-drag");
    }

    #[test]
    fn closed_path_anchor_count_matches_segments() {
        let s0 = CubicBezier::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(30.0, 10.0),
            Vec2::new(70.0, 10.0),
            Vec2::new(100.0, 0.0),
        );
        let s1 = CubicBezier::new(
            Vec2::new(100.0, 0.0),
            Vec2::new(70.0, -10.0),
            Vec2::new(30.0, -10.0),
            Vec2::new(0.0, 0.0),
        );
        let sk = BezierSkeleton::from_segments(vec![s0, s1], true);
        // Closed: anchors == segments, every anchor has both handles.
        assert_eq!(sk.anchor_count(), 2);
        for j in 0..sk.anchor_count() {
            assert!(sk.control_point(ControlPointRef::in_handle(j)).is_some());
            assert!(sk.control_point(ControlPointRef::out_handle(j)).is_some());
        }
    }

    #[test]
    fn arc_t_is_roughly_uniform_in_length() {
        // The midpoint in arc length should be near half the total length.
        let sk = BezierSkeleton::single(sample_curve());
        let half = sk.frame_at_arc_t(0.5).position;
        // Measure length to that point by walking fine samples.
        let mut walked = 0.0;
        let mut prev = sk.frame_at_arc_t(0.0).position;
        let mut len_to_half = None;
        for i in 1..=1000 {
            let t = i as f32 / 1000.0;
            let p = sk.frame_at_arc_t(t).position;
            walked += (p - prev).length();
            prev = p;
            if t >= 0.5 && len_to_half.is_none() {
                len_to_half = Some(walked);
            }
            let _ = half;
        }
        let total = walked;
        let lh = len_to_half.unwrap();
        assert!((lh / total - 0.5).abs() < 0.05, "arc-length midpoint off: {}", lh / total);
    }
}
