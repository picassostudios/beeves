/**
 * Vector-edit handle overlay rendering.
 *
 * The Rust `WasmApp.edit_overlay()` returns the geometry of the stroke currently open in
 * the Edit (direct-selection) tool, already projected into device-pixel screen space.
 * This module only paints it onto a 2D canvas sitting above the WebGPU surface — it owns
 * no state and makes no editing decisions. See the `edit_overlay` doc comment in
 * `crates/app_wasm/src/lib.rs` for the array layout.
 *
 * Layout (all values device pixels):
 *   [ anchorCount, pathPointCount,
 *     pathX0, pathY0, ... pathX(k-1), pathY(k-1),
 *     // then 6 floats per anchor:
 *     anchorX, anchorY, inX, inY, outX, outY ]    // a missing handle is NaN, NaN
 */

const ACCENT = "#5b9dff";
const ANCHOR_FILL = "#ffffff";

/** Per-anchor record decoded from the flat overlay array (device-pixel coords). */
interface Anchor {
  x: number;
  y: number;
  /** In-handle position, or null when the anchor has none (open-path start). */
  in: [number, number] | null;
  /** Out-handle position, or null when the anchor has none (open-path end). */
  out: [number, number] | null;
}

/** A handle is absent when either coordinate is NaN (the Rust-side sentinel). */
function handle(x: number, y: number): [number, number] | null {
  return Number.isNaN(x) || Number.isNaN(y) ? null : [x, y];
}

/** Decode the flat overlay array into a path polyline and a list of anchors. */
function decode(
  data: Float32Array
): { path: number[]; anchors: Anchor[] } | null {
  if (data.length < 2) return null;
  const anchorCount = data[0];
  const pathCount = data[1];
  let i = 2;
  const path: number[] = [];
  for (let p = 0; p < pathCount; p++) {
    path.push(data[i++], data[i++]);
  }
  const anchors: Anchor[] = [];
  for (let a = 0; a < anchorCount; a++) {
    const ax = data[i++];
    const ay = data[i++];
    const inH = handle(data[i++], data[i++]);
    const outH = handle(data[i++], data[i++]);
    anchors.push({ x: ax, y: ay, in: inH, out: outH });
  }
  return { path, anchors };
}

/**
 * Clear `ctx` and draw the edit overlay described by `data`. `dpr` scales line widths and
 * marker sizes so they stay a constant size on screen regardless of device-pixel ratio
 * (the canvas backing store is in device pixels, matching the coordinates in `data`).
 */
export function drawEditOverlay(
  ctx: CanvasRenderingContext2D,
  data: Float32Array,
  dpr: number
): void {
  const { canvas } = ctx;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const decoded = decode(data);
  if (decoded === null) return;
  const { path, anchors } = decoded;

  // 1) The curve itself, as a thin guide line tracing the skeleton.
  if (path.length >= 4) {
    ctx.beginPath();
    ctx.moveTo(path[0], path[1]);
    for (let p = 2; p < path.length; p += 2) {
      ctx.lineTo(path[p], path[p + 1]);
    }
    ctx.lineWidth = 1.25 * dpr;
    ctx.strokeStyle = ACCENT;
    ctx.stroke();
  }

  // 2) Handle lines (anchor -> handle), drawn under the markers.
  ctx.lineWidth = 1 * dpr;
  ctx.strokeStyle = ACCENT;
  for (const a of anchors) {
    for (const h of [a.in, a.out]) {
      if (h === null) continue;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(h[0], h[1]);
      ctx.stroke();
    }
  }

  // 3) Round handle markers.
  const handleR = 3.5 * dpr;
  ctx.fillStyle = ACCENT;
  for (const a of anchors) {
    for (const h of [a.in, a.out]) {
      if (h === null) continue;
      ctx.beginPath();
      ctx.arc(h[0], h[1], handleR, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // 4) Square anchor markers on top (hollow white with an accent border).
  const half = 3.5 * dpr;
  ctx.lineWidth = 1.5 * dpr;
  for (const a of anchors) {
    ctx.fillStyle = ANCHOR_FILL;
    ctx.strokeStyle = ACCENT;
    ctx.beginPath();
    ctx.rect(a.x - half, a.y - half, half * 2, half * 2);
    ctx.fill();
    ctx.stroke();
  }
}

/** Clear the overlay (used when the Edit tool is inactive). */
export function clearEditOverlay(ctx: CanvasRenderingContext2D): void {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
}

/**
 * Clear `ctx` and draw the vector-draw curvature-fit overlay described by `data`.
 *
 * The Rust `WasmApp.vector_overlay()` returns the in-progress adaptive curvature fit,
 * already projected into device-pixel screen space and ordered from the start of the
 * stroke to the **tip** (the point currently under the cursor). Each segment between two
 * anchors is one adaptive window — long/sparse where the drawing is straight, short/dense
 * where it curves.
 *
 * Rather than painting every window a flat red, we render the red as a "comet tail": fully
 * opaque at the tip (the section being put down right now) and smoothly fading to
 * transparent over `FADE_LEN` device pixels back along the stroke. As the user draws,
 * fresh points enter the active window at the tip and older windows slide past the fade
 * horizon, so the red dissolves cleanly behind the cursor. `dpr` scales widths, marker
 * sizes, and the fade length so the effect is a constant size on screen.
 *
 * Layout (all values device pixels):
 *   [ segCount,
 *     // repeated segCount times — one polyline per segment:
 *     ptCount, x0, y0, x1, y1, ...,
 *     // then anchors (3 floats each; isCorner is 1.0 or 0.0):
 *     anchorCount, ax0, ay0, isCorner0, ax1, ay1, isCorner1, ... ]
 */
export function drawVectorOverlay(
  ctx: CanvasRenderingContext2D,
  data: Float32Array,
  dpr: number
): void {
  const { canvas } = ctx;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (data.length < 1) return;

  // --- Decode the flat array into segment polylines + anchors ----------------
  let i = 0;
  const segCount = data[i++];
  const segments: number[][] = [];
  for (let s = 0; s < segCount; s++) {
    if (i >= data.length) break;
    const ptCount = data[i++];
    const pts: number[] = [];
    for (let p = 0; p < ptCount && i + 1 < data.length; p++) {
      pts.push(data[i++], data[i++]);
    }
    segments.push(pts);
  }
  const anchors: { x: number; y: number; corner: boolean }[] = [];
  if (i < data.length) {
    const anchorCount = data[i++];
    for (let a = 0; a < anchorCount && i + 2 < data.length; a++) {
      anchors.push({ x: data[i++], y: data[i++], corner: data[i++] === 1 });
    }
  }

  // --- Flatten the windows into one tip-anchored polyline --------------------
  // Consecutive windows share their join anchor (window N's last point == window
  // N+1's first point), so skip the duplicate. The tip — the most recently drawn
  // point — is the very last vertex.
  const poly: { x: number; y: number }[] = [];
  for (let s = 0; s < segments.length; s++) {
    const pts = segments[s];
    for (let p = 0; p < pts.length; p += 2) {
      if (s > 0 && p === 0) continue;
      poly.push({ x: pts[p], y: pts[p + 1] });
    }
  }
  if (poly.length === 0) return;

  // Cumulative arc length from the start; distance-from-tip = total - cum[k].
  const cum = new Array<number>(poly.length).fill(0);
  for (let k = 1; k < poly.length; k++) {
    cum[k] = cum[k - 1] + Math.hypot(poly[k].x - poly[k - 1].x, poly[k].y - poly[k - 1].y);
  }
  const total = cum[poly.length - 1];

  // The red is gone this many device pixels behind the tip. Smoothstep gives the
  // fade clean, derivative-free ends instead of a hard linear ramp.
  const FADE_LEN = 150 * dpr;
  const fadeAt = (distFromTip: number) => {
    const t = 1 - distFromTip / FADE_LEN;
    const c = t < 0 ? 0 : t > 1 ? 1 : t;
    return c * c * (3 - 2 * c);
  };

  // --- The red comet tail ----------------------------------------------------
  // Walk sub-segments from the tip backward, each painted at the alpha for its
  // distance-from-tip, until we cross the fade horizon. Round caps knit the pieces
  // together; subdividing keeps the gradient smooth across the long, straight
  // windows the fitter samples only coarsely.
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.strokeStyle = "#ff0000";
  for (let k = poly.length - 2; k >= 0; k--) {
    if (total - cum[k + 1] >= FADE_LEN) break; // everything further back is invisible
    const a = poly[k];
    const b = poly[k + 1];
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const sub = Math.max(1, Math.ceil(Math.hypot(dx, dy) / (6 * dpr)));
    for (let t = 0; t < sub; t++) {
      const t0 = t / sub;
      const t1 = (t + 1) / sub;
      const alpha = fadeAt(total - (cum[k] + (cum[k + 1] - cum[k]) * (t0 + t1) * 0.5));
      if (alpha <= 0.004) continue;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = (2 + 1.6 * alpha) * dpr;
      ctx.beginPath();
      ctx.moveTo(a.x + dx * t0, a.y + dy * t0);
      ctx.lineTo(a.x + dx * t1, a.y + dy * t1);
      ctx.stroke();
    }
  }

  // --- Anchors, faded by the same tip-relative falloff -----------------------
  // Anchor j sits at the start of window j, so its distance-from-tip is the summed
  // length of every window from j onward (the tip anchor's is 0 — full opacity).
  const suffix = new Array<number>(segments.length + 1).fill(0);
  for (let s = segments.length - 1; s >= 0; s--) {
    const pts = segments[s];
    let segLen = 0;
    for (let p = 2; p < pts.length; p += 2) {
      segLen += Math.hypot(pts[p] - pts[p - 2], pts[p + 1] - pts[p - 1]);
    }
    suffix[s] = suffix[s + 1] + segLen;
  }
  const half = 4 * dpr;
  for (let aIdx = 0; aIdx < anchors.length; aIdx++) {
    const { x, y, corner } = anchors[aIdx];
    const alpha = fadeAt(suffix[Math.min(aIdx, segments.length)]);
    if (alpha <= 0.004) continue;
    ctx.globalAlpha = alpha;
    if (corner) {
      // Corner: solid red square rotated 45° into a diamond.
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(Math.PI / 4);
      ctx.fillStyle = "#ff0000";
      ctx.beginPath();
      ctx.rect(-half, -half, half * 2, half * 2);
      ctx.fill();
      ctx.restore();
    } else {
      // Smooth: hollow white square with a red border.
      ctx.fillStyle = "#ffffff";
      ctx.strokeStyle = "#ff0000";
      ctx.lineWidth = 1.5 * dpr;
      ctx.beginPath();
      ctx.rect(x - half, y - half, half * 2, half * 2);
      ctx.fill();
      ctx.stroke();
    }
  }

  ctx.globalAlpha = 1;
}
