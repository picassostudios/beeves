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
 * already projected into device-pixel screen space. Each segment between two anchors is
 * one adaptive window — windows are long/sparse where the drawing is straight and
 * short/dense where it curves. `dpr` scales line widths and marker sizes so they stay a
 * constant size on screen regardless of device-pixel ratio.
 *
 * Layout (all values device pixels):
 *   [ segCount,
 *     // repeated segCount times — one polyline per segment:
 *     ptCount, x0, y0, x1, y1, ...,
 *     // then anchors (3 floats each; isCorner is 1.0 or 0.0):
 *     anchorCount, ax0, ay0, isCorner0, ax1, ay1, isCorner1, ... ]
 *
 * The last segment (index segCount-1) is the "active window" still being grown and is
 * drawn brighter/thicker than the committed ones.
 */
export function drawVectorOverlay(
  ctx: CanvasRenderingContext2D,
  data: Float32Array,
  dpr: number
): void {
  const { canvas } = ctx;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (data.length < 1) return;

  let i = 0;
  const segCount = data[i++];

  // 1) Segment polylines. The last segment is the active window — brighter & thicker.
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  for (let s = 0; s < segCount; s++) {
    if (i >= data.length) break;
    const ptCount = data[i++];
    const isActive = s === segCount - 1;
    ctx.beginPath();
    for (let p = 0; p < ptCount; p++) {
      if (i + 1 >= data.length) break;
      const x = data[i++];
      const y = data[i++];
      if (p === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = isActive ? "#ff0000" : "#ff5a52";
    ctx.lineWidth = (isActive ? 3.5 : 2) * dpr;
    ctx.stroke();
  }

  // 2) Anchors: smooth = hollow white square with red border; corner = solid red 45° diamond.
  if (i >= data.length) return;
  const anchorCount = data[i++];
  const half = 4 * dpr;
  ctx.lineWidth = 1.5 * dpr;
  for (let a = 0; a < anchorCount; a++) {
    if (i + 2 >= data.length) break;
    const ax = data[i++];
    const ay = data[i++];
    const isCorner = data[i++];
    if (isCorner === 1) {
      // Corner: solid red square rotated 45° into a diamond.
      ctx.save();
      ctx.translate(ax, ay);
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
      ctx.beginPath();
      ctx.rect(ax - half, ay - half, half * 2, half * 2);
      ctx.fill();
      ctx.stroke();
    }
  }
}
