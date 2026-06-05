/**
 * Thin web shell entry point. All editing logic lives in Rust (WasmApp);
 * this file only wires DOM events to the wasm interface and drives the
 * render loop.
 */
import { createApp, type WasmApp } from "./wasm";
import { createToolbar, type ToolName } from "./tools/toolbar";
import {
  drawEditOverlay,
  clearEditOverlay,
  drawVectorOverlay,
} from "./tools/handles";

/** Resolve a required element by id or throw (keeps the wiring honest). */
function mustGet<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (el === null) {
    throw new Error(`Missing required element #${id}`);
  }
  return el as T;
}

function showOverlay(title: string, message: string): void {
  const overlay = mustGet<HTMLDivElement>("overlay");
  mustGet<HTMLHeadingElement>("overlay-title").textContent = title;
  mustGet<HTMLParagraphElement>("overlay-msg").textContent = message;
  overlay.classList.add("show");
}

/** "#rrggbb" -> normalized [r, g, b] in 0..1. */
function hexToRgb(hex: string): [number, number, number] {
  const v = hex.replace("#", "");
  const r = parseInt(v.slice(0, 2), 16) / 255;
  const g = parseInt(v.slice(2, 4), 16) / 255;
  const b = parseInt(v.slice(4, 6), 16) / 255;
  return [r, g, b];
}

/**
 * Size the canvas backing store to its CSS box * devicePixelRatio, and tell
 * the wasm app about the device-pixel dimensions. Pointer coordinates are
 * scaled by DPR in JS to match (see eventCoords).
 */
function resizeCanvas(canvas: HTMLCanvasElement, app: WasmApp): void {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    app.resize(w, h);
  }
}

/** Pointer event -> device-pixel coords inside the canvas. */
function eventCoords(
  canvas: HTMLCanvasElement,
  ev: PointerEvent | WheelEvent
): [number, number] {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const x = (ev.clientX - rect.left) * dpr;
  const y = (ev.clientY - rect.top) * dpr;
  return [x, y];
}

function downloadJson(json: string, filename: string): void {
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

async function boot(): Promise<void> {
  const canvas = mustGet<HTMLCanvasElement>("stage");

  let app: WasmApp;
  try {
    // Seed the backing store before construction so the surface is created
    // at the right device-pixel size.
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(canvas.clientWidth * dpr));
    canvas.height = Math.max(1, Math.round(canvas.clientHeight * dpr));
    app = await createApp(canvas);
  } catch (err) {
    console.error("Failed to initialize WasmApp", err);
    const hasGpu = "gpu" in navigator;
    showOverlay(
      hasGpu ? "Failed to start" : "WebGPU not available",
      hasGpu
        ? `The renderer failed to initialize: ${
            err instanceof Error ? err.message : String(err)
          }`
        : "This app requires a browser with WebGPU support. Try the latest Chrome, Edge, or Safari Technology Preview."
    );
    return;
  }

  // --- Toolbar (tool selection forwarded to Rust) ---------------------------
  // Track the active tool locally too: the render loop only paints the vector-edit
  // handle overlay while the Edit tool is active.
  let currentTool: ToolName = "brush";
  const toolbar = createToolbar(
    mustGet<HTMLDivElement>("tools"),
    (tool: ToolName) => {
      currentTool = tool;
      app.set_tool(tool);
    },
    "brush"
  );

  // --- Vector-edit handle overlay (2D canvas above the WebGPU surface) -------
  const overlayCanvas = mustGet<HTMLCanvasElement>("edit-overlay");
  const overlayCtx = overlayCanvas.getContext("2d");
  if (overlayCtx === null) {
    throw new Error("2D canvas context unavailable for the edit overlay");
  }
  let overlayShown = false;

  // --- Color ----------------------------------------------------------------
  const colorInput = mustGet<HTMLInputElement>("brush-color");
  const applyColor = (): void => {
    const [r, g, b] = hexToRgb(colorInput.value);
    app.set_brush_color(r, g, b);
  };
  colorInput.addEventListener("input", applyColor);
  applyColor();

  // --- Brush size -----------------------------------------------------------
  const sizeInput = mustGet<HTMLInputElement>("brush-size");
  const sizeValue = mustGet<HTMLSpanElement>("brush-size-value");
  const applySize = (): void => {
    const radius = Number(sizeInput.value);
    sizeValue.textContent = sizeInput.value;
    app.set_brush_radius(radius);
  };
  sizeInput.addEventListener("input", applySize);
  applySize();

  // --- Brush hardness -------------------------------------------------------
  const hardnessInput = mustGet<HTMLInputElement>("brush-hardness");
  const hardnessValue = mustGet<HTMLSpanElement>("brush-hardness-value");
  const applyHardness = (): void => {
    const hardness = Number(hardnessInput.value);
    hardnessValue.textContent = hardness.toFixed(2);
    app.set_brush_hardness(hardness);
  };
  hardnessInput.addEventListener("input", applyHardness);
  applyHardness();

  // --- Edge (perimeter) hardness -------------------------------------------
  const edgeInput = mustGet<HTMLInputElement>("brush-edge-hardness");
  const edgeValue = mustGet<HTMLSpanElement>("brush-edge-hardness-value");
  const applyEdgeHardness = (): void => {
    const edge = Number(edgeInput.value);
    edgeValue.textContent = edge.toFixed(2);
    app.set_brush_edge_hardness(edge);
  };
  edgeInput.addEventListener("input", applyEdgeHardness);
  applyEdgeHardness();

  // --- Crisp perimeter toggle ----------------------------------------------
  const crispInput = mustGet<HTMLInputElement>("crisp-edges");
  const applyCrisp = (): void => {
    app.set_crisp_edges(crispInput.checked);
  };
  crispInput.addEventListener("change", applyCrisp);
  applyCrisp();

  // --- Render mode (Gaussian / Convex) -------------------------------------
  // Orthogonal to the active tool and to the crisp toggle: switching the kernel never touches
  // the document, so every tool — including Blend — works the same in either mode. The convex
  // shape controls are only meaningful in convex mode, so they are shown/hidden with it.
  const modeButtons = Array.from(
    mustGet<HTMLDivElement>("modes").querySelectorAll<HTMLButtonElement>(
      "button[data-mode]"
    )
  );
  const convexControls = mustGet<HTMLDivElement>("convex-controls");
  const triangleControls = mustGet<HTMLDivElement>("triangle-controls");
  const selectMode = (mode: string): void => {
    app.set_render_mode(mode);
    for (const b of modeButtons) {
      const isActive = b.dataset.mode === mode;
      b.classList.toggle("is-active", isActive);
      b.setAttribute("aria-pressed", String(isActive));
    }
    convexControls.hidden = mode !== "convex";
    triangleControls.hidden = mode !== "triangle";
  };
  for (const b of modeButtons) {
    b.addEventListener("click", () => selectMode(b.dataset.mode ?? "gaussian"));
  }
  selectMode("gaussian");

  // --- Convex shape: sides + corner smoothness -----------------------------
  const sidesInput = mustGet<HTMLInputElement>("convex-sides");
  const sidesValue = mustGet<HTMLSpanElement>("convex-sides-value");
  const applySides = (): void => {
    sidesValue.textContent = sidesInput.value;
    app.set_convex_sides(Number(sidesInput.value));
  };
  sidesInput.addEventListener("input", applySides);
  applySides();

  const smoothInput = mustGet<HTMLInputElement>("convex-smoothness");
  const smoothValue = mustGet<HTMLSpanElement>("convex-smoothness-value");
  const applySmooth = (): void => {
    smoothValue.textContent = Number(smoothInput.value).toFixed(2);
    app.set_convex_smoothness(Number(smoothInput.value));
  };
  smoothInput.addEventListener("input", applySmooth);
  applySmooth();

  const sharpInput = mustGet<HTMLInputElement>("convex-sharpness");
  const sharpValue = mustGet<HTMLSpanElement>("convex-sharpness-value");
  const applySharp = (): void => {
    sharpValue.textContent = Number(sharpInput.value).toFixed(2);
    app.set_convex_sharpness(Number(sharpInput.value));
  };
  sharpInput.addEventListener("input", applySharp);
  applySharp();

  // --- Triangle shape: rotation + window softness ---------------------------
  const triRotInput = mustGet<HTMLInputElement>("triangle-rotation");
  const triRotValue = mustGet<HTMLSpanElement>("triangle-rotation-value");
  const applyTriRot = (): void => {
    triRotValue.textContent = Number(triRotInput.value).toFixed(2);
    app.set_triangle_rotation(Number(triRotInput.value));
  };
  triRotInput.addEventListener("input", applyTriRot);
  applyTriRot();

  const triSoftInput = mustGet<HTMLInputElement>("triangle-softness");
  const triSoftValue = mustGet<HTMLSpanElement>("triangle-softness-value");
  const applyTriSoft = (): void => {
    triSoftValue.textContent = Number(triSoftInput.value).toFixed(2);
    app.set_triangle_softness(Number(triSoftInput.value));
  };
  triSoftInput.addEventListener("input", applyTriSoft);
  applyTriSoft();

  // --- Blend strength -------------------------------------------------------
  const blendInput = mustGet<HTMLInputElement>("blend-strength");
  const blendValue = mustGet<HTMLSpanElement>("blend-strength-value");
  const applyBlend = (): void => {
    const strength = Number(blendInput.value);
    blendValue.textContent = strength.toFixed(2);
    app.set_blend_strength(strength);
  };
  blendInput.addEventListener("input", applyBlend);
  applyBlend();

  // --- Save / Load ----------------------------------------------------------
  mustGet<HTMLButtonElement>("save-btn").addEventListener("click", () => {
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    downloadJson(app.save_json(), `beavus-${stamp}.gspf.json`);
  });

  const fileInput = mustGet<HTMLInputElement>("file-input");
  mustGet<HTMLButtonElement>("load-btn").addEventListener("click", () =>
    fileInput.click()
  );
  fileInput.addEventListener("change", async () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      app.load_json(text);
    } catch (err) {
      console.error("Failed to load document", err);
    } finally {
      // Reset so selecting the same file again re-fires `change`.
      fileInput.value = "";
    }
  });

  // --- Pointer input --------------------------------------------------------
  let pointerActive = false;
  canvas.addEventListener("pointerdown", (ev: PointerEvent) => {
    canvas.setPointerCapture(ev.pointerId);
    pointerActive = true;
    app.set_modifiers(ev.altKey, ev.shiftKey);
    const [x, y] = eventCoords(canvas, ev);
    app.pointer_down(x, y);
  });
  canvas.addEventListener("pointermove", (ev: PointerEvent) => {
    if (!pointerActive) return;
    app.set_modifiers(ev.altKey, ev.shiftKey);
    const [x, y] = eventCoords(canvas, ev);
    app.pointer_move(x, y);
  });
  const endPointer = (ev: PointerEvent): void => {
    if (!pointerActive) return;
    pointerActive = false;
    const [x, y] = eventCoords(canvas, ev);
    app.pointer_up(x, y);
    // In direct-edit mode a click (re)selects a path; reflect that path's colour in the
    // picker so the next colour change recolours it from a matching starting point.
    if (currentTool === "edit") {
      const hex = app.edit_target_color_hex();
      if (hex !== "") colorInput.value = hex;
    }
    if (canvas.hasPointerCapture(ev.pointerId)) {
      canvas.releasePointerCapture(ev.pointerId);
    }
  };
  canvas.addEventListener("pointerup", endPointer);
  canvas.addEventListener("pointercancel", endPointer);

  // --- Wheel (trackpad pan; Shift/pinch to zoom) ---------------------------
  // A plain two-finger swipe (or scroll wheel) pans the canvas — no Pan tool needed.
  // Hold Shift while scrolling to zoom at the cursor; a trackpad pinch (which the
  // browser reports as a wheel event with ctrlKey) zooms too.
  canvas.addEventListener(
    "wheel",
    (ev: WheelEvent) => {
      ev.preventDefault();
      const [x, y] = eventCoords(canvas, ev);
      if (ev.shiftKey || ev.ctrlKey) {
        // Zoom: use whichever axis carries the gesture — browsers route Shift+wheel
        // to deltaX on some platforms, and pinch comes through deltaY.
        const dz =
          Math.abs(ev.deltaY) >= Math.abs(ev.deltaX) ? ev.deltaY : ev.deltaX;
        app.wheel(x, y, dz);
      } else {
        // Pan: wheel deltas are in CSS pixels; scale by DPR to match the device-pixel
        // coordinate space the camera works in.
        const dpr = window.devicePixelRatio || 1;
        app.pan(ev.deltaX * dpr, ev.deltaY * dpr);
      }
    },
    { passive: false }
  );

  // --- Bezier/pen tool: finish or cancel the in-progress path ---------------
  // The pen tool builds a path across multiple clicks (each click drops an anchor;
  // press-and-drag pulls out curve handles). These gestures end the path:
  //   - double-click anywhere, or press Enter -> finish, keeping the path;
  //   - press Escape -> cancel, discarding the in-progress path;
  //   - clicking back on the first anchor closes the path (handled in Rust).
  canvas.addEventListener("dblclick", (ev: MouseEvent) => {
    ev.preventDefault();
    app.finish_path();
  });
  // Single-key tool shortcuts (skipped while typing in a form control).
  const TOOL_KEYS: Record<string, ToolName> = {
    b: "brush",
    d: "vectordraw",
    p: "bezier",
    a: "edit",
    s: "sculpt",
    l: "blend",
    v: "select",
  };
  window.addEventListener("keydown", (ev: KeyboardEvent) => {
    if (ev.key === "Enter") {
      app.finish_path();
      return;
    }
    if (ev.key === "Escape") {
      app.cancel_path();
      return;
    }
    const target = ev.target as HTMLElement | null;
    const typing =
      target !== null && (target.tagName === "INPUT" || target.isContentEditable);
    const tool = TOOL_KEYS[ev.key.toLowerCase()];
    if (tool !== undefined && !typing && !ev.metaKey && !ev.ctrlKey) {
      toolbar.select(tool);
    }
  });

  // --- Resize ---------------------------------------------------------------
  window.addEventListener("resize", () => resizeCanvas(canvas, app));
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(() => resizeCanvas(canvas, app)).observe(canvas);
  }

  // --- Render loop ----------------------------------------------------------
  const splatCount = mustGet<HTMLElement>("splat-count");
  let lastCount = -1;
  const frame = (): void => {
    resizeCanvas(canvas, app);
    app.render();

    // Keep the overlay backing store matched to the stage, then paint the handles
    // only while the Edit tool is active (clearing once when it's switched off).
    if (
      overlayCanvas.width !== canvas.width ||
      overlayCanvas.height !== canvas.height
    ) {
      overlayCanvas.width = canvas.width;
      overlayCanvas.height = canvas.height;
    }
    if (currentTool === "edit") {
      const dpr = window.devicePixelRatio || 1;
      drawEditOverlay(overlayCtx, app.edit_overlay(), dpr);
      overlayShown = true;
    } else if (currentTool === "vectordraw") {
      const dpr = window.devicePixelRatio || 1;
      drawVectorOverlay(overlayCtx, app.vector_overlay(), dpr);
      overlayShown = true;
    } else if (overlayShown) {
      clearEditOverlay(overlayCtx);
      overlayShown = false;
    }

    const count = app.splat_count();
    if (count !== lastCount) {
      lastCount = count;
      splatCount.textContent = String(count);
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

boot().catch((err) => {
  console.error(err);
  showOverlay(
    "Failed to start",
    err instanceof Error ? err.message : String(err)
  );
});
