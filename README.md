# gaussian-design-tool

A browser-based graphic design tool whose editable source model is a **Bézier-controlled
2D Gaussian splat field**. A stroke is stored not as pixels but as an editable curve plus
a cloud of anisotropic Gaussian splats attached to that curve — giving vector-like control
and raster-like marks at once.

## Status

The full vertical stack is in place and builds green end to end:

| Layer | Crate / dir | What's proven |
|---|---|---|
| Domain model + bidirectional sync | `crates/app_core` | 41 tests (`cargo test`) — incl. freehand→Bézier fitting + direct node editing |
| wgpu/WebGPU renderer | `crates/renderer` | 14 tests incl. **headless GPU render-to-texture + GPU picking**, pixel-verified on native (Metal) |
| WASM bridge (`WasmApp`) | `crates/app_wasm` | compiles to `wasm32`; `wasm-pack` produces `web/pkg` |
| TypeScript / Vite shell | `web/` | `tsc` clean; `vite build` produces `web/dist` |

**Verified automatically:** 55 Rust tests, clippy clean, the WASM package builds, and the
web bundle builds. The native render-to-texture tests exercise the *same* shader/pipeline
path the browser uses, so the GPU rendering logic is validated headlessly.

**Needs a browser to confirm visually:** live on-screen rendering and pointer interaction
(WebGPU canvas surface + event wiring) can only be eyeballed by running the dev server in a
WebGPU-capable browser — see [Run in the browser](#run-in-the-browser).

## Quick start (headless, no browser)

```bash
cargo test --workspace                  # 55 tests: app_core (41) + renderer (14, incl. GPU)
cargo run -p app_core --example demo    # prints the edit loop, writes examples/demo.gspf.json
cargo clippy --workspace --all-targets  # clean
```

## Run in the browser

Requires a WebGPU-capable browser (recent Chrome/Edge, or Safari Technology Preview).

```bash
cd web && npm install && npm run dev
```

That's it — `npm run dev` builds the Rust/WASM package first (`predev` hook →
`wasm-pack build` into `web/pkg`), then runs Vite **and** `cargo watch` concurrently. Edit
any Rust source (`app_core` / `renderer` / `app_wasm`) or a WGSL shader and save: the WASM
rebuilds (~1–2 s) and the browser **live-reloads automatically**. Then open the printed URL.

> Prerequisites for the WASM build: `wasm-pack`, the `wasm32-unknown-unknown` target
> (`rustup target add wasm32-unknown-unknown`), and `cargo-watch`
> (`cargo install cargo-watch`) for live reload. `npm run build` does the same auto-build
> with an optimized (release) WASM package.

Toolbar: Brush / Pen / Edit / Sculpt / Blend / Select / Pan, a color picker, a brush-size
slider, interior-hardness + edge-hardness sliders, a crisp-perimeter toggle, and a
blend-strength slider, plus Save/Load (`.gspf.json`).
Single-key shortcuts: `b` brush, `p` pen, `a` edit, `s` sculpt, `l` blend, `v` select,
`h` pan. The TypeScript shell is a thin pass-through — every edit is
handled in Rust by `WasmApp`, which owns the document, the renderer, and the bidirectional
sync.

**Edit (direct-selection) tool** — Illustrator-style vector editing of a stroke's Bézier
skeleton. Click a stroke to reveal its **handles**: square anchor markers on the curve,
round handle dots on the tangents, drawn on a 2D overlay above the WebGPU surface. Then:

- drag an **anchor** to move the node rigidly (its handles ride along);
- drag a **tangent handle** to reshape the curve — on a smooth anchor the opposite handle
  mirrors to stay collinear (preserving its length); **Alt-drag** breaks the tangent so the
  handle moves independently.
- **recolour the selected path** — with a stroke selected, pick a colour and the whole path
  (any pen curve or brush path) is recoloured to match. The picker also syncs to the selected
  path's colour when you click it. Recolouring is a pure colour edit (`stroke::set_base_color`,
  delta-shifted so per-splat texture jitter survives) — geometry and the skeleton are
  untouched.

Each edit reshapes the skeleton and **forward-syncs** the splats (`update_world_cache`), so
hand-painted residual detail survives the curve edit (acceptance test F). The overlay
geometry is computed in Rust (`WasmApp::edit_overlay`, screen-space) and only *painted* by
the shell — the model stays the single source of truth.

**Blend (smudge) tool** — bring neighbouring gaussians' **colours** into each other. Blending
here means colour diffusion, *not* geometry: every gaussian keeps its exact position and
covariance (size/orientation) — only its colour bleeds toward its neighbours', so adjacent
splats' colours mix. Nothing is moved, resized, or merged, so the ellipses never collapse into
larger blobs and (by construction, since geometry is untouched) every splat stays exactly on
the curve it lives on; the skeleton is never modified. It behaves like a real smudge brush:
on press it loads the colour under the cursor (`BlendCarry`), and as you drag (`smudge_splats`)
it deposits that carried colour onto the splats it passes — pulling their colour strongly
toward it — while the carry slowly drifts to pick up new colours, so colour visibly
*transports* along the drag. The neighbourhood is gathered across *all* strokes under the
brush, so two overlapping strokes' colours genuinely blend into each other. Because it edits
only the per-splat colour (no derived world cache), it flags each touched stroke for GPU
re-upload directly. The brush radius is the **Size** slider; the per-dab deposit/pickup amount
is the **Blend** strength slider. (`blend::blend_splats` is a stateless region-average colour
homogenise used by the headless API/tests.) The effect is most visible where there is colour
contrast — try it over two overlapping strokes of different colours, or bump **Size** up.

The demo output shows the classifier discriminating three reverse-sync scenarios:

```
coherent central drag: ... -> confidence=0.749  -> outcome=Structural   (curve bends 49px)
incoherent scatter   : ... -> confidence=0.370  -> outcome=Residual      (skeleton untouched)
symmetric widen      : ... -> confidence=0.512  -> outcome=Partial        (centerline holds)
```

## The model

The canonical, editable object is a `GaussianBezierStroke`:

```
GaussianBezierStroke
├── BezierSkeleton        chain of cubic segments + arc-length table
├── BrushModel            color, radius, spacing, width/opacity profiles, seed
├── Vec<GaussianSplat>    each attached by curve-local coords (t, u, v) + residuals
└── SyncPolicy            confidence thresholds, max structural error
```

Each splat's world center is

```
μ = P(t) + u·N(t) + v·T(t) + residual_local(in frame) + residual_world
```

where `P/T/N` come from the skeleton frame at **arc-length** coordinate `t` (not the raw
cubic parameter — arc-length keeps the splat distribution stable as the curve bends).
World center, covariance, and inverse covariance are **caches** derived from this model;
they are never serialized and never the source of truth.

### Two synchronization directions

**Forward (`stroke::update_splat_world_cache`)** — the easy direction. A skeleton edit
re-evaluates every splat from its `(t, u, v)` + residuals. Residuals ride the local
frame, so hand-painted detail survives vector edits.

**Reverse (`solver::apply_splat_edits_bidirectional`)** — the hard, ambiguous direction.
When the user sculpts splats directly, the system must infer *intent*. It scores five
coherence metrics and blends them into a confidence (weights per spec §9.1):

| metric | meaning |
|---|---|
| `same_stroke_ratio` | fraction of edited splats in this stroke |
| `t_interval_coverage` | **edit density** within the touched t-span (edited ÷ all splats in span) |
| `displacement_smoothness` | how smoothly displacement varies with `t` (scatter ⇒ low) |
| `core_weight_ratio` | how much edit weight is on structural *core* splats vs *edge*/detail |
| `fit_quality` | RMS of a trial Bézier fit to the implied centerline |

```
confidence = 0.25·same + 0.20·coverage + 0.20·smoothness + 0.20·core + 0.15·fit_quality
```

- **confidence ≥ 0.7** → update the skeleton (weighted least-squares Bézier fit), absorb
  the remainder as residuals.
- **0.4 ≤ confidence < 0.7** → apply a *partial* skeleton update blended with residuals.
- **confidence < 0.4** → skeleton untouched; store everything as residuals.

The key design refinement over the raw spec: `t_interval_coverage` measures edit
**density**, not raw span. That is what cleanly separates a contiguous brush-drag
(density ≈ 1 → structural) from scattered edge edits (density low → residual-only), even
when both touch a wide range of `t`.

### The curve fit

`fit_bezier_to_targets` solves a weighted, regularized least-squares system per segment:

```
min_C  Σ wᵢ‖P_C(sᵢ) − targetᵢ‖²  +  Σ λ_a‖C_a − C_a_old‖²
```

solved via a small Cholesky SPD solver (no `nalgebra` dependency). The regularization is
**relative per control point** (`λ_a ∝` that point's own data weight), so sparsely
constrained anchors follow the data as faithfully as densely constrained interior
handles — this is what makes a whole-stroke translate move the endpoints correctly.
`preserve_endpoints` pins anchors for a local bend, or frees them for a translate.

### Crisp line perimeters

A line is the *union* of many overlapping splats, so a crisp silhouette can't come from
hardening each splat individually and `over`-compositing them — the union of N translucent
crisp ellipses scallops and its edge opacity wobbles with overlap/draw order. Two cooperating
mechanisms keep the **perimeter** crisp while letting the **interior** stay fuzzy:

1. **Crisp edge ring (model, `stroke.rs`/`brush.rs`).** Hardness is now *per-splat*
   (`GaussianSplat::hardness`, derived from `role`): the outermost cross-section ring is thin
   (`SIGMA_NORMAL_EDGE_FACTOR`) and rendered at the brush's high `edge_hardness`, while the
   interior keeps the softer `hardness`. Texture/fuzz splats are placed *inside* ±width so
   painterly detail never softens the silhouette. Works in the single-pass `render_doc`.
2. **Two-pass coverage path (renderer, `coverage.rs`).** `render_doc_crisp` accumulates every
   splat into an offscreen premultiplied-color field (additive) plus a `Max`-blended coverage
   field, then a full-screen resolve pass thresholds the *accumulated* coverage **once**
   (`smoothstep` with a `fwidth`-normalized ~1px band) for a single crisp antialiased edge,
   and normalizes the accumulated color for a fuzzy interior. `Max` (not sum) keeps the union
   from inflating into a blobby metaball boundary.

Both are toggleable from the browser shell (`Edge` slider + `Crisp` checkbox →
`set_brush_edge_hardness` / `set_crisp_edges`). The headless test
`renderer/tests/crisp.rs` asserts the coverage path collapses a wide soft Gaussian edge to a
thin rim.

## Crate layout

```
crates/app_core/src/        ── headless domain model (no GPU, no browser)
  math.rs          glam re-exports, covariance, deterministic PRNG, Cholesky SPD solver
  bezier.rs        cubic eval, arc-length table, frame lookup (arc-t ↔ curve-s)
  splat.rs         GaussianSplat (+ curve-local coords/roles), GpuSplat packing (bytemuck)
  brush.rs         BrushModel + 1D profile curves
  stroke.rs        GaussianBezierStroke, splat generation, forward sync
  fitting.rs       freehand polyline → Bézier skeleton (RDP + cubic chain fit)  [brush tool]
  document.rs      Document / Layer / canvas, stroke store (SlotMap)
  selection.rs     selection state, CPU hit-testing, spatial grid
  solver.rs        ★ reverse sync: coherence, Bézier fit, residual handling
  blend.rs         ★ blend/smudge tool: fuse splats in curve-local space (skeleton-preserving)
  commands.rs      snapshot-based undo/redo
  serialization.rs .gspf JSON save/load (rebuilds caches on load)

crates/renderer/src/        ── wgpu/WebGPU renderer (native headless + browser surface)
  gpu.rs           context creation, offscreen target, texture readback
  camera.rs        2D pan/zoom camera + GPU uniform
  pipelines.rs     instanced splat render pipeline (premultiplied-alpha blend)
  coverage.rs      ★ two-pass crisp-perimeter path (accumulate coverage → resolve)
  buffers.rs       growable splat storage buffer + camera uniform
  picking.rs       GPU object-id pass → R32Uint id texture + readback
  shaders/         splat.wgsl, picking.wgsl, coverage_accum.wgsl, coverage_resolve.wgsl
  lib.rs           SplatRenderer (render_doc + render_doc_crisp) + collect_gpu_splats(&Document)

crates/app_wasm/src/lib.rs  ── #[wasm_bindgen] WasmApp: surface + tool dispatch + sync
                               (standalone crate; built with wasm-pack → web/pkg)

web/                        ── thin TypeScript / Vite shell
  index.html       canvas + toolbar (Brush/Bézier/Sculpt/Select/Pan, color, size, save/load)
  src/main.ts      wasm init, rAF render loop, pointer/wheel forwarding, save/load
  src/wasm.ts      typed loader   src/tools/toolbar.ts  toolbar wiring
```

## Acceptance tests (`crates/app_core/tests/sync.rs`)

| test | scenario | asserts |
|---|---|---|
| A | move a control handle | all splats follow; residual survives on its frame; continuity |
| B | drag a central band up | structural skeleton update; splats pinned; residual small |
| C | scatter 30 edge splats | residual-only; skeleton unchanged; residuals updated |
| D | push edges along normals | centerline does **not** shift (symmetric widen) |
| E | translate whole stroke | every control point (incl. anchors) translates by Δ |
| F | direct-edit a handle/anchor | drag via `move_control_point`; splats follow, residual survives, anchor drag is rigid |
| — | persistence | save/load survives structural + residual edits |

## Roadmap

Done: domain model + bidirectional sync, freehand→Bézier fitting, wgpu renderer (native +
WASM), GPU picking, the `WasmApp` bridge, and the web shell.

Next:
- **Browser polish:** visually confirm in-browser rendering; refine pointer/DPR handling.
  The direct-edit (node) tool now exposes anchor/handle editing with a 2D overlay; possible
  follow-ups are corner/smooth toggles, add/delete anchor, and re-tessellating splat density
  when a handle edit changes arc length a lot (today it forward-syncs the existing splats).
- **Selection via GPU picking:** wire `renderer::picking` into the input layer (replace CPU
  hit-testing); cache the id buffer per frame.
- **Width-profile inference** (spec §11) — promote symmetric edge edits from residuals to an
  explicit width-profile edit.
- **Scale:** tile binning / LOD for ≥100k splats; weighted-blended OIT compositing; persistent
  dirty-range GPU uploads instead of per-frame buffer rebuilds.
- **CI gap:** `app_wasm` is a standalone workspace, so `cargo --workspace` skips it; add a
  `cargo check --target wasm32-unknown-unknown` step so the bridge gets lint/type coverage.

## Known deviations from the spec

- `SlotMap` keys serialize as `{idx, version}` rather than flat integer ids. Round-trips
  correctly; a custom DTO can produce the spec's prettier JSON shape if needed.
- Width-profile *inference* (spec §11) is detected as "not a centerline move" and stored
  as residuals for now; promoting it to an explicit width-profile edit is a TODO.
- The curve fitter targets single-segment strokes robustly and handles multi-segment
  strokes per-segment with pinned anchors; a globally C¹-continuous multi-segment solve
  is future work.
- `WasmApp.new(canvas)` is a **static async factory**, not a JS `new` constructor —
  wasm-bindgen emits invalid TypeScript for async constructors. Call `await WasmApp.new(canvas)`.
- `create_stroke_from_points` returns a dense per-frame stroke index (matching
  `collect_gpu_splats` order), not the generational `StrokeId`. Stable for the current
  append-only session (no stroke deletion yet).
