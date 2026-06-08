//! WASM bridge: glues `app_core` (the editable model) and `renderer` (wgpu) to the
//! browser. The TypeScript shell drives a single [`WasmApp`] instance: it owns the GPU
//! context + surface, the `app_core::Document` (source of truth), a `Camera2D`, and the
//! transient tool/brush interaction state.
//!
//! Coordinate convention: pointer coordinates arrive in CSS/device pixels (the JS side is
//! responsible for applying device-pixel-ratio consistently with `resize`). We convert
//! screen <-> world through `Camera2D` so tools operate in document/world space, which is
//! what `app_core` expects.

use app_core::blend::{blend_splats, smudge_splats, BlendCarry};
use app_core::brush::BrushModel;
use app_core::document::Document;
use app_core::fitting::{
    fit_polyline_adaptive, fit_polyline_to_skeleton, AdaptiveFitParams, IncrementalAdaptiveFit,
};
use app_core::selection::hit_test_splat;
use app_core::solver::{apply_splat_edits_bidirectional, sculpt_move_splats};
use app_core::stroke::GaussianBezierStroke;
use app_core::{
    serialization, BezierSkeleton, ControlPointRef, CubicBezier, LayerId, StrokeId,
};

use glam::Vec2;
use renderer::{Camera2D, GpuContext, SplatRenderer};

use wasm_bindgen::prelude::*;

/// The active editing tool. Mirrors the string contract of [`WasmApp::set_tool`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tool {
    Brush,
    /// Vector-draw tool: a freehand brush whose path is fit with the curvature-adaptive
    /// fitter (see `app_core::fitting::fit_polyline_adaptive`) — long single cubics across
    /// straight runs, short ones across tight curves, corners preserved. While it is active
    /// the fitted segments (one per adaptive *window*) are highlighted in red via
    /// [`WasmApp::vector_overlay`].
    VectorDraw,
    /// Vector-blend tool: captured exactly like [`Tool::VectorDraw`] (the same freehand,
    /// curvature-adaptive fit), but the committed stroke is a *blend* path — it carries no colour
    /// and instead directionally smears the vector-stroke layer beneath it (see
    /// `renderer::vector_blend`). A live, non-destructive smudge described by a vector, distinct
    /// from the gaussian [`Tool::Blend`] which destructively rewrites splat colours.
    VectorBlend,
    Bezier,
    /// Direct-edit (node) tool: select a stroke, then drag its anchors and tangent
    /// handles — Illustrator's Direct-Selection / vector tool.
    Edit,
    Sculpt,
    /// Blend (smudge) tool: bring neighbouring gaussians' *colours* into each other (a colour
    /// diffusion/smudge). It edits colour only — positions, sizes, and the skeleton are never
    /// touched, so the ellipses stay put and never collapse into larger blobs.
    Blend,
    Select,
}

impl Tool {
    fn from_str(s: &str) -> Tool {
        match s {
            "vectordraw" => Tool::VectorDraw,
            "vectorblend" => Tool::VectorBlend,
            "bezier" => Tool::Bezier,
            "edit" => Tool::Edit,
            "sculpt" => Tool::Sculpt,
            "blend" => Tool::Blend,
            "select" => Tool::Select,
            // "brush" and any unknown value default to the brush.
            _ => Tool::Brush,
        }
    }
}

/// How a live preview / committed stroke is rendered, chosen by the tool that builds it. The
/// three kinds map onto the stroke flags consumed by the renderer (see [`apply_preview_kind`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PreviewKind {
    /// Gaussian splat cloud (brush / pen): `render_as_vector = false`.
    Splat,
    /// Plain tessellated vector path (vector-draw): `render_as_vector = true`.
    Vector,
    /// Vector-blend smear path (vector-blend): `render_as_vector = true`, `vector_blend = true`.
    Blend,
}

/// Which kernel the renderer draws splats with. This is *orthogonal* to the active tool and to
/// `crisp_edges`: every tool (brush/pen/edit/sculpt/blend/select) edits the same model and
/// works identically in either mode — the choice only changes the per-splat footprint at draw
/// time (elliptical Gaussian vs smooth convex polygon vs triangle splat).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RenderMode {
    Gaussian,
    Convex,
    Triangle,
}

impl RenderMode {
    fn from_str(s: &str) -> RenderMode {
        match s {
            "convex" => RenderMode::Convex,
            "triangle" => RenderMode::Triangle,
            // "gaussian" and any unknown value default to the Gaussian kernel.
            _ => RenderMode::Gaussian,
        }
    }
}

/// Transient state tracked across a single pointer interaction (down -> move* -> up).
#[derive(Default)]
struct PointerState {
    /// Whether a drag is in progress (a pointer_down was seen without a matching up).
    active: bool,
    /// Last pointer position in screen pixels (for computing per-move deltas).
    last_screen: Vec2,
    /// World-space points accumulated for the brush polyline.
    polyline: Vec<Vec2>,
    /// Stroke being sculpted/dragged this interaction (resolved on pointer_down).
    target_stroke: Option<StrokeId>,
    /// The brush tool's live preview stroke, rebuilt as the polyline grows so the
    /// stroke appears under the cursor while drawing rather than only on release.
    /// `None` until the polyline has at least two points to fit. On release this
    /// stroke is kept as the committed one (the handle is simply dropped).
    preview: Option<StrokeId>,
}

/// One placed anchor of the pen tool, with its symmetric tangent handles stored as
/// offsets relative to the anchor position (zero = a straight join on that side).
#[derive(Clone, Copy, Default)]
struct PenAnchor {
    position: Vec2,
    /// Handle leaving this anchor toward the next (controls the next segment's `p1`).
    out_handle: Vec2,
    /// Handle arriving at this anchor from the previous (controls that segment's `p2`).
    in_handle: Vec2,
}

/// State for an in-progress pen/Bezier path. Persists across multiple click
/// interactions until the path is finished (Enter / double-click / close) or
/// cancelled (Escape).
#[derive(Default)]
struct PenState {
    /// Anchors placed so far, in order.
    anchors: Vec<PenAnchor>,
    /// The live preview stroke rebuilt as anchors/handles change. `None` until the
    /// path has at least two anchors (one segment) to show.
    stroke: Option<StrokeId>,
    /// True while the just-placed anchor's handles are being dragged out.
    dragging_handle: bool,
    /// Whether the path closes back on its first anchor.
    closed: bool,
}

/// Transient state for the direct-edit (node) tool. A stroke is "opened for editing"
/// (`target`) so its control handles are shown; a press on one of those handles starts a
/// `drag` that lasts until release.
#[derive(Default)]
struct EditState {
    /// Stroke whose skeleton is currently shown/edited. `None` = nothing selected.
    target: Option<StrokeId>,
    /// Control point being dragged this interaction (anchor or tangent handle).
    drag: Option<ControlPointRef>,
}

#[wasm_bindgen]
pub struct WasmApp {
    // --- GPU + surface ---
    gpu: GpuContext,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    splat_renderer: SplatRenderer,

    // --- Model (source of truth) ---
    doc: Document,
    layer: LayerId,

    // --- View / interaction ---
    camera: Camera2D,
    tool: Tool,
    pointer: PointerState,
    pen: PenState,
    edit: EditState,
    /// Stroke whose adaptive-window decomposition the vector-draw overlay highlights in red.
    /// Points at the live preview while drawing and at the just-committed stroke afterwards,
    /// so the windows stay visible for inspection while the tool is selected. Read by
    /// [`WasmApp::vector_overlay`]; reset when a new vector stroke begins.
    vector_viz: Option<StrokeId>,
    /// Incremental curvature-adaptive fitter for the in-progress vector stroke. `Some` only
    /// during a vector-draw drag; it commits settled windows and re-fits just the open tail so
    /// per-move cost stays bounded instead of growing with the stroke. See `pointer_move`.
    vector_fit: Option<IncrementalAdaptiveFit>,
    /// Live keyboard-modifier state, pushed from JS via `set_modifiers` before each
    /// pointer event. `alt` breaks a smooth handle's tangent while dragging.
    mod_alt: bool,
    mod_shift: bool,

    // --- Brush parameters ---
    brush_color: [f32; 4],
    brush_radius: f32,
    brush_hardness: f32,
    /// Hardness of the stroke's outer boundary ring — keeps perimeters crisp even when
    /// `brush_hardness` (the interior) is soft. See `app_core::brush::BrushModel`.
    brush_edge_hardness: f32,
    /// Blend tool strength: fraction toward the carried/region paint applied per blend dab.
    blend_strength: f32,
    /// Vector-blend tool strength in `[0,1]`: the smear opacity stamped onto new blend strokes
    /// (`GaussianBezierStroke::blend_strength`). Distinct from `blend_strength` (the gaussian
    /// smudge tool's per-dab fraction).
    vector_blend_strength: f32,
    /// The blend (smudge) brush's carried appearance, so colour transports along a drag.
    /// Reset at the start of each blend stroke (on pointer_down).
    blend_carry: BlendCarry,
    /// When true, draw via the two-pass coverage path (`render_doc_crisp`) so line
    /// perimeters are a single crisp antialiased edge with a fuzzy interior; when false,
    /// use the direct per-splat path (`render_doc`). Toggled from JS via `set_crisp_edges`.
    crisp_edges: bool,
    /// Which kernel the renderer draws with (Gaussian vs convex polygon). Orthogonal to the
    /// tool and to `crisp_edges`, giving four draw paths total. Toggled via `set_render_mode`.
    render_mode: RenderMode,
    /// Convex hull side count (3..=8). Only affects `RenderMode::Convex`.
    convex_sides: f32,
    /// Convex corner smoothness in `[0,1]`: 0 = sharp polygon corners, 1 = rounded toward a
    /// circle. Mapped to the 3DCS smoothness δ by [`convex_delta`].
    convex_smoothness: f32,
    /// Convex edge sharpness in `[0,1]`: 0 = diffuse boundary, 1 = dense/crisp edge. Mapped to
    /// the 3DCS sharpness σ by [`convex_sigma`].
    convex_sharpness: f32,
    /// Triangle splat UI rotation in `[0,1]` (0 = apex up). Only affects `RenderMode::Triangle`.
    /// Mapped to radians by [`triangle_rotation_radians`].
    tri_rotation: f32,
    /// Triangle splat UI window smoothness in `[0,1]`: 0 = a solid, hard-edged triangle, 1 =
    /// soft/peaked falloff. Maps to the Triangle Splatting paper's σ by [`triangle_sigma`].
    tri_softness: f32,
    /// Vector-draw curve-fit softness in `[0,1]` (the curvature-adaptive fitter's `smoothness`):
    /// 0 = strict, hug-the-input tracking; 1 = anchors pinned only at curvature extrema (the
    /// apex of each bend — a sine's peaks and troughs) with the budget relaxed between them. See
    /// `app_core::fitting::AdaptiveFitParams::smoothness`.
    vector_smoothness: f32,
}

#[wasm_bindgen]
impl WasmApp {
    /// Async constructor. Creates the wgpu instance/surface/device, configures the
    /// surface to the canvas size, builds the splat renderer for the surface format, and
    /// seeds the document with one default layer.
    ///
    /// Exposed as the static async method `WasmApp.new(canvas)` (an async `constructor`
    /// generates invalid TypeScript in wasm-bindgen, so we keep it a plain factory).
    pub async fn new(canvas: web_sys::HtmlCanvasElement) -> Result<WasmApp, JsValue> {
        // Surface a Rust panic as a readable message in the browser console.
        console_error_panic_hook::set_once();

        let width = canvas.width().max(1);
        let height = canvas.height().max(1);

        let instance = wgpu::Instance::default();

        // The surface borrows the canvas; `SurfaceTarget::Canvas` clones the element
        // handle so the surface owns a `'static` reference.
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|e| JsValue::from_str(&format!("create_surface failed: {e}")))?;

        // Reuse the renderer's adapter/device setup, but supply our own instance + the
        // surface we just made so the adapter is surface-compatible. We mirror
        // `GpuContext::new` here because that helper builds its own instance.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .map_err(|e| JsValue::from_str(&format!("no suitable GPU adapter: {e}")))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("app_wasm device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| JsValue::from_str(&format!("request_device failed: {e}")))?;

        let gpu = GpuContext {
            instance,
            adapter,
            device,
            queue,
        };

        let caps = surface.get_capabilities(&gpu.adapter);
        let format = caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&gpu.device, &config);

        let splat_renderer = SplatRenderer::new(&gpu.device, format);

        // Convex-mode defaults: a crisp hexagon (sharp-ish corners, dense edge). Seed the
        // renderer's convex uniform now so the convex paths are ready the moment modes switch.
        let convex_sides = 6.0_f32;
        let convex_smoothness = 0.2_f32;
        let convex_sharpness = 0.85_f32;
        splat_renderer.set_convex_params(
            &gpu.queue,
            convex_sides,
            convex_delta(convex_smoothness),
            convex_sigma(convex_sharpness),
        );

        // Triangle-mode defaults: apex-up with a slightly soft window. Seed the renderer's
        // triangle uniform now so the triangle paths are ready the moment modes switch.
        let tri_rotation = 0.0_f32;
        let tri_softness = 0.25_f32;
        splat_renderer.set_triangle_params(
            &gpu.queue,
            triangle_rotation_radians(tri_rotation),
            triangle_sigma(tri_softness),
        );

        let mut doc = Document::new();
        let layer = doc.add_layer("Layer 1");

        let camera = Camera2D::new(Vec2::new(width as f32, height as f32));

        Ok(WasmApp {
            gpu,
            surface,
            config,
            splat_renderer,
            doc,
            layer,
            camera,
            tool: Tool::Brush,
            pointer: PointerState::default(),
            pen: PenState::default(),
            edit: EditState::default(),
            vector_viz: None,
            vector_fit: None,
            mod_alt: false,
            mod_shift: false,
            brush_color: [0.1, 0.2, 0.9, 1.0],
            brush_radius: 24.0,
            brush_hardness: BrushModel::default().hardness,
            brush_edge_hardness: BrushModel::default().edge_hardness,
            blend_strength: 0.5,
            vector_blend_strength: 1.0,
            blend_carry: BlendCarry::default(),
            crisp_edges: true,
            render_mode: RenderMode::Gaussian,
            convex_sides,
            convex_smoothness,
            convex_sharpness,
            tri_rotation,
            tri_softness,
            // Default to a moderately soft fit so the tool biases anchors toward curvature
            // extrema out of the box; the "Curve" slider tunes it.
            vector_smoothness: 0.5,
        })
    }

    /// Resize the surface and camera viewport. `width`/`height` are in physical pixels.
    pub fn resize(&mut self, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.gpu.device, &self.config);
        self.camera.viewport = Vec2::new(width as f32, height as f32);
    }

    /// Draw the current document to the surface.
    pub fn render(&mut self) {
        // Visible world rectangle for coarse per-stroke culling inside `render_doc`. With no
        // camera rotation the visible world AABB is bounded by the two screen corners mapped
        // to world; the y axis is flipped by the camera, so take the component-wise min/max
        // explicitly rather than assuming corner ordering.
        let tl = self.camera.screen_to_world(Vec2::ZERO);
        let br = self.camera.screen_to_world(self.camera.viewport);
        let view_min = tl.min(br);
        let view_max = tl.max(br);

        let frame = match self.surface.get_current_texture() {
            // `Suboptimal` still yields a usable texture (e.g. size mismatch mid-resize);
            // render it this frame and let the next configure fix the format.
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            // Surface lost/outdated/timed out (e.g. after a resize race): reconfigure and
            // skip this frame; the next render call will succeed.
            _ => {
                self.surface.configure(&self.gpu.device, &self.config);
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let bg = self.doc.canvas.background;
        let clear = wgpu::Color {
            r: bg[0] as f64,
            g: bg[1] as f64,
            b: bg[2] as f64,
            a: bg[3] as f64,
        };

        // Incremental, resident render: every path reconciles the GPU buffer against the
        // document (uploading only new/edited strokes via the per-stroke dirty flags) and
        // draws only the view-visible stroke ranges. A camera-only frame uploads no splats.
        // Two orthogonal toggles pick the path: `render_mode` (Gaussian vs convex kernel) and
        // `crisp_edges` (direct per-splat vs the two-pass coverage path for crisp perimeters),
        // giving four combinations — convex inherits the crisp path (and thus blending) for free.
        let camera = self.camera.uniform();
        let (w, h) = (self.config.width, self.config.height);
        match (self.render_mode, self.crisp_edges) {
            (RenderMode::Gaussian, false) => self.splat_renderer.render_doc(
                &self.gpu.device, &self.gpu.queue, &view, &mut self.doc, &camera, view_min,
                view_max, clear,
            ),
            (RenderMode::Gaussian, true) => self.splat_renderer.render_doc_crisp(
                &self.gpu.device, &self.gpu.queue, &view, &mut self.doc, &camera, view_min,
                view_max, w, h, clear,
            ),
            (RenderMode::Convex, false) => self.splat_renderer.render_doc_convex(
                &self.gpu.device, &self.gpu.queue, &view, &mut self.doc, &camera, view_min,
                view_max, clear,
            ),
            (RenderMode::Convex, true) => self.splat_renderer.render_doc_convex_crisp(
                &self.gpu.device, &self.gpu.queue, &view, &mut self.doc, &camera, view_min,
                view_max, w, h, clear,
            ),
            (RenderMode::Triangle, false) => self.splat_renderer.render_doc_triangle(
                &self.gpu.device, &self.gpu.queue, &view, &mut self.doc, &camera, view_min,
                view_max, clear,
            ),
            (RenderMode::Triangle, true) => self.splat_renderer.render_doc_triangle_crisp(
                &self.gpu.device, &self.gpu.queue, &view, &mut self.doc, &camera, view_min,
                view_max, w, h, clear,
            ),
        }

        // Conventional vector strokes (`render_as_vector`) are skipped by the splat passes
        // above; draw them as tessellated, antialiased stroked paths composited on top.
        self.splat_renderer.render_vector_paths(
            &self.gpu.device,
            &self.gpu.queue,
            &view,
            &self.doc,
            &camera,
        );

        frame.present();
    }

    /// Pointer pressed at screen `(x, y)`. Begins a tool-specific interaction.
    pub fn pointer_down(&mut self, x: f32, y: f32) {
        let screen = Vec2::new(x, y);
        let world = self.camera.screen_to_world(screen);

        self.pointer.active = true;
        self.pointer.last_screen = screen;
        self.pointer.polyline.clear();
        self.pointer.target_stroke = None;
        // Drop any preview left over from a gesture that didn't finish cleanly.
        self.brush_discard_preview();

        match self.tool {
            Tool::Brush => {
                self.pointer.polyline.push(world);
            }
            Tool::VectorDraw | Tool::VectorBlend => {
                // Same freehand capture as the brush; the fitter differs (see pointer_move).
                // Clear the previous stroke's window highlight so a new gesture starts clean.
                self.vector_viz = None;
                self.pointer.polyline.push(world);
                // Start a fresh incremental fit; the zoom-aware params are captured now and held
                // for the whole gesture (zoom rarely changes mid-draw).
                self.vector_fit = Some(IncrementalAdaptiveFit::new(self.vector_fit_params(), world));
            }
            Tool::Sculpt => {
                // Resolve which stroke this interaction acts on (nearest splat under the
                // cursor). Held for the duration of the drag.
                self.pointer.target_stroke = hit_test_splat(&self.doc, world).map(|h| h.stroke);
            }
            Tool::Blend => {
                // Blend works across whatever strokes fall under the brush, so there is no
                // single target to resolve. Start a fresh smudge: the first contact loads the
                // brush with the colour under the cursor, and the drag deposits it onward.
                self.blend_carry.reset();
                self.apply_blend(world);
            }
            Tool::Bezier => {
                // Pen tool: each press places an anchor. Clicking back onto the first
                // anchor (within a small screen-space radius) closes and finishes the
                // path. A press followed by a drag pulls out symmetric curve handles;
                // a press with no drag yields a straight (corner) join.
                if self.pen.anchors.len() >= 2 {
                    let first = self.pen.anchors[0].position;
                    let close_radius = 14.0 / self.camera.zoom;
                    if (world - first).length() <= close_radius {
                        self.finalize_pen(true);
                        return;
                    }
                }
                self.pen.anchors.push(PenAnchor {
                    position: world,
                    ..PenAnchor::default()
                });
                self.pen.dragging_handle = true;
                self.pen_rebuild_preview();
            }
            Tool::Edit => {
                // Direct-edit (node) tool. Priority: grab a control point of the stroke
                // already opened for editing; otherwise (re)select the stroke under the
                // cursor so its handles appear; an empty click deselects.
                if let Some(sid) = self.edit.target {
                    match self.doc.stroke(sid) {
                        Some(stroke) => {
                            if let Some(cp) = self.pick_control_point(stroke, world) {
                                self.edit.drag = Some(cp);
                                return;
                            }
                        }
                        // The opened stroke vanished (e.g. a load); fall through to reselect.
                        None => self.edit.target = None,
                    }
                }
                self.edit.drag = None;
                self.edit.target = hit_test_splat(&self.doc, world).map(|h| h.stroke);
                self.doc.selection.clear();
                if let Some(sid) = self.edit.target {
                    self.doc.selection.strokes.push(sid);
                }
            }
            Tool::Select => {
                self.doc.selection.clear();
                if let Some(hit) = hit_test_splat(&self.doc, world) {
                    self.doc.selection.strokes.push(hit.stroke);
                    self.doc.selection.splats.push((hit.stroke, hit.splat));
                }
            }
        }
    }

    /// Pointer moved to screen `(x, y)`.
    pub fn pointer_move(&mut self, x: f32, y: f32) {
        if !self.pointer.active {
            return;
        }
        let screen = Vec2::new(x, y);
        let world = self.camera.screen_to_world(screen);
        let prev_screen = self.pointer.last_screen;

        match self.tool {
            Tool::Brush => {
                self.pointer.polyline.push(world);
                // Live preview: refit the in-progress polyline and update the preview
                // stroke so it tracks the cursor as the user draws.
                self.brush_rebuild_preview();
            }
            Tool::VectorDraw | Tool::VectorBlend => {
                self.pointer.polyline.push(world);
                // Feed the incremental fitter one point; it commits settled windows and re-fits
                // only the open tail, so this stays cheap no matter how long the stroke gets.
                if let Some(fit) = self.vector_fit.as_mut() {
                    fit.push_point(world);
                }
                // Live preview from the incremental fit; also refreshes the red window overlay
                // (vector_viz) so anchors appear/dissolve as the path bends.
                self.vector_rebuild_preview();
            }
            Tool::Sculpt => {
                let delta_world = world - self.camera.screen_to_world(prev_screen);
                self.apply_sculpt(world, delta_world);
            }
            Tool::Blend => {
                // Continuous smudge: apply a blend dab at each move so the effect
                // accumulates under the dragging cursor.
                self.apply_blend(world);
            }
            Tool::Bezier => {
                // While the button is held after placing an anchor, drag pulls out the
                // anchor's tangent handles. They're mirrored (smooth anchor) so the curve
                // flows through the point, matching how a vector pen tool behaves.
                if self.pen.dragging_handle {
                    if let Some(anchor) = self.pen.anchors.last_mut() {
                        let handle = world - anchor.position;
                        anchor.out_handle = handle;
                        anchor.in_handle = -handle;
                    }
                    self.pen_rebuild_preview();
                }
            }
            Tool::Edit => {
                // Drag the grabbed control point to the cursor. A smooth anchor mirrors its
                // opposite handle unless Alt is held (which breaks the tangent). Forward
                // sync re-evaluates the splats from the reshaped skeleton, preserving any
                // hand-painted residuals.
                if let (Some(sid), Some(cp)) = (self.edit.target, self.edit.drag) {
                    let mirror = !self.mod_alt;
                    if let Some(stroke) = self.doc.stroke_mut(sid) {
                        stroke.skeleton.move_control_point(cp, world, mirror);
                        stroke.update_world_cache();
                    }
                }
            }
            Tool::Select => {}
        }

        self.pointer.last_screen = screen;
    }

    /// Pointer released at screen `(x, y)`. Finalizes the interaction.
    pub fn pointer_up(&mut self, x: f32, y: f32) {
        if !self.pointer.active {
            return;
        }
        let world = self.camera.screen_to_world(Vec2::new(x, y));

        if self.tool == Tool::Brush {
            if self.pointer.polyline.last() != Some(&world) {
                self.pointer.polyline.push(world);
            }
            // Need at least two distinct points to fit a skeleton.
            if self.pointer.polyline.len() >= 2 {
                // Refit with the final point, then keep the preview stroke as the
                // committed one — drop only our handle so it stays in the document.
                self.brush_rebuild_preview();
                self.pointer.preview = None;
            } else {
                // A click without a drag can't be fit: drop any preview.
                self.brush_discard_preview();
            }
        }

        if matches!(self.tool, Tool::VectorDraw | Tool::VectorBlend) {
            if self.pointer.polyline.last() != Some(&world) {
                self.pointer.polyline.push(world);
                if let Some(fit) = self.vector_fit.as_mut() {
                    fit.push_point(world);
                }
            }
            if self.pointer.polyline.len() >= 2 {
                self.vector_rebuild_preview();
                // Splats were skipped during the drag (the preview renders as a tessellated
                // vector path). Generate the final cloud once now so the committed stroke
                // supports hit-testing and direct-edit.
                if let Some(sid) = self.pointer.preview {
                    if let Some(stroke) = self.doc.stroke_mut(sid) {
                        stroke.regenerate_splats();
                    }
                }
                // Keep the committed stroke; keep pointing the red overlay at it so its
                // adaptive windows stay visible after release. Drop only our preview handle.
                self.vector_viz = self.pointer.preview;
                self.pointer.preview = None;
            } else {
                self.brush_discard_preview();
                self.vector_viz = None;
            }
            // The gesture is over; release the incremental fitter.
            self.vector_fit = None;
        }

        // Releasing ends the handle drag but keeps the pen path open for more anchors.
        if self.tool == Tool::Bezier {
            self.pen.dragging_handle = false;
        }

        // Direct-edit: releasing ends the node/handle drag; the stroke stays selected so
        // its handles remain visible for the next edit.
        if self.tool == Tool::Edit {
            self.edit.drag = None;
        }

        self.pointer.active = false;
        self.pointer.polyline.clear();
        self.pointer.target_stroke = None;
    }

    /// Zoom centered at screen `(x, y)`. `delta` is a wheel delta: positive zooms in.
    /// The shell routes zoom here only for Shift+scroll (and trackpad pinch); a plain
    /// scroll/two-finger swipe pans via [`WasmApp::pan`] instead.
    pub fn wheel(&mut self, x: f32, y: f32, delta: f32) {
        // Map the wheel delta to a multiplicative zoom factor. A typical wheel notch is
        // ~+/-100; exp keeps zooming symmetric and smooth.
        let factor = (delta * 0.0015).exp();
        self.camera.zoom_at(Vec2::new(x, y), factor);
    }

    /// Pan the view by a screen-pixel delta from a trackpad two-finger swipe (or scroll
    /// wheel). Positive `dx`/`dy` move the viewport right/down across the document, so the
    /// canvas tracks the swipe the way scrolling moves a page. This replaces the old Pan
    /// tool for everyday navigation — panning no longer needs a tool to be selected.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        self.camera.pan_pixels(Vec2::new(-dx, -dy));
    }

    /// Set the active tool. Unknown strings fall back to the brush.
    pub fn set_tool(&mut self, tool: &str) {
        let next = Tool::from_str(tool);
        // Leaving the pen tool commits whatever path is in progress so it isn't lost.
        if self.tool == Tool::Bezier && next != Tool::Bezier {
            self.finalize_pen(false);
        }
        // Any pending node drag is interaction-scoped; never carry it across a tool switch.
        self.edit.drag = None;
        // Entering the direct-edit tool with nothing open adopts the current selection, so
        // a stroke picked with the Select tool is immediately editable.
        if next == Tool::Edit && self.edit.target.is_none() {
            self.edit.target = self.doc.selection.strokes.first().copied();
        }
        self.tool = next;
    }

    /// Finish the in-progress pen path, keeping it as a committed stroke. No-op when no
    /// path is being drawn. Wire this to Enter / double-click on the JS side.
    pub fn finish_path(&mut self) {
        self.finalize_pen(false);
    }

    /// Abort the in-progress pen path, discarding its preview stroke. No-op when no path
    /// is being drawn. Wire this to Escape on the JS side.
    pub fn cancel_path(&mut self) {
        self.pen_discard_preview();
        self.pen = PenState::default();
    }

    /// Push the current keyboard-modifier state from JS. Call before forwarding a pointer
    /// event so tools see the right modifiers (the direct-edit tool uses `alt` to break a
    /// smooth handle's tangent while dragging).
    pub fn set_modifiers(&mut self, alt: bool, shift: bool) {
        self.mod_alt = alt;
        self.mod_shift = shift;
    }

    /// Geometry for the direct-edit handle overlay, in **device-pixel screen space**, as a
    /// flat `f32` array (a JS `Float32Array`). Empty when no stroke is open for editing.
    ///
    /// Layout:
    /// ```text
    /// [ anchor_count, path_point_count,
    ///   path_x0, path_y0, path_x1, path_y1, ...,      // polyline tracing the curve
    ///   // then 6 floats per anchor:
    ///   anchor_x, anchor_y,
    ///   in_handle_x,  in_handle_y,                    // NaN,NaN when absent
    ///   out_handle_x, out_handle_y ]                  // NaN,NaN when absent
    /// ```
    /// The JS overlay decodes the header, draws the path, then for each anchor draws the
    /// handle lines/dots and the anchor square. A missing handle is signalled by `NaN`.
    pub fn edit_overlay(&self) -> Vec<f32> {
        let mut out = Vec::new();
        let Some(sid) = self.edit.target else {
            return out;
        };
        let Some(stroke) = self.doc.stroke(sid) else {
            return out;
        };
        let sk = &stroke.skeleton;
        let n = sk.anchor_count();
        if n == 0 {
            return out;
        }

        // Trace the curve as a screen-space polyline (a handful of samples per segment).
        const STEPS: usize = 24;
        let mut path: Vec<Vec2> = Vec::new();
        for (si, seg) in sk.segments.iter().enumerate() {
            // Skip the shared join point so consecutive segments don't duplicate it.
            let start = if si == 0 { 0 } else { 1 };
            for k in start..=STEPS {
                let s = k as f32 / STEPS as f32;
                path.push(self.camera.world_to_screen(seg.point(s)));
            }
        }

        out.push(n as f32);
        out.push(path.len() as f32);
        for p in &path {
            out.push(p.x);
            out.push(p.y);
        }

        let push_pt = |out: &mut Vec<f32>, w: Option<Vec2>| match w {
            Some(w) => {
                let s = self.camera.world_to_screen(w);
                out.push(s.x);
                out.push(s.y);
            }
            None => {
                out.push(f32::NAN);
                out.push(f32::NAN);
            }
        };
        for j in 0..n {
            let a = self.camera.world_to_screen(sk.anchor_position(j));
            out.push(a.x);
            out.push(a.y);
            push_pt(&mut out, sk.control_point(ControlPointRef::in_handle(j)));
            push_pt(&mut out, sk.control_point(ControlPointRef::out_handle(j)));
        }
        out
    }

    /// Geometry that visualizes the vector-draw tool's adaptive windows, in **device-pixel
    /// screen space**, as a flat `f32` array (a JS `Float32Array`). Each fitted segment is one
    /// adaptive window: long/sparse across straight runs, short/dense across tight curves. The
    /// segments are emitted in draw order, so the last point of the last segment is the stroke
    /// **tip** (the point under the cursor). The JS overlay (`drawVectorOverlay`) renders the
    /// red as a comet tail — opaque at the tip and fading to nothing a fixed distance back —
    /// and draws each anchor as a marker (a corner anchor as a filled diamond, a smooth one as
    /// a hollow square), faded by the same tip-relative falloff. Empty when no vector stroke is
    /// being shown.
    ///
    /// Layout:
    /// ```text
    /// [ seg_count,
    ///   // repeated seg_count times — one polyline per segment:
    ///   pt_count, x0, y0, x1, y1, ...,
    ///   // then anchors:
    ///   anchor_count, ax0, ay0, is_corner0, ax1, ay1, is_corner1, ... ]
    /// ```
    /// `is_corner` is `1.0` for a hard corner (independent tangents) or `0.0` for a smooth join.
    pub fn vector_overlay(&self) -> Vec<f32> {
        let mut out = Vec::new();
        let Some(sid) = self.vector_viz else {
            return out;
        };
        let Some(stroke) = self.doc.stroke(sid) else {
            return out;
        };
        let sk = &stroke.skeleton;
        if sk.segments.is_empty() {
            return out;
        }

        const STEPS: usize = 16;
        out.push(sk.segments.len() as f32);
        for seg in &sk.segments {
            out.push((STEPS + 1) as f32);
            for k in 0..=STEPS {
                let s = k as f32 / STEPS as f32;
                let p = self.camera.world_to_screen(seg.point(s));
                out.push(p.x);
                out.push(p.y);
            }
        }

        let n = sk.anchor_count();
        out.push(n as f32);
        for j in 0..n {
            let a = self.camera.world_to_screen(sk.anchor_position(j));
            out.push(a.x);
            out.push(a.y);
            let is_corner = sk.anchors.get(j).map(|m| m.corner).unwrap_or(false);
            out.push(if is_corner { 1.0 } else { 0.0 });
        }
        out
    }

    /// Set the brush color (linear RGB, alpha preserved from the current brush).
    ///
    /// In the direct-edit (node) tool, this doubles as a "recolour the selected path"
    /// control: if a stroke is open for editing, its colour is updated to match — so picking
    /// a colour while a curve or brush path is selected recolours that whole path (geometry
    /// untouched). Outside the Edit tool it only sets the colour future strokes are drawn in.
    pub fn set_brush_color(&mut self, r: f32, g: f32, b: f32) {
        self.brush_color = [r, g, b, self.brush_color[3]];
        if self.tool == Tool::Edit {
            if let Some(sid) = self.edit.target {
                if let Some(stroke) = self.doc.stroke_mut(sid) {
                    stroke.set_base_color([r, g, b]);
                }
            }
        }
    }

    /// The colour (`#rrggbb`) of the stroke currently open in the direct-edit tool, or an
    /// empty string when none is selected. The shell reads this after a selection click so the
    /// colour picker reflects the chosen path's colour.
    pub fn edit_target_color_hex(&self) -> String {
        let Some(sid) = self.edit.target else {
            return String::new();
        };
        let Some(stroke) = self.doc.stroke(sid) else {
            return String::new();
        };
        let c = stroke.brush.base_color;
        let to = |v: f32| ((v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32).min(255);
        format!("#{:02x}{:02x}{:02x}", to(c[0]), to(c[1]), to(c[2]))
    }

    /// Set the brush radius (also used as the sculpt radius).
    pub fn set_brush_radius(&mut self, radius: f32) {
        self.brush_radius = radius.max(0.1);
    }

    /// Set the brush *interior* hardness in `[0,1]`. 0 = soft Gaussian falloff, 1 = crisp.
    /// This is the inside of the line; the perimeter is governed by `set_brush_edge_hardness`.
    pub fn set_brush_hardness(&mut self, hardness: f32) {
        self.brush_hardness = hardness.clamp(0.0, 1.0);
    }

    /// Set the hardness of the stroke's boundary ring in `[0,1]`. High keeps the line's
    /// **perimeter** crisp even when the interior (`set_brush_hardness`) is soft.
    pub fn set_brush_edge_hardness(&mut self, hardness: f32) {
        self.brush_edge_hardness = hardness.clamp(0.0, 1.0);
    }

    /// Toggle the crisp-perimeter render path. When on, lines are drawn through the two-pass
    /// coverage path so the silhouette is one crisp antialiased edge. Works in both render
    /// modes (Gaussian and convex).
    pub fn set_crisp_edges(&mut self, enabled: bool) {
        self.crisp_edges = enabled;
    }

    /// Switch the render kernel: `"gaussian"` (anisotropic Gaussian splats, the default),
    /// `"convex"` (smooth convex-polygon splats), or `"triangle"` (triangle splats). Orthogonal
    /// to the active tool and to the crisp-edges toggle — the document and every tool
    /// (brush/pen/edit/sculpt/blend/select) are untouched; only the per-splat footprint drawn
    /// each frame changes. Unknown strings fall back to Gaussian.
    pub fn set_render_mode(&mut self, mode: &str) {
        self.render_mode = RenderMode::from_str(mode);
    }

    /// The active render mode as the same string contract `set_render_mode` accepts.
    pub fn render_mode(&self) -> String {
        match self.render_mode {
            RenderMode::Gaussian => "gaussian".to_string(),
            RenderMode::Convex => "convex".to_string(),
            RenderMode::Triangle => "triangle".to_string(),
        }
    }

    /// Set the convex hull's side count, clamped to `3..=8` (matching the shader). Affects only
    /// convex mode; pushes the new shape to the GPU immediately.
    pub fn set_convex_sides(&mut self, sides: f32) {
        self.convex_sides = sides.clamp(3.0, 8.0);
        self.push_convex_params();
    }

    /// Set the convex *corner* smoothness in `[0,1]` (3DCS δ): 0 = sharp polygon corners, 1 =
    /// rounded toward a circle. Affects only convex mode; pushes the new shape immediately.
    pub fn set_convex_smoothness(&mut self, smoothness: f32) {
        self.convex_smoothness = smoothness.clamp(0.0, 1.0);
        self.push_convex_params();
    }

    /// Set the convex *edge* sharpness in `[0,1]` (3DCS σ): 0 = diffuse boundary, 1 = a dense,
    /// hard edge (the crisp, low-primitive-count regime convex splatting is for). Affects only
    /// convex mode; pushes the new shape immediately.
    pub fn set_convex_sharpness(&mut self, sharpness: f32) {
        self.convex_sharpness = sharpness.clamp(0.0, 1.0);
        self.push_convex_params();
    }

    /// Set the triangle orientation in `[0,1]` (0 = apex up, 1 = full turn). Triangle mode only.
    pub fn set_triangle_rotation(&mut self, rotation: f32) {
        self.tri_rotation = rotation.clamp(0.0, 1.0);
        self.push_triangle_params();
    }

    /// Set the triangle window smoothness in `[0,1]` (Triangle Splatting σ): 0 = a solid, hard-
    /// edged triangle, 1 = a soft falloff peaking at the incenter. Triangle mode only.
    pub fn set_triangle_softness(&mut self, softness: f32) {
        self.tri_softness = softness.clamp(0.0, 1.0);
        self.push_triangle_params();
    }

    /// Set the vector-draw curve-fit softness in `[0,1]`: 0 = strict (hug the input, anchors
    /// wherever the curvature-adaptive budget dictates), 1 = soft (anchors pinned at curvature
    /// extrema — the apex of each bend, e.g. a sine's peaks and troughs — and nowhere else).
    /// Applies to the vector-draw and vector-blend tools; the incremental fitter captures it at
    /// the start of a stroke, so it takes effect on the next stroke drawn.
    pub fn set_vector_smoothness(&mut self, smoothness: f32) {
        self.vector_smoothness = smoothness.clamp(0.0, 1.0);
    }

    /// Set the blend (smudge) tool strength in `[0,1]`: the fraction toward the local
    /// average applied on each blend dab. Higher = faster merging.
    pub fn set_blend_strength(&mut self, strength: f32) {
        self.blend_strength = strength.clamp(0.0, 1.0);
    }

    /// Set the vector-blend tool's smear strength in `[0,1]`: the opacity with which a blend
    /// stroke composites its directional smear of the underlying vector layer (0 = invisible,
    /// 1 = full smear). Applied to blend strokes drawn after this call. If a blend stroke is
    /// open in the direct-edit tool, its strength is updated live.
    pub fn set_vector_blend_strength(&mut self, strength: f32) {
        self.vector_blend_strength = strength.clamp(0.0, 1.0);
        if self.tool == Tool::Edit {
            if let Some(sid) = self.edit.target {
                if let Some(stroke) = self.doc.stroke_mut(sid) {
                    if stroke.vector_blend {
                        stroke.blend_strength = self.vector_blend_strength;
                    }
                }
            }
        }
    }

    /// Create a stroke from a flat `[x0,y0,x1,y1,...]` array of **world** coordinates.
    /// Fits a Bezier skeleton through the points and adds the stroke. Returns the new
    /// stroke's dense per-document index (its position in iteration / GPU order).
    pub fn create_stroke_from_points(&mut self, pts: &[f32]) -> u32 {
        let points: Vec<Vec2> = pts.chunks_exact(2).map(|c| Vec2::new(c[0], c[1])).collect();
        if points.len() < 2 {
            return u32::MAX;
        }
        let sid = self.add_brush_stroke(&points);
        self.stroke_index(sid).unwrap_or(u32::MAX)
    }

    /// Sculpt: move splats near world `(x, y)` by world `(dx, dy)` within `radius`, then
    /// run the bidirectional solver so coherent drags bend the skeleton and incoherent
    /// ones are absorbed as residual deformation.
    pub fn sculpt_splats(&mut self, x: f32, y: f32, dx: f32, dy: f32, radius: f32) {
        let center = Vec2::new(x, y);
        let delta = Vec2::new(dx, dy);
        let target = hit_test_splat(&self.doc, center).map(|h| h.stroke);
        self.sculpt_on_stroke(target, center, delta, radius);
    }

    /// Blend: fuse splats within `radius` of world `(x, y)` toward their local average
    /// (colour, opacity, size, and curve-local position), keeping every splat on its curve.
    /// `strength` is the fraction-toward-average applied by this call. Returns the number of
    /// splats moved.
    pub fn blend_at(&mut self, x: f32, y: f32, radius: f32, strength: f32) -> usize {
        blend_splats(&mut self.doc, Vec2::new(x, y), radius, strength)
    }

    /// Total splat count across the document.
    pub fn splat_count(&self) -> usize {
        self.doc.splat_count()
    }

    /// Serialize the document to JSON.
    pub fn save_json(&self) -> String {
        serialization::save_json(&self.doc).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
    }

    /// Replace the document from JSON. Invalid JSON leaves the current document intact.
    pub fn load_json(&mut self, json: &str) {
        if let Ok(doc) = serialization::load_json(json) {
            self.doc = doc;
            // The previous document's strokes are gone; drop the resident layout so its
            // stale per-stroke slots don't linger into the freshly-loaded document.
            self.splat_renderer.reset_scene();
            // The deserialized document brings its own layers; point editing at the first
            // one (creating one if the file had none) so subsequent strokes have a home.
            self.layer = match self.doc.layers.first() {
                Some(layer) => layer.id,
                None => self.doc.add_layer("Layer 1"),
            };
            self.doc.selection.clear();
        }
    }
}

// --- Internal helpers (not exported to JS) ---
impl WasmApp {
    /// Push the current convex shape parameters to the renderer's convex uniform. Cheap (one
    /// small `write_buffer`); called only when a convex control changes.
    fn push_convex_params(&self) {
        self.splat_renderer.set_convex_params(
            &self.gpu.queue,
            self.convex_sides,
            convex_delta(self.convex_smoothness),
            convex_sigma(self.convex_sharpness),
        );
    }

    /// Push the current triangle shape parameters to the renderer's triangle uniform.
    fn push_triangle_params(&self) {
        self.splat_renderer.set_triangle_params(
            &self.gpu.queue,
            triangle_rotation_radians(self.tri_rotation),
            triangle_sigma(self.tri_softness),
        );
    }

    /// Build a `BrushModel` from the current brush parameters.
    fn current_brush(&self) -> BrushModel {
        BrushModel {
            base_color: self.brush_color,
            radius: self.brush_radius,
            hardness: self.brush_hardness,
            edge_hardness: self.brush_edge_hardness,
            ..BrushModel::default()
        }
    }

    /// Fit `points` (world coords) into a skeleton with a zoom-aware tolerance, so the
    /// fit tracks what the user sees (~1px on screen regardless of zoom level).
    fn fit_brush_skeleton(&self, points: &[Vec2]) -> BezierSkeleton {
        let tolerance = (1.5 / self.camera.zoom).max(0.25);
        fit_polyline_to_skeleton(points, tolerance)
    }

    /// Zoom-aware parameters for the curvature-adaptive fitter: tolerance and resample step are
    /// held at a roughly constant size on screen, so the same gesture yields the same anchor
    /// density regardless of zoom level.
    fn vector_fit_params(&self) -> AdaptiveFitParams {
        AdaptiveFitParams {
            tolerance: (1.5 / self.camera.zoom).max(0.25),
            resample_step: (3.0 / self.camera.zoom).max(0.75),
            smoothness: self.vector_smoothness,
            ..AdaptiveFitParams::default()
        }
    }

    /// Fit `points` (world coords) with the curvature-adaptive fitter (batch). Used as a
    /// fallback; the live drag drives [`IncrementalAdaptiveFit`] instead (see `pointer_move`).
    fn fit_vector_skeleton(&self, points: &[Vec2]) -> BezierSkeleton {
        fit_polyline_adaptive(points, &self.vector_fit_params())
    }

    /// Fit `points` (world coords) to a skeleton and add a stroke on the current layer.
    fn add_brush_stroke(&mut self, points: &[Vec2]) -> StrokeId {
        let skeleton = self.fit_brush_skeleton(points);
        self.doc.add_stroke(self.layer, skeleton, self.current_brush())
    }

    /// Rebuild (or create) the brush tool's live preview stroke from the in-progress
    /// polyline. Once two or more points exist the stroke is a real, in-place document
    /// stroke updated each move; with fewer points there is nothing to fit yet, so any
    /// existing preview is dropped. Mirrors the pen tool's preview path.
    fn brush_rebuild_preview(&mut self) {
        if self.pointer.polyline.len() < 2 {
            self.brush_discard_preview();
            return;
        }
        let skeleton = self.fit_brush_skeleton(&self.pointer.polyline);
        self.set_preview_stroke(skeleton, PreviewKind::Splat);
    }

    /// Rebuild (or create) the vector-draw tool's live preview from the in-progress polyline
    /// using the curvature-adaptive fitter, and point the red window overlay at it. Mirrors
    /// [`Self::brush_rebuild_preview`] but with the adaptive fitter, and flags the stroke for
    /// conventional vector rendering (a stroked path, not splats).
    fn vector_rebuild_preview(&mut self) {
        if self.pointer.polyline.len() < 2 {
            self.brush_discard_preview();
            self.vector_viz = None;
            return;
        }
        // Prefer the incremental fitter (cheap, O(open tail)); fall back to a batch fit if it is
        // somehow absent (e.g. a stray rebuild outside a drag).
        let skeleton = match self.vector_fit.as_ref() {
            Some(fit) => fit.skeleton(),
            None => self.fit_vector_skeleton(&self.pointer.polyline),
        };
        // The vector-blend tool commits a smear path; the vector-draw tool a plain vector path.
        let kind = if self.tool == Tool::VectorBlend {
            PreviewKind::Blend
        } else {
            PreviewKind::Vector
        };
        self.set_preview_stroke(skeleton, kind);
        self.vector_viz = self.pointer.preview;
    }

    /// Update the active preview stroke (`pointer.preview`) to `skeleton` with the current
    /// brush, creating it on the current layer if it does not exist yet. `kind` selects how the
    /// stroke renders (splat cloud / plain vector / vector-blend smear). Shared by the brush,
    /// vector-draw, and vector-blend previews.
    ///
    /// Vector and blend previews are drawn by tessellating the skeleton each frame (the splat
    /// path skips `render_as_vector` strokes), so their splats are invisible mid-drag.
    /// Regenerating the whole splat cloud on every pointer move would be pure wasted O(n) work —
    /// the dominant cost of a long vector stroke — so for those previews we skip it and only
    /// clear the now stale splats. The final cloud (needed for hit-testing/editing) is generated
    /// once on release; see `pointer_up`.
    fn set_preview_stroke(&mut self, skeleton: BezierSkeleton, kind: PreviewKind) {
        let brush = self.current_brush();
        let strength = self.vector_blend_strength;
        match self.pointer.preview {
            Some(sid) if self.doc.stroke(sid).is_some() => {
                let stroke = self.doc.stroke_mut(sid).expect("checked present above");
                stroke.brush = brush;
                stroke.skeleton = skeleton;
                apply_preview_kind(stroke, kind, strength);
            }
            _ => {
                let sid = self.doc.add_stroke(self.layer, skeleton, brush);
                if let Some(stroke) = self.doc.stroke_mut(sid) {
                    apply_preview_kind(stroke, kind, strength);
                }
                self.pointer.preview = Some(sid);
            }
        }
    }

    /// Remove the brush tool's preview stroke from the document (used on cancel, or when
    /// the polyline no longer has enough points to be drawn).
    fn brush_discard_preview(&mut self) {
        if let Some(sid) = self.pointer.preview.take() {
            self.doc.remove_stroke(sid);
        }
    }

    /// Sculpt within the current drag (uses the resolved target stroke + brush radius).
    fn apply_sculpt(&mut self, center: Vec2, delta: Vec2) {
        let target = self.pointer.target_stroke;
        let radius = self.brush_radius;
        self.sculpt_on_stroke(target, center, delta, radius);
    }

    /// Smudge dab at `center`, using the current brush radius and blend strength. Carries
    /// colour between dabs (`blend_carry`) so it transports along the drag like a real blend
    /// brush; the carry is reset on pointer_down so each stroke starts clean.
    fn apply_blend(&mut self, center: Vec2) {
        smudge_splats(
            &mut self.doc,
            center,
            self.brush_radius,
            self.blend_strength,
            &mut self.blend_carry,
        );
    }

    /// Shared sculpt path used by both interactive sculpting and the `sculpt_splats` API.
    fn sculpt_on_stroke(
        &mut self,
        target: Option<StrokeId>,
        center: Vec2,
        delta: Vec2,
        radius: f32,
    ) {
        // Pick the stroke to sculpt: the explicitly resolved target, else the nearest.
        let sid = match target.or_else(|| hit_test_splat(&self.doc, center).map(|h| h.stroke)) {
            Some(sid) => sid,
            None => return,
        };
        let Some(stroke) = self.doc.stroke(sid) else {
            return;
        };
        let edits = sculpt_move_splats(stroke, center, delta, radius);
        if edits.is_empty() {
            return;
        }
        apply_splat_edits_bidirectional(&mut self.doc, &edits);
    }

    /// Rebuild (or create) the pen tool's live preview stroke from the placed anchors.
    /// Once two or more anchors exist the path is a real, editable stroke updated in
    /// place; with fewer anchors there is nothing to draw yet.
    fn pen_rebuild_preview(&mut self) {
        let Some(skeleton) = skeleton_from_pen(&self.pen.anchors, self.pen.closed) else {
            self.pen_discard_preview();
            return;
        };
        let brush = self.current_brush();
        match self.pen.stroke {
            Some(sid) if self.doc.stroke(sid).is_some() => {
                let stroke = self.doc.stroke_mut(sid).expect("checked present above");
                stroke.brush = brush;
                stroke.skeleton = skeleton;
                stroke.regenerate_splats();
            }
            _ => {
                let sid = self.doc.add_stroke(self.layer, skeleton, brush);
                self.pen.stroke = Some(sid);
            }
        }
    }

    /// Remove the pen tool's preview stroke from the document (used on cancel, or when
    /// the path no longer has enough anchors to be drawn).
    fn pen_discard_preview(&mut self) {
        if let Some(sid) = self.pen.stroke.take() {
            self.doc.remove_stroke(sid);
        }
    }

    /// Commit the in-progress pen path. With at least two anchors it stays as a stroke
    /// (optionally `close`d); otherwise any preview is discarded. Resets pen state so the
    /// next press starts a fresh path.
    fn finalize_pen(&mut self, close: bool) {
        // A double-click to finish lands a trailing anchor on top of the previous one
        // (browsers allow a few px of slop between the two clicks); drop it so we don't
        // emit a near-zero-length final segment. The threshold is a few device pixels in
        // world units so it tracks the current zoom.
        if self.pen.anchors.len() >= 2 {
            let n = self.pen.anchors.len();
            let merge_dist = 4.0 / self.camera.zoom;
            if (self.pen.anchors[n - 1].position - self.pen.anchors[n - 2].position).length()
                < merge_dist
            {
                self.pen.anchors.pop();
            }
        }

        if self.pen.anchors.len() >= 2 {
            self.pen.closed = close;
            self.pen_rebuild_preview();
            // Keep the committed stroke: drop only our handle to it so a new path starts
            // clean (PenState::default leaves `stroke: None`, which does not remove it).
        } else {
            self.pen_discard_preview();
        }
        self.pen = PenState::default();
    }

    /// Nearest control point (anchor or tangent handle) of `stroke` within a screen-space
    /// pick radius of world point `world`, or `None` if the cursor is over none. The
    /// radius is a fixed number of device pixels converted to world units via the current
    /// zoom, so the grab area stays constant on screen at any zoom level. `control_points`
    /// lists handles before anchors, so an overlapping handle wins the tie over the anchor
    /// beneath it.
    fn pick_control_point(
        &self,
        stroke: &GaussianBezierStroke,
        world: Vec2,
    ) -> Option<ControlPointRef> {
        const PICK_PX: f32 = 12.0;
        let pick = (PICK_PX / self.camera.zoom).max(0.25);
        let mut best: Option<(f32, ControlPointRef)> = None;
        for (r, p) in stroke.skeleton.control_points() {
            let d = (p - world).length();
            if d <= pick && best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, r));
            }
        }
        best.map(|(_, r)| r)
    }

    /// Dense per-document index of a stroke (its position in `strokes` iteration order),
    /// matching the `stroke_index` used by `renderer::collect_gpu_splats`.
    fn stroke_index(&self, sid: StrokeId) -> Option<u32> {
        self.doc
            .strokes
            .keys()
            .position(|k| k == sid)
            .map(|i| i as u32)
    }
}

/// Map the UI corner-smoothness slider `[0,1]` to the 3DCS smoothness δ: 0 → sharp polygon
/// corners (high δ, φ ≈ the exact hull signed distance), 1 → softly rounded vertices (low δ).
fn convex_delta(smoothness: f32) -> f32 {
    28.0 - 25.0 * smoothness.clamp(0.0, 1.0)
}

/// Map the UI edge-sharpness slider `[0,1]` to the 3DCS sharpness σ: 0 → a diffuse boundary
/// (low σ), 1 → a dense, crisp edge (high σ; the shader caps the on-screen transition to ~1px
/// so it antialiases rather than aliasing). [`renderer::ConvexUniform`] clamps σ to a safe range.
fn convex_sigma(sharpness: f32) -> f32 {
    3.0 + 57.0 * sharpness.clamp(0.0, 1.0)
}

/// Map the UI rotation slider `[0,1]` to radians `[0, 2π)`.
fn triangle_rotation_radians(rotation: f32) -> f32 {
    rotation.clamp(0.0, 1.0) * std::f32::consts::TAU
}

/// Map the UI softness slider `[0,1]` to the Triangle Splatting smoothness σ: 0 → a near-solid
/// triangle (small σ, hard top-hat window), 1 → a soft falloff peaking at the incenter (large σ).
fn triangle_sigma(softness: f32) -> f32 {
    0.06 + 2.4 * softness.clamp(0.0, 1.0).powf(1.4)
}

/// Apply a [`PreviewKind`] to `stroke`, setting the render flags the renderer consumes. A splat
/// stroke regenerates its cloud; vector and blend strokes drop their splats (they render as
/// tessellated paths) and a blend stroke additionally records its smear `strength`.
fn apply_preview_kind(stroke: &mut GaussianBezierStroke, kind: PreviewKind, strength: f32) {
    match kind {
        PreviewKind::Splat => {
            stroke.render_as_vector = false;
            stroke.vector_blend = false;
            stroke.regenerate_splats();
        }
        PreviewKind::Vector => {
            stroke.render_as_vector = true;
            stroke.vector_blend = false;
            stroke.splats.clear();
        }
        PreviewKind::Blend => {
            stroke.render_as_vector = true;
            stroke.vector_blend = true;
            stroke.blend_strength = strength;
            stroke.splats.clear();
        }
    }
}

/// Build a Bezier skeleton from pen anchors. Each consecutive pair becomes one cubic:
/// `p0 = a`, `p1 = a + a.out`, `p2 = b + b.in`, `p3 = b`. An anchor with no handle on a
/// given side falls back to a handle one-third along the chord, so a handle-less pair is
/// a straight line and a handle-less chain is a polyline. Returns `None` for fewer than
/// two anchors (no segment to draw). `closed` adds the wrap-around segment.
fn skeleton_from_pen(anchors: &[PenAnchor], closed: bool) -> Option<BezierSkeleton> {
    let n = anchors.len();
    if n == 0 {
        return None;
    }
    if n == 1 {
        // A single placed anchor draws as a tiny dab so the first click gives immediate
        // feedback (there is no segment yet). This preview is discarded on finish/cancel
        // if no second anchor is added (see `finalize_pen`).
        let p = anchors[0].position;
        let q = p + Vec2::new(1.0, 0.0);
        return Some(BezierSkeleton::single(CubicBezier::new(
            p,
            p + (q - p) / 3.0,
            p + (q - p) * (2.0 / 3.0),
            q,
        )));
    }
    let seg_count = if closed { n } else { n - 1 };
    let mut segments = Vec::with_capacity(seg_count);
    for i in 0..seg_count {
        let a = anchors[i];
        let b = anchors[(i + 1) % n];
        let chord = b.position - a.position;
        let out = if a.out_handle.length_squared() > 1e-12 {
            a.out_handle
        } else {
            chord / 3.0
        };
        let inc = if b.in_handle.length_squared() > 1e-12 {
            b.in_handle
        } else {
            -chord / 3.0
        };
        segments.push(CubicBezier::new(
            a.position,
            a.position + out,
            b.position + inc,
            b.position,
        ));
    }
    Some(BezierSkeleton::from_segments(segments, closed))
}
