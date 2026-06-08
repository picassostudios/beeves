//! Headless tests for the vector-blend (directional smear) pass. A `vector_blend` stroke must
//! (a) be skipped by the splat passes and the plain vector pass, and (b) draw a directional
//! smear of the *plain vector layer* beneath it — dragging colour along the path tangent past a
//! colour edge while leaving everything outside its ribbon untouched.

use app_core::brush::BrushModel;
use app_core::document::Document;
use app_core::{BezierSkeleton, CubicBezier, StrokeId};
use glam::Vec2;
use renderer::{camera::Camera2D, gpu, GpuContext, SplatRenderer};

const W: u32 = 256;
const H: u32 = 256;
const RADIUS: f32 = 8.0;
const BLUE: wgpu::Color = wgpu::Color { r: 0.0, g: 0.0, b: 1.0, a: 1.0 };
const HUGE: f32 = 1.0e4;

fn camera() -> Camera2D {
    let mut cam = Camera2D::new(Vec2::new(W as f32, H as f32));
    cam.center = Vec2::new(W as f32 / 2.0, H as f32 / 2.0);
    cam
}

fn pixel(buf: &[u8], x: u32, y: u32) -> &[u8] {
    let i = ((y * W + x) * 4) as usize;
    &buf[i..i + 4]
}

/// A straight cubic from `a` to `b` (handles a third of the way along the chord).
fn straight(a: Vec2, b: Vec2) -> CubicBezier {
    CubicBezier::new(a, a + (b - a) / 3.0, a + (b - a) * (2.0 / 3.0), b)
}

/// Add a horizontal stroke at y=center from `x0` to `x1`. `blend` flags it as a vector-blend
/// (smear) path; otherwise it is a plain `render_as_vector` path with `color`.
fn add_h_stroke(
    doc: &mut Document,
    layer: app_core::LayerId,
    x0: f32,
    x1: f32,
    color: [f32; 4],
    blend: bool,
) -> StrokeId {
    let y = H as f32 / 2.0;
    let brush = BrushModel {
        base_color: color,
        radius: RADIUS,
        opacity: 1.0,
        ..BrushModel::default()
    };
    let sid = doc.add_stroke(
        layer,
        BezierSkeleton::single(straight(Vec2::new(x0, y), Vec2::new(x1, y))),
        brush,
    );
    let stroke = doc.stroke_mut(sid).unwrap();
    stroke.render_as_vector = true;
    stroke.vector_blend = blend;
    sid
}

/// Render `doc` to an offscreen RGBA8 buffer: splat pass (clears blue, draws no vector strokes)
/// then the vector + blend passes.
fn render(doc: &mut Document) -> Vec<u8> {
    let ctx = GpuContext::new_headless_blocking();
    let mut r = SplatRenderer::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm);
    let target = gpu::create_offscreen_target(&ctx, W, H);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let cam = camera();
    r.render_doc(
        &ctx.device, &ctx.queue, &view, doc, &cam.uniform(),
        Vec2::new(-HUGE, -HUGE), Vec2::new(HUGE, HUGE), BLUE,
    );
    r.render_vector_paths(&ctx.device, &ctx.queue, &view, doc, &cam.uniform());
    gpu::read_texture_rgba8(&ctx, &target, W, H)
}

/// A red vector line ending at the canvas center, with a blend stroke that overlaps the end and
/// extends past it, oriented along the line. The smear must drag red *past* the line's end (where
/// the bare line would leave only background), proving it samples the layer along the tangent.
#[test]
fn blend_smears_color_past_a_color_edge() {
    let red = [1.0, 0.0, 0.0, 1.0];

    // The red line alone: a pixel past its end (and past the round cap) is plain background.
    let mut bare = Document::new();
    let layer = bare.add_layer("L");
    add_h_stroke(&mut bare, layer, W as f32 * 0.2, W as f32 * 0.5, red, false);
    let bare_img = render(&mut bare);
    // x=140 is ~4px past the cap reach (end 128 + radius 8 = 136), within the 16px smear reach;
    // without a smear it is plain background.
    let probe = pixel(&bare_img, 140, H / 2);
    assert!(
        probe[2] > 200 && probe[0] < 40,
        "without a blend stroke, past the line end should be background blue, got {probe:?}"
    );

    // Now add a blend stroke over the end, extending past it along the same direction.
    let mut doc = Document::new();
    let layer = doc.add_layer("L");
    add_h_stroke(&mut doc, layer, W as f32 * 0.2, W as f32 * 0.5, red, false);
    add_h_stroke(&mut doc, layer, W as f32 * 0.4, W as f32 * 0.75, [0.0, 0.0, 0.0, 1.0], true);
    let img = render(&mut doc);

    // Same probe pixel: the smear samples backward along the tangent into the red line, so red is
    // dragged forward past the end — the pixel now carries red it did not before.
    let smeared = pixel(&img, 140, H / 2);
    assert!(
        smeared[0] > 50,
        "the blend smear should drag red past the line's end, got {smeared:?}"
    );
    assert!(
        smeared[0] > probe[0] + 30,
        "smeared red ({}) should exceed the bare background red ({})",
        smeared[0],
        probe[0],
    );

    // Inside the overlap the line is solid red on both sides of the taps, so the smear of red is
    // still red (the effect does not wash a uniform region out).
    let inside = pixel(&img, 120, H / 2);
    assert!(
        inside[0] > 150 && inside[2] < 100,
        "over a solid-red region the smear stays red, got {inside:?}"
    );
}

/// The smear is confined to the blend ribbon: a pixel well outside the half-width is untouched
/// background, even directly above the smeared region.
#[test]
fn blend_leaves_background_outside_its_ribbon() {
    let red = [1.0, 0.0, 0.0, 1.0];
    let mut doc = Document::new();
    let layer = doc.add_layer("L");
    add_h_stroke(&mut doc, layer, W as f32 * 0.2, W as f32 * 0.5, red, false);
    add_h_stroke(&mut doc, layer, W as f32 * 0.4, W as f32 * 0.75, [0.0, 0.0, 0.0, 1.0], true);
    let img = render(&mut doc);

    // 40px above the centerline (far outside the 8px half-width of both strokes): pure blue.
    let off = pixel(&img, 148, H / 2 - 40);
    assert!(
        off[2] > 200 && off[0] < 40,
        "outside the blend ribbon should stay background blue, got {off:?}"
    );
}

/// A blend stroke contributes no splats to the splat pass (it sets `render_as_vector`), so the
/// splat field alone leaves the canvas at the background colour where the stroke lies.
#[test]
fn blend_stroke_is_skipped_by_the_splat_pass() {
    let ctx = GpuContext::new_headless_blocking();
    let mut doc = Document::new();
    let layer = doc.add_layer("L");
    add_h_stroke(&mut doc, layer, W as f32 * 0.2, W as f32 * 0.8, [0.0, 0.0, 0.0, 1.0], true);

    let mut r = SplatRenderer::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm);
    let target = gpu::create_offscreen_target(&ctx, W, H);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let cam = camera();
    // Splat pass only (no vector/blend pass): a blend stroke must add nothing here.
    r.render_doc(
        &ctx.device, &ctx.queue, &view, &mut doc, &cam.uniform(),
        Vec2::new(-HUGE, -HUGE), Vec2::new(HUGE, HUGE), BLUE,
    );
    let img = gpu::read_texture_rgba8(&ctx, &target, W, H);
    let c = pixel(&img, W / 2, H / 2);
    assert!(
        c[2] > 200 && c[0] < 40,
        "a blend stroke must contribute no splats; center should stay blue, got {c:?}"
    );
}
