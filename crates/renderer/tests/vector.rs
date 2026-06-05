//! Headless tests for the conventional vector-path render pass. A stroke flagged
//! `render_as_vector` must (a) be skipped by the splat passes and (b) draw as a solid,
//! width-correct, antialiased stroked outline via `render_vector_paths`.

use app_core::brush::BrushModel;
use app_core::document::Document;
use app_core::{BezierSkeleton, CubicBezier};
use glam::Vec2;
use renderer::{camera::Camera2D, gpu, GpuContext, SplatRenderer};

const W: u32 = 256;
const H: u32 = 256;
const RADIUS: f32 = 8.0; // stroke half-width in world px (== device px at zoom 1.0)

/// A document with one horizontal, red, vector-rendered stroke through the viewport center.
fn one_vector_stroke_doc() -> (Document, app_core::StrokeId) {
    let mut doc = Document::new();
    let layer = doc.add_layer("L");
    let y = H as f32 / 2.0;
    let curve = CubicBezier::new(
        Vec2::new(W as f32 * 0.2, y),
        Vec2::new(W as f32 * 0.4, y),
        Vec2::new(W as f32 * 0.6, y),
        Vec2::new(W as f32 * 0.8, y),
    );
    let brush = BrushModel {
        base_color: [1.0, 0.0, 0.0, 1.0],
        radius: RADIUS,
        opacity: 1.0,
        ..BrushModel::default()
    };
    let sid = doc.add_stroke(layer, BezierSkeleton::single(curve), brush);
    doc.stroke_mut(sid).unwrap().render_as_vector = true;
    (doc, sid)
}

fn camera() -> Camera2D {
    let mut cam = Camera2D::new(Vec2::new(W as f32, H as f32));
    cam.center = Vec2::new(W as f32 / 2.0, H as f32 / 2.0);
    cam
}

fn pixel(buf: &[u8], x: u32, y: u32) -> &[u8] {
    let i = ((y * W + x) * 4) as usize;
    &buf[i..i + 4]
}

const BLUE: wgpu::Color = wgpu::Color { r: 0.0, g: 0.0, b: 1.0, a: 1.0 };
const HUGE: f32 = 1.0e4;

/// The splat passes must NOT draw a vector stroke: rendering only `render_doc` over a blue
/// background must leave the line's pixels blue (the stroke contributes no splats to draw).
#[test]
fn vector_stroke_is_skipped_by_the_splat_pass() {
    let ctx = GpuContext::new_headless_blocking();
    let (mut doc, _sid) = one_vector_stroke_doc();
    let mut r = SplatRenderer::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm);
    let target = gpu::create_offscreen_target(&ctx, W, H);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let cam = camera();

    r.render_doc(
        &ctx.device, &ctx.queue, &view, &mut doc, &cam.uniform(),
        Vec2::new(-HUGE, -HUGE), Vec2::new(HUGE, HUGE), BLUE,
    );
    let img = gpu::read_texture_rgba8(&ctx, &target, W, H);

    // On the line's centerline: still pure background (no splats were drawn for it).
    let c = pixel(&img, W / 2, H / 2);
    assert!(
        c[2] > 200 && c[0] < 40,
        "a vector stroke must contribute no splats; center should stay blue, got {c:?}"
    );
}

/// `render_vector_paths` draws the stroke as a solid red band of the right width on top of the
/// (vector-skipped) splat field.
#[test]
fn vector_pass_draws_a_solid_width_correct_line() {
    let ctx = GpuContext::new_headless_blocking();
    let (mut doc, _sid) = one_vector_stroke_doc();
    let mut r = SplatRenderer::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm);
    let target = gpu::create_offscreen_target(&ctx, W, H);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let cam = camera();

    // Splat pass (clears to blue, draws nothing for the vector stroke), then the vector pass.
    r.render_doc(
        &ctx.device, &ctx.queue, &view, &mut doc, &cam.uniform(),
        Vec2::new(-HUGE, -HUGE), Vec2::new(HUGE, HUGE), BLUE,
    );
    r.render_vector_paths(&ctx.device, &ctx.queue, &view, &doc, &cam.uniform());
    let img = gpu::read_texture_rgba8(&ctx, &target, W, H);

    // Centerline pixel: solid red.
    let c = pixel(&img, W / 2, H / 2);
    assert!(c[0] > 200 && c[1] < 40 && c[2] < 40, "line center should be red, got {c:?}");

    // Far above the line (well outside the half-width): untouched blue background.
    let off = pixel(&img, W / 2, H / 2 - 40);
    assert!(off[2] > 200 && off[0] < 40, "off-line pixel should stay blue, got {off:?}");

    // Width: count the red run in the column through the center. It should be ~2*RADIUS tall
    // (allowing a couple of px of antialiased ramp on each rim).
    let red_rows = (0..H)
        .filter(|&yy| {
            let p = pixel(&img, W / 2, yy);
            p[0] > 150 && p[2] < 80
        })
        .count();
    let expected = (2.0 * RADIUS) as usize;
    assert!(
        red_rows >= expected - 3 && red_rows <= expected + 4,
        "line width should be ~{expected}px, measured {red_rows}px"
    );

    // Endcaps: before the round cap's reach (x < 0.2*W - radius) the column is background.
    let before = pixel(&img, (W as f32 * 0.1) as u32, H / 2);
    assert!(before[2] > 200 && before[0] < 40, "before the stroke start should be blue, got {before:?}");
}

/// Round caps bulge a half-width past the geometric endpoint: a pixel a few px beyond the
/// line's end on the centerline is painted (it would be background with butt caps).
#[test]
fn round_caps_extend_past_the_endpoints() {
    let ctx = GpuContext::new_headless_blocking();
    let (mut doc, _sid) = one_vector_stroke_doc();
    let mut r = SplatRenderer::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm);
    let target = gpu::create_offscreen_target(&ctx, W, H);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let cam = camera();

    r.render_doc(
        &ctx.device, &ctx.queue, &view, &mut doc, &cam.uniform(),
        Vec2::new(-HUGE, -HUGE), Vec2::new(HUGE, HUGE), BLUE,
    );
    r.render_vector_paths(&ctx.device, &ctx.queue, &view, &doc, &cam.uniform());
    let img = gpu::read_texture_rgba8(&ctx, &target, W, H);

    // Geometric end is x = 0.8*W = 204.8; with RADIUS=8 the round cap reaches ~212.8.
    let end_x = (W as f32 * 0.8) as u32; // 204
    let in_cap = pixel(&img, end_x + 4, H / 2); // ~3px past the end, inside the cap
    assert!(in_cap[0] > 150 && in_cap[2] < 80, "round cap should paint past the end, got {in_cap:?}");
    // Well beyond the cap's reach it is background again.
    let past = pixel(&img, end_x + 12, H / 2);
    assert!(past[2] > 200 && past[0] < 40, "beyond the round cap should be blue, got {past:?}");
}

/// A round join fills the outer corner of a sharp bend: a pixel just outside both straight
/// limbs but within a half-width of the corner is painted (a butt join leaves it background).
#[test]
fn round_join_fills_a_sharp_corner() {
    fn straight(a: Vec2, b: Vec2) -> CubicBezier {
        CubicBezier::new(a, a + (b - a) / 3.0, a + (b - a) * (2.0 / 3.0), b)
    }
    let ctx = GpuContext::new_headless_blocking();
    let mut doc = Document::new();
    let layer = doc.add_layer("L");
    // An "L": right to the corner (128,128), then down. 90° bend.
    let corner = Vec2::new(128.0, 128.0);
    let s1 = straight(Vec2::new(40.0, 128.0), corner);
    let s2 = straight(corner, Vec2::new(128.0, 216.0));
    let brush = BrushModel {
        base_color: [1.0, 0.0, 0.0, 1.0],
        radius: RADIUS,
        opacity: 1.0,
        ..BrushModel::default()
    };
    let sid = doc.add_stroke(layer, BezierSkeleton::from_segments(vec![s1, s2], false), brush);
    doc.stroke_mut(sid).unwrap().render_as_vector = true;

    let mut r = SplatRenderer::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm);
    let target = gpu::create_offscreen_target(&ctx, W, H);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let cam = camera();
    r.render_doc(
        &ctx.device, &ctx.queue, &view, &mut doc, &cam.uniform(),
        Vec2::new(-HUGE, -HUGE), Vec2::new(HUGE, HUGE), BLUE,
    );
    r.render_vector_paths(&ctx.device, &ctx.queue, &view, &doc, &cam.uniform());
    let img = gpu::read_texture_rgba8(&ctx, &target, W, H);

    // (130,126): in the outer-corner quadrant (x>128 and y<128, so outside both straight
    // limbs) but only ~3px from the corner — covered solely by the round-join disc.
    let notch = pixel(&img, 130, 126);
    assert!(notch[0] > 150 && notch[2] < 80, "round join should fill the corner notch, got {notch:?}");
    // Further out along the diagonal is past the join radius → background.
    let outside = pixel(&img, 138, 118);
    assert!(outside[2] > 200 && outside[0] < 40, "beyond the join radius should be blue, got {outside:?}");
}
