//! Throwaway: render strokes at high zoom to inspect edge quality. Remove after.
use app_core::brush::BrushModel;
use app_core::document::Document;
use app_core::fitting::{fit_polyline_adaptive, AdaptiveFitParams};
use app_core::{BezierSkeleton, CubicBezier};
use glam::Vec2;
use renderer::{camera::Camera2D, gpu, GpuContext, SplatRenderer};
use std::io::Write;

const W: u32 = 512;
const H: u32 = 512;

fn dump(name: &str, img: &[u8]) {
    let mut f = std::fs::File::create(format!("/tmp/{name}.ppm")).unwrap();
    write!(f, "P6\n{W} {H}\n255\n").unwrap();
    for px in img.chunks(4) {
        f.write_all(&px[..3]).unwrap();
    }
}

fn render(doc: &mut Document, zoom: f32, center: Vec2, name: &str) {
    let ctx = GpuContext::new_headless_blocking();
    let mut r = SplatRenderer::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm);
    let target = gpu::create_offscreen_target(&ctx, W, H);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let mut cam = Camera2D::new(Vec2::new(W as f32, H as f32));
    cam.center = center;
    cam.zoom = zoom;
    let white = wgpu::Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
    let huge = 1.0e5;
    r.render_doc(&ctx.device, &ctx.queue, &view, doc, &cam.uniform(), Vec2::new(-huge, -huge), Vec2::new(huge, huge), white);
    r.render_vector_paths(&ctx.device, &ctx.queue, &view, doc, &cam.uniform());
    let img = gpu::read_texture_rgba8(&ctx, &target, W, H);
    dump(name, &img);
}

#[test]
fn dump_smooth() {
    // A single smooth Bezier, rendered thick and zoomed in 4x.
    let mut doc = Document::new();
    let l = doc.add_layer("L");
    let c = CubicBezier::new(Vec2::new(0.0, 0.0), Vec2::new(80.0, -120.0), Vec2::new(160.0, 120.0), Vec2::new(240.0, 0.0));
    let brush = BrushModel { base_color: [0.05, 0.05, 0.05, 1.0], radius: 12.0, opacity: 1.0, ..BrushModel::default() };
    let sid = doc.add_stroke(l, BezierSkeleton::single(c), brush);
    doc.stroke_mut(sid).unwrap().render_as_vector = true;
    render(&mut doc, 4.0, Vec2::new(120.0, 0.0), "dbg_smooth");
}

#[test]
fn dump_fitted() {
    // A hand-drawn-like wavy polyline, fit by the adaptive fitter (many segments), zoomed in.
    let mut pts = Vec::new();
    let n = 220;
    for i in 0..n {
        let x = i as f32 * 1.2;
        // gentle large arc plus tiny hand tremor
        let y = 40.0 * (x * 0.012).sin() + 1.1 * ((x * 0.7).sin() + (x * 1.31).cos());
        pts.push(Vec2::new(x, y));
    }
    let sk = fit_polyline_adaptive(&pts, &AdaptiveFitParams::default());
    let mut doc = Document::new();
    let l = doc.add_layer("L");
    let brush = BrushModel { base_color: [0.05, 0.05, 0.05, 1.0], radius: 12.0, opacity: 1.0, ..BrushModel::default() };
    let sid = doc.add_stroke(l, sk, brush);
    doc.stroke_mut(sid).unwrap().render_as_vector = true;
    render(&mut doc, 3.0, Vec2::new(130.0, 0.0), "dbg_fitted");
}
