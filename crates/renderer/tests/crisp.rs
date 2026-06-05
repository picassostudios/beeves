//! Headless test for the two-pass crisp-perimeter path (`render_crisp`).
//!
//! It renders the *same* set of deliberately-soft splats two ways — the direct per-splat
//! `over` path (`render`) and the accumulate→resolve coverage path (`render_crisp`) — and
//! checks that the coverage path turns the line's soft Gaussian edge into a narrow, crisp
//! antialiased perimeter, while keeping the interior filled.

use app_core::math::covariance_from_sigmas;
use app_core::splat::{pack_rgba8, GpuSplat};
use glam::Vec2;
use renderer::{camera::Camera2D, gpu, GpuContext, SplatRenderer};

const W: u32 = 256;
const H: u32 = 256;

/// A horizontal bar of overlapping, isotropic, fully-soft (`hardness = 0`) splats. Soft on
/// purpose: the direct path then renders a wide gradual edge, so any narrowing the crisp
/// path achieves is attributable to the union-coverage threshold, not to per-splat hardness.
fn line_splats() -> Vec<GpuSplat> {
    let sigma = 12.0_f32;
    let cov = covariance_from_sigmas(0.0, sigma, sigma);
    let mut v = Vec::new();
    let mut x = 60.0_f32;
    while x <= 196.0 {
        v.push(GpuSplat {
            center: [x, (H / 2) as f32],
            cov_a: cov.x_axis.x,
            cov_b: cov.x_axis.y,
            cov_c: cov.y_axis.y,
            color: pack_rgba8([1.0, 0.0, 0.0, 1.0]),
            alpha: 1.0,
            radius: 3.0 * sigma,
            stroke_id: 0,
            flags: 0,
            hardness: 0.0,
        });
        x += 8.0;
    }
    v
}

fn centered_camera() -> Camera2D {
    let mut cam = Camera2D::new(Vec2::new(W as f32, H as f32));
    cam.center = Vec2::new((W / 2) as f32, (H / 2) as f32);
    cam
}

fn pixel(buf: &[u8], x: u32, y: u32) -> &[u8] {
    let i = ((y * W + x) * 4) as usize;
    &buf[i..i + 4]
}

fn render_direct(ctx: &GpuContext, splats: &[GpuSplat]) -> Vec<u8> {
    let mut r = SplatRenderer::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm);
    let target = gpu::create_offscreen_target(ctx, W, H);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    r.render(
        &ctx.device,
        &ctx.queue,
        &view,
        splats,
        &centered_camera().uniform(),
        wgpu::Color::TRANSPARENT,
    );
    gpu::read_texture_rgba8(ctx, &target, W, H)
}

fn render_crisp(ctx: &GpuContext, splats: &[GpuSplat]) -> Vec<u8> {
    let mut r = SplatRenderer::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm);
    let target = gpu::create_offscreen_target(ctx, W, H);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    r.render_crisp(
        &ctx.device,
        &ctx.queue,
        &view,
        splats,
        &centered_camera().uniform(),
        W,
        H,
        wgpu::Color::TRANSPARENT,
    );
    gpu::read_texture_rgba8(ctx, &target, W, H)
}

/// Number of partial-alpha pixels in the top half of the bar's center column — i.e. the
/// width (in px) of the top-edge antialiasing band. A crisp perimeter is a thin band.
fn top_edge_band(img: &[u8]) -> usize {
    let x = W / 2;
    (0..H / 2)
        .filter(|&y| {
            let a = pixel(img, x, y)[3];
            a > 25 && a < 230
        })
        .count()
}

#[test]
fn coverage_path_sharpens_the_perimeter() {
    let ctx = GpuContext::new_headless_blocking();
    let splats = line_splats();

    let direct = render_direct(&ctx, &splats);
    let crisp = render_crisp(&ctx, &splats);

    // Interior stays filled in both paths (center of the bar is opaque red).
    let dc = pixel(&direct, W / 2, H / 2);
    let cc = pixel(&crisp, W / 2, H / 2);
    assert!(dc[0] > 200 && dc[3] > 200, "direct interior should be opaque red, got {dc:?}");
    assert!(cc[0] > 200 && cc[3] > 200, "crisp interior should be opaque red, got {cc:?}");

    // Far outside the bar is clear in both.
    assert!(pixel(&direct, W / 2, 4)[3] < 20, "direct top should be clear");
    assert!(pixel(&crisp, W / 2, 4)[3] < 20, "crisp top should be clear");

    let soft_band = top_edge_band(&direct);
    let crisp_band = top_edge_band(&crisp);

    // The whole point: the soft Gaussian edge spans many px; the coverage threshold collapses
    // it to a thin antialiased rim.
    assert!(soft_band >= 8, "soft direct edge should be wide, got {soft_band}px");
    assert!(
        crisp_band <= 4,
        "crisp perimeter should be a thin rim, got {crisp_band}px (soft was {soft_band}px)"
    );
    assert!(
        crisp_band * 2 < soft_band,
        "coverage path must sharpen the edge: crisp={crisp_band}px vs soft={soft_band}px"
    );
}
