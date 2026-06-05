//! Headless GPU tests for the triangle-splat (2D Triangle Splatting) render paths.
//!
//! These run on the native GPU (Metal/Vulkan/DX) and exercise the *same* triangle shaders and
//! pipelines the browser uses, so passing here is strong evidence the triangle WASM path is
//! correct too. They also validate that the triangle shaders compile (naga validates the WGSL).
//!
//! 2D Triangle Splatting (Held, Vandeghen et al., 2025) renders a triangle via the window
//! function `I(p) = ReLU(φ/φ(s))^σ` over the TRUE-max edge SDF φ — peaking at the incenter and
//! vanishing at the boundary. The two properties these tests assert are: (1) the window **peaks
//! at the incenter and is bounded** (opaque center, transparent outside), and (2) a **triangular
//! (directional) silhouette** that reaches further toward a vertex than the opposite edge.

use app_core::math::covariance_from_sigmas;
use app_core::splat::{pack_rgba8, GpuSplat};
use glam::Vec2;
use renderer::{camera::Camera2D, gpu, GpuContext, SplatRenderer};

const W: u32 = 256;
const H: u32 = 256;
const CX: f32 = 128.0;
const CY: f32 = 128.0;
const SIGMA: f32 = 18.0;

fn red_center_splat() -> GpuSplat {
    let cov = covariance_from_sigmas(0.0, SIGMA, SIGMA);
    GpuSplat {
        center: [CX, CY],
        cov_a: cov.x_axis.x,
        cov_b: cov.x_axis.y,
        cov_c: cov.y_axis.y,
        color: pack_rgba8([1.0, 0.0, 0.0, 1.0]),
        alpha: 1.0,
        radius: 3.0 * SIGMA,
        stroke_id: 0,
        flags: 0,
        hardness: 0.0,
    }
}

fn centered_camera() -> Camera2D {
    let mut cam = Camera2D::new(Vec2::new(W as f32, H as f32));
    cam.center = Vec2::new(CX, CY);
    cam
}

fn pixel(buf: &[u8], x: u32, y: u32) -> &[u8] {
    let i = ((y * W + x) * 4) as usize;
    &buf[i..i + 4]
}

#[allow(dead_code)]
fn render_gaussian(ctx: &GpuContext, splats: &[GpuSplat]) -> Vec<u8> {
    let mut r = SplatRenderer::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm);
    let target = gpu::create_offscreen_target(ctx, W, H);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    r.render(&ctx.device, &ctx.queue, &view, splats, &centered_camera().uniform(), wgpu::Color::TRANSPARENT);
    gpu::read_texture_rgba8(ctx, &target, W, H)
}

/// Render through the triangle direct path: an equilateral triangle rotated by `rotation` with
/// window smoothness `sigma`.
fn render_triangle(ctx: &GpuContext, splats: &[GpuSplat], rotation: f32, sigma: f32) -> Vec<u8> {
    let mut r = SplatRenderer::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm);
    r.set_triangle_params(&ctx.queue, rotation, sigma);
    let target = gpu::create_offscreen_target(ctx, W, H);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    r.render_triangle(&ctx.device, &ctx.queue, &view, splats, &centered_camera().uniform(), wgpu::Color::TRANSPARENT);
    gpu::read_texture_rgba8(ctx, &target, W, H)
}

/// The triangle window `I = ReLU(φ/φ(s))^σ` peaks at the incenter (opaque) and vanishes at and
/// outside the boundary — it is bounded, not a radial Gaussian that decays everywhere. A
/// fairly-solid triangle (small σ) is checked: opaque at the incenter (image center), and
/// transparent well past the template circumradius (2.4σ) along +x.
#[test]
fn triangle_window_peaks_at_incenter_and_is_bounded() {
    let ctx = GpuContext::new_headless_blocking();
    let splat = [red_center_splat()];
    let tri = render_triangle(&ctx, &splat, 0.0, 0.3);

    // Incenter (image center) is opaque red.
    assert!(pixel(&tri, CX as u32, CY as u32)[0] > 200, "triangle incenter should be opaque red");

    // Well past the circumradius (2.4σ) along +x — outside the triangle in that direction.
    let off = (2.6 * SIGMA) as u32;
    let a = pixel(&tri, CX as u32 + off, CY as u32)[3];
    assert!(a < 40, "point beyond the template should be transparent, got alpha {a}");
}

/// A triangle is *directional*: along the direction of a vertex it reaches ~circumradius (2.4σ),
/// but in the diametrically opposite direction an edge faces outward so it only reaches the
/// inradius (1.2σ). This asymmetry between the +y and −y opaque reaches proves the silhouette is
/// a triangle (directional), not a circle/Gaussian (which would be symmetric).
#[test]
fn triangle_reaches_a_vertex_farther_than_the_opposite_edge() {
    let ctx = GpuContext::new_headless_blocking();
    let splat = [red_center_splat()];
    // Apex up (rotation 0); a near-solid (top-hat) triangle — small σ keeps the window ≈ 1
    // across the interior, so the opaque silhouette tracks the actual triangle geometry.
    let tri = render_triangle(&ctx, &splat, 0.0, 0.05);

    let limit = 3 * SIGMA as u32;
    // Opaque (alpha > 200) reach upward (toward −y in image space) and downward (+y), in px.
    let mut reach_up = 0u32;
    for d in 1..limit {
        if pixel(&tri, CX as u32, CY as u32 - d)[3] > 200 {
            reach_up = d;
        }
    }
    let mut reach_down = 0u32;
    for d in 1..limit {
        if pixel(&tri, CX as u32, CY as u32 + d)[3] > 200 {
            reach_down = d;
        }
    }

    let far = reach_up.max(reach_down);
    let near = reach_up.min(reach_down);
    assert!(near > 0, "triangle should be opaque in both ±y directions near the center");
    // Camera may flip Y, so assert the asymmetry abstractly rather than hardcoding apex side.
    assert!(
        far as f32 > 1.3 * near as f32,
        "vertex reach must clearly exceed the opposite-edge reach (triangular, not circular): far={far} near={near}"
    );
    // The far (vertex) direction should reach near the circumradius (~2.4σ ≈ 43px).
    assert!(
        far as f32 >= 2.0 * SIGMA,
        "vertex direction should reach ≥ ~2σ in px, got {far}"
    );
}

// ---- Triangle crisp-perimeter path -----------------------------------------------------

/// A horizontal bar of overlapping, isotropic splats (mirrors crisp.rs / convex.rs).
fn line_splats() -> Vec<GpuSplat> {
    let s = 12.0_f32;
    let cov = covariance_from_sigmas(0.0, s, s);
    let mut v = Vec::new();
    let mut x = 60.0_f32;
    while x <= 196.0 {
        v.push(GpuSplat {
            center: [x, CY],
            cov_a: cov.x_axis.x,
            cov_b: cov.x_axis.y,
            cov_c: cov.y_axis.y,
            color: pack_rgba8([1.0, 0.0, 0.0, 1.0]),
            alpha: 1.0,
            radius: 3.0 * s,
            stroke_id: 0,
            flags: 0,
            hardness: 0.0,
        });
        x += 8.0;
    }
    v
}

fn triangle_crisp(ctx: &GpuContext, splats: &[GpuSplat], rotation: f32, sigma: f32) -> Vec<u8> {
    let mut r = SplatRenderer::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm);
    r.set_triangle_params(&ctx.queue, rotation, sigma);
    let target = gpu::create_offscreen_target(ctx, W, H);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    r.render_triangle_crisp(
        &ctx.device, &ctx.queue, &view, splats, &centered_camera().uniform(), W, H, wgpu::Color::TRANSPARENT,
    );
    gpu::read_texture_rgba8(ctx, &target, W, H)
}

fn top_edge_band(img: &[u8]) -> usize {
    let x = W / 2;
    (0..H / 2).filter(|&y| { let a = pixel(img, x, y)[3]; a > 25 && a < 230 }).count()
}

/// The triangle crisp path (triangle accumulate + the shared resolve) collapses a *diffuse*
/// triangle union (large σ → a soft window falloff) into one thin, crisp antialiased perimeter —
/// exactly as the Gaussian/convex crisp path does — proving triangle splats inherit the
/// crisp-edge / blending toolset.
#[test]
fn triangle_crisp_path_sharpens_the_perimeter() {
    let ctx = GpuContext::new_headless_blocking();
    let splats = line_splats();

    // Larger σ → a deliberately diffuse window falloff, so the direct path renders a wide soft
    // edge and any narrowing the crisp path achieves is attributable to the union-coverage
    // threshold. (The window still keeps the bar interior opaque at this σ.)
    let soft = render_triangle(&ctx, &splats, 0.0, 1.8);
    let crisp = triangle_crisp(&ctx, &splats, 0.0, 1.8);

    assert!(pixel(&soft, W / 2, H / 2)[0] > 170, "triangle soft interior should be filled red");
    assert!(pixel(&crisp, W / 2, H / 2)[0] > 170, "triangle crisp interior should be filled red");
    assert!(pixel(&crisp, W / 2, 4)[3] < 20, "crisp top should be clear");

    let soft_band = top_edge_band(&soft);
    let crisp_band = top_edge_band(&crisp);
    assert!(soft_band >= 8, "diffuse triangle edge should be wide, got {soft_band}px");
    assert!(crisp_band <= 6, "triangle crisp perimeter should be a thin rim, got {crisp_band}px");
    // The crisp coverage path must clearly sharpen the edge vs the diffuse direct path.
    assert!(
        crisp_band * 2 <= soft_band && crisp_band < soft_band,
        "triangle coverage path must sharpen the edge: crisp={crisp_band}px vs soft={soft_band}px"
    );
}
