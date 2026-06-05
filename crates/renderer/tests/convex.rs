//! Headless GPU tests for the convex-splat (3D Convex Splatting) render paths.
//!
//! These run on the native GPU (Metal/Vulkan/DX) and exercise the *same* convex shaders and
//! pipelines the browser uses, so passing here is strong evidence the convex WASM path is
//! correct too. They also validate that the convex shaders compile (naga validates the WGSL).
//!
//! Convex splatting (following 3DCS / CvxNet) renders the convex hull of a point set via a
//! sigmoid of its LogSumExp signed distance — `I = sigmoid(−σ·φ)`. The two properties that
//! distinguish it from a Gaussian, and that these tests assert, are: (1) a **flat-topped
//! interior with a crisp edge** (the sigmoid of a signed distance, not a radial falloff), and
//! (2) a **polygonal silhouette** that extends further toward its corners than its flat sides.

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

fn render_gaussian(ctx: &GpuContext, splats: &[GpuSplat]) -> Vec<u8> {
    let mut r = SplatRenderer::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm);
    let target = gpu::create_offscreen_target(ctx, W, H);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    r.render(&ctx.device, &ctx.queue, &view, splats, &centered_camera().uniform(), wgpu::Color::TRANSPARENT);
    gpu::read_texture_rgba8(ctx, &target, W, H)
}

/// Render through the convex direct path: a `sides`-gon hull with smoothness δ and sharpness σ.
fn render_convex(ctx: &GpuContext, splats: &[GpuSplat], sides: f32, delta: f32, sigma: f32) -> Vec<u8> {
    let mut r = SplatRenderer::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm);
    r.set_convex_params(&ctx.queue, sides, delta, sigma);
    let target = gpu::create_offscreen_target(ctx, W, H);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    r.render_convex(&ctx.device, &ctx.queue, &view, splats, &centered_camera().uniform(), wgpu::Color::TRANSPARENT);
    gpu::read_texture_rgba8(ctx, &target, W, H)
}

/// The convex indicator is `sigmoid(−σ·φ)` of a hull signed distance, so its interior is
/// **flat-topped** (opaque well past where a Gaussian has decayed) and its edge is **crisp**
/// (a thin antialiasing band, not a long radial falloff). Both are the visual signature of a
/// convex primitive vs a Gaussian, and both are checked here against a Gaussian of equal size.
#[test]
fn convex_is_flat_topped_with_a_crisp_edge() {
    let ctx = GpuContext::new_headless_blocking();
    let splat = [red_center_splat()];

    let gauss = render_gaussian(&ctx, &splat);
    // Hexagon with sharp-ish corners (high δ) and a dense, crisp edge (high σ).
    let convex = render_convex(&ctx, &splat, 6.0, 28.0, 60.0);

    // Both are opaque at the center.
    assert!(pixel(&gauss, 128, 128)[0] > 200, "gaussian center red");
    assert!(pixel(&convex, 128, 128)[0] > 200, "convex center red");

    // Flat top: ~1.3σ off-center the convex interior is still essentially opaque, while the
    // Gaussian has clearly decayed (a radial profile).
    let off = (1.3 * SIGMA) as u32; // ~23px
    let c_in = pixel(&convex, CX as u32 + off, CY as u32)[0];
    let g_in = pixel(&gauss, CX as u32 + off, CY as u32)[0];
    assert!(c_in > 225, "convex interior should stay flat/opaque at 1.3σ, got {c_in}");
    assert!(g_in < 190, "gaussian should have decayed by 1.3σ, got {g_in}");

    // Crisp edge: the partial-alpha band along the +x ray (the soft transition width) is a thin
    // rim for the convex splat and a long gradient for the soft Gaussian.
    let band = |img: &[u8]| -> usize {
        (CX as u32..W).filter(|&x| { let a = pixel(img, x, CY as u32)[3]; a > 25 && a < 230 }).count()
    };
    let c_band = band(&convex);
    let g_band = band(&gauss);
    assert!(c_band <= 4, "convex edge should be a thin rim, got {c_band}px");
    assert!(g_band >= 12, "gaussian edge should be a wide falloff, got {g_band}px");
    assert!(c_band * 3 < g_band, "convex edge must be far crisper: convex={c_band} gauss={g_band}");
}

/// A convex *polygon* reaches further toward its corners than across its flat sides — a circle
/// (the Gaussian) does not. With an axis-aligned square hull we measure the opaque extent along
/// the +x axis (≈ the apothem) and along the +x+y diagonal (≈ the circumradius) and assert the
/// corner reach clearly exceeds the edge reach (the square's ratio is √2 ≈ 1.41).
#[test]
fn convex_square_reaches_further_at_its_corners() {
    let ctx = GpuContext::new_headless_blocking();
    let splat = [red_center_splat()];
    // Square: 4 sides, sharp corners (high δ), crisp edge (high σ).
    let convex = render_convex(&ctx, &splat, 4.0, 28.0, 60.0);

    // Last opaque pixel along the +x axis (edge/apothem reach), in px from center.
    let mut r_axis = 0u32;
    for d in 1..(3 * SIGMA as u32) {
        if pixel(&convex, CX as u32 + d, CY as u32)[3] > 200 {
            r_axis = d;
        }
    }
    // Last opaque pixel along the +x+y diagonal, as a true radius (per-axis step × √2).
    let mut r_diag = 0.0f32;
    for d in 1..(3 * SIGMA as u32) {
        if pixel(&convex, CX as u32 + d, CY as u32 + d)[3] > 200 {
            r_diag = d as f32 * std::f32::consts::SQRT_2;
        }
    }

    assert!(r_axis > 0 && r_diag > 0.0, "square should be opaque along both directions");
    assert!(
        r_diag > r_axis as f32 * 1.2,
        "corner reach must exceed edge reach (polygon, not circle): r_diag={r_diag:.1} r_axis={r_axis}"
    );
}

// ---- Convex crisp-perimeter path -------------------------------------------------------

/// A horizontal bar of overlapping, isotropic splats (mirrors crisp.rs).
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

fn convex_crisp(ctx: &GpuContext, splats: &[GpuSplat], sides: f32, delta: f32, sigma: f32) -> Vec<u8> {
    let mut r = SplatRenderer::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm);
    r.set_convex_params(&ctx.queue, sides, delta, sigma);
    let target = gpu::create_offscreen_target(ctx, W, H);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    r.render_convex_crisp(
        &ctx.device, &ctx.queue, &view, splats, &centered_camera().uniform(), W, H, wgpu::Color::TRANSPARENT,
    );
    gpu::read_texture_rgba8(ctx, &target, W, H)
}

fn top_edge_band(img: &[u8]) -> usize {
    let x = W / 2;
    (0..H / 2).filter(|&y| { let a = pixel(img, x, y)[3]; a > 25 && a < 230 }).count()
}

/// The convex crisp path (convex accumulate + the shared resolve) collapses a *diffuse* convex
/// union (low σ) into one thin, crisp antialiased perimeter — exactly as the Gaussian crisp
/// path does — proving convex splats inherit the crisp-edge / blending toolset.
#[test]
fn convex_crisp_path_sharpens_the_perimeter() {
    let ctx = GpuContext::new_headless_blocking();
    let splats = line_splats();

    // Low σ → a deliberately diffuse boundary, so the direct path renders a wide edge and any
    // narrowing the crisp path achieves is attributable to the union-coverage threshold.
    let soft = render_convex(&ctx, &splats, 6.0, 22.0, 2.0);
    let crisp = convex_crisp(&ctx, &splats, 6.0, 22.0, 2.0);

    assert!(pixel(&soft, W / 2, H / 2)[0] > 170, "convex soft interior should be filled red");
    assert!(pixel(&crisp, W / 2, H / 2)[0] > 170, "convex crisp interior should be filled red");
    assert!(pixel(&crisp, W / 2, 4)[3] < 20, "crisp top should be clear");

    let soft_band = top_edge_band(&soft);
    let crisp_band = top_edge_band(&crisp);
    assert!(soft_band >= 8, "diffuse convex edge should be wide, got {soft_band}px");
    assert!(crisp_band <= 4, "convex crisp perimeter should be a thin rim, got {crisp_band}px");
    assert!(
        crisp_band * 2 < soft_band,
        "convex coverage path must sharpen the edge: crisp={crisp_band}px vs soft={soft_band}px"
    );
}
