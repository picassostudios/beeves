//! Headless render-to-texture tests. These run on the native GPU (Metal/Vulkan/DX) and
//! verify the *same* shader/pipeline path that the browser uses, so a passing test here
//! is strong evidence the WASM renderer is correct too.

use app_core::math::covariance_from_sigmas;
use app_core::splat::{pack_rgba8, GpuSplat};
use glam::Vec2;
use renderer::{camera::Camera2D, gpu, GpuContext, SplatRenderer};

const W: u32 = 256;
const H: u32 = 256;

/// Build a single red, isotropic splat centered at the viewport center, with the given
/// edge `hardness` (0 = soft Gaussian, 1 = crisp edge).
fn red_center_splat(hardness: f32) -> GpuSplat {
    let cov = covariance_from_sigmas(0.0, 18.0, 18.0);
    GpuSplat {
        center: [(W / 2) as f32, (H / 2) as f32],
        cov_a: cov.x_axis.x,
        cov_b: cov.x_axis.y,
        cov_c: cov.y_axis.y,
        color: pack_rgba8([1.0, 0.0, 0.0, 1.0]),
        alpha: 1.0,
        radius: 3.0 * 18.0,
        stroke_id: 0,
        flags: 0,
        hardness,
    }
}

/// A red splat at the viewport center whose world-space sigma is far sub-pixel at zoom 1.0
/// (the test camera maps world 1:1 to pixels). Without the screen-space low-pass this splat
/// would project to sub-pixel size and produce ~0 / aliased coverage at its center.
fn tiny_subpixel_splat() -> GpuSplat {
    // sigma = 0.35 world units => at zoom 1.0 that is 0.35px: clearly sub-pixel. Without the
    // low-pass this projects to a single-pixel spike (exp(-0.5*(1/0.35)^2) ~= 0.017 just one
    // pixel away) that aliases/flickers as it moves; the ~0.55px floor turns it into a stable
    // antialiased dot.
    const SIGMA: f32 = 0.35;
    let cov = covariance_from_sigmas(0.0, SIGMA, SIGMA);
    GpuSplat {
        center: [(W / 2) as f32, (H / 2) as f32],
        cov_a: cov.x_axis.x,
        cov_b: cov.x_axis.y,
        cov_c: cov.y_axis.y,
        color: pack_rgba8([1.0, 0.0, 0.0, 1.0]),
        alpha: 1.0,
        // Radius is sub-pixel; the vertex shader pads it by the world-space low-pass radius.
        radius: 3.0 * SIGMA,
        stroke_id: 0,
        flags: 0,
        hardness: 0.0,
    }
}

/// Render one splat to an offscreen target and return the RGBA8 readback.
fn render_one(ctx: &GpuContext, splat: GpuSplat) -> Vec<u8> {
    let mut r = SplatRenderer::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm);
    let target = gpu::create_offscreen_target(ctx, W, H);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let mut cam = Camera2D::new(Vec2::new(W as f32, H as f32));
    cam.center = Vec2::new((W / 2) as f32, (H / 2) as f32);

    r.render(
        &ctx.device,
        &ctx.queue,
        &view,
        &[splat],
        &cam.uniform(),
        wgpu::Color::TRANSPARENT,
    );
    gpu::read_texture_rgba8(ctx, &target, W, H)
}

fn pixel(buf: &[u8], x: u32, y: u32) -> &[u8] {
    let i = ((y * W + x) * 4) as usize;
    &buf[i..i + 4]
}

#[test]
fn renders_a_gaussian_blob() {
    let ctx = GpuContext::new_headless_blocking();
    let mut r = SplatRenderer::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm);
    let target = gpu::create_offscreen_target(&ctx, W, H);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let mut cam = Camera2D::new(Vec2::new(W as f32, H as f32));
    cam.center = Vec2::new((W / 2) as f32, (H / 2) as f32);

    let splats = [red_center_splat(0.0)];
    r.render(
        &ctx.device,
        &ctx.queue,
        &view,
        &splats,
        &cam.uniform(),
        wgpu::Color::TRANSPARENT,
    );

    let img = gpu::read_texture_rgba8(&ctx, &target, W, H);

    // Center pixel: strong red.
    let c = pixel(&img, W / 2, H / 2);
    assert!(c[0] > 200, "center should be bright red, got {c:?}");
    assert!(c[1] < 40 && c[2] < 40, "center should be pure red, got {c:?}");

    // A pixel a few px off-center is dimmer than the center (Gaussian falloff).
    let near = pixel(&img, W / 2 + 25, H / 2);
    assert!(near[0] < c[0], "falloff: off-center should be dimmer");

    // Corner: outside 3-sigma, fully transparent/clear.
    let corner = pixel(&img, 4, 4);
    assert!(corner[0] < 10, "corner should be clear, got {corner:?}");
}

/// EWA / Mip-Splatting screen-space minimum-footprint low-pass: a splat whose world sigma
/// is far sub-pixel at the test zoom must still leave clearly non-zero coverage at its
/// center. Without the low-pass the dilated covariance would be absent and a 0.05px Gaussian
/// would round to ~0 coverage (aliased/flickering). The constant ~0.55px sigma floor makes
/// it render as a crisp ~1px dot instead.
#[test]
fn subpixel_splat_keeps_minimum_footprint() {
    let ctx = GpuContext::new_headless_blocking();
    let img = render_one(&ctx, tiny_subpixel_splat());

    // The center pixel must have clearly non-zero red coverage. The Mip-Splatting opacity
    // compensation deliberately keeps the *peak* modest (it preserves total mass over the
    // ~1px floor footprint), but it must stay well above the aliased/zero floor.
    let c = pixel(&img, W / 2, H / 2);
    assert!(
        c[0] > 40,
        "sub-pixel splat should still cover its center pixel via the low-pass, got {c:?}"
    );

    // The coverage should be localized to a ~1px footprint: a pixel a few px away from the
    // floor-sized dot is much dimmer (it is not a full-screen wash).
    let away = pixel(&img, W / 2 + 8, H / 2);
    assert!(
        away[0] < c[0],
        "low-pass footprint should stay local: center={c:?} away={away:?}"
    );
}

#[test]
fn empty_scene_clears_to_background() {
    let ctx = GpuContext::new_headless_blocking();
    let mut r = SplatRenderer::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm);
    let target = gpu::create_offscreen_target(&ctx, W, H);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let cam = Camera2D::new(Vec2::new(W as f32, H as f32));

    r.render(
        &ctx.device,
        &ctx.queue,
        &view,
        &[],
        &cam.uniform(),
        wgpu::Color {
            r: 0.0,
            g: 0.0,
            b: 1.0,
            a: 1.0,
        },
    );

    let img = gpu::read_texture_rgba8(&ctx, &target, W, H);
    let p = pixel(&img, W / 2, H / 2);
    assert!(p[2] > 200 && p[0] < 40, "empty scene should be the blue clear color, got {p:?}");
}

/// Higher brush `hardness` must produce a crisper splat: the same Gaussian rendered with
/// hardness 1.0 keeps a much larger near-opaque core (a flat-topped disk with a sharp
/// rim) than the soft hardness 0.0 version, whose alpha falls off gradually from the
/// center. This is the measurable meaning of "crisper, not blurry". It also exercises the
/// `fwidth`-based edge path in the shader (a WGSL uniformity-analysis hazard).
#[test]
fn hardness_sharpens_the_edge() {
    let ctx = GpuContext::new_headless_blocking();

    let soft = render_one(&ctx, red_center_splat(0.0));
    let hard = render_one(&ctx, red_center_splat(1.0));

    // Count near-opaque red pixels (the flat, fully-painted region).
    let opaque = |img: &[u8]| -> usize {
        (0..W * H)
            .filter(|i| {
                let p = &img[(*i as usize) * 4..*i as usize * 4 + 4];
                p[0] > 230 && p[1] < 40 && p[2] < 40
            })
            .count()
    };
    let soft_core = opaque(&soft);
    let hard_core = opaque(&hard);
    assert!(
        hard_core > soft_core * 2,
        "hard edge should keep a much larger opaque core: soft={soft_core}, hard={hard_core}"
    );

    // Both centers stay fully painted.
    assert!(pixel(&soft, W / 2, H / 2)[0] > 200);
    assert!(pixel(&hard, W / 2, H / 2)[0] > 200);

    // Just inside the ~1-sigma body (12px out), the hard splat is still near-full while the
    // soft Gaussian has already dimmed.
    let soft_body = pixel(&soft, W / 2 + 12, H / 2)[0];
    let hard_body = pixel(&hard, W / 2 + 12, H / 2)[0];
    assert!(
        hard_body > soft_body,
        "inside the body the hard edge should be brighter: soft={soft_body}, hard={hard_body}"
    );
}
