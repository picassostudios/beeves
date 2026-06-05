//! Headless GPU object-id picking tests. Render several splats with distinct
//! `stroke_id`s into the integer picking target, read it back, and assert that the
//! texel at each splat center decodes to that splat's `stroke_id` while an empty area
//! decodes to the `NO_ID` sentinel.
//!
//! Like `render.rs`, these run on the native GPU and exercise the *same* shader/
//! pipeline path the browser uses.

use app_core::math::covariance_from_sigmas;
use app_core::splat::{pack_rgba8, GpuSplat};
use glam::Vec2;
use renderer::camera::Camera2D;
use renderer::picking::{
    self, create_picking_target, pick_id_at, read_picking_ids, PickRequest, PickingRenderer,
    NO_ID,
};
use renderer::GpuContext;

const W: u32 = 256;
const H: u32 = 256;
const SIGMA: f32 = 14.0;

/// An isotropic, fully-opaque splat centered at pixel `(cx, cy)` carrying `stroke_id`.
/// Opaque/large so the center texel is solidly covered (q == 0 < 9).
fn splat_at(cx: f32, cy: f32, stroke_id: u32) -> GpuSplat {
    let cov = covariance_from_sigmas(0.0, SIGMA, SIGMA);
    GpuSplat {
        center: [cx, cy],
        cov_a: cov.x_axis.x,
        cov_b: cov.x_axis.y,
        cov_c: cov.y_axis.y,
        color: pack_rgba8([1.0, 1.0, 1.0, 1.0]),
        alpha: 1.0,
        radius: 3.0 * SIGMA,
        stroke_id,
        flags: 0,
        hardness: 0.0,
    }
}

/// Camera that maps world coordinates 1:1 to framebuffer pixels (top-left origin),
/// matching the convention used by `render.rs`.
fn pixel_camera() -> Camera2D {
    let mut cam = Camera2D::new(Vec2::new(W as f32, H as f32));
    cam.center = Vec2::new((W / 2) as f32, (H / 2) as f32);
    cam
}

fn id_at(ids: &[u32], x: u32, y: u32) -> u32 {
    ids[(y * W + x) as usize]
}

#[test]
fn decodes_stroke_ids_at_splat_centers() {
    let ctx = GpuContext::new_headless_blocking();
    let mut r = PickingRenderer::new(&ctx.device);
    let target = create_picking_target(&ctx, W, H);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let cam = pixel_camera();

    // Three well-separated splats with distinct, non-sequential stroke ids.
    let centers = [(64.0_f32, 64.0_f32), (192.0, 96.0), (110.0, 200.0)];
    let ids = [7_u32, 42, 1000];
    let splats: Vec<GpuSplat> = centers
        .iter()
        .zip(ids)
        .map(|(&(x, y), id)| splat_at(x, y, id))
        .collect();

    r.render(&ctx.device, &ctx.queue, &view, &splats, &cam.uniform());
    let buf = read_picking_ids(&ctx, &target, W, H);

    for (&(x, y), expected) in centers.iter().zip(ids) {
        let got = id_at(&buf, x as u32, y as u32);
        assert_eq!(
            got, expected,
            "pixel at splat center ({x}, {y}) should decode to stroke_id {expected}, got {got}"
        );
    }

    // An empty corner, far from every splat, must remain the sentinel.
    let empty = id_at(&buf, 4, 4);
    assert_eq!(
        empty, NO_ID,
        "empty area should decode to the NO_ID sentinel, got {empty}"
    );

    // The picking format is a single-channel integer target.
    assert_eq!(picking::PICKING_FORMAT, wgpu::TextureFormat::R32Uint);
}

#[test]
fn topmost_splat_wins_under_overlap() {
    // Two overlapping splats at the same center; the later-drawn instance (no depth
    // test, no blending) must win, matching painter's-order top-most selection.
    let ctx = GpuContext::new_headless_blocking();
    let mut r = PickingRenderer::new(&ctx.device);
    let target = create_picking_target(&ctx, W, H);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let cam = pixel_camera();

    let splats = [
        splat_at(128.0, 128.0, 3), // drawn first (underneath)
        splat_at(128.0, 128.0, 9), // drawn last (on top)
    ];
    r.render(&ctx.device, &ctx.queue, &view, &splats, &cam.uniform());
    let buf = read_picking_ids(&ctx, &target, W, H);

    assert_eq!(
        id_at(&buf, 128, 128),
        9,
        "the last-drawn (top-most) splat should win at the overlap center"
    );
}

#[test]
fn pick_id_at_samples_single_pixel() {
    // Exercise the one-shot cursor hit-test helper end to end.
    let ctx = GpuContext::new_headless_blocking();
    let mut r = PickingRenderer::new(&ctx.device);
    let target = create_picking_target(&ctx, W, H);
    let cam = pixel_camera();

    let splats = [splat_at(80.0, 170.0, 55)];
    let uniform = cam.uniform();

    // Pixel on the splat center decodes to its id.
    let hit = pick_id_at(
        &ctx,
        &mut r,
        PickRequest {
            target: &target,
            splats: &splats,
            camera: &uniform,
            x: 80,
            y: 170,
        },
    );
    assert_eq!(hit, 55, "cursor over splat center should pick its stroke_id");

    // Empty pixel decodes to the sentinel.
    let miss = pick_id_at(
        &ctx,
        &mut r,
        PickRequest {
            target: &target,
            splats: &splats,
            camera: &uniform,
            x: 8,
            y: 8,
        },
    );
    assert_eq!(miss, NO_ID, "cursor over empty area should pick NO_ID");

    // Out-of-bounds pixel short-circuits to the sentinel.
    let oob = pick_id_at(
        &ctx,
        &mut r,
        PickRequest {
            target: &target,
            splats: &splats,
            camera: &uniform,
            x: W,
            y: H,
        },
    );
    assert_eq!(oob, NO_ID, "out-of-bounds pixel should pick NO_ID");
}

#[test]
fn empty_scene_is_all_sentinel() {
    let ctx = GpuContext::new_headless_blocking();
    let mut r = PickingRenderer::new(&ctx.device);
    let target = create_picking_target(&ctx, W, H);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let cam = pixel_camera();

    r.render(&ctx.device, &ctx.queue, &view, &[], &cam.uniform());
    let buf = read_picking_ids(&ctx, &target, W, H);

    assert!(
        buf.iter().all(|&id| id == NO_ID),
        "with no splats every texel should be the NO_ID sentinel"
    );
}
