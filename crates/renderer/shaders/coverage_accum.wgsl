// Coverage accumulation pass — pass 1 of the crisp-perimeter path.
//
// Renders the same splat instances as `splat.wgsl`, but instead of compositing a final
// color it writes two offscreen fields that the resolve pass (pass 2) consumes:
//
//   @location(0) color  (Rgba16Float, ADDITIVE blend): premultiplied Σ(rgb·aᵢ, aᵢ) — the
//                       interior shading. Normalizing rgb by the summed weight in the
//                       resolve pass recovers the per-splat-jittered (fuzzy) interior color.
//   @location(1) cov    (R16Float, MAX blend): max over splats of the raw Gaussian wᵢ — a
//                       smooth, bounded, order-independent silhouette field. Thresholding
//                       *this accumulated field once* (resolve pass) yields a single crisp
//                       perimeter for the whole line, instead of N scalloped per-splat edges.
//
// Max (not sum) is used for coverage so overlapping splats don't inflate the union outward
// into a blobby metaball boundary; the field stays in [0,1] and its 0.5 iso-contour tracks
// the outer splats. This mirrors app_core::splat::GpuSplat (44-byte stride).

struct Splat {
    center_x: f32,
    center_y: f32,
    cov_a: f32,
    cov_b: f32,
    cov_c: f32,
    color: u32,
    alpha: f32,
    radius: f32,
    stroke_id: u32,
    flags: u32,
    hardness: f32,
};

struct Camera {
    scale: vec2<f32>,
    offset: vec2<f32>,
    params: vec4<f32>,
};

// Screen-space minimum-footprint low-pass, identical to splat.wgsl: keeps thin / sub-pixel
// strokes registering at least ~1px of coverage so they don't alias out of the silhouette.
const FILTER_PX2 = 0.3;

@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var<uniform> camera: Camera;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec2<f32>,
    @location(1) @interpolate(flat) instance_index: u32,
};

const CORNERS = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
);

@vertex
fn vs_accum(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VsOut {
    let s = splats[iid];
    let center = vec2<f32>(s.center_x, s.center_y);
    let corner = CORNERS[vid];
    let zoom = camera.params.x;
    let pad = 3.0 * sqrt(FILTER_PX2) / zoom;
    let world = center + corner * (s.radius + pad);

    var out: VsOut;
    out.position = vec4<f32>(world * camera.scale + camera.offset, 0.0, 1.0);
    out.world_pos = world;
    out.instance_index = iid;
    return out;
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) cov: vec4<f32>,
};

@fragment
fn fs_accum(in: VsOut) -> FsOut {
    let s = splats[in.instance_index];
    let center = vec2<f32>(s.center_x, s.center_y);
    let d = in.world_pos - center;

    // Dilated forward covariance → Mahalanobis q, exactly as in splat.wgsl.
    let zoom = camera.params.x;
    let dilate = FILTER_PX2 / (zoom * zoom);
    let dcov_a = s.cov_a + dilate;
    let dcov_b = s.cov_b;
    let dcov_c = s.cov_c + dilate;
    let det = max(dcov_a * dcov_c - dcov_b * dcov_b, 1e-12);
    let ia =  dcov_c / det;
    let ib = -dcov_b / det;
    let ic =  dcov_a / det;
    let q = ia * d.x * d.x + 2.0 * ib * d.x * d.y + ic * d.y * d.y;
    if (q > 9.0) {
        discard;
    }

    // Raw Gaussian weight (no Mip opacity compensation here: the resolve pass's crisp
    // threshold — not per-splat mass — governs thin-line visibility, and the geometric
    // silhouette must stay alpha/area-independent so EDGE=0.5 is a stable contour).
    let w = exp(-0.5 * q);

    let color = unpack4x8unorm(s.color); // .xyz = rgb, .w = color alpha
    let a = s.alpha * color.w * w;       // per-splat premultiplied weight

    var out: FsOut;
    // Additive: sums premultiplied color and weight across overlapping splats.
    out.color = vec4<f32>(color.xyz * a, a);
    // Max: silhouette field is the strongest geometric coverage at this pixel.
    out.cov = vec4<f32>(w, 0.0, 0.0, 0.0);
    return out;
}
