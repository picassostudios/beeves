// Convex coverage accumulation — pass 1 of the crisp-perimeter path, CONVEX (3DCS) variant.
//
// Structurally identical to coverage_accum.wgsl, writing the two offscreen fields the shared
// resolve pass (coverage_resolve.wgsl) consumes:
//
//   @location(0) color (Rgba16Float, ADDITIVE): premultiplied Σ(rgb·aᵢ, aᵢ) — the interior.
//   @location(1) cov   (R16Float, MAX):         max over splats of the convex indicator I — the
//                                               order-independent silhouette field.
//
// The per-splat weight is the 3DCS smooth-convex indicator I(q) = sigmoid(−σ·φ(q)) over the
// hull's LogSumExp signed distance φ (see convex_splat.wgsl for the full derivation; the φ and
// whitening helpers are kept in sync). Because I = 0.5 exactly on the hull boundary, the
// resolve pass's 0.5 threshold tracks the true convex silhouette, so a convex-splat stroke gets
// one crisp, flat-sided perimeter while its interior keeps per-splat color fuzz.

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

struct Convex {
    points: array<vec4<f32>, 4>,
    // shape.x = K, .y = δ smoothness, .z = σ sharpness, .w unused.
    shape: vec4<f32>,
};

const FILTER_PX2 = 0.3;

@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<uniform> convex: Convex;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec2<f32>,
    @location(1) @interpolate(flat) instance_index: u32,
};

const CORNERS = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
);

fn cvx_vert(i: i32) -> vec2<f32> {
    let v = convex.points[i / 2];
    if ((i & 1) == 0) { return v.xy; }
    return v.zw;
}

fn edge_normal(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let e = b - a;
    var n = normalize(vec2<f32>(e.y, -e.x));
    if (dot(n, a) < 0.0) { n = -n; }
    return n;
}

// Smooth signed distance φ to the convex hull — see convex_splat.wgsl::convex_phi.
fn convex_phi(p: vec2<f32>) -> f32 {
    let k = i32(convex.shape.x + 0.5);
    let delta = max(convex.shape.y, 0.5);
    var m = -1.0e30;
    for (var i = 0; i < k; i = i + 1) {
        let a = cvx_vert(i);
        let n = edge_normal(a, cvx_vert((i + 1) % k));
        m = max(m, dot(n, p - a));
    }
    var acc = 0.0;
    for (var i = 0; i < k; i = i + 1) {
        let a = cvx_vert(i);
        let n = edge_normal(a, cvx_vert((i + 1) % k));
        acc = acc + exp(delta * (dot(n, p - a) - m));
    }
    return m + log(acc) / delta;
}

// Whiten d under the dilated covariance — see convex_splat.wgsl::whiten.
fn whiten(d: vec2<f32>, dcov_a: f32, dcov_b: f32, dcov_c: f32) -> vec2<f32> {
    let det = max(dcov_a * dcov_c - dcov_b * dcov_b, 1e-12);
    let ia =  dcov_c / det;
    let ib = -dcov_b / det;
    let ic =  dcov_a / det;
    let l11 = sqrt(max(ia, 1e-12));
    let l21 = ib / l11;
    let l22 = sqrt(max(ic - l21 * l21, 1e-12));
    return vec2<f32>(l11 * d.x + l21 * d.y, l22 * d.y);
}

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

    let zoom = camera.params.x;
    let dilate = FILTER_PX2 / (zoom * zoom);
    let p = whiten(d, s.cov_a + dilate, s.cov_b, s.cov_c + dilate);

    let phi = convex_phi(p);
    // Smooth indicator (no fwidth cap here: the resolve pass thresholds the accumulated field
    // and applies its own ~1px AA, so this must stay a smooth, alpha-independent silhouette).
    let z = clamp(convex.shape.z * phi, -30.0, 30.0);
    let indicator = 1.0 / (1.0 + exp(z));
    if (indicator < 0.0039) {
        discard;
    }

    let color = unpack4x8unorm(s.color);
    let a = s.alpha * color.w * indicator;

    var out: FsOut;
    out.color = vec4<f32>(color.xyz * a, a);
    out.cov = vec4<f32>(indicator, 0.0, 0.0, 0.0);
    return out;
}
