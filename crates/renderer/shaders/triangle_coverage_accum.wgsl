// Triangle coverage accumulation — pass 1 of the crisp-perimeter path, TRIANGLE variant (a 2D
// realization of Triangle Splatting; Held, Vandeghen et al., 2025, arXiv:2505.19175). Twin of
// convex_coverage_accum.wgsl: same Splat/Camera/Triangle structs, bindings, whitening, vertex
// stage and the two offscreen targets the shared resolve pass (coverage_resolve.wgsl) consumes;
// only the window/coverage math differs.
//
// The split: the COLOR target accumulates the paper's window function I(p) = ReLU(φ/φ(s))^σ as
// the interior weight (1 at the incenter, 0 at the boundary). But that weight is 0 on the
// boundary, so it can't drive the resolve pass's 0.5 silhouette threshold. So the COVERAGE
// target instead carries a separate geometric silhouette field — a sigmoid of φ that is exactly
// 0.5 on the boundary — letting the resolve pass track the true triangle silhouette while the
// interior keeps per-splat color fuzz.

struct Splat {
    center_x: f32, center_y: f32, cov_a: f32, cov_b: f32, cov_c: f32,
    color: u32, alpha: f32, radius: f32, stroke_id: u32, flags: u32, hardness: f32,
};
struct Camera { scale: vec2<f32>, offset: vec2<f32>, params: vec4<f32> };
struct Triangle {
    // V0=verts[0].xy, V1=verts[0].zw, V2=verts[1].xy (whitened σ-unit frame).
    verts: array<vec4<f32>, 2>,
    // tmeta.x = φ(s) (incenter SDF value, negative); tmeta.y = σ smoothness; .zw unused.
    // (`meta` is a reserved WGSL keyword, so the field is named `tmeta` on the GPU side; the
    // byte layout still mirrors `TriangleUniform::meta`.)
    tmeta: vec4<f32>,
};

const FILTER_PX2 = 0.3;

@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<uniform> tri: Triangle;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec2<f32>,
    @location(1) @interpolate(flat) instance_index: u32,
};

const CORNERS = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
);

fn tri_vert(i: i32) -> vec2<f32> {
    if (i == 0) { return tri.verts[0].xy; }
    if (i == 1) { return tri.verts[0].zw; }
    return tri.verts[1].xy;
}

// Outward unit normal of edge a→b (the triangle centroid is at the origin in this frame).
fn edge_normal(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let e = b - a;
    var n = normalize(vec2<f32>(e.y, -e.x));
    if (dot(n, a) < 0.0) { n = -n; }
    return n;
}

// φ(p): triangle SDF as the TRUE max of the three edge distances (see triangle_splat.wgsl).
fn tri_phi(p: vec2<f32>) -> f32 {
    var m = -1.0e30;
    for (var i = 0; i < 3; i = i + 1) {
        let a = tri_vert(i);
        let n = edge_normal(a, tri_vert((i + 1) % 3));
        m = max(m, dot(n, p - a));
    }
    return m;
}

// Window function I(p) = ReLU(φ/φ(s))^σ : 1 at incenter, 0 at/outside the boundary.
fn tri_window(phi: f32) -> f32 {
    let phi_s = min(tri.tmeta.x, -1.0e-4);
    let sigma = max(tri.tmeta.y, 0.02);
    let ratio = clamp(phi / phi_s, 0.0, 1.0);
    if (ratio <= 0.0) { return 0.0; }
    return pow(ratio, sigma);
}

// Geometric silhouette field: sigmoid(−COV_SHARP·φ) = 0.5 on the boundary (φ=0), ~1 inside,
// ~0 outside, so the shared resolve pass's 0.5 threshold tracks the true triangle silhouette.
const COV_SHARP = 6.0;
fn tri_coverage(phi: f32) -> f32 {
    let z = clamp(COV_SHARP * phi, -30.0, 30.0);
    return 1.0 / (1.0 + exp(z));
}

// Whiten d under the dilated covariance — identical to convex_splat.wgsl::whiten.
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

struct FsOut { @location(0) color: vec4<f32>, @location(1) cov: vec4<f32> };

@fragment
fn fs_accum(in: VsOut) -> FsOut {
    let s = splats[in.instance_index];
    let center = vec2<f32>(s.center_x, s.center_y);
    let d = in.world_pos - center;
    let zoom = camera.params.x;
    let dilate = FILTER_PX2 / (zoom * zoom);
    let p = whiten(d, s.cov_a + dilate, s.cov_b, s.cov_c + dilate);

    let phi = tri_phi(p);
    let window = tri_window(phi); // interior color weight (the paper's window function)
    let cov = tri_coverage(phi);  // geometric silhouette (0.5 on the boundary)
    if (cov < 0.0039 && window < 0.0039) { discard; }

    let color = unpack4x8unorm(s.color);
    let a = s.alpha * color.w * window;
    var out: FsOut;
    out.color = vec4<f32>(color.xyz * a, a);
    out.cov = vec4<f32>(cov, 0.0, 0.0, 0.0);
    return out;
}
