// Instanced 2D TRIANGLE splat rendering — a 2D realization of Triangle Splatting (Held,
// Vandeghen et al., 2025, arXiv:2505.19175). Twin of convex_splat.wgsl: identical GpuSplat
// instance layout and whitening; only the window function differs.
//
// The triangle is 3 vertices {V0,V1,V2} in the splat's whitened (σ-unit) frame, so it inherits
// the stroke's orientation+anisotropy. The signed distance field is the TRUE max of the three
// edge half-plane distances (the paper rejects LogSumExp as a poor max for small triangles):
//
//   φ(p) = max_i L_i(p),  L_i(p) = n_i·(p − V_i)   (n_i outward unit normals)
//
// φ<0 inside, 0 on the boundary, >0 outside. With s the incenter (φ(s) = −inradius), the
// differentiable window function is  I(p) = ReLU(φ/φ(s))^σ : 1 at the incenter, 0 at/outside
// the boundary. σ→0 ⇒ a solid triangle, larger σ ⇒ a soft falloff. I depends only on the
// normalized ratio φ/φ(s), so it is depth/scale consistent (precomputed φ(s)).

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

// φ(p): triangle SDF as the TRUE max of the three edge distances.
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
fn vs_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VsOut {
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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let s = splats[in.instance_index];
    let center = vec2<f32>(s.center_x, s.center_y);
    let d = in.world_pos - center;
    let zoom = camera.params.x;
    let dilate = FILTER_PX2 / (zoom * zoom);
    let p = whiten(d, s.cov_a + dilate, s.cov_b, s.cov_c + dilate);

    let phi = tri_phi(p);
    let window = tri_window(phi);
    // Antialias the geometric boundary (φ=0) over ~1px so a near-solid triangle (small σ)
    // doesn't alias; for large σ the window already fades before the edge so this is a no-op.
    let aa = max(fwidth(phi), 1e-5);
    let edge = 1.0 - smoothstep(-aa, 0.0, phi); // 1 inside, 0 outside
    let weight = window * edge;
    if (weight < 0.0039) { discard; }

    let color = unpack4x8unorm(s.color);
    let a = clamp(s.alpha * color.w * weight, 0.0, 1.0);
    return vec4<f32>(color.xyz * a, a);
}
