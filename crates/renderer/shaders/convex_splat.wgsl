// Instanced smooth-CONVEX splat rendering — a 2D realization of 3D Convex Splatting (3DCS,
// Held et al., CVPR 2025) and CvxNet's smooth convexes.
//
// A convex primitive is the convex hull of a *point set* {V_0 … V_{K-1}}, NOT a Gaussian or a
// covariance-derived ellipse. Following 3DCS/CvxNet, the shape is rendered from the hull's
// edge lines:
//
//   • each hull edge j defines a signed distance   L_j(q) = n_j · q + d_j      (n_j outward),
//   • a smooth signed distance is the LogSumExp     φ(q) = (1/δ)·log Σ_j exp(δ·L_j(q)),
//   • the opacity is a sigmoid of that distance     I(q) = sigmoid(−σ·φ(q)).
//
// δ (smoothness) rounds the VERTICES (δ→∞ ⇒ a hard polygon, small δ ⇒ soft corners); σ
// (sharpness) controls the EDGE transition (large σ ⇒ a dense, hard boundary; small σ ⇒ a
// diffuse one). These are the two decoupled knobs from the paper (Fig. 4). Crucially, because
// I(q) is a sigmoid of a *line* signed distance — not a radial Gaussian falloff — the interior
// is FLAT-TOPPED and the edges are straight with sharp (or controllably rounded) corners. That
// is the whole point of convex splatting: hard edges / flat faces with far fewer primitives.
//
// The hull point set lives in the splat's whitened (covariance) frame in σ-units, so each
// splat inherits its stroke's orientation and anisotropy (a thin splat ⇒ a thin convex). The
// GpuSplat instance layout is unchanged (the Gaussian path is untouched); the hull point set
// and δ/σ come from the shared Convex uniform.

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
    // params.x = zoom (pixels-per-world-unit); rest reserved.
    params: vec4<f32>,
};

// The convex primitive's shape (see renderer::convex::ConvexUniform).
struct Convex {
    // Up to 8 hull vertices in CCW order, packed two-per-vec4 (v0 = .xy, v1 = .zw, …), given in
    // the whitened/local frame in σ-units. Their convex hull is the primitive.
    points: array<vec4<f32>, 4>,
    // shape.x = K (live vertex count); .y = δ smoothness; .z = σ sharpness; .w unused.
    shape: vec4<f32>,
};

// Screen-space minimum-footprint low-pass (EWA / Mip-Splatting), identical to splat.wgsl: keeps
// a far-sub-pixel splat covering ~1px so thin convex strokes don't alias out.
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

// Hull vertex `i` (0..K), unpacked from the two-per-vec4 packing.
fn cvx_vert(i: i32) -> vec2<f32> {
    let v = convex.points[i / 2];
    if ((i & 1) == 0) { return v.xy; }
    return v.zw;
}

// Outward unit normal of the CCW hull edge V_i → V_{i+1}. The polygon is centered near the
// origin (the whitened splat center), so we orient the normal to point away from it.
fn edge_normal(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let e = b - a;
    var n = normalize(vec2<f32>(e.y, -e.x));
    if (dot(n, a) < 0.0) { n = -n; }
    return n;
}

// Smooth signed distance φ(p) to the convex hull (CvxNet Eq. 2): the LogSumExp smooth-max of
// the per-edge signed distances, normalized by 1/δ so φ → the true signed distance as δ → ∞
// (positive outside, negative inside, 0 on the boundary). Evaluated with the max subtracted
// for numerical stability.
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

// Whiten the world offset `d` under the dilated forward covariance so the splat is isotropic
// (the 1σ ellipse maps to the unit circle): returns `p` with |p|² = dᵀ Σ⁻¹ d. The hull points
// are defined in this same σ-unit frame, so the convex inherits the splat's orientation +
// anisotropy. Cholesky Σ⁻¹ = L Lᵀ, p = Lᵀ d.
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
fn vs_main(
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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let s = splats[in.instance_index];
    let center = vec2<f32>(s.center_x, s.center_y);
    let d = in.world_pos - center;

    // Mip-Splatting screen-space low-pass dilation (same as splat.wgsl), then whiten.
    let zoom = camera.params.x;
    let dilate = FILTER_PX2 / (zoom * zoom);
    let p = whiten(d, s.cov_a + dilate, s.cov_b, s.cov_c + dilate);

    // Smooth signed distance to the convex hull, then the sigmoid indicator (CvxNet Eq. 3).
    let phi = convex_phi(p);
    // `phi` is in σ-units; `fwidth` is its screen-space gradient (σ-units per pixel). Cap the
    // effective sharpness so the sigmoid's [0.1,0.9] transition spans ~1px on screen — this
    // antialiases a maximally-crisp edge instead of letting it alias, while still honoring a
    // softer (smaller) σ for a diffuse boundary. 4.4 ≈ 2·logit(0.9): the sigmoid's [0.1,0.9]
    // width in units of `aa`, so the on-screen band lands at ~1px when σ is high.
    let aa = max(fwidth(phi), 1e-5);
    let sharp = min(convex.shape.z, 4.4 / aa);
    let z = clamp(sharp * phi, -30.0, 30.0);
    let indicator = 1.0 / (1.0 + exp(z)); // sigmoid(−σ·φ): ~1 inside, 0.5 on the boundary, ~0 out
    if (indicator < 0.0039) {
        discard;
    }

    let color = unpack4x8unorm(s.color); // .xyz = rgb, .w = color alpha
    let a = clamp(s.alpha * color.w * indicator, 0.0, 1.0);
    // Premultiplied alpha (paired with One / OneMinusSrcAlpha blending).
    return vec4<f32>(color.xyz * a, a);
}
