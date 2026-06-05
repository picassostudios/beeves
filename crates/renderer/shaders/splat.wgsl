// Instanced anisotropic 2D Gaussian splat rendering.
//
// One instance per splat; the vertex shader expands each into a screen-aligned quad
// covering ~3 sigma, and the fragment shader evaluates the Gaussian and outputs
// premultiplied alpha (paired with One / OneMinusSrcAlpha blending).

// Mirrors app_core::splat::GpuSplat exactly: all 4-byte scalars, 44-byte stride. The
// inverse covariance is not stored — this shader derives it from the (dilated) forward
// covariance below. Color is a packed RGBA8 word (unpack4x8unorm: .x = low byte = R).
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
    // clip.xy = world.xy * scale + offset
    scale: vec2<f32>,
    offset: vec2<f32>,
    // params.x = zoom (pixels-per-world-unit); rest reserved.
    params: vec4<f32>,
};

// Screen-space minimum-footprint low-pass (EWA / 3DGS / Mip-Splatting). We dilate every
// splat's projected covariance by a small constant *screen-space* variance so that even a
// far-sub-pixel splat (zoomed out, or a deliberately thin/sharp stroke) covers at least
// ~1px and renders as crisp antialiased coverage instead of aliasing/flickering. This also
// keeps the differentiable-rasterization gradient alive: a degenerate (point) splat would
// otherwise have a near-zero footprint and a dead gradient. FILTER_PX2 is the floor
// variance in px^2 (~0.55px sigma). It is converted to world^2 via 1/zoom^2 so the floor is
// constant on screen at any zoom, and paired with a Mip-Splatting opacity compensation so
// enlarging the Gaussian does not wash out thin lines.
const FILTER_PX2 = 0.3;

@group(0) @binding(0) var<storage, read> splats: array<Splat>;
@group(0) @binding(1) var<uniform> camera: Camera;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec2<f32>,
    @location(1) @interpolate(flat) instance_index: u32,
};

// Two triangles forming a [-1,1]^2 quad.
const CORNERS = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
);

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> VsOut {
    let s = splats[iid];
    let center = vec2<f32>(s.center_x, s.center_y);
    let corner = CORNERS[vid];
    // Enlarge the quad by the world-space low-pass radius so the dilated Gaussian (see the
    // FILTER_PX2 note above) is never clipped by the splat's own footprint.
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

    // EWA / Mip-Splatting screen-space low-pass: dilate the forward covariance by a constant
    // screen-space variance (FILTER_PX2, converted px^2 -> world^2 via 1/zoom^2) so every
    // splat is at least ~1px on screen, then invert the dilated covariance to evaluate the
    // Mahalanobis distance. This makes sub-pixel splats render as crisp antialiased coverage
    // and keeps the differentiable-rasterization gradient non-degenerate.
    let zoom = camera.params.x;
    let dilate = FILTER_PX2 / (zoom * zoom); // px^2 -> world^2: constant footprint on screen
    let dcov_a = s.cov_a + dilate;
    let dcov_b = s.cov_b;
    let dcov_c = s.cov_c + dilate;
    let det = max(dcov_a * dcov_c - dcov_b * dcov_b, 1e-12);
    let ia =  dcov_c / det;
    let ib = -dcov_b / det;
    let ic =  dcov_a / det;
    // Mahalanobis distance under the inverse of the dilated covariance [a b; b c].
    let q = ia * d.x * d.x + 2.0 * ib * d.x * d.y + ic * d.y * d.y;
    if (q > 9.0) {
        discard;
    }
    // Raw Gaussian falloff (soft halo out to ~3 sigma), with Mip-Splatting opacity
    // compensation: enlarging the Gaussian spreads its mass, so scale the weight by the
    // area-preserving factor sqrt(det_orig / det_dilated) (<= 1) to keep thin lines from
    // washing out.
    let det_orig = max(s.cov_a * s.cov_c - s.cov_b * s.cov_b, 1e-12);
    let comp = sqrt(det_orig / det); // <= 1
    let w = exp(-0.5 * q) * comp;

    // Edge hardness: remap the soft Gaussian toward a crisp, screen-space antialiased
    // edge. `hardness` = 0 leaves the Gaussian untouched; as it approaches 1 the splat
    // becomes a solid ellipse with a ~1px AA rim. `edge` is the iso-contour treated as
    // the boundary; `fwidth(w)` gives a resolution-independent ~1px transition (the small
    // epsilon keeps the smoothstep band non-degenerate when the splat is sub-pixel).
    let h = clamp(s.hardness, 0.0, 1.0);
    let edge = h * 0.6;
    let aa = fwidth(w) + 1e-4;
    let crisp = smoothstep(edge - aa, edge + aa, w);
    let weight = mix(w, crisp, h);

    let color = unpack4x8unorm(s.color); // .xyz = rgb, .w = color alpha
    let a = clamp(s.alpha * color.w * weight, 0.0, 1.0);
    // Premultiplied alpha.
    return vec4<f32>(color.xyz * a, a);
}
