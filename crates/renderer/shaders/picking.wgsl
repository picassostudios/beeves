// Object-id picking pass for instanced Gaussian splats.
//
// Vertex expansion mirrors splat.wgsl exactly (one screen-aligned quad per splat,
// ~3 sigma). The fragment shader does NOT blend or evaluate color; instead it
// writes the splat's identity (stroke_id) into a single-channel integer target
// (R32Uint). Texels outside the Gaussian's 3-sigma footprint (q > 9) are
// discarded, so only covered texels overwrite the cleared sentinel value. With no
// depth test and front-to-back instance order, the last-drawn covering splat wins
// (matching the painter's-order top-most stroke under the cursor).

// Mirrors app_core::splat::GpuSplat exactly: all 4-byte scalars, 44-byte stride.
// Layout is identical to splat.wgsl. `color`/`alpha`/`hardness` are unused here but must
// be present so the per-instance array stride matches the buffer.
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
    // params.x = zoom (pixels-per-world-unit); rest reserved. Unused here, present so the
    // uniform layout matches splat.wgsl / CameraUniform.
    params: vec4<f32>,
};

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
    let world = center + corner * s.radius;

    var out: VsOut;
    out.position = vec4<f32>(world * camera.scale + camera.offset, 0.0, 1.0);
    out.world_pos = world;
    out.instance_index = iid;
    return out;
}

// Single 32-bit unsigned integer target encoding the splat's stroke_id. Cleared
// to a sentinel (0xFFFFFFFF) before the pass; only texels inside the Gaussian
// footprint overwrite it.
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<u32> {
    let s = splats[in.instance_index];
    let center = vec2<f32>(s.center_x, s.center_y);
    let d = in.world_pos - center;
    // Invert the forward covariance [a b; b c] in-shader (no inverse is uploaded) and take
    // the Mahalanobis distance. Picking uses the raw (undilated) footprint, matching the
    // CPU `hit_test_splat`'s `q < 9` test.
    let det = max(s.cov_a * s.cov_c - s.cov_b * s.cov_b, 1e-12);
    let ia =  s.cov_c / det;
    let ib = -s.cov_b / det;
    let ic =  s.cov_a / det;
    let q = ia * d.x * d.x + 2.0 * ib * d.x * d.y + ic * d.y * d.y;
    if (q > 9.0) {
        discard;
    }
    // R32Uint target: only the .x channel is stored.
    return vec4<u32>(s.stroke_id, 0u, 0u, 0u);
}
