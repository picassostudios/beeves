// Vector-blend (directional smear) — resolve pass.
//
// A full-screen triangle reads the accumulated mask (from `vector_blend.wgsl`) and the offscreen
// vector-stroke layer, and for every covered pixel produces the directional smear *once*:
//
//   * recover the coverage-weighted average smear direction  dir = mask.rg / mask.b
//   * recover the soft coverage                              coverage = clamp(mask.b)
//   * average the layer along ±dir with ~1px tap spacing (adaptive, so a long smear does not
//     ghost into a few discrete copies of the underlying edges)
//   * composite the average over the view at `coverage` opacity (premultiplied `over`)
//
// Resolving once — instead of compositing every overlapping ribbon triangle — is what keeps the
// mark clean (no cap rings, no cross-hatch where strokes overlap) while staying soft at the rim.

struct Camera {
    scale: vec2<f32>,
    offset: vec2<f32>,
    // params.zw = render-target size in physical pixels.
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var layer: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var mask: texture_2d<f32>;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_resolve(@builtin(vertex_index) vi: u32) -> VsOut {
    var out: VsOut;
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    out.uv = vec2<f32>(x, y);
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

// Cap on the per-side tap count (total taps = 2*MAX_STEPS+1). Bounds the resolve cost for very
// long smears; the step count is otherwise ~1 per pixel of smear length.
const MAX_STEPS: i32 = 48;

@fragment
fn fs_resolve(in: VsOut) -> @location(0) vec4<f32> {
    let res = max(camera.params.zw, vec2<f32>(1.0, 1.0));
    let pix = vec2<i32>(in.position.xy);
    let m = textureLoad(mask, pix, 0);
    let w = m.b;
    if (w <= 1e-4) {
        // Untouched by any blend stroke: leave the view as-is.
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let dir = m.rg / w;                  // coverage-weighted average smear half-vector (uv)
    let coverage = clamp(w, 0.0, 1.0);

    let uv = in.position.xy / res;
    let half_len_px = length(dir * res);
    let steps = clamp(i32(ceil(half_len_px)), 1, MAX_STEPS);

    // The layer is premultiplied alpha, so a straight average is the correct premultiplied blend.
    var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var count = 0.0;
    for (var k: i32 = -steps; k <= steps; k = k + 1) {
        let f = f32(k) / f32(steps);
        sum = sum + textureSampleLevel(layer, samp, uv + dir * f, 0.0);
        count = count + 1.0;
    }
    let avg = sum / count;

    // `avg` is premultiplied; scaling by coverage keeps it premultiplied.
    return avg * coverage;
}
