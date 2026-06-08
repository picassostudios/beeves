// Vector-blend (directional smear) — resolve pass, run once per blend stroke in document z-order.
//
// A full-screen triangle reads the accumulated mask (one blend stroke's ribbon, from
// `vector_blend.wgsl`) and the running colour composite of everything *beneath* this stroke
// (`beneath`), and for every covered pixel produces the directional smear:
//
//   * recover the coverage-weighted average smear direction  dir = mask.rg / mask.b
//   * recover the soft coverage                              coverage = clamp(mask.b)
//   * take a Gaussian-weighted average of `beneath` along ±dir (nearby colour dominates, distant
//     colour falls off smoothly — softer and less muddy than a flat box average)
//   * composite that smear over `beneath` (premultiplied `over`) at `coverage` opacity
//
// Crucially the smear samples and composites over `beneath` — the layers below *this* stroke in
// document order — and the renderer ping-pongs the colour target so each successive (higher-z)
// blend stroke smears the result the lower-z strokes already produced. That is what makes the
// z-order matter: a back stroke smears first, a front stroke smears over it.
//
// Untouched pixels pass `beneath` straight through (the shader writes every pixel, with no
// hardware blend, so the output target is a complete copy of the layers below plus this smear).
// Resolving once per stroke — instead of compositing every overlapping ribbon triangle — keeps the
// mark clean (no cap rings, no cross-hatch) while staying soft at the rim.

struct Camera {
    scale: vec2<f32>,
    offset: vec2<f32>,
    // params.zw = render-target size in physical pixels.
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
// The running colour composite of every layer beneath this blend stroke (premultiplied alpha).
@group(0) @binding(1) var beneath: texture_2d<f32>;
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

// Gaussian sample weighting across the smear: weight = exp(-(t*t)/(2*SIGMA*SIGMA)) with t in
// [-1,1] over the smear half-length. Smaller SIGMA = tighter core (nearby colour dominates);
// SIGMA = 0.5 leaves the rim taps at exp(-2) ≈ 0.14, so the ends still contribute but softly.
const SIGMA: f32 = 0.5;

@fragment
fn fs_resolve(in: VsOut) -> @location(0) vec4<f32> {
    let res = max(camera.params.zw, vec2<f32>(1.0, 1.0));
    let pix = vec2<i32>(in.position.xy);
    // The layers beneath this stroke at this pixel; the smear composites over this, and untouched
    // pixels return it unchanged.
    let base = textureLoad(beneath, pix, 0);
    let m = textureLoad(mask, pix, 0);
    let w = m.b;
    if (w <= 1e-4) {
        return base;
    }
    let dir = m.rg / w;                  // coverage-weighted average smear half-vector (uv)
    let coverage = clamp(w, 0.0, 1.0);

    let uv = in.position.xy / res;
    let half_len_px = length(dir * res);
    let steps = clamp(i32(ceil(half_len_px)), 1, MAX_STEPS);

    // `beneath` is premultiplied alpha, so the Gaussian-weighted sum is the correct premultiplied
    // blend. Weighting by a Gaussian keeps the colour under the pixel dominant and lets the smear
    // fade rather than flatten into a uniform box average.
    let inv_two_sigma2 = 1.0 / (2.0 * SIGMA * SIGMA);
    var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var wsum = 0.0;
    for (var k: i32 = -steps; k <= steps; k = k + 1) {
        let t = f32(k) / f32(steps);
        let g = exp(-t * t * inv_two_sigma2);
        sum = sum + g * textureSampleLevel(beneath, samp, uv + dir * t, 0.0);
        wsum = wsum + g;
    }
    let avg = sum / max(wsum, 1e-6);

    // Composite this stroke's smear over the layers beneath it (premultiplied `over`) at the soft
    // coverage. `smear` is premultiplied, so scaling by coverage keeps it premultiplied.
    let smear = avg * coverage;
    return smear + base * (1.0 - smear.a);
}
