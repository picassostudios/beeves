// Coverage resolve pass — pass 2 of the crisp-perimeter path.
//
// A single full-screen triangle reads the two fields accumulated in pass 1 and produces the
// final premultiplied color. The whole line gets ONE crisp antialiased perimeter (a single
// threshold of the smooth, accumulated coverage field) while its interior keeps the
// per-splat color variation (fuzz) recovered by normalizing the accumulated color.
//
// The source textures are the same size as the output, so we sample them with `textureLoad`
// at the destination pixel (1 texel = 1 fragment, exact). `fwidth(cov)` is then the discrete
// per-pixel gradient of the coverage field, giving a resolution-independent ~1px AA band —
// i.e. the smoothstep width tracks how fast coverage changes, so the rim is uniformly thin
// regardless of stroke size (distance-normalized AA on the union, not per-splat).

@group(0) @binding(0) var color_tex: texture_2d<f32>;
@group(0) @binding(1) var coverage_tex: texture_2d<f32>;

// Iso-contour of the geometric coverage field treated as the silhouette boundary. 0.5 is the
// half-maximum contour of the Gaussian union — stable and area/alpha-independent.
const EDGE = 0.5;

struct VsOut {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_resolve(@builtin(vertex_index) vid: u32) -> VsOut {
    // Oversized triangle covering the viewport (clip space).
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out: VsOut;
    out.position = vec4<f32>(corners[vid], 0.0, 1.0);
    return out;
}

@fragment
fn fs_resolve(in: VsOut) -> @location(0) vec4<f32> {
    let p = vec2<i32>(i32(in.position.x), i32(in.position.y));
    let cov = textureLoad(coverage_tex, p, 0).r;
    let accum = textureLoad(color_tex, p, 0);

    // One crisp, smooth perimeter for the whole line: threshold the accumulated coverage
    // with a ~1px (gradient-normalized) antialiased band.
    let aa = max(fwidth(cov), 1e-4);
    let mask = smoothstep(EDGE - aa, EDGE + aa, cov);

    // Fuzzy interior: the per-splat-jittered color survives as the weight-normalized average
    // of everything that painted this pixel. `op` is the interior opacity (carries overlap /
    // texture variation); the crisp `mask` governs the outer edge.
    let rgb = accum.rgb / max(accum.a, 1e-5);
    let op = clamp(accum.a, 0.0, 1.0);
    let alpha = mask * op;

    return vec4<f32>(rgb * alpha, alpha);
}
