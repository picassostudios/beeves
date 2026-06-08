// Vector-blend (directional smear) — accumulate pass.
//
// A blend stroke is a tessellated ribbon that carries no colour of its own; it describes *where*
// and *which direction* to smear the vector-stroke layer beneath it. This pass rasterizes the
// ribbon into an offscreen mask with **additive** blending, so overlapping geometry (a stroke's
// own round caps/joins over its body, or two strokes crossing) accumulates rather than
// compositing twice — the resolve pass then saturates it, which is what removes the hard cap
// rings and the woven cross-hatch you get from drawing many translucent triangles with `over`.
//
// Per fragment it writes a feathered weight `w` and the smear direction scaled by `w`:
//   rg = smear_dir_uv * w,  b = w
// The resolve pass recovers the coverage-weighted average direction as rg/b and the coverage as
// clamp(b). Feathering `w` across the ribbon width gives the mark a soft edge.

struct Camera {
    // clip.xy = world.xy * scale + offset
    scale: vec2<f32>,
    offset: vec2<f32>,
    // params.x = zoom; params.zw = render-target size in physical pixels.
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

struct VsIn {
    @location(0) pos: vec2<f32>,      // world-space vertex
    @location(1) edge: f32,           // -1..1 across the ribbon width
    @location(2) tangent: vec2<f32>,  // world-space tangent, scaled to the smear half-length
    @location(3) strength: f32,       // blend strength [0,1]
};

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) edge: f32,
    @location(1) smear: vec2<f32>,    // smear half-vector in uv space
    @location(2) strength: f32,
};

@vertex
fn vs_accum(in: VsIn) -> VsOut {
    var out: VsOut;
    out.position = vec4<f32>(in.pos * camera.scale + camera.offset, 0.0, 1.0);
    out.edge = in.edge;
    // World -> clip is linear (translation drops out for a direction); clip -> uv maps a clip
    // displacement to (dx*0.5, -dy*0.5).
    let dclip = in.tangent * camera.scale;
    out.smear = vec2<f32>(dclip.x * 0.5, -dclip.y * 0.5);
    out.strength = in.strength;
    return out;
}

// Width fraction over which the mark fades to nothing at the rim (0 = hard edge, 1 = fades from
// the centreline). The inner `1 - FEATHER` of the half-width stays at full strength.
const FEATHER: f32 = 0.6;

@fragment
fn fs_accum(in: VsOut) -> @location(0) vec4<f32> {
    // Smooth falloff across the width: 1 in the core, ramping to 0 at the rim.
    let feather = 1.0 - smoothstep(1.0 - FEATHER, 1.0, abs(in.edge));
    let w = feather * clamp(in.strength, 0.0, 1.0);
    // Accumulate the weighted smear direction (rg) and the weight (b). Additive blend.
    return vec4<f32>(in.smear * w, w, w);
}
