// Full-screen blit: composite an offscreen texture onto the current target 1:1.
//
// Used by the vector-blend path to lay the offscreen vector-stroke layer back onto the view
// before the smear pass draws over it. A single full-screen triangle samples the source at the
// matching texel centre (so linear filtering is a pass-through, no blur) and emits it unchanged;
// the pipeline's premultiplied-alpha `over` blend composites it onto the view.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_blit(@builtin(vertex_index) vi: u32) -> VsOut {
    // The classic oversized triangle covering the viewport: uv spans [0,2] so the clipped
    // [0,1] region maps exactly onto the target.
    var out: VsOut;
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    out.uv = vec2<f32>(x, y);
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

@fragment
fn fs_blit(in: VsOut) -> @location(0) vec4<f32> {
    // The source is already premultiplied alpha (the vector pass outputs rgb*a, a).
    return textureSampleLevel(src, samp, in.uv, 0.0);
}
