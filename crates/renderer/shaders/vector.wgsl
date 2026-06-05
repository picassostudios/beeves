// Conventional vector path rendering.
//
// Unlike the splat shaders, this draws an ordinary triangle mesh: the CPU tessellates a
// stroke's Bézier skeleton into a stroked ribbon (centerline ± half-width), and this shader
// transforms it to clip space and antialiases the two long edges. `edge` runs -1 (left rim)
// → 0 (centerline) → +1 (right rim); the fragment shader fades a ~1px band at |edge| = 1 so
// the silhouette is crisp at any zoom. Output is premultiplied alpha (paired with
// One / OneMinusSrcAlpha blending), matching the splat path.

struct Camera {
    // clip.xy = world.xy * scale + offset
    scale: vec2<f32>,
    offset: vec2<f32>,
    // params.x = zoom (pixels-per-world-unit); rest reserved.
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

struct VsIn {
    @location(0) pos: vec2<f32>,   // world-space vertex
    @location(1) edge: f32,        // -1..1 across the stroke width
    @location(2) color: u32,       // packed RGBA8 (alpha already folded in)
};

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) edge: f32,
    @location(1) color: vec4<f32>, // interpolated so width/opacity profiles vary smoothly
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.position = vec4<f32>(in.pos * camera.scale + camera.offset, 0.0, 1.0);
    out.edge = in.edge;
    out.color = unpack4x8unorm(in.color); // .xyz = rgb, .w = alpha
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Distance (in `edge` units) from the nearest long rim, converted to pixels via the
    // screen-space derivative of `edge`, for a resolution-independent ~1px antialiased edge.
    let dist = 1.0 - abs(in.edge);
    let aa = max(fwidth(in.edge), 1e-5);
    let coverage = clamp(dist / aa + 0.5, 0.0, 1.0);

    let a = clamp(in.color.w * coverage, 0.0, 1.0);
    // Premultiplied alpha.
    return vec4<f32>(in.color.xyz * a, a);
}
