//! Conventional vector-path rendering: tessellate a stroke's Bézier skeleton into a stroked
//! ribbon and draw it as an antialiased triangle mesh, instead of as Gaussian splats.
//!
//! This is the render path for strokes flagged `render_as_vector` (the vector-draw tool). The
//! mesh is rebuilt on the CPU each frame from the skeleton — vector strokes are clean line art
//! (few, low-vertex), so per-frame tessellation is cheap and keeps the curve smooth at any
//! zoom (sample density scales with on-screen arc length). See `shaders/vector.wgsl`.

use app_core::document::Document;
use app_core::splat::pack_rgba8;
use app_core::stroke::GaussianBezierStroke;
use glam::Vec2;

/// One tessellated vertex of a stroked vector path. 16-byte stride, no padding (Pod-safe).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VectorVertex {
    /// World-space position.
    pub pos: [f32; 2],
    /// Cross-section coordinate: -1 at the left rim, 0 on the centerline, +1 at the right rim.
    pub edge: f32,
    /// Packed RGBA8 with the stroke's effective opacity already folded into the alpha byte.
    pub color: u32,
}

/// Render pipeline + growable vertex buffer for vector paths. The camera uniform is bound
/// externally (the renderer owns it and shares it across pipelines).
pub struct VectorPathPipeline {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub pipeline: wgpu::RenderPipeline,
    vertices: wgpu::Buffer,
    capacity: usize,
    len: usize,
}

impl VectorPathPipeline {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vector shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/vector.wgsl").into()),
        });

        // Just the camera uniform at binding 0 (consumed by the vertex stage).
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vector bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vector pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Premultiplied-alpha over blending — identical to the splat path so vector lines
        // composite the same way.
        let blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<VectorVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 8,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint32,
                    offset: 12,
                    shader_location: 2,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vector pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[vertex_layout],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let capacity = 256;
        let vertices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vector vertex buffer"),
            size: (capacity * std::mem::size_of::<VectorVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            bind_group_layout,
            pipeline,
            vertices,
            capacity,
            len: 0,
        }
    }

    /// Upload `verts` to the vertex buffer, growing it (with slack) only when capacity is
    /// exceeded. `len` then reflects how many vertices to draw.
    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, verts: &[VectorVertex]) {
        self.len = verts.len();
        if verts.is_empty() {
            return;
        }
        if verts.len() > self.capacity {
            // Grow with slack (×2) so repeated per-frame growth amortizes to O(1) reallocs.
            let new_cap = (verts.len() * 2).max(256);
            self.vertices = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vector vertex buffer"),
                size: (new_cap * std::mem::size_of::<VectorVertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.capacity = new_cap;
        }
        queue.write_buffer(&self.vertices, 0, bytemuck::cast_slice(verts));
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.vertices
    }

    pub fn vertex_count(&self) -> u32 {
        self.len as u32
    }
}

/// Append the tessellated stroked path for one vector stroke to `out`: a quad ribbon along
/// the centerline, round caps at both ends, and round joins at sharp corners.
///
/// The skeleton is sampled along arc length (sample count scales with on-screen length so
/// curves stay smooth at any zoom); each sample is offset by ±half-width along the curve
/// normal to form the ribbon. Width follows the brush's `width_profile`.
///
/// The fill is **solid and opaque**: vector line art is conventionally opaque, and — more
/// importantly — the ribbon is a *union* of overlapping primitives (the body quads, the round
/// caps, and a round-join disc at every sharp corner). Overlap is only invisible when each
/// primitive is fully opaque; a translucent fill (or a translucent antialiased rim) would darken
/// at every joint and shed thin slivers wherever two primitives' edges cross. So the brush
/// opacity slider is intentionally not applied here (only the colour's own alpha, normally 1.0),
/// and the silhouette is **not** feathered in the fragment shader. Instead the whole layer is
/// drawn into a multisampled target and resolved (see `VectorPathPipeline::encode`), which
/// antialiases the *true outer silhouette* of the union without ever making an interior edge
/// translucent. The `edge` channel (−1/0/+1 across the width, 0→1 radially in a disc) is retained
/// only for the secondary blend path, which still draws single-sampled.
pub fn tessellate_stroke(stroke: &GaussianBezierStroke, zoom: f32, out: &mut Vec<VectorVertex>) {
    let sk = &stroke.skeleton;
    if sk.segments.is_empty() {
        return;
    }
    let brush = &stroke.brush;
    let length_px = (sk.total_length() * zoom.max(1e-3)).max(1.0);
    // ~1 rib per 3 screen px, clamped so tiny strokes still get a few ribs and huge ones
    // don't explode the vertex count.
    let ribs = ((length_px / 3.0).ceil() as usize).clamp(2, 4096);

    let c = brush.base_color;
    let color = pack_rgba8([c[0], c[1], c[2], c[3].clamp(0.0, 1.0)]);

    // Sample the centerline once.
    let mut pos = Vec::with_capacity(ribs);
    let mut nrm = Vec::with_capacity(ribs);
    let mut half = Vec::with_capacity(ribs);
    for i in 0..ribs {
        let t = i as f32 / (ribs - 1) as f32;
        let frame = sk.frame_at_arc_t(t);
        pos.push(frame.position);
        nrm.push(frame.normal);
        half.push((brush.radius * brush.width_profile.eval(t)).max(0.05));
    }

    // Body: one quad between each consecutive rib (the smooth interior never self-overlaps).
    for i in 0..ribs - 1 {
        push_quad(
            out,
            pos[i] - nrm[i] * half[i],
            pos[i] + nrm[i] * half[i],
            pos[i + 1] + nrm[i + 1] * half[i + 1],
            pos[i + 1] - nrm[i + 1] * half[i + 1],
            color,
        );
    }

    // Round caps: a full disc at each end. The inward half overlaps the body harmlessly
    // (opaque fill); the outward half is the round cap.
    let last = ribs - 1;
    push_disc(out, pos[0], half[0], color, disc_segments(half[0] * zoom));
    push_disc(out, pos[last], half[last], color, disc_segments(half[last] * zoom));

    // Round joins: a disc wherever the centerline turns sharply (a fitter corner). Smooth
    // curves turn only a few degrees per dense step, so no disc is emitted along them.
    const JOIN_COS: f32 = 0.985; // ~10°
    for i in 1..last {
        let d0 = (pos[i] - pos[i - 1]).normalize_or_zero();
        let d1 = (pos[i + 1] - pos[i]).normalize_or_zero();
        if d0 != Vec2::ZERO && d1 != Vec2::ZERO && d0.dot(d1) < JOIN_COS {
            push_disc(out, pos[i], half[i], color, disc_segments(half[i] * zoom));
        }
    }
}

/// Push a width quad (two triangles) between two ribs. `al`/`bl` are the left rim (edge -1),
/// `ar`/`br` the right rim (edge +1).
fn push_quad(out: &mut Vec<VectorVertex>, al: Vec2, ar: Vec2, br: Vec2, bl: Vec2, color: u32) {
    let v = |p: Vec2, e: f32| VectorVertex { pos: [p.x, p.y], edge: e, color };
    out.push(v(al, -1.0));
    out.push(v(ar, 1.0));
    out.push(v(br, 1.0));
    out.push(v(al, -1.0));
    out.push(v(br, 1.0));
    out.push(v(bl, -1.0));
}

/// Push a filled disc as a triangle fan (`segments` wedges). The center carries edge 0 and the
/// rim edge 1.
fn push_disc(out: &mut Vec<VectorVertex>, center: Vec2, radius: f32, color: u32, segments: usize) {
    let c = VectorVertex { pos: [center.x, center.y], edge: 0.0, color };
    for k in 0..segments {
        let a0 = (k as f32 / segments as f32) * std::f32::consts::TAU;
        let a1 = ((k + 1) as f32 / segments as f32) * std::f32::consts::TAU;
        let p0 = center + Vec2::new(a0.cos(), a0.sin()) * radius;
        let p1 = center + Vec2::new(a1.cos(), a1.sin()) * radius;
        out.push(c);
        out.push(VectorVertex { pos: [p0.x, p0.y], edge: 1.0, color });
        out.push(VectorVertex { pos: [p1.x, p1.y], edge: 1.0, color });
    }
}

/// Triangle-fan resolution for a disc of on-screen radius `radius_px` — enough wedges that the
/// rim reads as round without spending vertices on tiny dabs.
fn disc_segments(radius_px: f32) -> usize {
    (radius_px.ceil() as usize).clamp(8, 48)
}

/// Tessellate every plain `render_as_vector` stroke in `doc` into `out` (cleared first).
/// Vector-*blend* strokes (which also set `render_as_vector`) are excluded — they carry no
/// colour of their own and are drawn by the dedicated smear pass (`crate::vector_blend`).
pub fn tessellate_document(doc: &Document, zoom: f32, out: &mut Vec<VectorVertex>) {
    out.clear();
    for stroke in doc.strokes.values() {
        if stroke.render_as_vector && !stroke.vector_blend {
            tessellate_stroke(stroke, zoom, out);
        }
    }
}
