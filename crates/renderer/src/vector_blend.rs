//! Vector-blend (directional smear) render path.
//!
//! A *blend stroke* is a vector path that paints no colour of its own. Instead it samples the
//! plain vector-stroke layer underneath it and smears those colours along the path tangent — a
//! live, non-destructive smudge whose region is described by a vector ribbon. This is distinct
//! from the gaussian [`app_core::blend`] tool, which destructively rewrites splat colours in the
//! document model; here nothing in the model changes and the smear is recomputed from the layer
//! each frame.
//!
//! Because core WebGPU cannot read the destination framebuffer in a fragment shader, the smear is
//! a four-pass dance (orchestrated in [`crate::SplatRenderer::render_vector_paths`]):
//!
//!   1. render the plain vector strokes into an offscreen colour texture (the "layer");
//!   2. blit the layer onto the view (so plain vector strokes still appear);
//!   3. **accumulate** the blend ribbons into an offscreen mask with *additive* blending —
//!      storing a feathered weight and the weight-scaled smear direction;
//!   4. **resolve** once per pixel: recover the averaged direction + soft coverage from the mask,
//!      sample the layer along ±direction, and composite the average over the view.
//!
//! The accumulate→resolve split (mirroring the crisp coverage path) is deliberate: drawing the
//! ribbon's many overlapping triangles straight onto the view with `over` blending double-
//! composites every overlap, which shows up as hard round-cap rings and a woven cross-hatch where
//! strokes cross. Accumulating then saturating removes that, and feathering the weight gives the
//! mark a soft edge.
//!
//! This module owns the offscreen layer + mask textures, the blit/accumulate/resolve pipelines,
//! the ribbon vertex buffer, and the tessellation that turns a blend stroke into ribbon geometry
//! carrying a per-vertex tangent. Shaders: `shaders/blit.wgsl`, `shaders/vector_blend.wgsl`
//! (accumulate), `shaders/vector_blend_resolve.wgsl` (resolve).

use app_core::document::Document;
use app_core::stroke::GaussianBezierStroke;
use glam::Vec2;

/// Smear half-length as a multiple of the stroke half-width (world units). The fragment shader
/// samples the layer over ±this distance along the path tangent, so a wider blend stroke smears
/// over a correspondingly longer run.
const SMEAR_FACTOR: f32 = 2.0;

/// One tessellated vertex of a blend ribbon. 24-byte stride, no padding (Pod-safe).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VectorBlendVertex {
    /// World-space position.
    pub pos: [f32; 2],
    /// Cross-section coordinate: -1/0/+1 across the ribbon width (drives rim antialiasing).
    pub edge: f32,
    /// World-space path tangent scaled to the smear half-length (`unit_tangent * half_width *
    /// SMEAR_FACTOR`); the shader maps it to a uv-space sampling offset.
    pub tangent: [f32; 2],
    /// Blend strength in `[0,1]` (the smear's composite opacity).
    pub strength: f32,
}

/// Accumulation-mask format: signed (the smear direction lands in rg) and additively blendable.
const MASK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Lazily-sized offscreen targets: the plain vector-stroke `layer` (sampled by the resolve pass)
/// and the additive smear `mask` (direction + weight). Both track the render-target size.
struct Targets {
    layer: wgpu::TextureView,
    mask: wgpu::TextureView,
    width: u32,
    height: u32,
}

/// Blit + accumulate + resolve pipelines, the offscreen layer/mask textures, and a growable
/// ribbon vertex buffer. The camera uniform is owned by the renderer and bound externally.
pub struct VectorBlendPipeline {
    blit_bind_group_layout: wgpu::BindGroupLayout,
    blit_pipeline: wgpu::RenderPipeline,
    accum_bind_group_layout: wgpu::BindGroupLayout,
    accum_pipeline: wgpu::RenderPipeline,
    resolve_bind_group_layout: wgpu::BindGroupLayout,
    resolve_pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    format: wgpu::TextureFormat,
    targets: Option<Targets>,
    vertices: wgpu::Buffer,
    capacity: usize,
    len: usize,
}

impl VectorBlendPipeline {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/blit.wgsl").into()),
        });
        let blend_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vector blend shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/vector_blend.wgsl").into()),
        });

        // Premultiplied-alpha over — identical to the splat and plain-vector paths.
        let over = wgpu::BlendState {
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

        // --- Blit pipeline: full-screen triangle sampling the layer (no vertex buffers). ---
        let blit_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("blit bind group layout"),
                entries: &[filterable_texture_entry(0), sampler_entry(1)],
            });
        let blit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit pipeline layout"),
            bind_group_layouts: &[Some(&blit_bind_group_layout)],
            immediate_size: 0,
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit pipeline"),
            layout: Some(&blit_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_blit"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_blit"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(over),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        // --- Accumulate pipeline: ribbon mesh -> mask, additive (direction + weight). ---
        let accum_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vector blend accum bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    // The vertex stage transforms positions and maps the tangent to uv; the
                    // fragment stage only needs the interpolated edge/smear/strength.
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let accum_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vector blend accum pipeline layout"),
            bind_group_layouts: &[Some(&accum_bind_group_layout)],
            immediate_size: 0,
        });

        // Additive accumulation of the weighted direction + weight.
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<VectorBlendVertex>() as u64,
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
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 12,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 20,
                    shader_location: 3,
                },
            ],
        };

        let accum_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vector blend accum pipeline"),
            layout: Some(&accum_layout),
            vertex: wgpu::VertexState {
                module: &blend_shader,
                entry_point: Some("vs_accum"),
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
                module: &blend_shader,
                entry_point: Some("fs_accum"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: MASK_FORMAT,
                    blend: Some(additive),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        // --- Resolve pipeline: full-screen triangle reading mask + layer, smearing once. ---
        let resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vector blend resolve shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/vector_blend_resolve.wgsl").into(),
            ),
        });
        let resolve_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vector blend resolve bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        // The fragment stage reads the render-target size from params.zw.
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    filterable_texture_entry(1),
                    sampler_entry(2),
                    // The mask is read via textureLoad (no sampler), so filterability is moot.
                    nonfilterable_texture_entry(3),
                ],
            });
        let resolve_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vector blend resolve pipeline layout"),
            bind_group_layouts: &[Some(&resolve_bind_group_layout)],
            immediate_size: 0,
        });
        let resolve_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vector blend resolve pipeline"),
            layout: Some(&resolve_layout),
            vertex: wgpu::VertexState {
                module: &resolve_shader,
                entry_point: Some("vs_resolve"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &resolve_shader,
                entry_point: Some("fs_resolve"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(over),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        // Linear filtering with clamped addressing so taps that fall outside the layer read the
        // edge texel rather than wrapping.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("vector blend sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let capacity = 256;
        let vertices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vector blend vertex buffer"),
            size: (capacity * std::mem::size_of::<VectorBlendVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            blit_bind_group_layout,
            blit_pipeline,
            accum_bind_group_layout,
            accum_pipeline,
            resolve_bind_group_layout,
            resolve_pipeline,
            sampler,
            format: target_format,
            targets: None,
            vertices,
            capacity,
            len: 0,
        }
    }

    /// Ensure the offscreen layer + mask textures exist at `width`×`height`, recreating only on
    /// resize.
    pub fn ensure(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        let stale = self
            .targets
            .as_ref()
            .map(|t| t.width != width || t.height != height)
            .unwrap_or(true);
        if stale {
            let make = |label: &str, format: wgpu::TextureFormat| {
                device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some(label),
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    })
                    .create_view(&wgpu::TextureViewDescriptor::default())
            };
            self.targets = Some(Targets {
                layer: make("vector blend layer", self.format),
                mask: make("vector blend mask", MASK_FORMAT),
                width,
                height,
            });
        }
    }

    fn targets(&self) -> &Targets {
        self.targets
            .as_ref()
            .expect("VectorBlendPipeline::ensure must precede target access")
    }

    /// The offscreen vector-layer view (render target for the plain vectors; sampled by resolve).
    pub fn layer_view(&self) -> &wgpu::TextureView {
        &self.targets().layer
    }

    /// The offscreen smear-mask view (additive accumulate target; read by resolve).
    pub fn mask_view(&self) -> &wgpu::TextureView {
        &self.targets().mask
    }

    pub fn blit_pipeline(&self) -> &wgpu::RenderPipeline {
        &self.blit_pipeline
    }

    pub fn accum_pipeline(&self) -> &wgpu::RenderPipeline {
        &self.accum_pipeline
    }

    pub fn resolve_pipeline(&self) -> &wgpu::RenderPipeline {
        &self.resolve_pipeline
    }

    /// Bind group for the blit pass: the layer texture (0) + sampler (1).
    pub fn blit_bind_group(&self, device: &wgpu::Device, layer: &wgpu::TextureView) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit bind group"),
            layout: &self.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(layer),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    /// Bind group for the accumulate pass: just the camera uniform at binding 0 (vertex stage).
    pub fn accum_bind_group(&self, device: &wgpu::Device, camera: &wgpu::Buffer) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vector blend accum bind group"),
            layout: &self.accum_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera.as_entire_binding(),
            }],
        })
    }

    /// Bind group for the resolve pass: camera (0) + layer texture (1) + sampler (2) + mask (3).
    pub fn resolve_bind_group(&self, device: &wgpu::Device, camera: &wgpu::Buffer) -> wgpu::BindGroup {
        let targets = self.targets();
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vector blend resolve bind group"),
            layout: &self.resolve_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&targets.layer),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&targets.mask),
                },
            ],
        })
    }

    /// Upload `verts` to the smear vertex buffer, growing it (with slack) only when capacity is
    /// exceeded. `len` then reflects how many vertices to draw. Mirrors [`crate::VectorPathPipeline::upload`].
    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, verts: &[VectorBlendVertex]) {
        self.len = verts.len();
        if verts.is_empty() {
            return;
        }
        if verts.len() > self.capacity {
            let new_cap = (verts.len() * 2).max(256);
            self.vertices = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vector blend vertex buffer"),
                size: (new_cap * std::mem::size_of::<VectorBlendVertex>()) as u64,
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

/// A filterable 2D float texture binding (sampled with a linear sampler).
fn filterable_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// A non-filterable 2D float texture binding (read via `textureLoad`, no sampler).
fn nonfilterable_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// A filtering sampler binding.
fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

/// Tessellate every `vector_blend` stroke in `doc` into `out` (cleared first).
pub fn tessellate_blend_document(doc: &Document, zoom: f32, out: &mut Vec<VectorBlendVertex>) {
    out.clear();
    for stroke in doc.strokes.values() {
        if stroke.vector_blend {
            tessellate_blend_stroke(stroke, zoom, out);
        }
    }
}

/// Append the tessellated ribbon for one blend stroke to `out`: a single quad ribbon along the
/// centerline, every vertex carrying the local path tangent (scaled to the smear half-length) and
/// an effective blend strength.
///
/// Unlike [`crate::vector::tessellate_stroke`], this emits **no round caps or joins**. Those are
/// triangle fans that carry one constant smear direction, which would create hard direction
/// discontinuities against the smoothly-interpolated body — and because the resolve pass does a
/// long, high-contrast directional sample, any direction break shows up as a jagged shard. The
/// body ribbon's tangent interpolates continuously rib-to-rib (C0 across every shared edge), so
/// the direction field is smooth everywhere. Soft *ends* come from tapering the strength to zero
/// over a short run at each end (`end_fade`) rather than from cap discs.
pub fn tessellate_blend_stroke(
    stroke: &GaussianBezierStroke,
    zoom: f32,
    out: &mut Vec<VectorBlendVertex>,
) {
    let sk = &stroke.skeleton;
    if sk.segments.is_empty() {
        return;
    }
    let brush = &stroke.brush;
    let strength = stroke.blend_strength.clamp(0.0, 1.0);
    let length_px = (sk.total_length() * zoom.max(1e-3)).max(1.0);
    let ribs = ((length_px / 3.0).ceil() as usize).clamp(2, 4096);

    // Sample the centerline once: position, normal, tangent, and half-width per rib.
    let mut pos = Vec::with_capacity(ribs);
    let mut nrm = Vec::with_capacity(ribs);
    let mut tan = Vec::with_capacity(ribs);
    let mut half = Vec::with_capacity(ribs);
    for i in 0..ribs {
        let t = i as f32 / (ribs - 1) as f32;
        let frame = sk.frame_at_arc_t(t);
        pos.push(frame.position);
        nrm.push(frame.normal);
        tan.push(frame.tangent);
        half.push((brush.radius * brush.width_profile.eval(t)).max(0.05));
    }

    // End taper: fade strength to 0 over a short run (~ the half-width, in ribs) at each end so
    // the mark has soft ends. Capped to 40% of the span so even a short stroke keeps a full-
    // strength core. `ribs >= 2`, so `ribs - 1 >= 1` and the span is well-defined.
    let half_px = brush.radius.max(0.05) * zoom;
    let max_span = (0.4 * (ribs - 1) as f32).max(1.0);
    let fade_span = (half_px / 3.0).clamp(1.0, max_span);
    let end_fade = |i: usize| -> f32 {
        let head = smoothstep01(i as f32 / fade_span);
        let tail = smoothstep01((ribs - 1 - i) as f32 / fade_span);
        head.min(tail)
    };

    // Body ribbon: one quad between consecutive ribs.
    for i in 0..ribs - 1 {
        push_quad(
            out,
            pos[i] - nrm[i] * half[i],
            pos[i] + nrm[i] * half[i],
            pos[i + 1] + nrm[i + 1] * half[i + 1],
            pos[i + 1] - nrm[i + 1] * half[i + 1],
            smear(tan[i], half[i]),
            smear(tan[i + 1], half[i + 1]),
            strength * end_fade(i),
            strength * end_fade(i + 1),
        );
    }
}

/// The per-vertex smear vector: the unit tangent scaled to the smear half-length.
fn smear(tangent: Vec2, half: f32) -> Vec2 {
    tangent.normalize_or_zero() * half * SMEAR_FACTOR
}

/// Smooth Hermite step on `[0,1]` (clamps first). Used for the end-of-stroke strength taper.
fn smoothstep01(x: f32) -> f32 {
    let t = x.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Push a width quad (two triangles) between two ribs. `al`/`bl` are the left rim (edge -1),
/// `ar`/`br` the right rim (edge +1). Rib i's corners (`al`,`ar`) carry `smear_i`/`strength_i`;
/// rib i+1's (`br`,`bl`) carry `smear_j`/`strength_j`. The two triangles share the `al`–`br`
/// diagonal, and interpolation along it depends only on those two shared corners, so the payload
/// is continuous across the diagonal (no crease) and across shared ribs (no inter-quad seam).
#[allow(clippy::too_many_arguments)]
fn push_quad(
    out: &mut Vec<VectorBlendVertex>,
    al: Vec2,
    ar: Vec2,
    br: Vec2,
    bl: Vec2,
    smear_i: Vec2,
    smear_j: Vec2,
    strength_i: f32,
    strength_j: f32,
) {
    let v = |p: Vec2, e: f32, s: Vec2, st: f32| VectorBlendVertex {
        pos: [p.x, p.y],
        edge: e,
        tangent: [s.x, s.y],
        strength: st,
    };
    out.push(v(al, -1.0, smear_i, strength_i));
    out.push(v(ar, 1.0, smear_i, strength_i));
    out.push(v(br, 1.0, smear_j, strength_j));
    out.push(v(al, -1.0, smear_i, strength_i));
    out.push(v(br, 1.0, smear_j, strength_j));
    out.push(v(bl, -1.0, smear_j, strength_j));
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_core::brush::BrushModel;
    use app_core::{BezierSkeleton, CubicBezier};

    fn straight(a: Vec2, b: Vec2) -> CubicBezier {
        CubicBezier::new(a, a + (b - a) / 3.0, a + (b - a) * (2.0 / 3.0), b)
    }

    /// A document with one plain vector stroke and one blend stroke; the tessellator must pick
    /// only the blend stroke, and only when `vector_blend` is set.
    #[test]
    fn tessellate_document_selects_only_blend_strokes() {
        let mut doc = Document::new();
        let layer = doc.add_layer("L");
        let brush = BrushModel { radius: 8.0, ..BrushModel::default() };
        // Plain vector stroke (render_as_vector but not a blend) — must be ignored here.
        let plain = doc.add_stroke(
            layer,
            BezierSkeleton::single(straight(Vec2::new(0.0, 0.0), Vec2::new(100.0, 0.0))),
            brush.clone(),
        );
        doc.stroke_mut(plain).unwrap().render_as_vector = true;
        // Blend stroke.
        let blend = doc.add_stroke(
            layer,
            BezierSkeleton::single(straight(Vec2::new(0.0, 50.0), Vec2::new(100.0, 50.0))),
            brush,
        );
        {
            let s = doc.stroke_mut(blend).unwrap();
            s.render_as_vector = true;
            s.vector_blend = true;
        }

        let mut out = Vec::new();
        tessellate_blend_document(&doc, 1.0, &mut out);
        assert!(!out.is_empty(), "the blend stroke should produce ribbon vertices");

        // Drop the blend flag: now nothing is tessellated.
        doc.stroke_mut(blend).unwrap().vector_blend = false;
        tessellate_blend_document(&doc, 1.0, &mut out);
        assert!(out.is_empty(), "no blend strokes => no vertices");
    }

    /// Every emitted vertex carries a finite, non-degenerate smear vector aligned with the path,
    /// and the stroke's blend strength is folded into the vertices.
    #[test]
    fn blend_vertices_carry_a_tangent_aligned_smear() {
        let mut doc = Document::new();
        let layer = doc.add_layer("L");
        // A horizontal stroke => tangent ~ +x, so the smear vector should be ~ (±, 0).
        let brush = BrushModel { radius: 8.0, ..BrushModel::default() };
        let sid = doc.add_stroke(
            layer,
            BezierSkeleton::single(straight(Vec2::new(0.0, 0.0), Vec2::new(200.0, 0.0))),
            brush,
        );
        {
            let s = doc.stroke_mut(sid).unwrap();
            s.render_as_vector = true;
            s.vector_blend = true;
            s.blend_strength = 0.5;
        }

        let mut out = Vec::new();
        tessellate_blend_document(&doc, 1.0, &mut out);
        assert!(!out.is_empty());

        let expected_len = 8.0 * SMEAR_FACTOR; // half-width * factor
        let mut max_strength = 0.0_f32;
        for v in &out {
            assert!(v.pos[0].is_finite() && v.pos[1].is_finite());
            let smear = Vec2::new(v.tangent[0], v.tangent[1]);
            assert!(smear.length() > 1e-3, "smear vector must be non-degenerate");
            // Horizontal path: the smear is essentially along x.
            assert!(smear.x.abs() > smear.y.abs(), "smear should follow the tangent (x)");
            // Length matches half-width * SMEAR_FACTOR everywhere (body-only, no cap variance).
            assert!(
                (smear.length() - expected_len).abs() < 1.0,
                "smear length {} should be ~{expected_len}",
                smear.length()
            );
            // The end taper scales strength down at the ends, so it spans (0, blend_strength].
            assert!(
                v.strength >= 0.0 && v.strength <= 0.5 + 1e-6,
                "strength {} out of range (end taper of 0.5)",
                v.strength
            );
            max_strength = max_strength.max(v.strength);
        }
        // The full-strength core reaches the stroke's blend strength.
        assert!(
            (max_strength - 0.5).abs() < 1e-3,
            "the core should reach the full blend strength 0.5, got {max_strength}"
        );
    }
}
