//! Two-pass crisp-perimeter render path.
//!
//! The default [`crate::pipelines::SplatPipeline`] composites each splat's hardened Gaussian
//! with `over` blending. That makes individual splats crisp but cannot make the *line*
//! perimeter crisp: the union of N translucent crisp ellipses scallops and its edge opacity
//! wobbles with overlap/draw order. This module computes the silhouette as a property of the
//! whole splat union instead:
//!
//!   1. **Accumulate** (`coverage_accum.wgsl`) every splat into two offscreen fields — an
//!      additive premultiplied color and a `Max`-blended coverage scalar.
//!   2. **Resolve** (`coverage_resolve.wgsl`) thresholds the accumulated coverage *once* for
//!      a single crisp antialiased perimeter, and normalizes the accumulated color for a
//!      fuzzy interior.
//!
//! The accumulate pipeline shares the splat-pass bind-group shape (storage splats + camera),
//! so it draws the same instance buffers; only the targets and fragment differ.

/// Accumulated premultiplied interior color. 16-bit float is renderable + blendable
/// (additive), and holds the summed weight in alpha for the resolve-pass normalize.
pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Silhouette coverage field (single channel). 16-bit float supports `Max` blending so the
/// field is the per-pixel maximum geometric Gaussian over all splats.
pub const COVERAGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;

/// Offscreen color + coverage targets for the accumulate pass. Recreated when the output
/// size changes; otherwise reused across frames.
pub struct CoverageTargets {
    pub color_view: wgpu::TextureView,
    pub coverage_view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl CoverageTargets {
    fn create(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let make = |label: &str, format: wgpu::TextureFormat| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: width.max(1),
                        height: height.max(1),
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
        Self {
            color_view: make("coverage color target", COLOR_FORMAT),
            coverage_view: make("coverage silhouette target", COVERAGE_FORMAT),
            width: width.max(1),
            height: height.max(1),
        }
    }

    fn matches(&self, width: u32, height: u32) -> bool {
        self.width == width.max(1) && self.height == height.max(1)
    }
}

/// Holds the accumulate + resolve pipelines and lazily-sized offscreen targets.
pub struct CoveragePipeline {
    pub accum_bind_group_layout: wgpu::BindGroupLayout,
    pub accum_pipeline: wgpu::RenderPipeline,
    pub resolve_bind_group_layout: wgpu::BindGroupLayout,
    pub resolve_pipeline: wgpu::RenderPipeline,
    targets: Option<CoverageTargets>,
}

impl CoveragePipeline {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let accum_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("coverage accum shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/coverage_accum.wgsl").into(),
            ),
        });
        let resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("coverage resolve shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/coverage_resolve.wgsl").into(),
            ),
        });

        // --- Accumulate pipeline: same bind-group shape as the splat pass. ---
        let accum_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("coverage accum bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let accum_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("coverage accum pipeline layout"),
            bind_group_layouts: &[Some(&accum_bind_group_layout)],
            immediate_size: 0,
        });

        // Additive accumulation of premultiplied color + weight.
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
        // Per-pixel maximum of the geometric coverage (factors are ignored for Max).
        let max_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Max,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Max,
            },
        };

        let accum_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("coverage accum pipeline"),
            layout: Some(&accum_layout),
            vertex: wgpu::VertexState {
                module: &accum_shader,
                entry_point: Some("vs_accum"),
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
                module: &accum_shader,
                entry_point: Some("fs_accum"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: COLOR_FORMAT,
                        blend: Some(additive),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: COVERAGE_FORMAT,
                        blend: Some(max_blend),
                        write_mask: wgpu::ColorWrites::RED,
                    }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });

        // --- Resolve pipeline: full-screen triangle reading the two fields. ---
        let resolve_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("coverage resolve bind group layout"),
                entries: &[
                    texture_entry(0),
                    texture_entry(1),
                ],
            });

        let resolve_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("coverage resolve pipeline layout"),
            bind_group_layouts: &[Some(&resolve_bind_group_layout)],
            immediate_size: 0,
        });

        // Premultiplied-alpha over blending so the resolved line composites over the cleared
        // background in the final target.
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

        let resolve_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("coverage resolve pipeline"),
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

        Self {
            accum_bind_group_layout,
            accum_pipeline,
            resolve_bind_group_layout,
            resolve_pipeline,
            targets: None,
        }
    }

    /// Ensure the offscreen targets exist at `width`×`height`, recreating only on resize.
    /// Split from [`Self::current_targets`] so callers can hold an immutable borrow of the
    /// targets while also immutably borrowing the pipelines/bind-group builder.
    pub fn ensure(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let stale = self.targets.as_ref().map(|t| !t.matches(width, height)).unwrap_or(true);
        if stale {
            self.targets = Some(CoverageTargets::create(device, width, height));
        }
    }

    /// The current targets. Panics if [`Self::ensure`] has not been called yet.
    pub fn current_targets(&self) -> &CoverageTargets {
        self.targets.as_ref().expect("CoveragePipeline::ensure must precede current_targets")
    }

    /// Build the resolve-pass bind group (color + coverage textures) for the current targets.
    pub fn resolve_bind_group(&self, device: &wgpu::Device, targets: &CoverageTargets) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("coverage resolve bind group"),
            layout: &self.resolve_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&targets.color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&targets.coverage_view),
                },
            ],
        })
    }
}

/// A read-only float texture binding sampled via `textureLoad` (no sampler, so filterability
/// is irrelevant — `false` is the portable choice for 16-bit float targets).
fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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
