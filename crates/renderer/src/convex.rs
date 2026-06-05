//! Convex-splat render pipelines.
//!
//! A *convex splat* is the same instance data as a Gaussian splat ([`app_core::splat::GpuSplat`],
//! unchanged) drawn with a different kernel: its level sets are smooth convex polygons instead
//! of ellipses (see `shaders/convex_splat.wgsl`). This module provides the two convex
//! pipelines — a direct per-splat pipeline (twin of [`crate::pipelines::SplatPipeline`]) and a
//! coverage-accumulate pipeline (twin of the accum half of [`crate::coverage::CoveragePipeline`]).
//! The crisp-path *resolve* pass and its offscreen targets are mode-agnostic, so the convex
//! crisp path reuses them from [`crate::coverage::CoveragePipeline`] verbatim.
//!
//! The convex pipelines add one binding over the Gaussian ones: a [`ConvexUniform`] at
//! binding 2 carrying the polygon shape (sides, sharpness, rotation, corner scale).

use crate::coverage::{COLOR_FORMAT, COVERAGE_FORMAT};

/// Maximum hull vertices the shader (and the packed uniform) support.
pub const MAX_CONVEX_POINTS: usize = 8;

/// Template hull-vertex radius in σ-units. The vertices sit at this radius in the splat's
/// whitened frame; the splat's `radius` (≈3σ) quad comfortably covers them plus the σ tail.
const CONVEX_RADIUS: f32 = 2.4;

/// GPU-side convex-primitive uniform (80 bytes), mirrored by the `Convex` struct in the convex
/// shaders. This is a faithful 2D realization of 3D Convex Splatting (3DCS / CvxNet): the
/// primitive is the convex hull of a **point set**, rendered from its edge lines via a
/// LogSumExp smooth signed distance and a sigmoid indicator — not a covariance-derived ellipse.
///
/// - `points`: up to [`MAX_CONVEX_POINTS`] hull vertices in CCW order, packed two per `vec4`
///   (`v0 = points[0].xy`, `v1 = points[0].zw`, …), in the whitened/local σ-unit frame.
/// - `meta = [K, δ, σ, 0]`: live vertex count, smoothness `δ` (rounds the corners; large →
///   hard polygon), and sharpness `σ` (edge transition; large → dense/crisp boundary).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ConvexUniform {
    pub points: [[f32; 4]; 4],
    pub meta: [f32; 4],
}

impl ConvexUniform {
    /// Build the uniform for a regular `sides`-gon hull with smoothness `delta` and sharpness
    /// `sigma`. `sides` is clamped to `[3, MAX_CONVEX_POINTS]`; `delta`/`sigma` to shader-safe
    /// positive ranges. (The representation is a free point set, so an irregular hull is also
    /// expressible — a regular polygon is just the natural brush default.)
    pub fn new(sides: f32, delta: f32, sigma: f32) -> Self {
        let k = (sides.round().clamp(3.0, MAX_CONVEX_POINTS as f32)) as usize;
        let mut points = [[0.0f32; 4]; 4];
        for i in 0..k {
            // (2i+1)π/K spaces the vertices so an even-K hull has a vertex on each diagonal
            // (e.g. an axis-aligned square's corners sit on the ±45° diagonals); CCW as i grows.
            let ang = std::f32::consts::PI * (2.0 * i as f32 + 1.0) / k as f32;
            let (x, y) = (CONVEX_RADIUS * ang.cos(), CONVEX_RADIUS * ang.sin());
            let slot = &mut points[i / 2];
            if i % 2 == 0 {
                slot[0] = x;
                slot[1] = y;
            } else {
                slot[2] = x;
                slot[3] = y;
            }
        }
        Self {
            points,
            meta: [k as f32, delta.clamp(0.5, 60.0), sigma.clamp(0.5, 200.0), 0.0],
        }
    }
}

impl Default for ConvexUniform {
    /// A crisp hexagon: sharp-ish corners (high δ) and a dense, hard edge (high σ).
    fn default() -> Self {
        Self::new(6.0, 22.0, 48.0)
    }
}

/// Bind-group layout shared by both convex pipelines: storage splats (0), camera uniform (1),
/// convex uniform (2). Identical shape to [`crate::pipelines::SplatPipeline`]'s layout plus the
/// convex uniform.
fn convex_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let buffer = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("convex bind group layout"),
        entries: &[
            buffer(0, wgpu::BufferBindingType::Storage { read_only: true }),
            buffer(1, wgpu::BufferBindingType::Uniform),
            buffer(2, wgpu::BufferBindingType::Uniform),
        ],
    })
}

/// The direct convex-splat render pipeline (premultiplied-alpha `over` blend) plus its
/// bind-group layout. Twin of [`crate::pipelines::SplatPipeline`].
pub struct ConvexSplatPipeline {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub pipeline: wgpu::RenderPipeline,
}

impl ConvexSplatPipeline {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("convex splat shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/convex_splat.wgsl").into()),
        });

        let bind_group_layout = convex_bind_group_layout(device);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("convex splat pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Premultiplied-alpha over blending — identical to the Gaussian pipeline.
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

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("convex splat pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
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

        Self {
            bind_group_layout,
            pipeline,
        }
    }
}

/// The convex coverage-*accumulate* pipeline (pass 1 of the crisp-perimeter path) plus its
/// bind-group layout. Writes the same two offscreen targets as the Gaussian accum pipeline
/// (additive premultiplied color + `Max` coverage), so the shared resolve pass consumes it
/// unchanged. Twin of the accum half of [`crate::coverage::CoveragePipeline`].
pub struct ConvexAccumPipeline {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub pipeline: wgpu::RenderPipeline,
}

impl ConvexAccumPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("convex coverage accum shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/convex_coverage_accum.wgsl").into(),
            ),
        });

        let bind_group_layout = convex_bind_group_layout(device);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("convex coverage accum pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
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

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("convex coverage accum pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
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
                module: &shader,
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

        Self {
            bind_group_layout,
            pipeline,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unpack hull vertex `i` from the two-per-vec4 packing (mirrors `cvx_vert` in the shader).
    fn vert(u: &ConvexUniform, i: usize) -> (f32, f32) {
        let v = u.points[i / 2];
        if i % 2 == 0 {
            (v[0], v[1])
        } else {
            (v[2], v[3])
        }
    }

    #[test]
    fn stores_k_ccw_vertices_on_the_template_circle() {
        let u = ConvexUniform::new(5.0, 22.0, 48.0);
        assert_eq!(u.meta[0], 5.0, "K is stored in meta.x");
        // Every live vertex sits on the σ-unit template circle, and they wind CCW (the signed
        // area of the polygon is positive).
        let mut area = 0.0f32;
        for i in 0..5 {
            let (x, y) = vert(&u, i);
            assert!((x.hypot(y) - CONVEX_RADIUS).abs() < 1e-4, "vertex on template circle");
            let (nx, ny) = vert(&u, (i + 1) % 5);
            area += x * ny - nx * y;
        }
        assert!(area > 0.0, "vertices must wind counter-clockwise, got signed area {area}");
    }

    #[test]
    fn params_are_clamped_to_shader_safe_ranges() {
        let u = ConvexUniform::new(2.0, 1000.0, 1000.0); // sides too low, δ/σ too high
        assert_eq!(u.meta[0], 3.0, "sides clamp up to the triangle minimum");
        assert!(u.meta[1] <= 60.0 && u.meta[2] <= 200.0, "δ/σ clamp to finite maxima");
        let v = ConvexUniform::new(99.0, 0.0, 0.0); // sides too high, δ/σ too low
        assert_eq!(v.meta[0], MAX_CONVEX_POINTS as f32, "sides clamp down to the max hull size");
        assert!(v.meta[1] >= 0.5 && v.meta[2] >= 0.5, "δ/σ clamp to positive floors");
    }
}
