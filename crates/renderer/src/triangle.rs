//! Triangle-splat render pipelines (2D Triangle Splatting). Twin of `convex.rs`: same
//! GpuSplat instance data, whitening, and crisp-path reuse — only the window function differs.
//!
//! A *triangle splat* is the same instance data as a Gaussian splat
//! ([`app_core::splat::GpuSplat`], unchanged) drawn with the kernel from "Triangle Splatting for
//! Real-Time Radiance Field Rendering" (Held, Vandeghen et al., 2025, arXiv:2505.19175),
//! realized in 2D. The primitive is 3 vertices {V0,V1,V2} in the splat's whitened σ-unit frame,
//! so it inherits the stroke's orientation + anisotropy. Its signed distance field is the TRUE
//! max of the three edge half-plane distances (the paper rejects LogSumExp here), and the
//! differentiable window function `I(p) = ReLU(φ/φ(s))^σ` peaks at the incenter and vanishes at
//! the boundary (see `shaders/triangle_splat.wgsl`). This module provides the two triangle
//! pipelines — a direct per-splat pipeline (twin of [`crate::convex::ConvexSplatPipeline`]) and a
//! coverage-accumulate pipeline (twin of [`crate::convex::ConvexAccumPipeline`]). The crisp-path
//! *resolve* pass and its offscreen targets are mode-agnostic, so the triangle crisp path reuses
//! them from [`crate::coverage::CoveragePipeline`] verbatim.

use crate::coverage::{COLOR_FORMAT, COVERAGE_FORMAT};

/// Template circumradius in σ-units (vertices sit here in the whitened frame; the splat's
/// ~3σ `radius` quad covers them). Matches the convex template radius.
const TRI_RADIUS: f32 = 2.4;

/// GPU-side triangle uniform (48 bytes), mirrored by `Triangle` in the triangle shaders.
/// - `verts`: V0=verts[0].xy, V1=verts[0].zw, V2=verts[1].xy (whitened σ-unit frame).
/// - `meta = [φ(s), σ, 0, 0]`: incenter SDF value (negative, precomputed) and window smoothness.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TriangleUniform {
    pub verts: [[f32; 4]; 2],
    pub meta: [f32; 4],
}

impl TriangleUniform {
    /// Equilateral triangle, circumradius `TRI_RADIUS`, rotated by `rotation` radians (0 = apex
    /// up). `sigma` is the window smoothness exponent, clamped to a shader-safe range. φ(s) is
    /// precomputed from the vertices as −inradius (inradius = 2·Area/perimeter).
    pub fn new(rotation: f32, sigma: f32) -> Self {
        let base = std::f32::consts::FRAC_PI_2; // apex up at rotation = 0
        let mut v = [[0.0f32; 2]; 3];
        for (i, vert) in v.iter_mut().enumerate() {
            let ang = rotation + base + i as f32 * std::f32::consts::TAU / 3.0;
            *vert = [TRI_RADIUS * ang.cos(), TRI_RADIUS * ang.sin()];
        }
        let side = |a: [f32; 2], b: [f32; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
        let perim = side(v[1], v[2]) + side(v[2], v[0]) + side(v[0], v[1]);
        let area = 0.5
            * ((v[1][0] - v[0][0]) * (v[2][1] - v[0][1])
                - (v[2][0] - v[0][0]) * (v[1][1] - v[0][1]))
                .abs();
        let phi_s = -(2.0 * area / perim.max(1e-6)); // −inradius
        Self {
            verts: [[v[0][0], v[0][1], v[1][0], v[1][1]], [v[2][0], v[2][1], 0.0, 0.0]],
            meta: [phi_s, sigma.clamp(0.02, 8.0), 0.0, 0.0],
        }
    }
}

impl Default for TriangleUniform {
    fn default() -> Self {
        Self::new(0.0, 0.4)
    }
}

/// Bind-group layout shared by both triangle pipelines: storage splats (0), camera uniform (1),
/// triangle uniform (2). Identical shape to the convex layout plus the triangle uniform.
fn triangle_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
        label: Some("triangle bind group layout"),
        entries: &[
            buffer(0, wgpu::BufferBindingType::Storage { read_only: true }),
            buffer(1, wgpu::BufferBindingType::Uniform),
            buffer(2, wgpu::BufferBindingType::Uniform),
        ],
    })
}

/// The direct triangle-splat render pipeline (premultiplied-alpha `over` blend) plus its
/// bind-group layout. Twin of [`crate::convex::ConvexSplatPipeline`].
pub struct TriangleSplatPipeline {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub pipeline: wgpu::RenderPipeline,
}

impl TriangleSplatPipeline {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("triangle splat shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/triangle_splat.wgsl").into(),
            ),
        });

        let bind_group_layout = triangle_bind_group_layout(device);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("triangle splat pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Premultiplied-alpha over blending — identical to the Gaussian/convex pipeline.
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
            label: Some("triangle splat pipeline"),
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

/// The triangle coverage-*accumulate* pipeline (pass 1 of the crisp-perimeter path) plus its
/// bind-group layout. Writes the same two offscreen targets as the Gaussian/convex accum
/// pipeline (additive premultiplied color + `Max` coverage), so the shared resolve pass consumes
/// it unchanged. Twin of [`crate::convex::ConvexAccumPipeline`].
pub struct TriangleAccumPipeline {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub pipeline: wgpu::RenderPipeline,
}

impl TriangleAccumPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("triangle coverage accum shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/triangle_coverage_accum.wgsl").into(),
            ),
        });

        let bind_group_layout = triangle_bind_group_layout(device);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("triangle coverage accum pipeline layout"),
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
            label: Some("triangle coverage accum pipeline"),
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

    /// Unpack triangle vertex `i` (mirrors `tri_vert` in the shader).
    fn vert(u: &TriangleUniform, i: usize) -> (f32, f32) {
        match i {
            0 => (u.verts[0][0], u.verts[0][1]),
            1 => (u.verts[0][2], u.verts[0][3]),
            _ => (u.verts[1][0], u.verts[1][1]),
        }
    }

    #[test]
    fn incenter_phi_is_negative_inradius() {
        let u = TriangleUniform::new(0.0, 0.4);
        assert!(u.meta[0] < 0.0, "φ(s) must be negative (inside the triangle)");
        // Inradius of an equilateral triangle with circumradius R is R/2 = 1.2.
        assert!(
            (u.meta[0] - (-1.2)).abs() < 1e-3,
            "φ(s) should be ≈ −1.2, got {}",
            u.meta[0]
        );
    }

    #[test]
    fn vertices_lie_on_the_template_circle() {
        let u = TriangleUniform::new(0.7, 0.4);
        for i in 0..3 {
            let (x, y) = vert(&u, i);
            assert!(
                (x.hypot(y) - TRI_RADIUS).abs() < 1e-4,
                "vertex {i} must lie on the template circle (r ≈ {TRI_RADIUS}), got {}",
                x.hypot(y)
            );
        }
    }

    #[test]
    fn sigma_is_clamped_to_shader_safe_range() {
        assert!(TriangleUniform::new(0.0, 100.0).meta[1] <= 8.0, "σ clamps to a finite max");
        assert!(TriangleUniform::new(0.0, 0.0).meta[1] >= 0.02, "σ clamps to a positive floor");
    }
}
