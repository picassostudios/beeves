//! GPU object-id picking pass.
//!
//! Renders splat identities (`stroke_id`) into an offscreen single-channel integer
//! texture (`R32Uint`), then reads back the id under a given pixel. This lets the
//! editor answer "which stroke is under the cursor?" with the *same* coverage that
//! the visible splat pass produces (same vertex expansion, same `q > 9` cutoff),
//! rather than re-deriving hit-testing on the CPU.
//!
//! The pipeline is structurally a sibling of [`crate::pipelines::SplatPipeline`] but
//! with no blending, an integer color target, and a fragment that writes ids instead
//! of premultiplied color.

use app_core::splat::GpuSplat;

use crate::buffers::{self, SplatBuffer};
use crate::camera::CameraUniform;
use crate::gpu::GpuContext;

/// The integer format used for the picking target. `R32Uint` holds the full 32-bit
/// `stroke_id` in a single 4-byte texel, which keeps the readback row math identical
/// to the `Rgba8Unorm` case (4 bytes per texel).
pub const PICKING_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;

/// Sentinel id written to texels not covered by any splat (the clear value). `stroke_id`
/// is a dense per-frame index starting at 0, so `u32::MAX` is safely "no stroke".
pub const NO_ID: u32 = u32::MAX;

/// The picking render pipeline plus its bind-group layout (storage buffer of splats +
/// camera uniform). Mirrors [`crate::pipelines::SplatPipeline`].
pub struct PickingPipeline {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub pipeline: wgpu::RenderPipeline,
}

impl PickingPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("picking shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/picking.wgsl").into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("picking bind group layout"),
                entries: &[
                    // binding 0: read-only storage buffer of splats.
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
                    // binding 1: camera uniform.
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("picking pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("picking pipeline"),
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
                // No blending: integer targets cannot blend, and ids must be written
                // verbatim.
                targets: &[Some(wgpu::ColorTargetState {
                    format: PICKING_FORMAT,
                    blend: None,
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

/// Renders splat ids to an offscreen integer target and reads back the id under a single
/// pixel. Owns the pipeline + reusable GPU buffers, mirroring `SplatRenderer`.
pub struct PickingRenderer {
    pipeline: PickingPipeline,
    splats: SplatBuffer,
    camera_buffer: wgpu::Buffer,
}

impl PickingRenderer {
    pub fn new(device: &wgpu::Device) -> Self {
        let pipeline = PickingPipeline::new(device);
        let splats = SplatBuffer::new(device, &[]);
        let camera_buffer = buffers::camera_buffer(device, &CameraUniform::default());
        Self {
            pipeline,
            splats,
            camera_buffer,
        }
    }

    /// Render `splats` id-coverage into `view`, clearing to [`NO_ID`] first.
    ///
    /// `view` must be a view of an `R32Uint` ([`PICKING_FORMAT`]) texture.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        splats: &[GpuSplat],
        camera: &CameraUniform,
    ) {
        self.splats.update(device, queue, splats);
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("picking bind group"),
            layout: &self.pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.splats.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.camera_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("picking encoder"),
        });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("picking pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Integer targets clear with a u32 component value: the sentinel.
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: NO_ID as f64,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&self.pipeline.pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..6, 0..self.splats.len as u32);
        }
        queue.submit(Some(encoder.finish()));
    }
}

/// Create an offscreen [`PICKING_FORMAT`] target usable as a color attachment and copy
/// source for picking readback.
pub fn create_picking_target(ctx: &GpuContext, width: u32, height: u32) -> wgpu::Texture {
    ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("picking target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: PICKING_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// Round `value` up to the next multiple of `align`.
fn align_up(value: u32, align: u32) -> u32 {
    value.div_ceil(align) * align
}

/// Copy the whole [`PICKING_FORMAT`] picking texture back to the CPU as a tightly-packed
/// `width*height` array of ids (`u32` per texel). Handles the 256-byte `bytes_per_row`
/// alignment requirement. `R32Uint` is 4 bytes per texel, matching the RGBA8 row math.
pub fn read_picking_ids(
    ctx: &GpuContext,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Vec<u32> {
    const BYTES_PER_TEXEL: u32 = 4; // R32Uint
    let unpadded_bytes_per_row = width * BYTES_PER_TEXEL;
    let padded_bytes_per_row = align_up(unpadded_bytes_per_row, 256);

    let out_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("picking readback"),
        size: (padded_bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("picking readback encoder"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &out_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    ctx.queue.submit(Some(encoder.finish()));

    let slice = out_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll failed");
    rx.recv().expect("map channel").expect("map failed");

    let data = slice.get_mapped_range();
    let mut out = Vec::with_capacity((width * height) as usize);
    for row in 0..height {
        let start = (row * padded_bytes_per_row) as usize;
        let end = start + unpadded_bytes_per_row as usize;
        // Each row is `width` little-endian u32s, tightly packed within the row.
        for chunk in data[start..end].chunks_exact(4) {
            out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
    }
    drop(data);
    out_buffer.unmap();
    out
}

/// The scene to pick against in a one-shot [`pick_id_at`] call.
pub struct PickRequest<'a> {
    pub target: &'a wgpu::Texture,
    pub splats: &'a [GpuSplat],
    pub camera: &'a CameraUniform,
    /// Pixel to sample (origin top-left).
    pub x: u32,
    pub y: u32,
}

/// Render the picking pass and read back the id at a single pixel (origin top-left).
/// Returns [`NO_ID`] if the pixel is outside the viewport or covered by no splat.
/// `target` dimensions define the viewport. This is the one-shot convenience used by
/// cursor hit-testing.
pub fn pick_id_at(
    ctx: &GpuContext,
    renderer: &mut PickingRenderer,
    req: PickRequest<'_>,
) -> u32 {
    let width = req.target.width();
    let height = req.target.height();
    if req.x >= width || req.y >= height {
        return NO_ID;
    }
    let view = req
        .target
        .create_view(&wgpu::TextureViewDescriptor::default());
    renderer.render(&ctx.device, &ctx.queue, &view, req.splats, req.camera);
    let ids = read_picking_ids(ctx, req.target, width, height);
    ids[(req.y * width + req.x) as usize]
}
