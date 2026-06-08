//! wgpu/WebGPU renderer for the Gaussian-Bezier splat field.
//!
//! The renderer is deliberately a thin, stateless-ish pass over GPU instances built from
//! `app_core::splat::GpuSplat`. The editable model in `app_core` is the source of truth;
//! GPU buffers here are disposable caches. The same pipeline/shader path runs headless on
//! native (for tests/thumbnails) and against a browser surface in `app_wasm`.

pub mod buffers;
pub mod camera;
pub mod convex;
pub mod coverage;
pub mod gpu;
pub mod picking;
pub mod pipelines;
pub mod scene;
pub mod triangle;
pub mod vector;
pub mod vector_blend;

pub use camera::{Camera2D, CameraUniform};
pub use convex::ConvexUniform;
pub use gpu::GpuContext;
pub use scene::SceneLayout;
pub use triangle::TriangleUniform;
pub use vector::{VectorPathPipeline, VectorVertex};
pub use vector_blend::{VectorBlendPipeline, VectorBlendVertex};

use app_core::document::Document;
use app_core::splat::GpuSplat;
use buffers::SplatBuffer;
use convex::{ConvexAccumPipeline, ConvexSplatPipeline};
use pipelines::SplatPipeline;
use triangle::{TriangleAccumPipeline, TriangleSplatPipeline};
use scene::SceneLayout as Layout;

/// Initial resident-buffer capacity (in splats). Sized so a handful of strokes fit before
/// the first growth; growth doubles, so this only affects the very first frames.
const MIN_RESIDENT_CAPACITY: usize = 256;

/// Create a zeroed, GPU-resident storage buffer sized for `cap_splats` instances.
fn create_resident_buffer(device: &wgpu::Device, cap_splats: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("resident splat buffer"),
        size: (cap_splats.max(1) * std::mem::size_of::<GpuSplat>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Draws instanced Gaussian splats into a color target.
///
/// Two paths share the pipeline: [`SplatRenderer::render_doc`] is the incremental, resident
/// fast-path used by the app (reconciles + uploads only what changed, draws only what's in
/// view); [`SplatRenderer::render`] is an immediate-mode path over a raw slice, kept for the
/// headless shader tests.
pub struct SplatRenderer {
    pipeline: SplatPipeline,
    /// Convex-splat twin of `pipeline` (smooth convex-polygon kernel). Selected per-frame by
    /// the `render_doc_convex*` paths; shares the resident buffer + camera uniform.
    convex: ConvexSplatPipeline,
    /// Two-pass crisp-perimeter path (accumulate coverage + resolve). Owns its own offscreen
    /// targets; see [`coverage`]. The resolve pass is mode-agnostic and shared by both kernels.
    coverage: coverage::CoveragePipeline,
    /// Convex coverage-accumulate pipeline (pass 1 of the crisp path, convex kernel). Reuses
    /// `coverage`'s offscreen targets and resolve pass.
    convex_accum: ConvexAccumPipeline,
    /// Convex-shape uniform (sides/sharpness/rotation/corner-scale), updated via
    /// [`Self::set_convex_params`]; bound at binding 2 by the convex pipelines.
    convex_buffer: wgpu::Buffer,
    /// Triangle-splat twin of `convex` (2D Triangle Splatting kernel). Selected per-frame by
    /// the `render_doc_triangle*` paths; shares the resident buffer + camera uniform.
    triangle: TriangleSplatPipeline,
    /// Triangle coverage-accumulate pipeline (pass 1 of the crisp path, triangle kernel). Reuses
    /// `coverage`'s offscreen targets and resolve pass.
    triangle_accum: TriangleAccumPipeline,
    /// Triangle-shape uniform (φ(s)/σ), updated via [`Self::set_triangle_params`]; bound at
    /// binding 2 by the triangle pipelines.
    triangle_buffer: wgpu::Buffer,
    /// Immediate-mode transient buffer for `render(&[GpuSplat])`.
    splats: SplatBuffer,
    camera_buffer: wgpu::Buffer,
    /// Persistent CPU layout mirror for the incremental `render_doc` path.
    layout: Layout,
    /// Persistent GPU buffer backing `render_doc`; grown with slack, never rebuilt per frame.
    resident: wgpu::Buffer,
    resident_capacity: usize,
    /// Conventional vector-path pipeline + growable vertex buffer, used to draw strokes flagged
    /// `render_as_vector` as antialiased stroked outlines on top of the splat field.
    vector: VectorPathPipeline,
    /// Reused per-frame tessellation scratch so the vector pass never allocates.
    vector_scratch: Vec<VectorVertex>,
    /// Vector-blend (directional smear) pipelines + offscreen vector layer + smear vertex buffer.
    /// Used to draw `vector_blend` strokes, which sample the plain vector layer beneath them.
    vector_blend: VectorBlendPipeline,
    /// Reused per-frame smear tessellation scratch so the blend pass never allocates.
    blend_scratch: Vec<VectorBlendVertex>,
}

impl SplatRenderer {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let pipeline = SplatPipeline::new(device, target_format);
        let convex = ConvexSplatPipeline::new(device, target_format);
        let coverage = coverage::CoveragePipeline::new(device, target_format);
        let convex_accum = ConvexAccumPipeline::new(device);
        let triangle = TriangleSplatPipeline::new(device, target_format);
        let triangle_accum = TriangleAccumPipeline::new(device);
        let vector = VectorPathPipeline::new(device, target_format);
        let vector_blend = VectorBlendPipeline::new(device, target_format);
        let splats = SplatBuffer::new(device, &[]);
        let camera_buffer = buffers::camera_buffer(device, &CameraUniform::default());
        let convex_buffer = buffers::convex_buffer(device, &ConvexUniform::default());
        let triangle_buffer = buffers::triangle_buffer(device, &TriangleUniform::default());
        let resident_capacity = MIN_RESIDENT_CAPACITY;
        let resident = create_resident_buffer(device, resident_capacity);
        Self {
            pipeline,
            convex,
            coverage,
            convex_accum,
            convex_buffer,
            triangle,
            triangle_accum,
            triangle_buffer,
            splats,
            camera_buffer,
            layout: Layout::default(),
            resident,
            resident_capacity,
            vector,
            vector_scratch: Vec::new(),
            vector_blend,
            blend_scratch: Vec::new(),
        }
    }

    /// Update the convex-primitive uniform (no-op for the Gaussian paths). Builds a regular
    /// `sides`-gon hull with smoothness `delta` (rounds the corners; higher → harder polygon)
    /// and sharpness `sigma` (edge transition; higher → denser/crisper boundary) — the two
    /// decoupled knobs from 3D Convex Splatting. Cheap; call only when the shape changes.
    pub fn set_convex_params(&self, queue: &wgpu::Queue, sides: f32, delta: f32, sigma: f32) {
        let u = ConvexUniform::new(sides, delta, sigma);
        queue.write_buffer(&self.convex_buffer, 0, bytemuck::bytes_of(&u));
    }

    /// Update the triangle-primitive uniform (no-op for the Gaussian/convex paths). Builds an
    /// equilateral triangle (circumradius in σ-units) rotated by `rotation` radians (0 = apex up)
    /// with window smoothness `sigma` (σ→0 ⇒ a solid top-hat triangle, larger σ ⇒ a soft falloff
    /// peaking at the incenter) — the window function from 2D Triangle Splatting. Cheap; call
    /// only when the shape changes.
    pub fn set_triangle_params(&self, queue: &wgpu::Queue, rotation: f32, sigma: f32) {
        let u = TriangleUniform::new(rotation, sigma);
        queue.write_buffer(&self.triangle_buffer, 0, bytemuck::bytes_of(&u));
    }

    /// Incrementally render the whole document into `view`, clearing to `clear` first.
    ///
    /// Reconciles the resident buffer against `doc` — uploading only new/edited strokes via
    /// `queue.write_buffer` at their stable offsets — then issues one draw per view-visible
    /// stroke range (contiguous ranges merged). A camera-only frame uploads no splat data;
    /// only the camera uniform and the O(strokes) visibility sweep run. `view_min`/`view_max`
    /// is the visible world-space rectangle (used for coarse per-stroke culling).
    /// Reconcile the resident buffer against `doc` (uploading only new/edited stroke slices)
    /// and upload the camera uniform. Shared by [`Self::render_doc`] and
    /// [`Self::render_doc_crisp`]; leaves the GPU buffers ready, draws nothing.
    fn upload_doc(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        doc: &mut Document,
        camera: &CameraUniform,
    ) {
        let dirty = self.layout.reconcile(doc);
        let needed = self.layout.len();

        if needed > self.resident_capacity {
            // Grow with slack so repeated appends amortize to O(1). The whole mirror is
            // re-uploaded once here, which subsumes the per-stroke dirty ranges.
            let new_cap = (needed * 2).max(MIN_RESIDENT_CAPACITY);
            self.resident = create_resident_buffer(device, new_cap);
            self.resident_capacity = new_cap;
            queue.write_buffer(&self.resident, 0, bytemuck::cast_slice(self.layout.mirror()));
        } else {
            let stride = std::mem::size_of::<GpuSplat>();
            for (off, len) in dirty {
                let slice = &self.layout.mirror()[off..off + len];
                queue.write_buffer(
                    &self.resident,
                    (off * stride) as u64,
                    bytemuck::cast_slice(slice),
                );
            }
        }

        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_doc(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        doc: &mut Document,
        camera: &CameraUniform,
        view_min: glam::Vec2,
        view_max: glam::Vec2,
        clear: wgpu::Color,
    ) {
        self.upload_doc(device, queue, doc, camera);
        let bind_group = self.splat_bind_group(device, &self.resident);
        let ranges = self.layout.visible_ranges(view_min, view_max);
        encode_resident_pass(
            device,
            queue,
            view,
            &self.pipeline.pipeline,
            &bind_group,
            &ranges,
            clear,
        );
    }

    /// Convex-kernel twin of [`Self::render_doc`]: identical resident upload + view culling,
    /// but draws through the convex-splat pipeline so the document renders as smooth convex
    /// polygons. The convex shape comes from the uniform set by [`Self::set_convex_params`].
    #[allow(clippy::too_many_arguments)]
    pub fn render_doc_convex(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        doc: &mut Document,
        camera: &CameraUniform,
        view_min: glam::Vec2,
        view_max: glam::Vec2,
        clear: wgpu::Color,
    ) {
        self.upload_doc(device, queue, doc, camera);
        let bind_group = self.convex_bind_group(device, &self.resident);
        let ranges = self.layout.visible_ranges(view_min, view_max);
        encode_resident_pass(
            device,
            queue,
            view,
            &self.convex.pipeline,
            &bind_group,
            &ranges,
            clear,
        );
    }

    /// Triangle-kernel twin of [`Self::render_doc`]: identical resident upload + view culling,
    /// but draws through the triangle-splat pipeline so the document renders as 2D triangle
    /// splats. The triangle shape comes from the uniform set by [`Self::set_triangle_params`].
    #[allow(clippy::too_many_arguments)]
    pub fn render_doc_triangle(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        doc: &mut Document,
        camera: &CameraUniform,
        view_min: glam::Vec2,
        view_max: glam::Vec2,
        clear: wgpu::Color,
    ) {
        self.upload_doc(device, queue, doc, camera);
        let bind_group = self.triangle_bind_group(device, &self.resident);
        let ranges = self.layout.visible_ranges(view_min, view_max);
        encode_resident_pass(
            device,
            queue,
            view,
            &self.triangle.pipeline,
            &bind_group,
            &ranges,
            clear,
        );
    }

    /// Draw the document's vector strokes on top of whatever is already in `view` (it **loads**,
    /// never clears). Two kinds of vector stroke are handled:
    ///
    ///   * plain `render_as_vector` strokes — tessellated, antialiased stroked paths;
    ///   * `vector_blend` strokes — ribbons that carry no colour of their own and instead
    ///     directionally smear the plain vector layer beneath them (see [`crate::vector_blend`]).
    ///
    /// When the document has no blend strokes this is the cheap single-pass path (draw the plain
    /// vectors straight onto the view). When it does, the plain vectors are first rendered into an
    /// offscreen layer so the smear pass can sample them: render layer → blit layer onto view →
    /// draw the smear ribbons sampling the layer. Tessellation runs on the CPU each frame (vector
    /// strokes are few, clean line art). Call *after* the splat pass and before presenting; a
    /// document with no vector strokes of either kind is a no-op (no pass is encoded).
    pub fn render_vector_paths(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        doc: &Document,
        camera: &CameraUniform,
    ) {
        let zoom = camera.params[0];
        vector::tessellate_document(doc, zoom, &mut self.vector_scratch);
        vector_blend::tessellate_blend_document(doc, zoom, &mut self.blend_scratch);
        let has_vectors = !self.vector_scratch.is_empty();
        let has_blend = !self.blend_scratch.is_empty();
        if !has_vectors && !has_blend {
            return;
        }
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));

        if !has_blend {
            // Fast path: no smear strokes, so draw the plain vectors directly onto the view.
            self.vector.upload(device, queue, &self.vector_scratch);
            self.encode_vector_only(device, queue, view);
            return;
        }

        // Smear strokes present: the plain vector layer must live in a sampleable texture. Size
        // the offscreen layer to the render target (carried in the camera uniform's params.zw).
        let w = camera.params[2].max(1.0) as u32;
        let h = camera.params[3].max(1.0) as u32;
        self.vector_blend.ensure(device, w, h);
        self.vector.upload(device, queue, &self.vector_scratch);
        self.vector_blend.upload(device, queue, &self.blend_scratch);
        self.encode_blend(device, queue, view, has_vectors);
    }

    /// Single-pass plain-vector draw: composite the tessellated ribbons onto `view` (load).
    fn encode_vector_only(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
    ) {
        let bind_group = self.vector_camera_bind_group(device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vector encoder"),
        });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vector pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&self.vector.pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.set_vertex_buffer(0, self.vector.buffer().slice(..));
            rpass.draw(0..self.vector.vertex_count(), 0..1);
        }
        queue.submit(Some(encoder.finish()));
    }

    /// Four-pass smear draw: (1) render the plain vectors into the offscreen layer, (2) blit the
    /// layer onto `view`, (3) accumulate the blend ribbons into the offscreen mask (additive
    /// direction + weight), (4) resolve the mask against the layer once and composite the smear
    /// onto `view`. Buffers must already be uploaded and the targets sized via
    /// [`vector_blend::VectorBlendPipeline::ensure`].
    fn encode_blend(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        has_vectors: bool,
    ) {
        let layer = self.vector_blend.layer_view();
        let mask = self.vector_blend.mask_view();
        let vec_bind_group = self.vector_camera_bind_group(device);
        let blit_bind_group = self.vector_blend.blit_bind_group(device, layer);
        let accum_bind_group = self.vector_blend.accum_bind_group(device, &self.camera_buffer);
        let resolve_bind_group = self.vector_blend.resolve_bind_group(device, &self.camera_buffer);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vector blend encoder"),
        });

        // Pass 1: plain vectors -> offscreen layer (cleared transparent so untouched texels read
        // as empty for the smear taps).
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vector layer pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: layer,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if has_vectors {
                rpass.set_pipeline(&self.vector.pipeline);
                rpass.set_bind_group(0, &vec_bind_group, &[]);
                rpass.set_vertex_buffer(0, self.vector.buffer().slice(..));
                rpass.draw(0..self.vector.vertex_count(), 0..1);
            }
        }

        // Pass 2: blit the layer onto the view so the plain vectors still appear.
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vector blit pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(self.vector_blend.blit_pipeline());
            rpass.set_bind_group(0, &blit_bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        // Pass 3: accumulate the blend ribbons into the mask (additive, cleared transparent).
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vector blend accum pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: mask,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(self.vector_blend.accum_pipeline());
            rpass.set_bind_group(0, &accum_bind_group, &[]);
            rpass.set_vertex_buffer(0, self.vector_blend.buffer().slice(..));
            rpass.draw(0..self.vector_blend.vertex_count(), 0..1);
        }

        // Pass 4: resolve the mask against the layer once and composite the smear onto the view.
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vector blend resolve pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(self.vector_blend.resolve_pipeline());
            rpass.set_bind_group(0, &resolve_bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        queue.submit(Some(encoder.finish()));
    }

    /// Bind group for the plain-vector pipeline: just the shared camera uniform at binding 0.
    fn vector_camera_bind_group(&self, device: &wgpu::Device) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vector bind group"),
            layout: &self.vector.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.camera_buffer.as_entire_binding(),
            }],
        })
    }

    /// Drop resident scene state. Call when the document is replaced wholesale (e.g. on
    /// load) so slots from the previous document don't linger.
    pub fn reset_scene(&mut self) {
        self.layout.clear();
    }

    /// Immediate-mode render of a raw `splats` slice into `view`, clearing to `clear` first.
    /// Used by the headless shader tests; the app uses [`Self::render_doc`].
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        splats: &[GpuSplat],
        camera: &CameraUniform,
        clear: wgpu::Color,
    ) {
        self.splats.update(device, queue, splats);
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));
        let bind_group = self.splat_bind_group(device, &self.splats.buffer);
        encode_resident_pass(
            device,
            queue,
            view,
            &self.pipeline.pipeline,
            &bind_group,
            &full_range(self.splats.len),
            clear,
        );
    }

    /// Immediate-mode convex twin of [`Self::render`] (headless tests / thumbnails). The shape
    /// comes from the uniform set by [`Self::set_convex_params`].
    pub fn render_convex(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        splats: &[GpuSplat],
        camera: &CameraUniform,
        clear: wgpu::Color,
    ) {
        self.splats.update(device, queue, splats);
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));
        let bind_group = self.convex_bind_group(device, &self.splats.buffer);
        encode_resident_pass(
            device,
            queue,
            view,
            &self.convex.pipeline,
            &bind_group,
            &full_range(self.splats.len),
            clear,
        );
    }

    /// Immediate-mode triangle twin of [`Self::render`] (headless tests / thumbnails). The shape
    /// comes from the uniform set by [`Self::set_triangle_params`].
    pub fn render_triangle(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        splats: &[GpuSplat],
        camera: &CameraUniform,
        clear: wgpu::Color,
    ) {
        self.splats.update(device, queue, splats);
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));
        let bind_group = self.triangle_bind_group(device, &self.splats.buffer);
        encode_resident_pass(
            device,
            queue,
            view,
            &self.triangle.pipeline,
            &bind_group,
            &full_range(self.splats.len),
            clear,
        );
    }

    /// Crisp-perimeter twin of [`Self::render_doc`]: runs the two-pass coverage path
    /// (accumulate → resolve) so the line's silhouette is one crisp antialiased edge while
    /// its interior keeps per-splat color fuzz. `target_width`/`target_height` are the output
    /// view's pixel size (used to size the offscreen accumulation targets).
    #[allow(clippy::too_many_arguments)]
    pub fn render_doc_crisp(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        doc: &mut Document,
        camera: &CameraUniform,
        view_min: glam::Vec2,
        view_max: glam::Vec2,
        target_width: u32,
        target_height: u32,
        clear: wgpu::Color,
    ) {
        self.upload_doc(device, queue, doc, camera);
        let ranges = self.layout.visible_ranges(view_min, view_max);
        self.coverage.ensure(device, target_width, target_height);
        let accum_bind_group = self.coverage_accum_bind_group(device, &self.resident);
        encode_crisp_passes(
            device,
            queue,
            &self.coverage,
            &self.coverage.accum_pipeline,
            &accum_bind_group,
            &ranges,
            view,
            clear,
        );
    }

    /// Convex-kernel twin of [`Self::render_doc_crisp`]: the two-pass coverage path with the
    /// convex accumulate shader, so a convex-splat line gets one crisp (flat-sided) perimeter
    /// while its interior keeps per-splat color fuzz. The convex direct path and this crisp
    /// path together give convex splatting the same blending/crisp toolset as the Gaussian.
    #[allow(clippy::too_many_arguments)]
    pub fn render_doc_convex_crisp(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        doc: &mut Document,
        camera: &CameraUniform,
        view_min: glam::Vec2,
        view_max: glam::Vec2,
        target_width: u32,
        target_height: u32,
        clear: wgpu::Color,
    ) {
        self.upload_doc(device, queue, doc, camera);
        let ranges = self.layout.visible_ranges(view_min, view_max);
        self.coverage.ensure(device, target_width, target_height);
        let accum_bind_group = self.convex_accum_bind_group(device, &self.resident);
        encode_crisp_passes(
            device,
            queue,
            &self.coverage,
            &self.convex_accum.pipeline,
            &accum_bind_group,
            &ranges,
            view,
            clear,
        );
    }

    /// Triangle-kernel twin of [`Self::render_doc_crisp`]: the two-pass coverage path with the
    /// triangle accumulate shader, so a triangle-splat line gets one crisp perimeter (tracking
    /// the triangle silhouette) while its interior keeps per-splat color fuzz.
    #[allow(clippy::too_many_arguments)]
    pub fn render_doc_triangle_crisp(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        doc: &mut Document,
        camera: &CameraUniform,
        view_min: glam::Vec2,
        view_max: glam::Vec2,
        target_width: u32,
        target_height: u32,
        clear: wgpu::Color,
    ) {
        self.upload_doc(device, queue, doc, camera);
        let ranges = self.layout.visible_ranges(view_min, view_max);
        self.coverage.ensure(device, target_width, target_height);
        let accum_bind_group = self.triangle_accum_bind_group(device, &self.resident);
        encode_crisp_passes(
            device,
            queue,
            &self.coverage,
            &self.triangle_accum.pipeline,
            &accum_bind_group,
            &ranges,
            view,
            clear,
        );
    }

    /// Immediate-mode crisp render of a raw `splats` slice (headless tests / thumbnails). The
    /// resident `render_doc_crisp` is the app path; this mirrors [`Self::render`].
    #[allow(clippy::too_many_arguments)]
    pub fn render_crisp(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        splats: &[GpuSplat],
        camera: &CameraUniform,
        width: u32,
        height: u32,
        clear: wgpu::Color,
    ) {
        self.splats.update(device, queue, splats);
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));
        self.coverage.ensure(device, width, height);
        let accum_bind_group = self.coverage_accum_bind_group(device, &self.splats.buffer);
        encode_crisp_passes(
            device,
            queue,
            &self.coverage,
            &self.coverage.accum_pipeline,
            &accum_bind_group,
            &full_range(self.splats.len),
            view,
            clear,
        );
    }

    /// Immediate-mode convex crisp twin of [`Self::render_crisp`] (headless tests / thumbnails).
    #[allow(clippy::too_many_arguments)]
    pub fn render_convex_crisp(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        splats: &[GpuSplat],
        camera: &CameraUniform,
        width: u32,
        height: u32,
        clear: wgpu::Color,
    ) {
        self.splats.update(device, queue, splats);
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));
        self.coverage.ensure(device, width, height);
        let accum_bind_group = self.convex_accum_bind_group(device, &self.splats.buffer);
        encode_crisp_passes(
            device,
            queue,
            &self.coverage,
            &self.convex_accum.pipeline,
            &accum_bind_group,
            &full_range(self.splats.len),
            view,
            clear,
        );
    }

    /// Immediate-mode triangle crisp twin of [`Self::render_crisp`] (headless tests / thumbnails).
    #[allow(clippy::too_many_arguments)]
    pub fn render_triangle_crisp(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        splats: &[GpuSplat],
        camera: &CameraUniform,
        width: u32,
        height: u32,
        clear: wgpu::Color,
    ) {
        self.splats.update(device, queue, splats);
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));
        self.coverage.ensure(device, width, height);
        let accum_bind_group = self.triangle_accum_bind_group(device, &self.splats.buffer);
        encode_crisp_passes(
            device,
            queue,
            &self.coverage,
            &self.triangle_accum.pipeline,
            &accum_bind_group,
            &full_range(self.splats.len),
            view,
            clear,
        );
    }

    // --- Bind-group builders -------------------------------------------------------------
    // Each builds the per-pass bind group against the supplied splat buffer (the resident
    // buffer for the `render_doc*` paths, the transient buffer for the immediate-mode ones).
    // They are split by pipeline because a bind group must be created against the exact layout
    // its pipeline uses, and the convex pipelines add the convex uniform at binding 2.

    /// Gaussian direct pipeline: storage splats (0) + camera (1).
    fn splat_bind_group(&self, device: &wgpu::Device, splats: &wgpu::Buffer) -> wgpu::BindGroup {
        self.two_binding_group(device, &self.pipeline.bind_group_layout, splats, "splat bind group")
    }

    /// Gaussian coverage-accum pipeline: storage splats (0) + camera (1).
    fn coverage_accum_bind_group(
        &self,
        device: &wgpu::Device,
        splats: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.two_binding_group(
            device,
            &self.coverage.accum_bind_group_layout,
            splats,
            "coverage accum bind group",
        )
    }

    /// Convex direct pipeline: storage splats (0) + camera (1) + convex uniform (2).
    fn convex_bind_group(&self, device: &wgpu::Device, splats: &wgpu::Buffer) -> wgpu::BindGroup {
        self.three_binding_group(
            device,
            &self.convex.bind_group_layout,
            splats,
            &self.convex_buffer,
            "convex bind group",
        )
    }

    /// Convex coverage-accum pipeline: storage splats (0) + camera (1) + convex uniform (2).
    fn convex_accum_bind_group(
        &self,
        device: &wgpu::Device,
        splats: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.three_binding_group(
            device,
            &self.convex_accum.bind_group_layout,
            splats,
            &self.convex_buffer,
            "convex accum bind group",
        )
    }

    /// Triangle direct pipeline: storage splats (0) + camera (1) + triangle uniform (2).
    fn triangle_bind_group(
        &self,
        device: &wgpu::Device,
        splats: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.three_binding_group(
            device,
            &self.triangle.bind_group_layout,
            splats,
            &self.triangle_buffer,
            "triangle bind group",
        )
    }

    /// Triangle coverage-accum pipeline: storage splats (0) + camera (1) + triangle uniform (2).
    fn triangle_accum_bind_group(
        &self,
        device: &wgpu::Device,
        splats: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.three_binding_group(
            device,
            &self.triangle_accum.bind_group_layout,
            splats,
            &self.triangle_buffer,
            "triangle accum bind group",
        )
    }

    fn two_binding_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        splats: &wgpu::Buffer,
        label: &str,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: splats.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.camera_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn three_binding_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        splats: &wgpu::Buffer,
        uniform: &wgpu::Buffer,
        label: &str,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: splats.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        })
    }
}

/// Visible draw range for an immediate-mode (single-slice) draw: the whole buffer, or empty
/// when there are no splats (so the draw call requests zero instances).
fn full_range(len: usize) -> Vec<(u32, u32)> {
    if len == 0 {
        Vec::new()
    } else {
        vec![(0, len as u32)]
    }
}

/// Encode + submit one instanced color pass: bind `pipeline` + `bind_group` and draw each
/// visible `(start, count)` instance range into `view`, clearing to `clear` first. Shared by
/// the Gaussian and convex direct paths (`render_doc*` / `render*`).
fn encode_resident_pass(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    view: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    ranges: &[(u32, u32)],
    clear: wgpu::Color,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("splat encoder"),
    });
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("splat pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(pipeline);
        rpass.set_bind_group(0, bind_group, &[]);
        for (start, count) in ranges {
            rpass.draw(0..6, *start..*start + *count);
        }
    }
    queue.submit(Some(encoder.finish()));
}

/// Encode and submit the two crisp-path passes: accumulate the splat instances in `ranges`
/// into the coverage pipeline's offscreen targets (using the caller-supplied `accum_pipeline`
/// and `accum_bind_group`), then resolve them into `view` (cleared to `clear` first). The
/// accumulate pipeline differs by kernel (Gaussian vs convex) while the resolve pass and
/// offscreen targets — owned by `coverage` — are mode-agnostic and shared. The caller must
/// have sized the targets via [`coverage::CoveragePipeline::ensure`] and uploaded the buffers.
#[allow(clippy::too_many_arguments)]
fn encode_crisp_passes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    coverage: &coverage::CoveragePipeline,
    accum_pipeline: &wgpu::RenderPipeline,
    accum_bind_group: &wgpu::BindGroup,
    ranges: &[(u32, u32)],
    view: &wgpu::TextureView,
    clear: wgpu::Color,
) {
    let targets = coverage.current_targets();
    let resolve_bind_group = coverage.resolve_bind_group(device, targets);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("crisp encoder"),
    });

    // Pass 1: accumulate color (additive) + coverage (max) into the offscreen targets.
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("coverage accum pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.coverage_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(accum_pipeline);
        rpass.set_bind_group(0, accum_bind_group, &[]);
        for (start, count) in ranges {
            rpass.draw(0..6, *start..*start + *count);
        }
    }

    // Pass 2: resolve — threshold the accumulated coverage once for a crisp perimeter and
    // composite the line over the cleared background.
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("coverage resolve pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&coverage.resolve_pipeline);
        rpass.set_bind_group(0, &resolve_bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }

    queue.submit(Some(encoder.finish()));
}

/// Collect every splat in a document into GPU instance form (in layer/stroke order).
pub fn collect_gpu_splats(doc: &app_core::document::Document) -> Vec<GpuSplat> {
    let mut out = Vec::with_capacity(doc.splat_count());
    for (idx, stroke) in doc.strokes.values().enumerate() {
        for splat in &stroke.splats {
            out.push(GpuSplat::from_splat(splat, idx as u32));
        }
    }
    out
}

/// Like `collect_gpu_splats`, but skips splats whose world-space AABB does not
/// intersect `[view_min, view_max]` (the visible world rectangle). Stroke indices are
/// preserved (every stroke is still enumerated) so `stroke_id` stays stable for picking.
///
/// This is a pure performance optimization on the disposable render cache: the canonical
/// document is untouched and any splat even partially inside the view is kept (we test a
/// per-splat AABB expanded by `radius_px + 2.0` world units, where the `+2` generously
/// covers the EWA low-pass screen-space pad at any reasonable zoom).
pub fn collect_gpu_splats_in_view(
    doc: &app_core::document::Document,
    view_min: glam::Vec2,
    view_max: glam::Vec2,
) -> Vec<GpuSplat> {
    let mut out = Vec::with_capacity(doc.splat_count());
    for (idx, stroke) in doc.strokes.values().enumerate() {
        for splat in &stroke.splats {
            let r = splat.radius_px + 2.0;
            let c = splat.center;
            let visible = c.x + r >= view_min.x
                && c.x - r <= view_max.x
                && c.y + r >= view_min.y
                && c.y - r <= view_max.y;
            if visible {
                out.push(GpuSplat::from_splat(splat, idx as u32));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_core::brush::BrushModel;
    use app_core::document::Document;
    use app_core::{BezierSkeleton, CubicBezier};
    use glam::Vec2;

    /// Build a deterministic document with two strokes at known, well-separated world
    /// positions so we can reason about which view rects overlap which splats. The first
    /// stroke lives near the origin (~x in [0,160]); the second is shifted +1000 in x so
    /// the two never overlap.
    fn two_stroke_doc() -> Document {
        let mut doc = Document::new();
        let layer = doc.add_layer("L");
        let a = CubicBezier::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(40.0, 80.0),
            Vec2::new(120.0, 80.0),
            Vec2::new(160.0, 0.0),
        );
        doc.add_stroke(layer, BezierSkeleton::single(a), BrushModel::default());
        let b = CubicBezier::new(
            Vec2::new(1000.0, 0.0),
            Vec2::new(1040.0, 80.0),
            Vec2::new(1120.0, 80.0),
            Vec2::new(1160.0, 0.0),
        );
        doc.add_stroke(layer, BezierSkeleton::single(b), BrushModel::default());
        doc
    }

    #[test]
    fn view_covering_everything_matches_unculled() {
        let doc = two_stroke_doc();
        let full = collect_gpu_splats(&doc);
        let in_view = collect_gpu_splats_in_view(
            &doc,
            Vec2::new(-10_000.0, -10_000.0),
            Vec2::new(10_000.0, 10_000.0),
        );
        assert_eq!(in_view.len(), full.len());
        assert!(!full.is_empty(), "test doc should produce splats");
    }

    #[test]
    fn view_far_away_culls_everything() {
        let doc = two_stroke_doc();
        let in_view = collect_gpu_splats_in_view(
            &doc,
            Vec2::new(100_000.0, 100_000.0),
            Vec2::new(110_000.0, 110_000.0),
        );
        assert_eq!(in_view.len(), 0);
    }

    #[test]
    fn partial_view_keeps_some_drops_others() {
        let doc = two_stroke_doc();
        let full = collect_gpu_splats(&doc);
        // Tight box around only the first stroke (near the origin); the second stroke at
        // x ~ 1000+ lies entirely outside, so we must keep some but not all.
        let in_view = collect_gpu_splats_in_view(
            &doc,
            Vec2::new(-50.0, -50.0),
            Vec2::new(200.0, 150.0),
        );
        assert!(!in_view.is_empty(), "first stroke should be visible");
        assert!(
            in_view.len() < full.len(),
            "second far stroke should be culled"
        );
        // Every kept splat must reference a stroke index that exists in the full set,
        // i.e. enumerate-based stroke_ids are unchanged by culling.
        for s in &in_view {
            assert!(
                full.iter().any(|f| f.stroke_id == s.stroke_id),
                "stroke_id {} must be preserved",
                s.stroke_id
            );
        }
    }
}
