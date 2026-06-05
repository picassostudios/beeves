//! wgpu context creation and headless texture readback.

/// Owns the core wgpu objects. Works on native (Metal/Vulkan/DX) and in the browser
/// (WebGPU) — only the surface differs, which is created separately.
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Request an adapter + device, optionally compatible with a surface (for on-screen
    /// rendering). Pass `None` for a headless context.
    pub async fn new(compatible_surface: Option<&wgpu::Surface<'_>>) -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface,
            })
            .await
            .expect("no suitable GPU adapter found");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("gaussian-design-tool device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("failed to create device");

        Self {
            instance,
            adapter,
            device,
            queue,
        }
    }

    /// Block on [`GpuContext::new`] (native only).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_headless_blocking() -> Self {
        pollster::block_on(Self::new(None))
    }
}

/// Round `value` up to the next multiple of `align`.
fn align_up(value: u32, align: u32) -> u32 {
    value.div_ceil(align) * align
}

/// Copy an `Rgba8Unorm` texture back to the CPU as tightly-packed RGBA bytes
/// (`width*height*4`). Handles the 256-byte `bytes_per_row` alignment requirement.
pub fn read_texture_rgba8(
    ctx: &GpuContext,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let unpadded_bytes_per_row = width * 4;
    let padded_bytes_per_row = align_up(unpadded_bytes_per_row, 256);

    let out_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded_bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback encoder"),
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

    // Map and wait.
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
    let mut out = Vec::with_capacity((unpadded_bytes_per_row * height) as usize);
    for row in 0..height {
        let start = (row * padded_bytes_per_row) as usize;
        let end = start + unpadded_bytes_per_row as usize;
        out.extend_from_slice(&data[start..end]);
    }
    drop(data);
    out_buffer.unmap();
    out
}

/// Create an offscreen `Rgba8Unorm` render target usable as a color attachment and
/// copy source (for headless rendering / tests / thumbnails).
pub fn create_offscreen_target(
    ctx: &GpuContext,
    width: u32,
    height: u32,
) -> wgpu::Texture {
    ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}
