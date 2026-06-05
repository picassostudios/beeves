//! GPU buffer helpers for splat instances and the camera uniform.

use wgpu::util::DeviceExt;

use app_core::splat::GpuSplat;

use crate::camera::CameraUniform;

/// A growable storage buffer holding the splat instances. Reallocates when the splat
/// count exceeds capacity; otherwise updates in place via `queue.write_buffer`.
pub struct SplatBuffer {
    pub buffer: wgpu::Buffer,
    pub capacity: usize,
    pub len: usize,
}

impl SplatBuffer {
    pub fn new(device: &wgpu::Device, splats: &[GpuSplat]) -> Self {
        let capacity = splats.len().max(1);
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("splat storage buffer"),
            contents: padded_contents(splats, capacity),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            buffer,
            capacity,
            len: splats.len(),
        }
    }

    /// Upload a new set of splats, reallocating only if capacity is exceeded.
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, splats: &[GpuSplat]) {
        if splats.len() > self.capacity {
            *self = Self::new(device, splats);
            return;
        }
        self.len = splats.len();
        if !splats.is_empty() {
            queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(splats));
        }
    }
}

/// Ensure at least `capacity` instances back the buffer so it is never zero-sized.
fn padded_contents(splats: &[GpuSplat], capacity: usize) -> &[u8] {
    if splats.is_empty() {
        // One zeroed instance keeps the storage buffer non-empty; `len = 0` means the
        // draw call requests zero instances, so it is never read.
        static ONE: [GpuSplat; 1] = [GpuSplat {
            center: [0.0, 0.0],
            cov_a: 1.0,
            cov_b: 0.0,
            cov_c: 1.0,
            color: 0,
            alpha: 0.0,
            radius: 0.0,
            stroke_id: 0,
            flags: 0,
            hardness: 0.0,
        }];
        let _ = capacity;
        bytemuck::cast_slice(&ONE)
    } else {
        bytemuck::cast_slice(splats)
    }
}

/// A camera uniform buffer.
pub fn camera_buffer(device: &wgpu::Device, uniform: &CameraUniform) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("camera uniform"),
        contents: bytemuck::bytes_of(uniform),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}
