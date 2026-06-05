//! 2D pan/zoom camera. Maps world coordinates to clip space via `clip = world*scale +
//! offset`, with the y-axis flipped so world-y points downward (canvas convention).

use glam::Vec2;

/// GPU-side camera uniform (32 bytes: two `vec2<f32>` + one `vec4<f32>`).
///
/// `params` carries auxiliary scalars in a 16-byte-aligned slot so the Rust↔WGSL layout
/// stays byte-identical (a bare trailing `f32` would mismatch std140/wgsl vec4 alignment).
/// `params.x` = `zoom` (pixels-per-world-unit), consumed by the screen-space low-pass.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub scale: [f32; 2],
    pub offset: [f32; 2],
    pub params: [f32; 4],
}

/// An orthographic 2D camera.
#[derive(Clone, Copy, Debug)]
pub struct Camera2D {
    /// World-space point at the center of the viewport.
    pub center: Vec2,
    /// Pixels-per-world-unit (1.0 = world units are device pixels).
    pub zoom: f32,
    /// Viewport size in physical pixels.
    pub viewport: Vec2,
}

impl Camera2D {
    pub fn new(viewport: Vec2) -> Self {
        Self {
            center: viewport * 0.5,
            zoom: 1.0,
            viewport,
        }
    }

    pub fn uniform(&self) -> CameraUniform {
        let w = self.viewport.x.max(1.0);
        let h = self.viewport.y.max(1.0);
        // clip = (world - center) * (2*zoom/size), with y flipped.
        let scale = Vec2::new(2.0 * self.zoom / w, -2.0 * self.zoom / h);
        let offset = -self.center * scale;
        CameraUniform {
            scale: scale.into(),
            offset: offset.into(),
            // params.x = zoom (pixels-per-world-unit); remaining lanes reserved.
            params: [self.zoom, 0.0, 0.0, 0.0],
        }
    }

    /// Convert a screen-pixel position (origin top-left) to a world coordinate.
    pub fn screen_to_world(&self, screen: Vec2) -> Vec2 {
        (screen - self.viewport * 0.5) / self.zoom + self.center
    }

    /// Convert a world coordinate to a screen-pixel position (inverse of
    /// [`screen_to_world`]). Used to place editing overlays (handles, anchors) drawn on a
    /// 2D layer above the GPU surface.
    pub fn world_to_screen(&self, world: Vec2) -> Vec2 {
        (world - self.center) * self.zoom + self.viewport * 0.5
    }

    /// Pan by a screen-space delta (e.g. a drag in pixels).
    pub fn pan_pixels(&mut self, delta_px: Vec2) {
        self.center -= delta_px / self.zoom;
    }

    /// Zoom by `factor` while keeping `anchor_screen` fixed on screen.
    pub fn zoom_at(&mut self, anchor_screen: Vec2, factor: f32) {
        let before = self.screen_to_world(anchor_screen);
        self.zoom = (self.zoom * factor).clamp(0.01, 100.0);
        let after = self.screen_to_world(anchor_screen);
        self.center += before - after;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn center_maps_to_clip_origin() {
        let cam = Camera2D::new(Vec2::new(256.0, 256.0));
        let u = cam.uniform();
        let c = cam.center;
        let clip_x = c.x * u.scale[0] + u.offset[0];
        let clip_y = c.y * u.scale[1] + u.offset[1];
        assert!(clip_x.abs() < 1e-5 && clip_y.abs() < 1e-5);
    }

    #[test]
    fn world_to_screen_inverts_screen_to_world() {
        let mut cam = Camera2D::new(Vec2::new(800.0, 600.0));
        cam.center = Vec2::new(123.0, -45.0);
        cam.zoom = 2.5;
        for p in [
            Vec2::new(0.0, 0.0),
            Vec2::new(400.0, 300.0),
            Vec2::new(-200.0, 750.0),
        ] {
            let round = cam.world_to_screen(cam.screen_to_world(p));
            assert!((round - p).length() < 1e-3, "round trip off at {p:?}");
        }
    }

    #[test]
    fn zoom_at_keeps_anchor_fixed() {
        let mut cam = Camera2D::new(Vec2::new(800.0, 600.0));
        let anchor = Vec2::new(200.0, 150.0);
        let world_before = cam.screen_to_world(anchor);
        cam.zoom_at(anchor, 2.5);
        let world_after = cam.screen_to_world(anchor);
        assert!((world_before - world_after).length() < 1e-3);
    }
}
