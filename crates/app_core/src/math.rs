//! Small math helpers: covariance construction, a deterministic PRNG, and a dense
//! symmetric-positive-definite solver for the least-squares curve fit.

pub use glam::{Mat2, Vec2};

/// Build a 2x2 covariance matrix for an anisotropic Gaussian whose principal axes are
/// rotated by `theta` and have standard deviations `sigma_x` (tangent) / `sigma_y`
/// (normal). `Sigma = R S R^T` with `S = diag(sigma_x^2, sigma_y^2)`.
pub fn covariance_from_sigmas(theta: f32, sigma_x: f32, sigma_y: f32) -> Mat2 {
    let r = Mat2::from_angle(theta);
    let s = Mat2::from_cols(
        Vec2::new(sigma_x * sigma_x, 0.0),
        Vec2::new(0.0, sigma_y * sigma_y),
    );
    r * s * r.transpose()
}

/// Deterministic, allocation-free PRNG (SplitMix64). Used for brush jitter so that a
/// given brush `seed` always reproduces the same splat cloud — important for stable
/// regeneration and for reproducible tests across platforms.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // Use the top 24 bits for a uniform float with full mantissa precision.
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }

    /// Uniform in `[lo, hi)`.
    #[inline]
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    /// Uniform in `[-1, 1)`.
    #[inline]
    pub fn signed(&mut self) -> f32 {
        self.range(-1.0, 1.0)
    }
}

/// Solve `A x = b` for a small dense symmetric-positive-definite `A` via Cholesky
/// factorization (`A = L L^T`). Returns `None` if `A` is not numerically SPD.
///
/// Used for the regularized normal equations of the Bezier fit, which are SPD by
/// construction (`A^T W A + lambda I`, `lambda > 0`). Computed in `f64` for stability.
// Triangular factorization/substitution are clearest written with explicit matrix
// indices; the iterator rewrites clippy suggests would obscure them.
#[allow(clippy::needless_range_loop)]
pub fn solve_spd(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    debug_assert!(a.len() == n && a.iter().all(|row| row.len() == n));

    // Cholesky: lower-triangular L with A = L L^T.
    let mut l = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 1e-12 {
                    return None; // not positive definite
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    // Forward substitution: L y = b.
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i][k] * y[k];
        }
        y[i] = sum / l[i][i];
    }

    // Back substitution: L^T x = y.
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[k][i] * x[k];
        }
        x[i] = sum / l[i][i];
    }
    Some(x)
}

#[inline]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covariance_is_symmetric_and_spd() {
        let cov = covariance_from_sigmas(0.7, 6.0, 2.0);
        // Symmetric.
        assert!((cov.x_axis.y - cov.y_axis.x).abs() < 1e-4);
        // Positive determinant (= product of eigenvalues = (sigma_x*sigma_y)^2).
        assert!(cov.determinant() > 0.0);
        // Inverse round-trips.
        let inv = cov.inverse();
        let id = cov * inv;
        assert!((id.x_axis.x - 1.0).abs() < 1e-3);
        assert!((id.y_axis.y - 1.0).abs() < 1e-3);
        assert!(id.x_axis.y.abs() < 1e-3);
    }

    #[test]
    fn rng_is_deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_f32(), b.next_f32());
        }
        // And produces values in range.
        let mut r = Rng::new(7);
        for _ in 0..1000 {
            let v = r.next_f32();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn solve_spd_recovers_known_solution() {
        // A = [[4,1],[1,3]], x = [1,2] -> b = [6,7].
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let b = vec![6.0, 7.0];
        let x = solve_spd(&a, &b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-9);
        assert!((x[1] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn solve_spd_rejects_non_pd() {
        let a = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let b = vec![1.0, 1.0];
        assert!(solve_spd(&a, &b).is_none());
    }
}
