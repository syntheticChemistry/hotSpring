// SPDX-License-Identifier: AGPL-3.0-only

//! Wilson gauge action for SU(3) lattice gauge theory.
//!
//! The fundamental building block is the plaquette — the smallest closed loop
//! of link variables on the lattice:
//!
//!   `P_μν`(x) = `U_μ`(x) `U_ν`(x+μ) `U_μ`†(x+ν) `U_ν`†(x)
//!
//! The Wilson action is:
//!
//!   S = β × Σ\_{x,μ<ν} (1 - Re Tr `P_μν`(x) / 3)
//!
//! where β = 6/g² is the inverse bare coupling.
//!
//! # References
//!
//! - Wilson, PRD 10, 2445 (1974) — original formulation
//! - Gattringer & Lang, "QCD on the Lattice" (2010), Ch. 3

use super::complex_f64::Complex64;
use super::su3::Su3Matrix;

/// 4D lattice of SU(3) link variables.
///
/// Links are stored as `links[site_index][mu]` where mu ∈ {0,1,2,3}
/// represents the four spacetime directions.
#[allow(missing_docs)]
pub struct Lattice {
    pub dims: [usize; 4],
    /// Link variables: links[site * 4 + mu]
    pub links: Vec<Su3Matrix>,
    /// Inverse bare coupling β = 6/g²
    pub beta: f64,
}

impl Lattice {
    /// Total number of lattice sites.
    #[must_use]
    pub const fn volume(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2] * self.dims[3]
    }

    /// Convert 4D coordinates to linear site index.
    ///
    /// Convention: `dims = [Nx, Ny, Nz, Nt]`, `x = [x, y, z, t]`.
    /// Index order: z fastest, then y, then x, then t (slowest).
    /// `idx = t*NxNyNz + x*NyNz + y*Nz + z`
    ///
    /// This matches toadStool's upstream lattice ops, enabling direct use
    /// of upstream DF64 shaders without buffer reordering.
    #[must_use]
    pub const fn site_index(&self, x: [usize; 4]) -> usize {
        x[3] * (self.dims[0] * self.dims[1] * self.dims[2])
            + x[0] * (self.dims[1] * self.dims[2])
            + x[1] * self.dims[2]
            + x[2]
    }

    /// Convert linear site index to 4D coordinates.
    ///
    /// Returns `[x, y, z, t]` where `dims = [Nx, Ny, Nz, Nt]`.
    #[must_use]
    pub const fn site_coords(&self, idx: usize) -> [usize; 4] {
        let nxyz = self.dims[0] * self.dims[1] * self.dims[2];
        let t = idx / nxyz;
        let rem = idx % nxyz;
        let x0 = rem / (self.dims[1] * self.dims[2]);
        let rem2 = rem % (self.dims[1] * self.dims[2]);
        let x1 = rem2 / self.dims[2];
        let x2 = rem2 % self.dims[2];
        [x0, x1, x2, t]
    }

    /// Neighbor in direction mu with periodic boundary conditions.
    #[must_use]
    pub const fn neighbor(&self, x: [usize; 4], mu: usize, forward: bool) -> [usize; 4] {
        let mut y = x;
        if forward {
            y[mu] = (x[mu] + 1) % self.dims[mu];
        } else {
            y[mu] = (x[mu] + self.dims[mu] - 1) % self.dims[mu];
        }
        y
    }

    /// Get link `U_mu`(x).
    pub fn link(&self, x: [usize; 4], mu: usize) -> Su3Matrix {
        let idx = self.site_index(x);
        self.links[idx * 4 + mu]
    }

    /// Set link `U_mu`(x).
    pub fn set_link(&mut self, x: [usize; 4], mu: usize, u: Su3Matrix) {
        let idx = self.site_index(x);
        self.links[idx * 4 + mu] = u;
    }

    /// Initialize to cold start: all links = identity (ordered configuration).
    #[must_use]
    pub fn cold_start(dims: [usize; 4], beta: f64) -> Self {
        let vol = dims[0] * dims[1] * dims[2] * dims[3];
        Self {
            dims,
            links: vec![Su3Matrix::IDENTITY; vol * 4],
            beta,
        }
    }

    /// Initialize to hot start: random SU(3) links (disordered configuration).
    #[must_use]
    pub fn hot_start(dims: [usize; 4], beta: f64, seed: u64) -> Self {
        let vol = dims[0] * dims[1] * dims[2] * dims[3];
        let mut rng_seed = seed;
        let links: Vec<Su3Matrix> = (0..vol * 4)
            .map(|_| Su3Matrix::random_near_identity(&mut rng_seed, 1.5))
            .collect();
        Self { dims, links, beta }
    }

    /// Compute plaquette `P_μν`(x) = `U_μ`(x) `U_ν`(x+μ) `U_μ`†(x+ν) `U_ν`†(x).
    pub fn plaquette(&self, x: [usize; 4], mu: usize, nu: usize) -> Su3Matrix {
        let x_mu = self.neighbor(x, mu, true);
        let x_nu = self.neighbor(x, nu, true);

        let u1 = self.link(x, mu);
        let u2 = self.link(x_mu, nu);
        let u3 = self.link(x_nu, mu).adjoint();
        let u4 = self.link(x, nu).adjoint();

        u1 * u2 * u3 * u4
    }

    /// Average plaquette: <Re Tr P / 3> averaged over all plaquettes.
    ///
    /// For a cold (ordered) start, this is 1.0.
    /// For a hot (disordered) start, this is near 0.
    /// At equilibrium with coupling β, this follows the strong-coupling expansion.
    #[must_use]
    pub fn average_plaquette(&self) -> f64 {
        let vol = self.volume();
        let mut sum = 0.0;
        let mut count = 0usize;

        for idx in 0..vol {
            let x = self.site_coords(idx);
            for mu in 0..4 {
                for nu in (mu + 1)..4 {
                    let p = self.plaquette(x, mu, nu);
                    sum += p.re_trace() / 3.0;
                    count += 1;
                }
            }
        }

        sum / count as f64
    }

    /// Compute the staple sum for link `U_μ`(x).
    ///
    /// The staple is the sum of the 6 "horseshoe" paths that complete
    /// a plaquette with `U_μ`(x). The force on `U_μ`(x) is proportional to
    /// the staple sum.
    ///
    /// For each ν ≠ μ:
    ///   upper staple: `U_ν`(x+μ) `U_μ`†(x+ν) `U_ν`†(x)
    ///   lower staple: `U_ν`†(x+μ-ν) `U_μ`†(x-ν) `U_ν`(x-ν)
    pub fn staple(&self, x: [usize; 4], mu: usize) -> Su3Matrix {
        let mut s = Su3Matrix::ZERO;
        let x_mu = self.neighbor(x, mu, true);

        for nu in 0..4 {
            if nu == mu {
                continue;
            }
            let x_nu = self.neighbor(x, nu, true);
            let x_mu_bnu = self.neighbor(x_mu, nu, false);
            let x_bnu = self.neighbor(x, nu, false);

            // Upper staple
            let upper =
                self.link(x_mu, nu) * self.link(x_nu, mu).adjoint() * self.link(x, nu).adjoint();

            // Lower staple
            let lower = self.link(x_mu_bnu, nu).adjoint()
                * self.link(x_bnu, mu).adjoint()
                * self.link(x_bnu, nu);

            s = s + upper + lower;
        }

        s
    }

    /// Wilson gauge action: S = β × Σ_{x,μ<ν} (1 - Re Tr P / 3)
    #[must_use]
    pub fn wilson_action(&self) -> f64 {
        let vol = self.volume();
        let mut sum = 0.0;

        for idx in 0..vol {
            let x = self.site_coords(idx);
            for mu in 0..4 {
                for nu in (mu + 1)..4 {
                    let p = self.plaquette(x, mu, nu);
                    sum += 1.0 - p.re_trace() / 3.0;
                }
            }
        }

        self.beta * sum
    }

    /// Gauge force dP/dt = -(β/3) × `Proj_TA`(U × V)
    ///
    /// where V is the staple sum and `Proj_TA` is the traceless anti-Hermitian
    /// projection. Derived from H = S(U) + T(P) with the metric
    /// g(X,Y) = -Tr(XY) on su(3).
    pub fn gauge_force(&self, x: [usize; 4], mu: usize) -> Su3Matrix {
        let u = self.link(x, mu);
        let v = self.staple(x, mu);
        let w = u * v;

        let wd = w.adjoint();
        let diff = (w - wd).scale(0.5);
        let tr = diff.trace();
        let tr_over_3 = tr.scale(1.0 / 3.0);
        let mut proj = diff;
        for i in 0..3 {
            proj.m[i][i] -= tr_over_3;
        }

        proj.scale(-self.beta / 3.0)
    }

    /// Polyakov loop: product of temporal links at a fixed spatial position.
    ///
    /// `L`(x\_s) = Tr( Π\_{t=0}^{N\_t-1} `U_0`(t, x\_s) )
    ///
    /// The Polyakov loop is an order parameter for the deconfinement transition.
    /// <|L|> ≈ 0 in the confined phase, <|L|> > 0 in the deconfined phase.
    pub fn polyakov_loop(&self, x_spatial: [usize; 3]) -> Complex64 {
        let nt = self.dims[3];
        let mut prod = Su3Matrix::IDENTITY;
        for t in 0..nt {
            let x = [x_spatial[0], x_spatial[1], x_spatial[2], t];
            prod = prod * self.link(x, 3);
        }
        prod.trace().scale(1.0 / 3.0)
    }

    /// Average Polyakov loop magnitude: spatial average of |`L`(x\_s)|.
    #[must_use]
    pub fn average_polyakov_loop(&self) -> f64 {
        let ns = [self.dims[0], self.dims[1], self.dims[2]];
        let spatial_vol = ns[0] * ns[1] * ns[2];
        let mut sum = 0.0;

        for ix in 0..ns[0] {
            for iy in 0..ns[1] {
                for iz in 0..ns[2] {
                    let l = self.polyakov_loop([ix, iy, iz]);
                    sum += l.abs();
                }
            }
        }

        sum / spatial_vol as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_start_plaquette_is_one() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let p = lat.average_plaquette();
        assert!(
            (p - 1.0).abs() < 1e-14,
            "cold start plaquette should be 1.0, got {p}"
        );
    }

    #[test]
    fn cold_start_action_is_zero() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let s = lat.wilson_action();
        assert!(s.abs() < 1e-12, "cold start action should be 0, got {s}");
    }

    #[test]
    fn hot_start_plaquette_below_one() {
        let lat = Lattice::hot_start([4, 4, 4, 4], 6.0, 42);
        let p = lat.average_plaquette();
        assert!(p < 1.0, "hot start plaquette should be < 1.0, got {p}");
        assert!(p > -1.0, "plaquette should be > -1.0, got {p}");
    }

    #[test]
    fn site_index_roundtrip() {
        let lat = Lattice::cold_start([4, 6, 8, 10], 6.0);
        for idx in 0..lat.volume() {
            let coords = lat.site_coords(idx);
            let back = lat.site_index(coords);
            assert_eq!(idx, back, "site index roundtrip failed at {idx}");
        }
    }

    #[test]
    fn neighbor_periodic() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let x = [3, 0, 0, 0];
        let fwd = lat.neighbor(x, 0, true);
        assert_eq!(fwd, [0, 0, 0, 0], "periodic forward wrap");
        let bwd = lat.neighbor([0, 0, 0, 0], 0, false);
        assert_eq!(bwd, [3, 0, 0, 0], "periodic backward wrap");
    }

    #[test]
    fn plaquette_is_unitary() {
        let lat = Lattice::hot_start([4, 4, 4, 4], 6.0, 99);
        let p = lat.plaquette([0, 0, 0, 0], 0, 1);
        let pp = p * p.adjoint();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (pp.m[i][j].re - expected).abs() < 1e-8,
                    "plaquette should be unitary"
                );
            }
        }
    }

    #[test]
    fn cold_polyakov_loop_is_one() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let l = lat.polyakov_loop([0, 0, 0]);
        assert!(
            (l.abs() - 1.0).abs() < 1e-14,
            "cold Polyakov loop should be 1.0"
        );
    }

    #[test]
    fn gauge_force_cold_start_is_zero() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let f = lat.gauge_force([1, 1, 1, 1], 0);
        assert!(f.norm_sq() < 1e-20, "force on cold start should be zero");
    }
}
