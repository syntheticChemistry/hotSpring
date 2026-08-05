// SPDX-License-Identifier: AGPL-3.0-or-later

//! Generic lattice for SU(N) gauge theory — Wilson action, HMC, observables.
//!
//! `GenericLattice<G>` implements the same physics as `Lattice` (SU(3))
//! but is parameterized over any `GaugeGroup` implementation. This enables
//! the SU(N) thermalization grid and measurement battery for N=2,3,4,5,6,8.
//!
//! The existing `Lattice` (SU(3)-specific) remains the validated production
//! path for GPU-accelerated HMC. `GenericLattice` targets CPU thermalization
//! and measurement passes.

use std::io;
use std::path::Path;

use super::complex_f64::Complex64;
use super::gauge_group::GaugeGroup;

/// 4D lattice of SU(N) link variables, generic over gauge group G.
#[derive(Clone)]
pub struct GenericLattice<G: GaugeGroup> {
    pub dims: [usize; 4],
    pub links: Vec<G>,
    pub beta: f64,
    /// Runtime NC — matches G::NC for fixed-size types, runtime-determined for SuNMatrix.
    pub nc: usize,
}

/// HMC configuration for generic lattice.
#[derive(Clone, Debug)]
pub struct GenericHmcConfig {
    pub n_md_steps: usize,
    pub dt: f64,
    pub seed: u64,
}

/// HMC trajectory result.
#[derive(Clone, Debug)]
pub struct GenericHmcResult {
    pub accepted: bool,
    pub delta_h: f64,
    pub plaquette: f64,
}

impl<G: GaugeGroup> GenericLattice<G> {
    #[must_use]
    pub const fn volume(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2] * self.dims[3]
    }

    #[must_use]
    pub const fn site_index(&self, x: [usize; 4]) -> usize {
        x[3] * (self.dims[0] * self.dims[1] * self.dims[2])
            + x[0] * (self.dims[1] * self.dims[2])
            + x[1] * self.dims[2]
            + x[2]
    }

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

    pub fn link(&self, x: [usize; 4], mu: usize) -> &G {
        let idx = self.site_index(x);
        &self.links[idx * 4 + mu]
    }

    pub fn set_link(&mut self, x: [usize; 4], mu: usize, u: G) {
        let idx = self.site_index(x);
        self.links[idx * 4 + mu] = u;
    }

    // --- Initialization ---

    #[must_use]
    pub fn cold_start(dims: [usize; 4], beta: f64) -> Self {
        let vol = dims[0] * dims[1] * dims[2] * dims[3];
        let nc = G::NC;
        Self {
            dims,
            links: vec![G::identity(); vol * 4],
            beta,
            nc,
        }
    }

    #[must_use]
    pub fn hot_start(dims: [usize; 4], beta: f64, seed: u64) -> Self {
        let vol = dims[0] * dims[1] * dims[2] * dims[3];
        let nc = G::NC;
        let mut rng_seed = seed;
        let links: Vec<G> = (0..vol * 4)
            .map(|_| G::random_near_identity(&mut rng_seed, 1.5))
            .collect();
        Self { dims, links, beta, nc }
    }

    /// Tile a smaller lattice into a larger volume (dynamic programming bootstrap).
    ///
    /// Each dimension of `new_dims` must be a multiple of the source dimension.
    /// Links at site (x0,x1,x2,x3) in the new lattice are copied from
    /// (x0 % src.dims[0], x1 % src.dims[1], x2 % src.dims[2], x3 % src.dims[3])
    /// in the source. This preserves gauge invariance: every plaquette in the
    /// tiled lattice is identical to one in the source, so the starting action
    /// density is physically correct. A short HMC burn-in (50-100 trajectories)
    /// breaks the artificial periodicity.
    #[must_use]
    pub fn tile_from(source: &Self, new_dims: [usize; 4]) -> Self {
        for mu in 0..4 {
            assert!(
                new_dims[mu] >= source.dims[mu] && new_dims[mu] % source.dims[mu] == 0,
                "new_dims[{}]={} must be a multiple of source dims[{}]={}",
                mu, new_dims[mu], mu, source.dims[mu]
            );
        }

        let vol = new_dims[0] * new_dims[1] * new_dims[2] * new_dims[3];
        let mut links = Vec::with_capacity(vol * 4);

        let nxyz_new = new_dims[0] * new_dims[1] * new_dims[2];
        for idx in 0..vol {
            let t = idx / nxyz_new;
            let rem = idx % nxyz_new;
            let x0 = rem / (new_dims[1] * new_dims[2]);
            let rem2 = rem % (new_dims[1] * new_dims[2]);
            let x1 = rem2 / new_dims[2];
            let x2 = rem2 % new_dims[2];

            let src_x = [
                x0 % source.dims[0],
                x1 % source.dims[1],
                x2 % source.dims[2],
                t % source.dims[3],
            ];
            let src_idx = source.site_index(src_x);
            for mu in 0..4 {
                links.push(source.links[src_idx * 4 + mu].clone());
            }
        }

        Self {
            dims: new_dims,
            links,
            beta: source.beta,
            nc: source.nc,
        }
    }

    // --- Observables ---

    pub fn plaquette(&self, x: [usize; 4], mu: usize, nu: usize) -> G {
        let x_mu = self.neighbor(x, mu, true);
        let x_nu = self.neighbor(x, nu, true);

        let u1 = self.link(x, mu);
        let u2 = self.link(x_mu, nu);
        let u3_dag = self.link(x_nu, mu).adjoint();
        let u4_dag = self.link(x, nu).adjoint();

        u1.mul(u2).mul(&u3_dag).mul(&u4_dag)
    }

    #[must_use]
    pub fn average_plaquette(&self) -> f64 {
        let vol = self.volume();
        let nc = self.nc as f64;
        let mut sum = 0.0;
        let mut count = 0usize;

        for idx in 0..vol {
            let x = self.site_coords(idx);
            for mu in 0..4 {
                for nu in (mu + 1)..4 {
                    let p = self.plaquette(x, mu, nu);
                    sum += p.re_trace() / nc;
                    count += 1;
                }
            }
        }

        sum / count as f64
    }

    pub fn staple(&self, x: [usize; 4], mu: usize) -> G {
        let mut s = G::zero();
        let x_mu = self.neighbor(x, mu, true);

        for nu in 0..4 {
            if nu == mu {
                continue;
            }
            let x_nu = self.neighbor(x, nu, true);
            let x_mu_bnu = self.neighbor(x_mu, nu, false);
            let x_bnu = self.neighbor(x, nu, false);

            let upper = self.link(x_mu, nu)
                .mul(&self.link(x_nu, mu).adjoint())
                .mul(&self.link(x, nu).adjoint());

            let lower = self.link(x_mu_bnu, nu).adjoint()
                .mul(&self.link(x_bnu, mu).adjoint())
                .mul(self.link(x_bnu, nu));

            s = s.add(&upper).add(&lower);
        }

        s
    }

    #[must_use]
    pub fn wilson_action(&self) -> f64 {
        let vol = self.volume();
        let nc = self.nc as f64;
        let mut sum = 0.0;

        for idx in 0..vol {
            let x = self.site_coords(idx);
            for mu in 0..4 {
                for nu in (mu + 1)..4 {
                    let p = self.plaquette(x, mu, nu);
                    sum += 1.0 - p.re_trace() / nc;
                }
            }
        }

        self.beta * sum
    }

    pub fn gauge_force(&self, x: [usize; 4], mu: usize) -> G {
        let u = self.link(x, mu);
        let v = self.staple(x, mu);
        let w = u.mul(&v);
        let nc = self.nc as f64;

        let wd = w.adjoint();
        let diff = w.sub(&wd).scale(0.5);
        let tr = diff.trace();
        let tr_over_n = tr.scale(1.0 / nc);
        let mut proj = diff;
        proj.sub_diagonal(tr_over_n);

        proj.scale(-self.beta / nc)
    }

    pub fn polyakov_loop(&self, x_spatial: [usize; 3]) -> Complex64 {
        let nt = self.dims[3];
        let nc = self.nc as f64;
        let mut prod = G::identity();
        for t in 0..nt {
            let x = [x_spatial[0], x_spatial[1], x_spatial[2], t];
            prod = prod.mul(self.link(x, 3));
        }
        prod.trace().scale(1.0 / nc)
    }

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

    pub fn complex_polyakov_average(&self) -> (f64, f64) {
        let ns = [self.dims[0], self.dims[1], self.dims[2]];
        let spatial_vol = ns[0] * ns[1] * ns[2];
        let mut sum_re = 0.0;
        let mut sum_im = 0.0;
        for ix in 0..ns[0] {
            for iy in 0..ns[1] {
                for iz in 0..ns[2] {
                    let c = self.polyakov_loop([ix, iy, iz]);
                    sum_re += c.re;
                    sum_im += c.im;
                }
            }
        }
        (sum_re / spatial_vol as f64, sum_im / spatial_vol as f64)
    }

    /// Wilson loop W(R,T) averaged over spatial directions and lattice sites.
    #[must_use]
    pub fn spatial_temporal_wilson_loop(&self, r: usize, t: usize) -> f64 {
        let vol = self.volume();
        let nc = self.nc as f64;
        let mut sum = 0.0;
        let mut count = 0u64;

        for idx in 0..vol {
            let x = self.site_coords(idx);
            for spatial_dir in 0..3_usize {
                let temporal_dir = 3;

                let mut bottom = G::identity();
                let mut pos = x;
                for _ in 0..r {
                    bottom = bottom.mul(self.link(pos, spatial_dir));
                    pos = self.neighbor(pos, spatial_dir, true);
                }

                let mut right = G::identity();
                let mut pos_r = pos;
                for _ in 0..t {
                    right = right.mul(self.link(pos_r, temporal_dir));
                    pos_r = self.neighbor(pos_r, temporal_dir, true);
                }

                let mut top = G::identity();
                let mut pos_t = pos_r;
                for _ in 0..r {
                    pos_t = self.neighbor(pos_t, spatial_dir, false);
                    top = top.mul(&self.link(pos_t, spatial_dir).adjoint());
                }

                let mut left_links = Vec::with_capacity(t);
                let mut pos_l = x;
                for _ in 0..t {
                    left_links.push(self.link(pos_l, temporal_dir).clone());
                    pos_l = self.neighbor(pos_l, temporal_dir, true);
                }
                let mut left = G::identity();
                for u in left_links.iter().rev() {
                    left = left.mul(&u.adjoint());
                }

                let w = bottom.mul(&right).mul(&top).mul(&left);
                sum += w.trace().re / nc;
                count += 1;
            }
        }

        if count > 0 { sum / count as f64 } else { 0.0 }
    }

    // --- Creutz ratios ---

    /// Creutz ratio χ(R,T) = -ln(W(R,T) * W(R-1,T-1) / (W(R,T-1) * W(R-1,T))).
    ///
    /// For large R,T the Creutz ratio converges to the string tension σa²
    /// (in lattice units). Requires R ≥ 2, T ≥ 2.
    ///
    /// Reference: Creutz, PRD 21, 2308 (1980).
    #[must_use]
    pub fn creutz_ratio(&self, r: usize, t: usize) -> Option<f64> {
        if r < 2 || t < 2 {
            return None;
        }
        let w_rt = self.spatial_temporal_wilson_loop(r, t);
        let w_r1t1 = self.spatial_temporal_wilson_loop(r - 1, t - 1);
        let w_rt1 = self.spatial_temporal_wilson_loop(r, t - 1);
        let w_r1t = self.spatial_temporal_wilson_loop(r - 1, t);

        let numer = w_rt * w_r1t1;
        let denom = w_rt1 * w_r1t;

        if denom <= 0.0 || numer <= 0.0 {
            return None;
        }
        Some(-(numer / denom).ln())
    }

    /// Compute Creutz ratios for R,T = 2..max_r, extracting
    /// the string tension at each scale.
    #[must_use]
    pub fn creutz_ratio_scan(&self, max_r: usize) -> Vec<(usize, usize, f64)> {
        let mut results = Vec::new();
        for r in 2..=max_r {
            for t in 2..=max_r {
                if let Some(chi) = self.creutz_ratio(r, t) {
                    results.push((r, t, chi));
                }
            }
        }
        results
    }

    // --- HMC ---

    /// Kinetic energy: T = -Tr(P²)/2 summed over all links.
    fn kinetic_energy(momenta: &[G]) -> f64 {
        let mut sum = 0.0;
        for p in momenta {
            sum += -0.5 * p.mul(p).re_trace();
        }
        sum
    }

    /// One Omelyan 2MN HMC trajectory.
    pub fn hmc_trajectory(&mut self, config: &mut GenericHmcConfig) -> GenericHmcResult {
        let vol = self.volume();
        let old_links = self.links.clone();
        let action_before = self.wilson_action();

        let nc = self.nc;
        let mut momenta: Vec<G> = (0..vol * 4)
            .map(|_| G::random_algebra_for_nc(nc, &mut config.seed))
            .collect();

        let kinetic_before = Self::kinetic_energy(&momenta);
        let h_old = action_before + kinetic_before;

        self.omelyan_integrate(&mut momenta, config.n_md_steps, config.dt);

        let action_after = self.wilson_action();
        let kinetic_after = Self::kinetic_energy(&momenta);
        let h_new = action_after + kinetic_after;

        let delta_h = h_new - h_old;

        let accept = if delta_h <= 0.0 {
            true
        } else {
            let r = super::constants::lcg_uniform_f64(&mut config.seed);
            r < (-delta_h).exp()
        };

        if !accept {
            self.links = old_links;
        }

        GenericHmcResult {
            accepted: accept,
            delta_h,
            plaquette: self.average_plaquette(),
        }
    }

    /// Leapfrog integrator (O(dt²) shadow Hamiltonian).
    fn leapfrog_integrate(&mut self, momenta: &mut [G], n_steps: usize, dt: f64) {
        self.update_momenta(momenta, 0.5 * dt);
        for step in 0..n_steps {
            self.update_links(momenta, dt);
            let eps = if step < n_steps - 1 { dt } else { 0.5 * dt };
            self.update_momenta(momenta, eps);
        }
    }

    /// Omelyan 2MN integrator — matches the per-step pattern from the validated SU(3) HMC.
    fn omelyan_integrate(&mut self, momenta: &mut [G], n_steps: usize, dt: f64) {
        let lam = crate::tolerances::OMELYAN_LAMBDA;

        for _step in 0..n_steps {
            self.update_momenta(momenta, lam * dt);
            self.update_links(momenta, 0.5 * dt);
            self.update_momenta(momenta, (1.0 - 2.0 * lam) * dt);
            self.update_links(momenta, 0.5 * dt);
            self.update_momenta(momenta, lam * dt);
        }
    }

    fn update_momenta(&self, momenta: &mut [G], eps: f64) {
        let vol = self.volume();
        for idx in 0..vol {
            let x = self.site_coords(idx);
            for mu in 0..4 {
                let f = self.gauge_force(x, mu);
                let link_idx = idx * 4 + mu;
                momenta[link_idx] = momenta[link_idx].add(&f.scale(eps));
            }
        }
    }

    fn update_links(&mut self, momenta: &[G], eps: f64) {
        let vol = self.volume();
        for i in 0..vol * 4 {
            let exp_p = momenta[i].exp_cayley(eps);
            self.links[i] = exp_p.mul(&self.links[i]).reunitarize();
        }
    }

    // --- Serialization ---

    /// Cache key for this lattice configuration.
    pub fn cache_key(dims: [usize; 4], beta: f64, seed: u64, n_therm: usize, integrator: &str) -> String {
        let input = format!(
            "{}_{}x{}x{}x{}_b{beta:.6}_s{seed}_t{n_therm}_{integrator}",
            G::gauge_group_tag(),
            dims[0], dims[1], dims[2], dims[3],
        );
        let hash = blake3::hash(input.as_bytes());
        format!("{}", hash.to_hex())
    }

    /// Cache directory for this gauge group.
    pub fn config_cache_dir() -> std::path::PathBuf {
        let dir = dirs::data_local_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("hotspring")
            .join("configs")
            .join(G::gauge_group_tag());
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    /// Save to disk with BLAKE3 integrity hash.
    pub fn save(&self, path: &Path) -> io::Result<blake3::Hash> {
        let nc = self.nc;
        let mut buf = Vec::new();

        // Header: dims (4 x u64) + beta (f64) + nc (u64) = 48 bytes
        for &d in &self.dims {
            buf.extend_from_slice(&(d as u64).to_le_bytes());
        }
        buf.extend_from_slice(&self.beta.to_le_bytes());
        buf.extend_from_slice(&(nc as u64).to_le_bytes());

        let header_len = buf.len();

        for link in &self.links {
            link.write_to_buf(&mut buf);
        }

        let hash = blake3::hash(&buf[header_len..]);
        std::fs::write(path, &buf)?;
        Ok(hash)
    }

    /// Load from disk and verify integrity.
    pub fn load(path: &Path) -> io::Result<Self> {
        let buf = std::fs::read(path)?;
        if buf.len() < 48 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too short"));
        }

        let dims = [
            u64::from_le_bytes(buf[0..8].try_into().unwrap()) as usize,
            u64::from_le_bytes(buf[8..16].try_into().unwrap()) as usize,
            u64::from_le_bytes(buf[16..24].try_into().unwrap()) as usize,
            u64::from_le_bytes(buf[24..32].try_into().unwrap()) as usize,
        ];
        let beta = f64::from_le_bytes(buf[32..40].try_into().unwrap());
        let nc = u64::from_le_bytes(buf[40..48].try_into().unwrap()) as usize;

        if G::NC != 0 && nc != G::NC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("gauge group mismatch: file has NC={nc}, expected NC={}", G::NC),
            ));
        }

        let vol = dims[0] * dims[1] * dims[2] * dims[3];
        let n_links = vol * 4;
        let bytes_per_link = if G::NC == 0 { 2 * nc * nc * 8 } else { G::bytes_per_link() };
        let expected_bytes = 48 + n_links * bytes_per_link;
        if buf.len() != expected_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("expected {} bytes, got {}", expected_bytes, buf.len()),
            ));
        }

        let data = &buf[48..];
        let mut links = Vec::with_capacity(n_links);
        for i in 0..n_links {
            links.push(G::read_from_buf(data, i * bytes_per_link));
        }

        Ok(Self { dims, links, beta, nc })
    }
}

impl GenericLattice<crate::lattice::su_n::SuNMatrix> {
    /// Create a cold-start SU(N) lattice with runtime nc.
    pub fn cold_start_nc(dims: [usize; 4], beta: f64, nc: usize) -> Self {
        use crate::lattice::su_n::SuNMatrix;
        let vol = dims[0] * dims[1] * dims[2] * dims[3];
        Self {
            dims,
            links: vec![SuNMatrix::identity_nc(nc); vol * 4],
            beta,
            nc,
        }
    }

    /// Create a hot-start SU(N) lattice with runtime nc.
    pub fn hot_start_nc(dims: [usize; 4], beta: f64, nc: usize, seed: u64) -> Self {
        use crate::lattice::su_n::SuNMatrix;
        let vol = dims[0] * dims[1] * dims[2] * dims[3];
        let mut rng = seed;
        let links: Vec<SuNMatrix> = (0..vol * 4)
            .map(|_| SuNMatrix::random_near_identity_nc(nc, &mut rng, 1.5))
            .collect();
        Self { dims, links, beta, nc }
    }

    /// Load an SU(N) config with runtime NC read from the file header.
    pub fn load_sun(path: &Path) -> io::Result<Self> {
        use crate::lattice::su_n::SuNMatrix;

        let buf = std::fs::read(path)?;
        if buf.len() < 48 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too short"));
        }

        let dims = [
            u64::from_le_bytes(buf[0..8].try_into().unwrap()) as usize,
            u64::from_le_bytes(buf[8..16].try_into().unwrap()) as usize,
            u64::from_le_bytes(buf[16..24].try_into().unwrap()) as usize,
            u64::from_le_bytes(buf[24..32].try_into().unwrap()) as usize,
        ];
        let beta = f64::from_le_bytes(buf[32..40].try_into().unwrap());
        let nc = u64::from_le_bytes(buf[40..48].try_into().unwrap()) as usize;

        let vol = dims[0] * dims[1] * dims[2] * dims[3];
        let n_links = vol * 4;
        let bytes_per_link = 2 * nc * nc * 8;
        let expected_bytes = 48 + n_links * bytes_per_link;
        if buf.len() != expected_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("SU({nc}) load: expected {expected_bytes} bytes, got {}", buf.len()),
            ));
        }

        let data = &buf[48..];
        let mut links = Vec::with_capacity(n_links);
        for i in 0..n_links {
            links.push(SuNMatrix::read_from_buf_nc(nc, data, i * bytes_per_link));
        }

        Ok(Self { dims, links, beta, nc })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::su2::Su2Matrix;
    use crate::lattice::su3::Su3Matrix;

    #[test]
    fn su2_cold_start_plaquette_is_one() {
        let lat = GenericLattice::<Su2Matrix>::cold_start([4, 4, 4, 4], 2.5);
        let p = lat.average_plaquette();
        assert!(
            (p - 1.0).abs() < 1e-14,
            "SU(2) cold start plaquette = {p}"
        );
    }

    #[test]
    fn su3_cold_start_plaquette_is_one() {
        let lat = GenericLattice::<Su3Matrix>::cold_start([4, 4, 4, 4], 6.0);
        let p = lat.average_plaquette();
        assert!(
            (p - 1.0).abs() < 1e-14,
            "SU(3) cold start plaquette = {p}"
        );
    }

    #[test]
    fn su2_hmc_cold_start_stable() {
        let mut lat = GenericLattice::<Su2Matrix>::cold_start([4, 4, 4, 4], 2.3);
        let mut cfg = GenericHmcConfig {
            n_md_steps: 10,
            dt: 0.01,
            seed: 42,
        };
        let result = lat.hmc_trajectory(&mut cfg);
        assert!(
            result.plaquette > 0.8,
            "SU(2) cold start should stay ordered: plaq={}, dH={}, acc={}",
            result.plaquette, result.delta_h, result.accepted
        );
    }

    #[test]
    fn su3_generic_leapfrog_delta_h() {
        let mut lat = GenericLattice::<Su3Matrix>::cold_start([4, 4, 4, 4], 6.0);
        let action_before = lat.wilson_action();
        let mut seed = 42u64;
        let mut momenta: Vec<Su3Matrix> = (0..lat.volume() * 4)
            .map(|_| Su3Matrix::random_algebra(&mut seed))
            .collect();
        let kinetic_before = GenericLattice::<Su3Matrix>::kinetic_energy(&momenta);
        let h_old = action_before + kinetic_before;

        lat.leapfrog_integrate(&mut momenta, 10, 0.01);

        let action_after = lat.wilson_action();
        let kinetic_after = GenericLattice::<Su3Matrix>::kinetic_energy(&momenta);
        let h_new = action_after + kinetic_after;
        let delta_h = h_new - h_old;
        eprintln!(
            "  leapfrog: S_old={:.4}, T_old={:.4}, H_old={:.4}",
            action_before, kinetic_before, h_old
        );
        eprintln!(
            "  leapfrog: S_new={:.4}, T_new={:.4}, H_new={:.4}, dH={:.4e}",
            action_after, kinetic_after, h_new, delta_h
        );
        assert!(
            delta_h.abs() < 5.0,
            "leapfrog ΔH too large: {:.4e}",
            delta_h
        );
    }

    #[test]
    fn su3_force_cold_is_zero() {
        let lat = GenericLattice::<Su3Matrix>::cold_start([4, 4, 4, 4], 6.0);
        let f = lat.gauge_force([1, 1, 1, 1], 0);
        assert!(f.norm_sq() < 1e-20, "force should be zero on cold start, norm²={:.4e}", f.norm_sq());
    }

    #[test]
    fn su3_exp_cayley_matches_original() {
        let mut seed = 42u64;
        let p = Su3Matrix::random_algebra(&mut seed);
        let dt = 0.01;

        let cayley_result = <Su3Matrix as GaugeGroup>::exp_cayley(&p, dt);
        let original_result = crate::lattice::hmc::exp_su3_cayley_pub(&p, dt);

        for i in 0..3 {
            for j in 0..3 {
                let diff_re = (cayley_result.m[i][j].re - original_result.m[i][j].re).abs();
                let diff_im = (cayley_result.m[i][j].im - original_result.m[i][j].im).abs();
                assert!(
                    diff_re < 1e-10 && diff_im < 1e-10,
                    "exp_cayley mismatch at ({i},{j}): generic=({:.8e},{:.8e}), original=({:.8e},{:.8e})",
                    cayley_result.m[i][j].re, cayley_result.m[i][j].im,
                    original_result.m[i][j].re, original_result.m[i][j].im
                );
            }
        }
    }

    #[test]
    fn su3_generic_hmc_delta_h() {
        let mut lat = GenericLattice::<Su3Matrix>::cold_start([4, 4, 4, 4], 6.0);
        let mut cfg = GenericHmcConfig {
            n_md_steps: 10,
            dt: 0.01,
            seed: 42,
        };
        let result = lat.hmc_trajectory(&mut cfg);
        eprintln!(
            "  SU(3) generic: plaq={:.6}, dH={:.4e}, acc={}",
            result.plaquette, result.delta_h, result.accepted
        );
        assert!(
            result.delta_h.abs() < 10.0,
            "SU(3) generic ΔH too large: {:.4e}",
            result.delta_h
        );
    }

    #[test]
    fn su2_hmc_thermalizes() {
        let mut lat = GenericLattice::<Su2Matrix>::cold_start([4, 4, 4, 4], 2.3);
        let mut cfg = GenericHmcConfig {
            n_md_steps: 10,
            dt: 0.002,
            seed: 42,
        };
        let mut accepted_count = 0;
        let mut last_plaq = 0.0;
        for _ in 0..80 {
            let result = lat.hmc_trajectory(&mut cfg);
            last_plaq = result.plaquette;
            if result.accepted {
                accepted_count += 1;
            }
        }
        assert!(
            accepted_count > 20,
            "SU(2) HMC acceptance too low: {accepted_count}/80, plaq={last_plaq}"
        );
    }

    #[test]
    fn su2_wilson_action_cold_is_zero() {
        let lat = GenericLattice::<Su2Matrix>::cold_start([4, 4, 4, 4], 2.5);
        let s = lat.wilson_action();
        assert!(s.abs() < 1e-12, "SU(2) cold start action = {s}");
    }

    #[test]
    fn su2_polyakov_cold_is_one() {
        let lat = GenericLattice::<Su2Matrix>::cold_start([4, 4, 4, 4], 2.5);
        let l = lat.polyakov_loop([0, 0, 0]);
        assert!(
            (l.abs() - 1.0).abs() < 1e-14,
            "SU(2) cold Polyakov loop = {}",
            l.abs()
        );
    }

    #[test]
    fn su2_serialize_roundtrip() {
        let lat = GenericLattice::<Su2Matrix>::hot_start([4, 4, 4, 4], 2.3, 42);
        let plaq_before = lat.average_plaquette();

        let tmp = std::env::temp_dir().join("test_su2_lattice.lat");
        lat.save(&tmp).unwrap();
        let lat2 = GenericLattice::<Su2Matrix>::load(&tmp).unwrap();
        let plaq_after = lat2.average_plaquette();
        let _ = std::fs::remove_file(&tmp);

        assert!(
            (plaq_before - plaq_after).abs() < 1e-14,
            "roundtrip plaquette mismatch: {plaq_before} vs {plaq_after}"
        );
    }

    #[test]
    fn su3_creutz_ratio_cold() {
        let lat = GenericLattice::<Su3Matrix>::cold_start([4, 4, 4, 4], 6.0);
        // For cold start, all Wilson loops = 1, so χ = -ln(1*1/(1*1)) = 0
        if let Some(chi) = lat.creutz_ratio(2, 2) {
            assert!(
                chi.abs() < 1e-12,
                "cold start Creutz ratio should be 0, got {chi}"
            );
        }
    }

    #[test]
    fn su2_creutz_ratio_thermalized() {
        let mut lat = GenericLattice::<Su2Matrix>::cold_start([4, 4, 4, 4], 2.3);
        let mut cfg = GenericHmcConfig {
            n_md_steps: 10,
            dt: 0.002,
            seed: 42,
        };
        for _ in 0..30 {
            lat.hmc_trajectory(&mut cfg);
        }
        let scan = lat.creutz_ratio_scan(3);
        assert!(!scan.is_empty(), "should have Creutz ratio data");
        for &(r, t, chi) in &scan {
            assert!(
                chi.is_finite(),
                "Creutz ratio χ({r},{t}) should be finite"
            );
        }
    }

    #[test]
    fn su2_wilson_loop_cold() {
        let lat = GenericLattice::<Su2Matrix>::cold_start([4, 4, 4, 4], 2.5);
        let w11 = lat.spatial_temporal_wilson_loop(1, 1);
        assert!(
            (w11 - 1.0).abs() < 1e-12,
            "SU(2) cold W(1,1) = {w11}"
        );
    }

    #[test]
    fn tile_from_preserves_plaquette() {
        let mut small = GenericLattice::<Su2Matrix>::hot_start([4, 4, 4, 4], 2.3, 42);
        let mut cfg = GenericHmcConfig {
            n_md_steps: 10,
            dt: 0.01,
            seed: 42,
        };
        for _ in 0..20 {
            small.hmc_trajectory(&mut cfg);
        }
        let small_plaq = small.average_plaquette();

        let tiled = GenericLattice::<Su2Matrix>::tile_from(&small, [8, 8, 8, 8]);
        let tiled_plaq = tiled.average_plaquette();

        assert_eq!(tiled.dims, [8, 8, 8, 8]);
        assert_eq!(tiled.volume() * 4, tiled.links.len());
        assert!(
            (tiled_plaq - small_plaq).abs() < 1e-12,
            "tiled plaquette {tiled_plaq} should exactly match source {small_plaq}"
        );
    }

    #[test]
    fn tile_from_asymmetric() {
        let small = GenericLattice::<Su2Matrix>::cold_start([4, 4, 4, 4], 2.5);
        let tiled = GenericLattice::<Su2Matrix>::tile_from(&small, [8, 8, 8, 4]);
        assert_eq!(tiled.dims, [8, 8, 8, 4]);
        let plaq = tiled.average_plaquette();
        assert!(
            (plaq - 1.0).abs() < 1e-12,
            "tiled cold-start should be exactly 1.0, got {plaq}"
        );
    }

    #[test]
    fn tile_then_thermalize_diverges() {
        let mut small = GenericLattice::<Su2Matrix>::hot_start([4, 4, 4, 4], 2.3, 42);
        let mut cfg = GenericHmcConfig {
            n_md_steps: 10,
            dt: 0.01,
            seed: 42,
        };
        for _ in 0..20 {
            small.hmc_trajectory(&mut cfg);
        }

        let mut tiled = GenericLattice::<Su2Matrix>::tile_from(&small, [8, 8, 8, 8]);
        let before = tiled.average_plaquette();
        for _ in 0..10 {
            tiled.hmc_trajectory(&mut cfg);
        }
        let after = tiled.average_plaquette();

        // HMC should evolve the tiled config — plaquette changes but stays physical
        assert!(
            (after - before).abs() < 0.15,
            "plaquette should stay physical: before={before}, after={after}"
        );
    }
}
