// SPDX-License-Identifier: AGPL-3.0-or-later

//! SU(N) parallel CPU thermalizer — memo table producer for the full gauge group ladder.
//!
//! Thermalizes lattice configurations for SU(2), SU(3), SU(4), SU(5), SU(6), SU(8)
//! across volumes and β values specified in the plan. Each config is stored in a
//! gauge-group-specific subdirectory with BLAKE3 content addressing.
//!
//! Usage:
//!   cargo run --release --bin arxiv_thermalize_sun
//!
//! Environment:
//!   THERM_THREADS=N   — override rayon thread count
//!   SUN_GROUP=2       — only thermalize SU(2) (skip others)
//!   SUN_GROUP=3       — only thermalize SU(3) (etc.)

use hotspring_barracuda::lattice::generic_lattice::{GenericHmcConfig, GenericLattice};
use hotspring_barracuda::lattice::su2::Su2Matrix;
use hotspring_barracuda::lattice::su_n::SuNMatrix;
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

struct ThermSpec {
    gauge_group: &'static str,
    nc: usize,
    dims: [usize; 4],
    beta: f64,
    seed: u64,
    n_therm: usize,
    dt: f64,
    n_md_steps: usize,
}

impl ThermSpec {
    fn label(&self) -> String {
        let l = self.dims[0];
        let l_t = self.dims[3];
        if l == l_t {
            format!(
                "SU({}) {}⁴ β={:.2} seed={}",
                self.nc, l, self.beta, self.seed
            )
        } else {
            format!(
                "SU({}) {}³×{} β={:.2} seed={}",
                self.nc, l, l_t, self.beta, self.seed
            )
        }
    }

    fn cache_key(&self) -> String {
        let input = format!(
            "{}_{}x{}x{}x{}_b{:.6}_s{}_t{}_omelyan",
            self.gauge_group,
            self.dims[0], self.dims[1], self.dims[2], self.dims[3],
            self.beta, self.seed, self.n_therm,
        );
        let hash = blake3::hash(input.as_bytes());
        format!("{}", hash.to_hex())
    }

    fn cache_dir(&self) -> PathBuf {
        let dir = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("hotspring")
            .join("configs")
            .join(self.gauge_group);
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    fn cache_path(&self) -> PathBuf {
        self.cache_dir().join(format!("{}.lat", &self.cache_key()[..16]))
    }

    fn mem_bytes(&self) -> usize {
        let vol: usize = self.dims.iter().product();
        vol * 4 * 2 * self.nc * self.nc * 8
    }
}

fn build_grid(filter_nc: Option<usize>) -> Vec<ThermSpec> {
    let mut specs = Vec::new();
    let seeds: &[u64] = &[42, 137, 271];

    // ═══ SU(2) ═══
    // Published comparison: Liddle-Teper (2008) deconfinement, β_c at Nt=4 ~ 2.30
    if filter_nc.is_none() || filter_nc == Some(2) {
        let su2_betas = [2.2, 2.3, 2.5];
        for &l in &[16, 24, 32] {
            let n_seeds = if l <= 24 { 3 } else { 1 };
            let (n_therm, dt, n_md) = match l {
                16 => (200, 0.01, 20),
                24 => (300, 0.008, 30),
                _ => (400, 0.005, 40),
            };
            for &beta in &su2_betas {
                for &seed in &seeds[..n_seeds] {
                    specs.push(ThermSpec {
                        gauge_group: "su2", nc: 2,
                        dims: [l, l, l, l], beta, seed, n_therm, dt, n_md_steps: n_md,
                    });
                }
            }
        }
    }

    // ═══ SU(3) ═══ (already handled by arxiv_thermalize_grid, but included for completeness)
    if filter_nc.is_none() || filter_nc == Some(3) {
        let su3_betas = [5.9, 6.0, 6.2];
        for &l in &[16, 24, 32] {
            let n_seeds = if l <= 24 { 3 } else { 1 };
            let (n_therm, dt, n_md) = match l {
                16 => (200, 0.01, 40),
                24 => (200, 0.008, 50),
                _ => (300, 0.005, 60),
            };
            for &beta in &su3_betas {
                for &seed in &seeds[..n_seeds] {
                    specs.push(ThermSpec {
                        gauge_group: "su3", nc: 3,
                        dims: [l, l, l, l], beta, seed, n_therm, dt, n_md_steps: n_md,
                    });
                }
            }
        }
    }

    // ═══ SU(4) ═══
    // Published: Lucini-Teper-Wenger (2004), β_c(Nt=4) ~ 10.7
    if filter_nc.is_none() || filter_nc == Some(4) {
        let su4_betas = [10.5, 10.7, 11.0];
        for &l in &[16, 24] {
            let (n_therm, dt, n_md) = if l == 16 {
                (200, 0.008, 50)
            } else {
                (300, 0.005, 60)
            };
            for &beta in &su4_betas {
                specs.push(ThermSpec {
                    gauge_group: "su4", nc: 4,
                    dims: [l, l, l, l], beta, seed: 42, n_therm, dt, n_md_steps: n_md,
                });
            }
        }
    }

    // ═══ SU(5) ═══
    // Published: Gonzalez-Arroyo-Okawa (2014)
    if filter_nc.is_none() || filter_nc == Some(5) {
        let su5_betas = [16.5, 17.0, 17.5];
        for &beta in &su5_betas {
            specs.push(ThermSpec {
                gauge_group: "su5", nc: 5,
                dims: [16, 16, 16, 16], beta, seed: 42,
                n_therm: 200, dt: 0.006, n_md_steps: 50,
            });
        }
    }

    // ═══ SU(6) ═══
    // Published: Lucini-Teper-Wenger (2004)
    if filter_nc.is_none() || filter_nc == Some(6) {
        let su6_betas = [24.0, 25.0, 26.0];
        for &beta in &su6_betas {
            specs.push(ThermSpec {
                gauge_group: "su6", nc: 6,
                dims: [16, 16, 16, 16], beta, seed: 42,
                n_therm: 200, dt: 0.005, n_md_steps: 60,
            });
        }
    }

    // ═══ SU(8) ═══
    // Published: Gonzalez-Arroyo-Okawa, sparse data
    if filter_nc.is_none() || filter_nc == Some(8) {
        let su8_betas = [44.0, 45.0, 46.0];
        for &beta in &su8_betas {
            specs.push(ThermSpec {
                gauge_group: "su8", nc: 8,
                dims: [16, 16, 16, 16], beta, seed: 42,
                n_therm: 200, dt: 0.004, n_md_steps: 80,
            });
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Finite-Temperature: Asymmetric Ns³ × Nt for deconfinement scans
    // ═══════════════════════════════════════════════════════════════
    // Polyakov loop ⟨|L|⟩ signals deconfinement at Tc.
    // Tc/√σ approaches 0.5949(17) at large N ('t Hooft conjecture).

    // SU(3) finite-T: 24³ × Nt, scan β across the Nt=6,8,10 transitions
    // β_c(Nt=6) ≈ 5.89, β_c(Nt=8) ≈ 6.06, β_c(Nt=10) ≈ 6.20
    if filter_nc.is_none() || filter_nc == Some(3) {
        let su3_ft_betas: &[&[f64]] = &[
            &[5.80, 5.85, 5.89, 5.93, 5.98],   // Nt=6 window
            &[5.96, 6.01, 6.06, 6.11, 6.16],   // Nt=8 window
            &[6.10, 6.15, 6.20, 6.25, 6.30],   // Nt=10 window
        ];
        for (nt_idx, &nt) in [6, 8, 10].iter().enumerate() {
            for &beta in su3_ft_betas[nt_idx] {
                specs.push(ThermSpec {
                    gauge_group: "su3", nc: 3,
                    dims: [24, 24, 24, nt], beta, seed: 42,
                    n_therm: 300, dt: 0.008, n_md_steps: 50,
                });
            }
        }
    }

    // SU(2) finite-T: 24³ × Nt, scan β across the Nt=6,8,10 transitions
    // β_c(Nt=4) ≈ 2.30, β_c(Nt=6) ≈ 2.43, β_c(Nt=8) ≈ 2.51
    if filter_nc.is_none() || filter_nc == Some(2) {
        let su2_ft_betas: &[&[f64]] = &[
            &[2.35, 2.39, 2.43, 2.47, 2.51],   // Nt=6 window
            &[2.43, 2.47, 2.51, 2.55, 2.59],   // Nt=8 window
            &[2.51, 2.55, 2.59, 2.63, 2.67],   // Nt=10 window
        ];
        for (nt_idx, &nt) in [6, 8, 10].iter().enumerate() {
            for &beta in su2_ft_betas[nt_idx] {
                specs.push(ThermSpec {
                    gauge_group: "su2", nc: 2,
                    dims: [24, 24, 24, nt], beta, seed: 42,
                    n_therm: 300, dt: 0.008, n_md_steps: 30,
                });
            }
        }
    }

    specs
}

fn thermalize_su2(spec: &ThermSpec) -> Result<blake3::Hash, String> {
    let mut lat = GenericLattice::<Su2Matrix>::hot_start(spec.dims, spec.beta, spec.seed);
    let mut cfg = GenericHmcConfig {
        n_md_steps: spec.n_md_steps, dt: spec.dt, seed: spec.seed,
    };
    for t in 0..spec.n_therm {
        lat.hmc_trajectory(&mut cfg);
        if (t + 1) % 50 == 0 {
            eprintln!("      {} — step {}/{}, ⟨P⟩={:.6}", spec.label(), t + 1, spec.n_therm, lat.average_plaquette());
        }
    }
    lat.save(&spec.cache_path()).map_err(|e| e.to_string())
}

fn thermalize_su3(spec: &ThermSpec) -> Result<blake3::Hash, String> {
    use hotspring_barracuda::lattice::hmc::{self, HmcConfig, IntegratorType};
    use hotspring_barracuda::lattice::wilson::Lattice;

    let mut lat = Lattice::hot_start(spec.dims, spec.beta, spec.seed);
    let cfg = &mut HmcConfig {
        n_md_steps: spec.n_md_steps, dt: spec.dt, seed: spec.seed,
        integrator: IntegratorType::Omelyan,
    };
    for t in 0..spec.n_therm {
        hmc::hmc_trajectory(&mut lat, cfg);
        if (t + 1) % 50 == 0 {
            eprintln!("      {} — step {}/{}, ⟨P⟩={:.6}", spec.label(), t + 1, spec.n_therm, lat.average_plaquette());
        }
    }
    lat.save(&spec.cache_path()).map_err(|e| e.to_string())
}

fn thermalize_sun(spec: &ThermSpec) -> Result<blake3::Hash, String> {
    let nc = spec.nc;
    let mut lat = GenericLattice::<SuNMatrix>::hot_start_nc(spec.dims, spec.beta, nc, spec.seed);
    let mut cfg = GenericHmcConfig {
        n_md_steps: spec.n_md_steps, dt: spec.dt, seed: spec.seed,
    };
    for t in 0..spec.n_therm {
        lat.hmc_trajectory(&mut cfg);
        if (t + 1) % 50 == 0 {
            eprintln!("      {} — step {}/{}, ⟨P⟩={:.6}", spec.label(), t + 1, spec.n_therm, lat.average_plaquette());
        }
    }
    // SuNMatrix save: use the generic lattice save
    lat.save(&spec.cache_path()).map_err(|e| e.to_string())
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  SU(N) Parallel CPU Thermalizer — Full Gauge Group Ladder   ║");
    println!("║  N=2,3,4,5,6,8 · Omelyan 2MN · BLAKE3 content-addressed    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let filter_nc: Option<usize> = std::env::var("SUN_GROUP")
        .ok()
        .and_then(|s| s.parse().ok());

    if let Some(nc) = filter_nc {
        println!("  Filter: SU({nc}) only");
    }

    let n_threads: usize = std::env::var("THERM_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            let cpus = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
            (cpus / 2).max(2)
        });

    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .expect("rayon thread pool");

    println!("  Rayon threads: {n_threads}");

    let grid = build_grid(filter_nc);

    let cached: Vec<bool> = grid.iter().map(|s| s.cache_path().exists()).collect();
    let n_cached = cached.iter().filter(|&&c| c).count();
    let n_total = grid.len();
    let n_todo = n_total - n_cached;

    println!("  Grid: {n_total} configs ({n_cached} cached, {n_todo} to thermalize)");
    println!();

    for (i, spec) in grid.iter().enumerate() {
        let status = if cached[i] { "[CACHED]" } else { "[TODO]  " };
        let mem_mb = spec.mem_bytes() / (1024 * 1024);
        println!(
            "    {status} {} — {n_therm} therm, dt={dt}, ~{mem_mb} MB",
            spec.label(), n_therm = spec.n_therm, dt = spec.dt,
        );
    }
    println!();

    if n_todo == 0 {
        println!("  All configs already cached. Nothing to do.");
        return;
    }

    let completed = AtomicUsize::new(0);
    let total_start = Instant::now();

    grid.par_iter().enumerate().for_each(|(i, spec)| {
        if cached[i] {
            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            eprintln!("  [{done}/{n_total}] [CACHED] {}", spec.label());
            return;
        }

        let start = Instant::now();
        eprintln!("  [START] {} — thermalizing...", spec.label());

        let result = match spec.nc {
            2 => thermalize_su2(spec),
            3 => thermalize_su3(spec),
            _ => thermalize_sun(spec),
        };

        let elapsed = start.elapsed().as_secs_f64();
        let done = completed.fetch_add(1, Ordering::Relaxed) + 1;

        match result {
            Ok(hash) => {
                eprintln!(
                    "  [{done}/{n_total}] [SAVED] {} — {elapsed:.1}s, hash={}",
                    spec.label(), &hash.to_hex()[..16]
                );
            }
            Err(e) => {
                eprintln!("  [{done}/{n_total}] [ERROR] {} — {e}", spec.label());
            }
        }
    });

    let total = total_start.elapsed().as_secs_f64();
    println!();
    println!("═══ SU(N) Thermalization Complete ═══");
    println!("  Total wall time: {total:.1}s ({:.1} min)", total / 60.0);
}
