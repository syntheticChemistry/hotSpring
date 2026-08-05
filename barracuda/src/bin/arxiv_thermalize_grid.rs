// SPDX-License-Identifier: AGPL-3.0-or-later

//! Parallel CPU thermalizer — producer side of the config cache pipeline.
//!
//! Thermalizes lattice configurations across all available CPU cores using rayon,
//! saving each thermalized config to a content-addressed cache directory.
//! GPU production binaries can then load these configs instantly, eliminating
//! the single-threaded CPU thermalization bottleneck.
//!
//! Usage:
//!   cargo run --release --bin arxiv_thermalize_grid --features barracuda-local
//!
//! Environment:
//!   THERM_THREADS=N    — override rayon thread count (default: num_cpus / 2)

use hotspring_barracuda::lattice::hmc::{self, HmcConfig, IntegratorType};
use hotspring_barracuda::lattice::wilson::Lattice;
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

struct ThermJob {
    dims: [usize; 4],
    beta: f64,
    seed: u64,
    n_therm: usize,
    dt: f64,
    n_md_steps: usize,
}

impl ThermJob {
    fn label(&self) -> String {
        format!(
            "{}⁴ β={:.1} seed={}",
            self.dims[0], self.beta, self.seed
        )
    }

    fn cache_key(&self) -> String {
        Lattice::cache_key(
            self.dims,
            self.beta,
            self.seed,
            self.n_therm,
            "omelyan",
        )
    }

    fn cache_path(&self, cache_dir: &PathBuf) -> PathBuf {
        cache_dir.join(format!("{}.lat", &self.cache_key()[..16]))
    }

    fn legacy_cache_path(&self) -> PathBuf {
        let key = Lattice::legacy_cache_key(
            self.dims,
            self.beta,
            self.seed,
            self.n_therm,
            "omelyan",
        );
        Lattice::config_cache_root().join(format!("{}.lat", &key[..16]))
    }
}

fn build_grid() -> Vec<ThermJob> {
    let mut jobs = Vec::new();

    // (beta, n_therm, dt, n_md_steps)
    let scan_16: &[(f64, usize, f64, usize)] = &[
        (5.9, 200, 0.01, 40),
        (6.0, 200, 0.01, 40),
        (6.2, 200, 0.01, 40),
    ];

    let scan_24: &[(f64, usize, f64, usize)] = &[
        (6.0, 200, 0.008, 50),
    ];

    // 32⁴: minimal publishable volume. Smaller dt for stability at larger volume.
    let scan_32: &[(f64, usize, f64, usize)] = &[
        (5.9, 300, 0.005, 60),
        (6.0, 300, 0.005, 60),
        (6.2, 300, 0.005, 60),
    ];

    let seeds: &[u64] = &[42, 137, 271];

    for &(beta, n_therm, dt, n_md) in scan_16 {
        for &seed in seeds {
            jobs.push(ThermJob {
                dims: [16, 16, 16, 16],
                beta,
                seed,
                n_therm,
                dt,
                n_md_steps: n_md,
            });
        }
    }

    for &(beta, n_therm, dt, n_md) in scan_24 {
        for &seed in &seeds[..1] {
            jobs.push(ThermJob {
                dims: [24, 24, 24, 24],
                beta,
                seed,
                n_therm,
                dt,
                n_md_steps: n_md,
            });
        }
    }

    for &(beta, n_therm, dt, n_md) in scan_32 {
        for &seed in &seeds[..1] {
            jobs.push(ThermJob {
                dims: [32, 32, 32, 32],
                beta,
                seed,
                n_therm,
                dt,
                n_md_steps: n_md,
            });
        }
    }

    jobs
}

fn migrate_legacy_configs(new_dir: &PathBuf) {
    let root = Lattice::config_cache_root();
    let Ok(entries) = std::fs::read_dir(&root) else { return };
    let mut migrated = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("lat") && path.is_file() {
            let fname = path.file_name().unwrap().to_string_lossy().to_string();
            let dest = new_dir.join(&fname);
            if !dest.exists() {
                if let Err(e) = std::fs::rename(&path, &dest) {
                    eprintln!("  [MIGRATE] failed to move {fname}: {e}");
                } else {
                    migrated += 1;
                }
            }
        }
    }
    if migrated > 0 {
        println!("  Migrated {migrated} legacy configs → {}", new_dir.display());
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Parallel CPU Thermalizer — Config Cache Producer           ║");
    println!("║  SU(3) pure gauge, Omelyan 2MN, BLAKE3-addressed           ║");
    println!("║  Target: 32⁴ minimal publish / 48⁴ stretch                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let cache_dir = Lattice::config_cache_dir();
    println!("  Cache directory: {}", cache_dir.display());

    migrate_legacy_configs(&cache_dir);

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
    println!();

    let grid = build_grid();

    let cached: Vec<bool> = grid
        .iter()
        .map(|j| j.cache_path(&cache_dir).exists() || j.legacy_cache_path().exists())
        .collect();
    let n_cached = cached.iter().filter(|&&c| c).count();
    let n_total = grid.len();
    let n_todo = n_total - n_cached;

    println!("  Grid: {n_total} configs ({n_cached} cached, {n_todo} to thermalize)");
    println!();

    if n_todo == 0 {
        println!("  All configs already cached. Nothing to do.");
        return;
    }

    for (i, job) in grid.iter().enumerate() {
        let status = if cached[i] { "[CACHED]" } else { "[TODO]  " };
        let mem_mb = job.dims[0].pow(4) * 4 * 18 * 8 / (1024 * 1024);
        println!(
            "    {status} {} — {n_therm} therm, dt={dt}, N_md={n_md}, ~{mem_mb} MB",
            job.label(),
            n_therm = job.n_therm,
            dt = job.dt,
            n_md = job.n_md_steps,
        );
    }
    println!();

    let completed = AtomicUsize::new(0);
    let total_start = Instant::now();

    grid.par_iter().enumerate().for_each(|(i, job)| {
        if cached[i] {
            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            eprintln!("  [{done}/{n_total}] [CACHED] {}", job.label());
            return;
        }

        let start = Instant::now();
        eprintln!("  [START] {} — thermalizing...", job.label());

        let mut lat = Lattice::hot_start(job.dims, job.beta, job.seed);
        let cfg = &mut HmcConfig {
            n_md_steps: job.n_md_steps,
            dt: job.dt,
            seed: job.seed,
            integrator: IntegratorType::Omelyan,
        };

        for t in 0..job.n_therm {
            hmc::hmc_trajectory(&mut lat, cfg);
            if (t + 1) % 50 == 0 {
                let plaq = lat.average_plaquette();
                let elapsed = start.elapsed().as_secs_f64();
                let rate = (t + 1) as f64 / elapsed;
                eprintln!(
                    "    {} — step {}/{}, ⟨P⟩={plaq:.6}, {rate:.1} traj/s",
                    job.label(),
                    t + 1,
                    job.n_therm
                );
            }
        }

        let path = job.cache_path(&cache_dir);
        match lat.save(&path) {
            Ok(hash) => {
                let elapsed = start.elapsed().as_secs_f64();
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                let plaq = lat.average_plaquette();
                eprintln!(
                    "  [{done}/{n_total}] [SAVED] {} — ⟨P⟩={plaq:.6}, {elapsed:.1}s, hash={}",
                    job.label(),
                    &hash.to_hex()[..16]
                );
            }
            Err(e) => {
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                eprintln!("  [{done}/{n_total}] [ERROR] {} — {e}", job.label());
            }
        }
    });

    let total = total_start.elapsed().as_secs_f64();
    println!();
    println!("═══ Thermalization Complete ═══");
    println!("  Total wall time: {total:.1}s ({:.1} min)", total / 60.0);
    println!("  Configs cached: {}", cache_dir.display());
    println!(
        "  Files: {}",
        std::fs::read_dir(&cache_dir)
            .map(|rd| rd.count())
            .unwrap_or(0)
    );
    println!();
    println!("  Next: cargo run --release --bin arxiv_volume_scan --features barracuda-local");
    println!("  (GPU production will load cached configs instantly)");
}
