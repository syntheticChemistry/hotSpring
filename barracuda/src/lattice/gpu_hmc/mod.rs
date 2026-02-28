// SPDX-License-Identifier: AGPL-3.0-only

//! Pure GPU HMC: all math on GPU via fp64 WGSL shaders.
//!
//! The CPU only orchestrates dispatches and reads back scalars (ΔH,
//! plaquette) for Metropolis accept/reject and observables. Links and
//! momenta live in GPU buffers between MD steps — no round-trip.
//!
//! # Shader pipeline (per Omelyan MD step)
//!
//! 1. `su3_gauge_force_f64` — F(U) for all links
//! 2. `su3_momentum_update_f64` — P += dt * F
//! 3. `su3_link_update_f64` — U = exp(dt·P) * U (Cayley + reunitarize)
//!
//! # Observable shaders
//!
//! - `wilson_plaquette_f64` — per-site plaquette sum
//! - `su3_kinetic_energy_f64` — per-link -½ Re Tr(P²)
//!
//! # Module structure
//!
//! | Module | Responsibility |
//! |--------|---------------|
//! | root (this file) | Shader constants, shared types, dispatch helpers |
//! | `dynamical` | Dynamical fermion HMC (full QCD with staggered quarks) |
//! | `streaming` | Zero-dispatch-overhead streaming via batched encoders |
//! | `resident_cg` | GPU-resident CG with minimal scalar readback |
//! | `observables` | Three-substrate stream integration and NPU monitoring |

pub mod dynamical;
pub mod observables;
pub mod resident_cg;
pub mod streaming;

pub use dynamical::gpu_dynamical_hmc_trajectory;
pub use dynamical::{
    GpuDynHmcPipelines, GpuDynHmcResult, GpuDynHmcState, WGSL_AXPY, WGSL_COMPLEX_DOT_RE,
    WGSL_DIRAC_STAGGERED, WGSL_FERMION_FORCE, WGSL_RANDOM_MOMENTA, WGSL_XPAY,
};
pub use observables::{BidirectionalStream, StreamObservables};
pub use resident_cg::{
    gpu_cg_solve_brain, gpu_cg_solve_resident, gpu_cg_solve_resident_async,
    gpu_dynamical_hmc_trajectory_brain, gpu_dynamical_hmc_trajectory_resident, AsyncCgReadback,
    BrainInterrupt, CgResidualUpdate, GpuResidentCgBuffers, GpuResidentCgPipelines,
    WGSL_CG_COMPUTE_ALPHA, WGSL_CG_COMPUTE_BETA, WGSL_CG_UPDATE_P, WGSL_CG_UPDATE_XR,
    WGSL_SUM_REDUCE,
};
pub use streaming::{
    gpu_dynamical_hmc_trajectory_streaming, gpu_hmc_trajectory_streaming,
    gpu_hmc_trajectory_streaming_cpu_mom, GpuDynHmcStreamingPipelines, GpuHmcStreamingPipelines,
    WGSL_GAUSSIAN_FERMION,
};

use super::wilson::Lattice;
use crate::gpu::GpuF64;

// Hardware-adaptive DF64 core streaming (hotSpring Exp 012 → toadStool S58-S60).
// Auto-selects between native f64 and DF64 (f32-pair) based on GPU hardware:
//   - Titan V, V100, A100, MI250 → native f64 (1:2 FP64:FP32)
//   - RTX 3090, 4070, consumer → DF64 on FP32 cores (force + plaquette + KE)
use barracuda::device::driver_profile::Fp64Strategy;

/// WGSL shader: Wilson plaquette per site (6 planes, Re Tr P/3).
pub const WGSL_WILSON_PLAQUETTE: &str = include_str!("../shaders/wilson_plaquette_f64.wgsl");

/// WGSL shader: SU(3) gauge force (staple + traceless anti-Hermitian projection).
pub const WGSL_GAUGE_FORCE: &str = include_str!("../shaders/su3_gauge_force_f64.wgsl");

/// WGSL shader: DF64 gauge force — staple computation on FP32 cores.
pub const WGSL_GAUGE_FORCE_DF64: &str = include_str!("../shaders/su3_gauge_force_df64.wgsl");

/// WGSL shader: momentum update P += dt * F.
pub const WGSL_MOMENTUM_UPDATE: &str = include_str!("../shaders/su3_momentum_update_f64.wgsl");

/// WGSL shader: link update U = exp(dt·P) * U via Cayley + reunitarize.
pub const WGSL_LINK_UPDATE: &str = include_str!("../shaders/su3_link_update_f64.wgsl");

/// WGSL shader: kinetic energy -½ Re Tr(P²) per link.
pub const WGSL_KINETIC_ENERGY: &str = include_str!("../shaders/su3_kinetic_energy_f64.wgsl");

/// WGSL shader: DF64 plaquette — SU(3) products on FP32 cores (neighbor-buffer indexing).
pub const WGSL_PLAQUETTE_DF64: &str = include_str!("../shaders/wilson_plaquette_df64.wgsl");

/// WGSL shader: DF64 kinetic energy — P² on FP32 cores (toadStool S60).
pub const WGSL_KINETIC_ENERGY_DF64: &str = include_str!("../shaders/su3_kinetic_energy_df64.wgsl");

/// WGSL shader: GPU Polyakov loop — temporal Wilson line on GPU.
pub const WGSL_POLYAKOV_LOOP: &str = include_str!("../shaders/polyakov_loop_f64.wgsl");

/// WGSL shader preamble: complex_f64 + su3_math_f64 (safe for composition, no ptr I/O).
const WGSL_COMPLEX_F64: &str = include_str!("../shaders/complex_f64.wgsl");
const WGSL_SU3_MATH_F64: &str = include_str!("../shaders/su3_math_f64.wgsl");

// ═══════════════════════════════════════════════════════════════════
//  Pipeline compilation
// ═══════════════════════════════════════════════════════════════════

/// GPU HMC pipeline: local shaders with hardware-adaptive DF64 for SU(3) matmul
/// kernels (force, plaquette, kinetic energy).
///
/// On consumer GPUs (RTX 3090, 4070 — 1:64 FP64:FP32), the gauge force,
/// plaquette, and kinetic energy shaders route SU(3) matrix products through
/// the FP32 core array via DF64 (f32-pair), yielding ~2.8× total trajectory
/// speedup. On compute-class GPUs (Titan V, V100 — 1:2 hardware), native f64
/// is used directly.
pub struct GpuHmcPipelines {
    /// Wilson plaquette per-site kernel
    pub plaquette_pipeline: wgpu::ComputePipeline,
    /// Gauge force kernel (DF64-capable on consumer GPUs)
    pub force_pipeline: wgpu::ComputePipeline,
    /// Momentum update kernel
    pub momentum_pipeline: wgpu::ComputePipeline,
    /// Link update (Cayley exp) kernel
    pub link_pipeline: wgpu::ComputePipeline,
    /// Kinetic energy per-link kernel
    pub kinetic_pipeline: wgpu::ComputePipeline,
    /// Polyakov loop (temporal Wilson line) — full GPU, no CPU readback
    pub polyakov_pipeline: wgpu::ComputePipeline,
    /// Which FP64 strategy was selected for this hardware
    pub fp64_strategy: Fp64Strategy,
}

impl GpuHmcPipelines {
    /// Compile all HMC shader pipelines.
    ///
    /// Automatically selects DF64 (f32-pair) shaders for the gauge force
    /// on consumer GPUs, routing staple SU(3) multiplications through
    /// the FP32 core array for ~10× throughput.
    #[must_use]
    pub fn new(gpu: &GpuF64) -> Self {
        let strategy = gpu.driver_profile().fp64_strategy();

        let df64_preamble = barracuda::ops::lattice::su3::su3_df64_preamble();

        let force_src = match strategy {
            Fp64Strategy::Native => WGSL_GAUGE_FORCE.to_string(),
            Fp64Strategy::Hybrid => format!("{df64_preamble}\n{WGSL_GAUGE_FORCE_DF64}"),
        };

        let plaq_src = match strategy {
            Fp64Strategy::Native => WGSL_WILSON_PLAQUETTE.to_string(),
            Fp64Strategy::Hybrid => format!("{df64_preamble}\n{WGSL_PLAQUETTE_DF64}"),
        };

        let ke_src = match strategy {
            Fp64Strategy::Native => WGSL_KINETIC_ENERGY.to_string(),
            Fp64Strategy::Hybrid => format!("{df64_preamble}\n{WGSL_KINETIC_ENERGY_DF64}"),
        };

        eprintln!(
            "[HMC] FP64 strategy: {:?} — {}",
            strategy,
            match strategy {
                Fp64Strategy::Native => "native f64 on all cores",
                Fp64Strategy::Hybrid =>
                    "DF64 on FP32 cores for force + plaquette + KE (~2.8× trajectory speedup)",
            }
        );

        let poly_src = format!("{WGSL_COMPLEX_F64}\n{WGSL_SU3_MATH_F64}\n{WGSL_POLYAKOV_LOOP}");

        Self {
            plaquette_pipeline: gpu.create_pipeline_f64(&plaq_src, "hmc_plaq"),
            force_pipeline: gpu.create_pipeline_f64(&force_src, "hmc_force"),
            momentum_pipeline: gpu.create_pipeline_f64(WGSL_MOMENTUM_UPDATE, "hmc_mom_update"),
            link_pipeline: gpu.create_pipeline_f64(WGSL_LINK_UPDATE, "hmc_link_update"),
            kinetic_pipeline: gpu.create_pipeline_f64(&ke_src, "hmc_ke"),
            polyakov_pipeline: gpu.create_pipeline_f64(&poly_src, "hmc_polyakov"),
            fp64_strategy: strategy,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Layout helpers (SU(3) ↔ flat f64 buffers)
// ═══════════════════════════════════════════════════════════════════

/// Flatten lattice links to f64 array (same layout as `DiracGpuLayout`).
#[must_use]
pub fn flatten_links(lattice: &Lattice) -> Vec<f64> {
    let vol = lattice.volume();
    let mut flat = vec![0.0_f64; vol * 4 * 18];
    for idx in 0..vol {
        let x = lattice.site_coords(idx);
        for mu in 0..4 {
            let u = lattice.link(x, mu);
            let base = (idx * 4 + mu) * 18;
            for row in 0..3 {
                for col in 0..3 {
                    flat[base + row * 6 + col * 2] = u.m[row][col].re;
                    flat[base + row * 6 + col * 2 + 1] = u.m[row][col].im;
                }
            }
        }
    }
    flat
}

/// Build neighbor table (same layout as `DiracGpuLayout`).
#[must_use]
pub fn build_neighbors(lattice: &Lattice) -> Vec<u32> {
    let vol = lattice.volume();
    let mut neighbors = vec![0_u32; vol * 8];
    for idx in 0..vol {
        let x = lattice.site_coords(idx);
        for mu in 0..4 {
            let fwd = lattice.site_index(lattice.neighbor(x, mu, true));
            let bwd = lattice.site_index(lattice.neighbor(x, mu, false));
            neighbors[idx * 8 + mu * 2] = fwd as u32;
            neighbors[idx * 8 + mu * 2 + 1] = bwd as u32;
        }
    }
    neighbors
}

/// Flatten SU(3) momenta to f64 array.
#[must_use]
pub fn flatten_momenta(momenta: &[super::su3::Su3Matrix]) -> Vec<f64> {
    let mut flat = vec![0.0_f64; momenta.len() * 18];
    for (i, p) in momenta.iter().enumerate() {
        let base = i * 18;
        for row in 0..3 {
            for col in 0..3 {
                flat[base + row * 6 + col * 2] = p.m[row][col].re;
                flat[base + row * 6 + col * 2 + 1] = p.m[row][col].im;
            }
        }
    }
    flat
}

/// Unflatten f64 array back to SU(3) link matrices and update lattice.
pub fn unflatten_links_into(lattice: &mut Lattice, flat: &[f64]) {
    let vol = lattice.volume();
    for idx in 0..vol {
        let x = lattice.site_coords(idx);
        for mu in 0..4 {
            let base = (idx * 4 + mu) * 18;
            let mut m = super::su3::Su3Matrix::ZERO;
            for row in 0..3 {
                for col in 0..3 {
                    m.m[row][col] = super::complex_f64::Complex64::new(
                        flat[base + row * 6 + col * 2],
                        flat[base + row * 6 + col * 2 + 1],
                    );
                }
            }
            lattice.set_link(x, mu, m);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Pure gauge HMC types and trajectory
// ═══════════════════════════════════════════════════════════════════

/// Result of a single GPU HMC trajectory.
pub struct GpuHmcResult {
    /// Whether the Metropolis test accepted this trajectory.
    pub accepted: bool,
    /// ΔH = H_new - H_old
    pub delta_h: f64,
    /// Average plaquette after trajectory (whether accepted or rejected).
    pub plaquette: f64,
}

/// GPU-resident HMC state: buffers that persist across trajectories.
pub struct GpuHmcState {
    /// SU(3) link field U_μ(x), flattened to f64 (n_links × 18).
    pub link_buf: wgpu::Buffer,
    /// Backup copy of link_buf for Metropolis reject rollback.
    pub link_backup: wgpu::Buffer,
    /// Conjugate momenta P_μ(x), same layout as link_buf.
    pub mom_buf: wgpu::Buffer,
    /// Gauge force ∂S/∂U accumulated per link.
    pub force_buf: wgpu::Buffer,
    /// Per-link kinetic energy T = Tr(P†P)/2 output buffer.
    pub ke_out_buf: wgpu::Buffer,
    /// Per-site plaquette sum output buffer.
    pub plaq_out_buf: wgpu::Buffer,
    /// Per-site Polyakov loop output buffer: (Re, Im) per spatial site.
    pub poly_out_buf: wgpu::Buffer,
    /// Uniform parameter buffer for Polyakov loop shader.
    pub poly_params_buf: wgpu::Buffer,
    /// Neighbor index table: 8 neighbors per site (±μ for μ=0..3).
    pub nbr_buf: wgpu::Buffer,
    /// Lattice dimensions `[Nx, Ny, Nz, Nt]`.
    pub dims: [usize; 4],
    /// Number of lattice sites (product of all dimensions).
    pub volume: usize,
    /// Number of gauge links (volume × N_DIM).
    pub n_links: usize,
    /// Gauge coupling β = 2N_c/g².
    pub beta: f64,
    /// Spatial volume Nx × Ny × Nz.
    pub spatial_vol: usize,
    /// GPU workgroup count for link-indexed dispatches.
    pub wg_links: u32,
    /// GPU workgroup count for site-indexed dispatches.
    pub wg_vol: u32,
}

/// Polyakov loop shader uniform params — must match WGSL PolyParams layout.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PolyParams {
    nt: u32,
    nx: u32,
    ny: u32,
    nz: u32,
    volume: u32,
    spatial_vol: u32,
    _pad0: u32,
    _pad1: u32,
}

impl GpuHmcState {
    /// Upload a lattice to GPU and create all persistent buffers.
    ///
    /// Includes NVK allocation guard (toadStool cross-spring evolution):
    /// checks total estimated VRAM against nouveau driver PTE fault limit.
    #[must_use]
    pub fn from_lattice(gpu: &GpuF64, lattice: &Lattice, beta: f64) -> Self {
        let vol = lattice.volume();
        let n_links = vol * 4;
        let [nx, ny, nz, nt] = lattice.dims;
        let spatial_vol = nx * ny * nz;

        let link_bytes = (n_links * 18 * 8) as u64;
        let total_estimate = 6 * link_bytes
            + (vol as u64 * 8)
            + (n_links as u64 * 8)
            + (spatial_vol as u64 * 2 * 8)
            + (vol as u64 * 8 * 4);
        let profile = gpu.driver_profile();
        if let Err(e) = profile.check_allocation_safe(total_estimate) {
            eprintln!("[HMC] NVK allocation guard: {e}");
            eprintln!(
                "[HMC] Total estimated: {:.1} MB",
                total_estimate as f64 / 1e6
            );
        }

        let links_flat = flatten_links(lattice);
        let neighbors = build_neighbors(lattice);

        let link_buf = gpu.create_f64_output_buffer(n_links * 18, "hmc_links");
        gpu.upload_f64(&link_buf, &links_flat);
        let link_backup = gpu.create_f64_output_buffer(n_links * 18, "hmc_links_backup");
        let mom_buf = gpu.create_f64_output_buffer(n_links * 18, "hmc_momenta");
        let force_buf = gpu.create_f64_output_buffer(n_links * 18, "hmc_force");
        let ke_out_buf = gpu.create_f64_output_buffer(n_links, "hmc_ke");
        let plaq_out_buf = gpu.create_f64_output_buffer(vol, "hmc_plaq");
        let poly_out_buf = gpu.create_f64_output_buffer(spatial_vol * 2, "hmc_polyakov");
        let nbr_buf = gpu.create_u32_buffer(&neighbors, "hmc_nbr");

        let poly_params = PolyParams {
            nt: nt as u32,
            nx: nx as u32,
            ny: ny as u32,
            nz: nz as u32,
            volume: vol as u32,
            spatial_vol: spatial_vol as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let poly_params_buf = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("hmc_poly_params"),
            size: std::mem::size_of::<PolyParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue()
            .write_buffer(&poly_params_buf, 0, bytemuck::bytes_of(&poly_params));

        Self {
            link_buf,
            link_backup,
            mom_buf,
            force_buf,
            ke_out_buf,
            plaq_out_buf,
            poly_out_buf,
            poly_params_buf,
            nbr_buf,
            dims: lattice.dims,
            volume: vol,
            n_links,
            beta,
            spatial_vol,
            wg_links: n_links.div_ceil(64) as u32,
            wg_vol: vol.div_ceil(64) as u32,
        }
    }
}

/// Run one pure-GPU Omelyan HMC trajectory.
///
/// All gauge force, momentum update, link update, kinetic energy, and
/// plaquette math happens on GPU. CPU only generates random momenta
/// (uploaded once), reads back H_old/H_new (scalar sums), and makes
/// the Metropolis decision.
pub fn gpu_hmc_trajectory(
    gpu: &GpuF64,
    pipelines: &GpuHmcPipelines,
    state: &GpuHmcState,
    n_md_steps: usize,
    dt: f64,
    seed: &mut u64,
) -> GpuHmcResult {
    let n_links = state.n_links;

    let momenta: Vec<super::su3::Su3Matrix> = (0..n_links)
        .map(|_| super::su3::Su3Matrix::random_algebra(seed))
        .collect();
    let mom_flat = flatten_momenta(&momenta);
    gpu.upload_f64(&state.mom_buf, &mom_flat);

    {
        let mut enc = gpu.begin_encoder("backup_links");
        enc.copy_buffer_to_buffer(
            &state.link_buf,
            0,
            &state.link_backup,
            0,
            (n_links * 18 * 8) as u64,
        );
        gpu.submit_encoder(enc);
    }

    let s_old = gpu_wilson_action(gpu, pipelines, state);
    let t_old = gpu_kinetic_energy(gpu, pipelines, state);
    let h_old = s_old + t_old;

    let lam = crate::tolerances::OMELYAN_LAMBDA;

    for step in 0..n_md_steps {
        gpu_force_dispatch(gpu, pipelines, state);
        gpu_mom_update_dispatch(gpu, pipelines, state, lam * dt);
        gpu_link_update_dispatch(gpu, pipelines, state, 0.5 * dt);
        gpu_force_dispatch(gpu, pipelines, state);
        gpu_mom_update_dispatch(gpu, pipelines, state, (1.0 - 2.0 * lam) * dt);
        gpu_link_update_dispatch(gpu, pipelines, state, 0.5 * dt);
        gpu_force_dispatch(gpu, pipelines, state);
        gpu_mom_update_dispatch(gpu, pipelines, state, lam * dt);

        if step < n_md_steps - 1 {
            // Fusion of step 5/1 is an optimization for later.
        }
    }

    let s_new = gpu_wilson_action(gpu, pipelines, state);
    let t_new = gpu_kinetic_energy(gpu, pipelines, state);
    let h_new = s_new + t_new;

    let delta_h = h_new - h_old;

    let r: f64 = super::constants::lcg_uniform_f64(seed);
    let accepted = delta_h <= 0.0 || r < (-delta_h).exp();

    if !accepted {
        let mut enc = gpu.begin_encoder("restore_links");
        enc.copy_buffer_to_buffer(
            &state.link_backup,
            0,
            &state.link_buf,
            0,
            (n_links * 18 * 8) as u64,
        );
        gpu.submit_encoder(enc);
    }

    let plaquette = gpu_plaquette(gpu, pipelines, state);

    GpuHmcResult {
        accepted,
        delta_h,
        plaquette,
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Shared dispatch helpers (used by all submodules)
// ═══════════════════════════════════════════════════════════════════

pub(super) fn make_force_params(vol: usize, beta: f64) -> Vec<u8> {
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&(vol as u32).to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes());
    v.extend_from_slice(&beta.to_le_bytes());
    v
}

pub(super) fn make_link_mom_params(n_links: usize, dt: f64) -> Vec<u8> {
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&(n_links as u32).to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes());
    v.extend_from_slice(&dt.to_le_bytes());
    v
}

pub(super) fn make_u32x4_params(val: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&val.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes());
    v
}

pub(super) fn make_prng_params(n_links: u32, traj_id: u32, seed: &mut u64) -> Vec<u8> {
    super::constants::lcg_step(seed);
    let s = *seed;
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&n_links.to_le_bytes());
    v.extend_from_slice(&traj_id.to_le_bytes());
    v.extend_from_slice(&(s as u32).to_le_bytes());
    v.extend_from_slice(&((s >> 32) as u32).to_le_bytes());
    v
}

pub(super) fn gpu_force_dispatch(gpu: &GpuF64, p: &GpuHmcPipelines, s: &GpuHmcState) {
    let params = make_force_params(s.volume, s.beta);
    let param_buf = gpu.create_uniform_buffer(&params, "force_p");
    let bg = gpu.create_bind_group(
        &p.force_pipeline,
        &[&param_buf, &s.link_buf, &s.nbr_buf, &s.force_buf],
    );
    gpu.dispatch(&p.force_pipeline, &bg, s.wg_links);
}

pub(super) fn gpu_mom_update_dispatch(gpu: &GpuF64, p: &GpuHmcPipelines, s: &GpuHmcState, dt: f64) {
    let params = make_link_mom_params(s.n_links, dt);
    let param_buf = gpu.create_uniform_buffer(&params, "mom_p");
    let bg = gpu.create_bind_group(
        &p.momentum_pipeline,
        &[&param_buf, &s.force_buf, &s.mom_buf],
    );
    gpu.dispatch(&p.momentum_pipeline, &bg, s.wg_links);
}

pub(super) fn gpu_link_update_dispatch(
    gpu: &GpuF64,
    p: &GpuHmcPipelines,
    s: &GpuHmcState,
    dt: f64,
) {
    let params = make_link_mom_params(s.n_links, dt);
    let param_buf = gpu.create_uniform_buffer(&params, "link_p");
    let bg = gpu.create_bind_group(&p.link_pipeline, &[&param_buf, &s.mom_buf, &s.link_buf]);
    gpu.dispatch(&p.link_pipeline, &bg, s.wg_links);
}

pub(super) fn gpu_wilson_action(gpu: &GpuF64, p: &GpuHmcPipelines, s: &GpuHmcState) -> f64 {
    let params = make_u32x4_params(s.volume as u32);
    let param_buf = gpu.create_uniform_buffer(&params, "plaq_p");
    let bg = gpu.create_bind_group(
        &p.plaquette_pipeline,
        &[&param_buf, &s.link_buf, &s.nbr_buf, &s.plaq_out_buf],
    );
    gpu.dispatch(&p.plaquette_pipeline, &bg, s.wg_vol);
    let Ok(per_site) = gpu.read_back_f64(&s.plaq_out_buf, s.volume) else {
        return f64::NAN;
    };
    let plaq_sum: f64 = per_site.iter().sum();
    s.beta * (6.0 * s.volume as f64 - plaq_sum)
}

pub(super) fn gpu_kinetic_energy(gpu: &GpuF64, p: &GpuHmcPipelines, s: &GpuHmcState) -> f64 {
    let params = make_u32x4_params(s.n_links as u32);
    let param_buf = gpu.create_uniform_buffer(&params, "ke_p");
    let bg = gpu.create_bind_group(
        &p.kinetic_pipeline,
        &[&param_buf, &s.mom_buf, &s.ke_out_buf],
    );
    gpu.dispatch(&p.kinetic_pipeline, &bg, s.wg_links);
    let Ok(per_link) = gpu.read_back_f64(&s.ke_out_buf, s.n_links) else {
        return f64::NAN;
    };
    per_link.iter().sum()
}

pub(super) fn gpu_plaquette(gpu: &GpuF64, p: &GpuHmcPipelines, s: &GpuHmcState) -> f64 {
    let params = make_u32x4_params(s.volume as u32);
    let param_buf = gpu.create_uniform_buffer(&params, "plaq_obs");
    let bg = gpu.create_bind_group(
        &p.plaquette_pipeline,
        &[&param_buf, &s.link_buf, &s.nbr_buf, &s.plaq_out_buf],
    );
    gpu.dispatch(&p.plaquette_pipeline, &bg, s.wg_vol);
    let Ok(per_site) = gpu.read_back_f64(&s.plaq_out_buf, s.volume) else {
        return f64::NAN;
    };
    let plaq_sum: f64 = per_site.iter().sum();
    plaq_sum / (6.0 * s.volume as f64)
}

// ═══════════════════════════════════════════════════════════════════
//  Fermion dispatch helpers (shared by dynamical + resident_cg)
// ═══════════════════════════════════════════════════════════════════

pub(super) fn gpu_dirac_dispatch(
    gpu: &GpuF64,
    pipelines: &dynamical::GpuDynHmcPipelines,
    state: &dynamical::GpuDynHmcState,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    hop_sign: f64,
) {
    let vol = state.gauge.volume;
    let wg = (vol as u32).div_ceil(64);
    let mut params = Vec::with_capacity(24);
    params.extend_from_slice(&(vol as u32).to_le_bytes());
    params.extend_from_slice(&0u32.to_le_bytes());
    params.extend_from_slice(&state.mass.to_le_bytes());
    params.extend_from_slice(&hop_sign.to_le_bytes());
    let pbuf = gpu.create_uniform_buffer(&params, "dirac_p");
    let bg = gpu.create_bind_group(
        &pipelines.dirac_pipeline,
        &[
            &pbuf,
            &state.gauge.link_buf,
            input,
            output,
            &state.gauge.nbr_buf,
            &state.phases_buf,
        ],
    );
    gpu.dispatch(&pipelines.dirac_pipeline, &bg, wg);
}

pub(super) fn gpu_fermion_force_dispatch(
    gpu: &GpuF64,
    pipelines: &dynamical::GpuDynHmcPipelines,
    state: &dynamical::GpuDynHmcState,
) {
    let vol = state.gauge.volume;
    let wg = (vol as u32).div_ceil(64);
    let params = make_u32x4_params(vol as u32);
    let pbuf = gpu.create_uniform_buffer(&params, "fforce_p");
    let bg = gpu.create_bind_group(
        &pipelines.fermion_force_pipeline,
        &[
            &pbuf,
            &state.gauge.link_buf,
            &state.x_buf,
            &state.y_buf,
            &state.gauge.nbr_buf,
            &state.phases_buf,
            &state.ferm_force_buf,
        ],
    );
    gpu.dispatch(&pipelines.fermion_force_pipeline, &bg, wg);
}

pub(super) fn gpu_dot_re(
    gpu: &GpuF64,
    dot_pl: &wgpu::ComputePipeline,
    dot_buf: &wgpu::Buffer,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    n_pairs: usize,
) -> f64 {
    let wg = (n_pairs as u32).div_ceil(64);
    let params = make_u32x4_params(n_pairs as u32);
    let pbuf = gpu.create_uniform_buffer(&params, "dot_p");
    let bg = gpu.create_bind_group(dot_pl, &[&pbuf, a, b, dot_buf]);
    gpu.dispatch(dot_pl, &bg, wg);
    match gpu.read_back_f64(dot_buf, n_pairs) {
        Ok(v) => v.iter().sum(),
        Err(_) => f64::NAN,
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Utility functions
// ═══════════════════════════════════════════════════════════════════

/// Read back current GPU links into a lattice.
pub fn gpu_links_to_lattice(gpu: &GpuF64, state: &GpuHmcState, lattice: &mut Lattice) {
    if let Ok(flat) = gpu.read_back_f64(&state.link_buf, state.n_links * 18) {
        unflatten_links_into(lattice, &flat);
    }
}

/// Compute the average Polyakov loop on GPU (no CPU readback of full link buffer).
///
/// Cross-spring evolution: toadStool GpuPolyakovLoop → hotSpring v0.6.13.
/// Dispatches the Polyakov loop shader on GPU, then reads back only the
/// spatial_vol × 2 output (Re, Im) instead of the full V × 4 × 18 link buffer.
/// Returns (magnitude, phase) averaged over spatial sites.
pub fn gpu_polyakov_loop(
    gpu: &GpuF64,
    pipelines: &GpuHmcPipelines,
    state: &GpuHmcState,
) -> (f64, f64) {
    let spatial_vol = state.spatial_vol;
    let wg = spatial_vol.div_ceil(64) as u32;

    {
        let mut enc = gpu.begin_encoder("polyakov_dispatch");
        let bind_group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("polyakov_bg"),
            layout: &pipelines.polyakov_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state.poly_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state.link_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: state.poly_out_buf.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("polyakov_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.polyakov_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wg, 1, 1);
        }
        gpu.submit_encoder(enc);
    }

    let Ok(poly_data) = gpu.read_back_f64(&state.poly_out_buf, spatial_vol * 2) else {
        return (0.0, 0.0);
    };

    let mut sum_re = 0.0;
    let mut sum_im = 0.0;
    for i in 0..spatial_vol {
        let re = poly_data[i * 2];
        let im = poly_data[i * 2 + 1];
        sum_re += (re * re + im * im).sqrt();
        sum_im += im.atan2(re);
    }
    (sum_re / spatial_vol as f64, sum_im / spatial_vol as f64)
}
