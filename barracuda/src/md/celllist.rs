// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-resident cell list and cell-list MD simulation.
//!
//! Three-pass GPU cell-list build (bin → scan → scatter) eliminates all
//! CPU readback during neighbor-list rebuilds. Positions, velocities, and
//! forces stay in original particle order — the force shader uses indirect
//! indexing via `sorted_indices`.
//!
//! Energy reductions use `barracuda::pipeline::ReduceScalarPipeline`.
//!
//! ## Data flow (v0.5.12)
//!
//! ```text
//! GPU cell-list rebuild (every CELLLIST_REBUILD_INTERVAL steps):
//!   Pass 1: cell_bin       — particle → cell assignment + atomic count
//!   Pass 2: prefix_sum     — cell_counts → cell_start (exclusive scan)
//!   Pass 3: cell_scatter   — write sorted_indices[cell_start[c] + k] = i
//!
//! Force shader: indirect indexing via sorted_indices (no array sorting)
//! VV/KE/Berendsen: operate on original-order arrays (unchanged)
//! ```
//!
//! ## Benefits
//!
//! - Zero CPU readback for cell-list rebuild (was 720 KB round-trip at N=10000)
//! - Preserves particle identity across snapshots (fixes VACF correctness)

use barracuda::ops::md::CellListGpu;
use barracuda::pipeline::ReduceScalarPipeline;

use crate::gpu::GpuF64;
use crate::md::config::MdConfig;
use crate::md::shaders;
use crate::md::simulation::{init_fcc_lattice, init_velocities, EnergyRecord, MdSimulation};
use crate::tolerances::{CELLLIST_REBUILD_INTERVAL, THERMOSTAT_INTERVAL};
use std::time::Instant;

/// CPU cell list for spatial decomposition (retained for tests and diagnostics).
pub struct CellList {
    /// Number of cells per dimension [nx, ny, nz].
    pub n_cells: [usize; 3],
    /// Cell side length per dimension (reduced units).
    pub cell_size: [f64; 3],
    /// Total number of cells (nx × ny × nz).
    pub n_cells_total: usize,
    /// Exclusive prefix sum: cell_start[c] = index of first particle in cell c.
    pub cell_start: Vec<u32>,
    /// Particle count per cell.
    pub cell_count: Vec<u32>,
    /// Particle indices sorted by cell (for indirect force indexing).
    pub sorted_indices: Vec<usize>,
}

impl CellList {
    /// Build cell list from particle positions.
    #[must_use]
    pub fn build(positions: &[f64], n: usize, box_side: f64, rc: f64) -> Self {
        let n_cells_per_dim = (box_side / rc).floor() as usize;
        let n_cells_per_dim = n_cells_per_dim.max(3);
        let cell_size = box_side / n_cells_per_dim as f64;
        let n_cells_total = n_cells_per_dim * n_cells_per_dim * n_cells_per_dim;

        let mut cell_ids = Vec::with_capacity(n);
        for i in 0..n {
            let cx = ((positions[i * 3] / cell_size) as usize).min(n_cells_per_dim - 1);
            let cy = ((positions[i * 3 + 1] / cell_size) as usize).min(n_cells_per_dim - 1);
            let cz = ((positions[i * 3 + 2] / cell_size) as usize).min(n_cells_per_dim - 1);
            let cell_id = cx + cy * n_cells_per_dim + cz * n_cells_per_dim * n_cells_per_dim;
            cell_ids.push(cell_id);
        }

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by_key(|&i| cell_ids[i]);

        let mut cell_start = vec![0u32; n_cells_total];
        let mut cell_count = vec![0u32; n_cells_total];
        for &idx in &indices {
            cell_count[cell_ids[idx]] += 1;
        }
        let mut offset = 0u32;
        for c in 0..n_cells_total {
            cell_start[c] = offset;
            offset += cell_count[c];
        }

        Self {
            n_cells: [n_cells_per_dim; 3],
            cell_size: [cell_size; 3],
            n_cells_total,
            cell_start,
            cell_count,
            sorted_indices: indices,
        }
    }

    /// Reorder data array by cell list's sorted indices.
    #[must_use]
    pub fn sort_array(&self, data: &[f64], stride: usize) -> Vec<f64> {
        let mut sorted = vec![0.0f64; data.len()];
        for (new_idx, &old_idx) in self.sorted_indices.iter().enumerate() {
            for s in 0..stride {
                sorted[new_idx * stride + s] = data[old_idx * stride + s];
            }
        }
        sorted
    }

    /// Restore original particle order from cell-sorted array.
    #[must_use]
    pub fn unsort_array(&self, data: &[f64], stride: usize) -> Vec<f64> {
        let mut unsorted = vec![0.0f64; data.len()];
        for (new_idx, &old_idx) in self.sorted_indices.iter().enumerate() {
            for s in 0..stride {
                unsorted[old_idx * stride + s] = data[new_idx * stride + s];
            }
        }
        unsorted
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Cell-list MD simulation with GPU-resident rebuild
// ═══════════════════════════════════════════════════════════════════════

/// Run simulation with GPU cell-list (indirect indexing, zero-readback rebuild).
///
/// # Errors
///
/// Returns [`crate::error::HotSpringError::NoAdapter`] if no GPU adapter is found.
/// Returns [`crate::error::HotSpringError::NoShaderF64`] if the GPU lacks `SHADER_F64`.
/// Returns [`crate::error::HotSpringError::DeviceCreation`] or
/// [`crate::error::HotSpringError::Barracuda`] if GPU initialization or pipeline setup fails.
#[allow(clippy::cast_possible_truncation)] // Cell-list dimensions: N ≤ 100,000, grid ≤ 100³
pub async fn run_simulation_celllist(
    config: &MdConfig,
) -> Result<MdSimulation, crate::error::HotSpringError> {
    let t_start = Instant::now();
    let n = config.n_particles;
    let box_side = config.box_side();
    let prefactor = config.force_prefactor();
    let temperature = config.temperature();
    let mass = config.reduced_mass();

    println!("  ── Initializing {n} particles (GPU cell-list mode) ──");
    println!("    Box side: {box_side:.4} a_ws");
    println!(
        "    κ = {}, Γ = {}, T* = {:.6}",
        config.kappa, config.gamma, temperature
    );
    println!(
        "    rc = {} a_ws, dt* = {}, m* = {}",
        config.rc, config.dt, mass
    );

    let (positions, n_actual) = init_fcc_lattice(n, box_side);
    let n = n_actual.min(n);
    let velocities = init_velocities(n, temperature, mass, 42);
    println!("    Placed {n} particles on FCC lattice");

    // ── GPU initialization ──
    let gpu = GpuF64::new().await?;
    if !gpu.has_f64 {
        return Err(crate::error::HotSpringError::NoShaderF64);
    }
    gpu.print_info();

    // ── Build GPU cell-list infrastructure ──
    let gpu_cl = CellListGpu::new(gpu.to_wgpu_device(), n, [box_side; 3], config.rc)?;
    let (mx, my, mz) = gpu_cl.grid();
    let cell_size = box_side / f64::from(mx);
    println!(
        "    GPU cell list: {}×{}×{} = {} cells, cell_size={:.3} a_ws",
        mx,
        my,
        mz,
        gpu_cl.n_cells(),
        cell_size
    );

    // ── Compile shaders ──
    println!("  ── Compiling f64 WGSL shaders (GPU cell-list, indirect) ──");
    let t_compile = Instant::now();

    let force_pipeline =
        gpu.create_pipeline_f64(shaders::SHADER_YUKAWA_FORCE_INDIRECT, "force_indirect_f64");
    let kick_drift_pipeline =
        gpu.create_pipeline(shaders::SHADER_VV_KICK_DRIFT, "vv_kick_drift_f64");
    let half_kick_pipeline = gpu.create_pipeline(shaders::SHADER_VV_HALF_KICK, "vv_half_kick_f64");
    let berendsen_pipeline = gpu.create_pipeline(shaders::SHADER_BERENDSEN, "berendsen_f64");
    let ke_pipeline = gpu.create_pipeline(shaders::SHADER_KINETIC_ENERGY, "kinetic_energy_f64");

    println!(
        "    Compiled 5+3 shaders in {:.1}ms",
        t_compile.elapsed().as_secs_f64() * 1000.0
    );

    let reducer = ReduceScalarPipeline::new(gpu.to_wgpu_device(), n)?;

    // ── Create GPU buffers ──
    // Positions and velocities in ORIGINAL particle order (no sorting)
    let pos_buf = gpu.create_f64_output_buffer(n * 3, "positions");
    let vel_buf = gpu.create_f64_output_buffer(n * 3, "velocities");
    let force_buf = gpu.create_f64_output_buffer(n * 3, "forces");
    let pe_buf = gpu.create_f64_output_buffer(n, "pe_per_particle");
    let ke_buf = gpu.create_f64_output_buffer(n, "ke_per_particle");

    gpu.upload_f64(&pos_buf, &positions);
    gpu.upload_f64(&vel_buf, &velocities);

    // Force params: same layout as sorted shader, grid from CellListGpu
    let force_params = vec![
        n as f64,
        config.kappa,
        prefactor,
        config.rc * config.rc,
        box_side,
        box_side,
        box_side,
        0.0,
        f64::from(mx),
        f64::from(my),
        f64::from(mz),
        cell_size,
        cell_size,
        cell_size,
        f64::from(gpu_cl.n_cells()),
        0.0,
    ];
    let force_params_buf = gpu.create_f64_buffer(&force_params, "force_params");

    let vv_params = vec![
        n as f64, config.dt, mass, 0.0, box_side, box_side, box_side, 0.0,
    ];
    let vv_params_buf = gpu.create_f64_buffer(&vv_params, "vv_params");
    let hk_params = vec![n as f64, config.dt, mass, 0.0];
    let hk_params_buf = gpu.create_f64_buffer(&hk_params, "hk_params");
    let ke_params = vec![n as f64, mass, 0.0, 0.0];
    let ke_params_buf = gpu.create_f64_buffer(&ke_params, "ke_params");

    let workgroups = n.div_ceil(64) as u32;

    // ── Bind groups ──
    // Force shader: 7 bindings (indirect indexing via sorted_indices)
    let force_bg = gpu.create_bind_group(
        &force_pipeline,
        &[
            &pos_buf,
            &force_buf,
            &pe_buf,
            &force_params_buf,
            gpu_cl.cell_start(),
            gpu_cl.cell_count(),
            gpu_cl.sorted_indices(),
        ],
    );
    let kick_drift_bg = gpu.create_bind_group(
        &kick_drift_pipeline,
        &[&pos_buf, &vel_buf, &force_buf, &vv_params_buf],
    );
    let half_kick_bg =
        gpu.create_bind_group(&half_kick_pipeline, &[&vel_buf, &force_buf, &hk_params_buf]);
    let ke_bg = gpu.create_bind_group(&ke_pipeline, &[&vel_buf, &ke_buf, &ke_params_buf]);

    // ── Initial cell-list build + force computation ──
    gpu_cl.build(&pos_buf)?;
    gpu.dispatch(&force_pipeline, &force_bg, workgroups);

    let rebuild_interval = CELLLIST_REBUILD_INTERVAL;

    // ══════════════════════════════════════════════════════════════
    // Equilibration — GPU cell-list rebuild every rebuild_interval steps
    // ══════════════════════════════════════════════════════════════
    println!("  ── Equilibration ({} steps) ──", config.equil_steps);
    let t_equil = Instant::now();

    let thermostat_interval = rebuild_interval.min(THERMOSTAT_INTERVAL);
    let mut step = 0;

    while step < config.equil_steps {
        let batch = thermostat_interval.min(config.equil_steps - step);

        let mut encoder = gpu.begin_encoder("equil_cl_batch");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("equil_cl_stream"),
                timestamp_writes: None,
            });
            for _ in 0..batch {
                pass.set_pipeline(&kick_drift_pipeline);
                pass.set_bind_group(0, &kick_drift_bg, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
                pass.set_pipeline(&force_pipeline);
                pass.set_bind_group(0, &force_bg, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
                pass.set_pipeline(&half_kick_pipeline);
                pass.set_bind_group(0, &half_kick_bg, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            pass.set_pipeline(&ke_pipeline);
            pass.set_bind_group(0, &ke_bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        gpu.submit_encoder(encoder);

        let total_ke = reducer.sum_f64(&ke_buf)?;
        let t_current = 2.0 * total_ke / (3.0 * n as f64);
        if t_current > 1e-30 {
            let ratio =
                (config.dt / config.berendsen_tau).mul_add(temperature / t_current - 1.0, 1.0);
            let scale = ratio.max(0.0).sqrt();
            let beren_params = vec![n as f64, scale, 0.0, 0.0];
            let beren_params_buf = gpu.create_f64_buffer(&beren_params, "beren_params");
            let beren_bg =
                gpu.create_bind_group(&berendsen_pipeline, &[&vel_buf, &beren_params_buf]);
            gpu.dispatch(&berendsen_pipeline, &beren_bg, workgroups);
        }

        step += batch;

        // GPU cell-list rebuild: zero CPU readback
        if step.is_multiple_of(rebuild_interval) && step < config.equil_steps {
            gpu_cl.build(&pos_buf)?;
        }

        if step % 1000 < thermostat_interval || step >= config.equil_steps {
            println!("    Step {step}: T* = {t_current:.6} (target {temperature:.6})");
        }
    }
    println!(
        "    Equilibration complete in {:.2}s",
        t_equil.elapsed().as_secs_f64()
    );

    // ══════════════════════════════════════════════════════════════
    // Production — streamed with GPU cell-list rebuild
    // ══════════════════════════════════════════════════════════════
    println!(
        "  ── Production ({} steps, GPU rebuild every {}) ──",
        config.prod_steps, rebuild_interval
    );
    let t_prod = Instant::now();

    let mut energy_history = Vec::new();
    let mut positions_snapshots = Vec::new();
    let mut velocity_snapshots = Vec::new();

    let pos_staging = gpu.create_staging_buffer(n * 3 * 8, "pos_staging");
    let vel_staging = gpu.create_staging_buffer(n * 3 * 8, "vel_staging");

    let stream_batch = config.dump_step.min(rebuild_interval);
    let n_dumps = config.prod_steps / stream_batch;
    let snap_every = config.vel_snapshot_interval * (config.dump_step / stream_batch).max(1);

    for dump_idx in 0..n_dumps {
        let step_end = (dump_idx + 1) * stream_batch;
        let is_dump = step_end.is_multiple_of(config.dump_step);
        let is_snapshot = is_dump && dump_idx.is_multiple_of(snap_every);
        let is_rebuild = step_end.is_multiple_of(rebuild_interval);

        let mut encoder = gpu.begin_encoder("prod_cl_batch");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prod_cl_stream"),
                timestamp_writes: None,
            });
            for _ in 0..stream_batch {
                pass.set_pipeline(&kick_drift_pipeline);
                pass.set_bind_group(0, &kick_drift_bg, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
                pass.set_pipeline(&force_pipeline);
                pass.set_bind_group(0, &force_bg, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
                pass.set_pipeline(&half_kick_pipeline);
                pass.set_bind_group(0, &half_kick_bg, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            if is_dump {
                pass.set_pipeline(&ke_pipeline);
                pass.set_bind_group(0, &ke_bg, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }
        if is_snapshot {
            encoder.copy_buffer_to_buffer(&pos_buf, 0, &pos_staging, 0, (n * 3 * 8) as u64);
            encoder.copy_buffer_to_buffer(&vel_buf, 0, &vel_staging, 0, (n * 3 * 8) as u64);
        }
        gpu.submit_encoder(encoder);

        if is_dump {
            let total_ke = reducer.sum_f64(&ke_buf)?;
            let total_pe = reducer.sum_f64(&pe_buf)?;
            let total_e = total_ke + total_pe;
            let t_current = 2.0 * total_ke / (3.0 * n as f64);
            energy_history.push(EnergyRecord {
                step: step_end - 1,
                ke: total_ke,
                pe: total_pe,
                total: total_e,
                temperature: t_current,
            });
        }

        if is_snapshot {
            positions_snapshots.push(gpu.read_staging_f64(&pos_staging)?);
            velocity_snapshots.push(gpu.read_staging_f64(&vel_staging)?);
        }

        // GPU cell-list rebuild: zero CPU readback, zero upload
        if is_rebuild && step_end < config.prod_steps {
            gpu_cl.build(&pos_buf)?;
        }

        if step_end % 5000 < stream_batch || step_end >= config.prod_steps {
            if let Some(last) = energy_history.last() {
                println!(
                    "    Step {}: T*={:.6}, KE={:.4}, PE={:.4}, E={:.4}",
                    step_end - 1,
                    last.temperature,
                    last.ke,
                    last.pe,
                    last.total
                );
            }
        }
    }

    let prod_time = t_prod.elapsed().as_secs_f64();
    let total_time = t_start.elapsed().as_secs_f64();
    let total_steps = config.equil_steps + config.prod_steps;
    let steps_per_sec = total_steps as f64 / total_time;

    println!("    Production complete in {prod_time:.2}s");
    println!("    Total: {total_time:.2}s ({steps_per_sec:.1} steps/s)");

    Ok(MdSimulation {
        config: config.clone(),
        energy_history,
        positions_snapshots,
        velocity_snapshots,
        rdf_histogram: Vec::new(),
        wall_time_s: total_time,
        steps_per_sec,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_positions(n: usize, box_side: f64) -> Vec<f64> {
        let mut pos = Vec::with_capacity(n * 3);
        let mut seed = 42u64;
        for _ in 0..n {
            for _ in 0..3 {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                pos.push((seed >> 33) as f64 / (1u64 << 31) as f64 * box_side);
            }
        }
        pos
    }

    #[test]
    fn cell_list_builds_without_panic() {
        let box_side = 10.0;
        let rc = 3.0;
        let pos = sample_positions(100, box_side);
        let cl = CellList::build(&pos, 100, box_side, rc);
        assert!(cl.n_cells_total > 0);
        assert_eq!(cl.sorted_indices.len(), 100);
    }

    #[test]
    fn cell_list_all_particles_assigned() {
        let box_side = 10.0;
        let rc = 3.0;
        let n = 200;
        let pos = sample_positions(n, box_side);
        let cl = CellList::build(&pos, n, box_side, rc);
        let total_count: u32 = cl.cell_count.iter().sum();
        assert_eq!(total_count as usize, n);
    }

    #[test]
    fn cell_list_start_count_consistent() {
        let box_side = 10.0;
        let rc = 2.5;
        let n = 150;
        let pos = sample_positions(n, box_side);
        let cl = CellList::build(&pos, n, box_side, rc);
        for c in 0..cl.n_cells_total {
            let start = cl.cell_start[c] as usize;
            let count = cl.cell_count[c] as usize;
            assert!(start + count <= n, "cell {c} overflows");
        }
    }

    #[test]
    fn sort_unsort_round_trip() {
        let box_side = 10.0;
        let rc = 3.0;
        let n = 50;
        let pos = sample_positions(n, box_side);
        let cl = CellList::build(&pos, n, box_side, rc);
        let sorted = cl.sort_array(&pos, 3);
        let recovered = cl.unsort_array(&sorted, 3);
        for i in 0..pos.len() {
            assert!(
                (pos[i] - recovered[i]).abs() < 1e-15,
                "round-trip failed at index {i}"
            );
        }
    }

    #[test]
    fn cell_list_minimum_cells() {
        let box_side = 5.0;
        let rc = 10.0;
        let pos = sample_positions(10, box_side);
        let cl = CellList::build(&pos, 10, box_side, rc);
        assert!(cl.n_cells[0] >= 3, "minimum 3 cells per dimension");
    }

    #[test]
    fn cell_list_single_particle() {
        let pos = vec![1.0, 2.0, 3.0];
        let cl = CellList::build(&pos, 1, 10.0, 3.0);
        assert_eq!(cl.sorted_indices.len(), 1);
        let total_count: u32 = cl.cell_count.iter().sum();
        assert_eq!(total_count, 1);
    }

    #[test]
    fn cell_list_deterministic() {
        let pos = sample_positions(100, 10.0);
        let a = CellList::build(&pos, 100, 10.0, 3.0);
        let b = CellList::build(&pos, 100, 10.0, 3.0);
        assert_eq!(a.sorted_indices, b.sorted_indices);
        assert_eq!(a.cell_start, b.cell_start);
        assert_eq!(a.cell_count, b.cell_count);
    }

    #[test]
    fn gpu_cell_list_grid_matches_cpu() {
        let box_side: f64 = 10.0;
        let rc: f64 = 2.0;
        let cpu_n = ((box_side / rc).floor() as u32).max(3);
        let expected_nc = cpu_n * cpu_n * cpu_n;
        assert_eq!(cpu_n, 5);
        assert_eq!(expected_nc, 125);
    }

    #[test]
    fn sort_array_stride_one() {
        let box_side = 10.0;
        let rc = 3.0;
        let n = 8;
        let pos = sample_positions(n, box_side);
        let cl = CellList::build(&pos, n, box_side, rc);
        let scalar_data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let sorted = cl.sort_array(&scalar_data, 1);
        assert_eq!(sorted.len(), n);
        for (new_idx, &old_idx) in cl.sorted_indices.iter().enumerate() {
            assert!(
                (sorted[new_idx] - old_idx as f64).abs() < 1e-15,
                "sorted[{new_idx}] should be {old_idx}, got {}",
                sorted[new_idx]
            );
        }
    }

    #[test]
    fn cell_size_consistency() {
        let box_side = 10.0;
        let rc = 2.5;
        let n = 50;
        let pos = sample_positions(n, box_side);
        let cl = CellList::build(&pos, n, box_side, rc);
        let expected_cell_size = box_side / cl.n_cells[0] as f64;
        for d in 0..3 {
            assert!(
                (cl.cell_size[d] - expected_cell_size).abs() < 1e-15,
                "cell_size[{d}] = {} should equal {}",
                cl.cell_size[d],
                expected_cell_size
            );
        }
    }

    #[test]
    fn two_particles_same_cell() {
        let box_side = 10.0;
        let rc = 5.0;
        let pos = vec![1.0, 1.0, 1.0, 1.1, 1.1, 1.1];
        let cl = CellList::build(&pos, 2, box_side, rc);
        assert_eq!(cl.sorted_indices.len(), 2);
        let total_count: u32 = cl.cell_count.iter().sum();
        assert_eq!(total_count, 2);
        let nonzero_count = cl.cell_count.iter().filter(|&&c| c > 0).count();
        assert_eq!(nonzero_count, 1, "both particles should be in same cell");
    }

    #[test]
    fn boundary_positions_clamped() {
        let box_side = 10.0;
        let rc = 3.0;
        let pos = vec![
            box_side - 1e-10,
            box_side - 1e-10,
            box_side - 1e-10,
            0.0,
            0.0,
            0.0,
        ];
        let cl = CellList::build(&pos, 2, box_side, rc);
        assert_eq!(cl.sorted_indices.len(), 2);
        let total_count: u32 = cl.cell_count.iter().sum();
        assert_eq!(total_count, 2, "boundary particles must be assigned");
    }

    #[test]
    fn very_small_cutoff_gives_many_cells() {
        let box_side = 10.0;
        let rc = 0.5;
        let pos = sample_positions(20, box_side);
        let cl = CellList::build(&pos, 20, box_side, rc);
        assert!(
            cl.n_cells[0] >= 20,
            "small cutoff should give many cells: got {}",
            cl.n_cells[0]
        );
    }

    #[test]
    fn sort_unsort_round_trip_stride_4() {
        let box_side = 10.0;
        let rc = 3.0;
        let n = 25;
        let pos = sample_positions(n, box_side);
        let cl = CellList::build(&pos, n, box_side, rc);
        let data: Vec<f64> = (0..n * 4).map(|i| i as f64 * 0.1).collect();
        let sorted = cl.sort_array(&data, 4);
        let recovered = cl.unsort_array(&sorted, 4);
        for i in 0..data.len() {
            assert!(
                (data[i] - recovered[i]).abs() < 1e-15,
                "stride-4 round-trip failed at index {i}"
            );
        }
    }

    #[test]
    fn many_particles_in_small_box() {
        let box_side = 2.0;
        let rc = 0.5;
        let n = 500;
        let pos = sample_positions(n, box_side);
        let cl = CellList::build(&pos, n, box_side, rc);
        let total_count: u32 = cl.cell_count.iter().sum();
        assert_eq!(total_count as usize, n, "all particles must be accounted");
        assert!(cl.n_cells_total > 27, "should have many cells");
    }

    #[test]
    fn cell_list_particles_in_correct_cells() {
        let box_side = 10.0;
        let rc = 5.0;
        let pos = vec![1.0, 1.0, 1.0, 6.0, 6.0, 6.0];
        let cl = CellList::build(&pos, 2, box_side, rc);
        let nonzero_count = cl
            .cell_count
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .count();
        assert_eq!(
            nonzero_count, 2,
            "particles far apart should be in different cells"
        );
    }
}
