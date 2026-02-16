// SPDX-License-Identifier: AGPL-3.0-only

//! GPU MD Simulation Engine
//!
//! Runs a full Yukawa OCP molecular dynamics simulation on GPU using
//! f64 WGSL shaders. The MD loop is:
//!
//!   1. Half-kick + drift + PBC wrap  (fused kernel)
//!   2. Force computation              (Yukawa all-pairs kernel)
//!   3. Second half-kick               (kernel)
//!   4. (optional) Berendsen thermostat
//!
//! All particle data lives on GPU. CPU reads back energy/positions
//! only at dump intervals for observable computation.

use crate::gpu::GpuF64;
use crate::md::config::MdConfig;
use crate::md::shaders;

use std::f64::consts::PI;
use std::time::Instant;

/// Per-step energy record
#[derive(Clone, Debug)]
pub struct EnergyRecord {
    pub step: usize,
    pub ke: f64,
    pub pe: f64,
    pub total: f64,
    pub temperature: f64,
}

/// Simulation state and results
#[derive(Debug)]
pub struct MdSimulation {
    pub config: MdConfig,
    pub energy_history: Vec<EnergyRecord>,
    pub positions_snapshots: Vec<Vec<f64>>,
    pub velocity_snapshots: Vec<Vec<f64>>,
    pub rdf_histogram: Vec<u64>,
    pub wall_time_s: f64,
    pub steps_per_sec: f64,
}

/// Initialize particles on an FCC lattice, then add Maxwell-Boltzmann velocities
pub fn init_fcc_lattice(n: usize, box_side: f64) -> (Vec<f64>, usize) {
    // Find smallest n_cell such that 4*n_cell^3 >= n (FCC has 4 atoms per unit cell)
    let mut n_cell = 1usize;
    while 4 * n_cell * n_cell * n_cell < n {
        n_cell += 1;
    }
    let n_actual = 4 * n_cell * n_cell * n_cell;
    let a = box_side / n_cell as f64; // lattice constant

    // FCC basis vectors (in units of lattice constant a)
    let basis = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];

    let mut positions = Vec::with_capacity(n_actual * 3);
    for ix in 0..n_cell {
        for iy in 0..n_cell {
            for iz in 0..n_cell {
                for b in &basis {
                    let x = (ix as f64 + b[0]) * a;
                    let y = (iy as f64 + b[1]) * a;
                    let z = (iz as f64 + b[2]) * a;
                    positions.push(x);
                    positions.push(y);
                    positions.push(z);
                }
            }
        }
    }

    // Trim to exactly n particles if n_actual > n
    positions.truncate(n * 3);
    (positions, n.min(n_actual))
}

/// Generate Maxwell-Boltzmann velocities for given temperature
/// T* = 1/Gamma in reduced units.  KE = (3/2) N T*
/// Per component: <v²_α> = T*/m*  →  σ = sqrt(T*/m*)
pub fn init_velocities(n: usize, temperature: f64, mass: f64, seed: u64) -> Vec<f64> {
    // Simple Box-Muller PRNG for Gaussian distribution
    let sigma = (temperature / mass).sqrt(); // sqrt(T*/m*) per component
    let mut velocities = Vec::with_capacity(n * 3);

    // LCG random number generator (good enough for initialization)
    let mut rng_state = seed;
    let lcg_next = |state: &mut u64| -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as f64 / (1u64 << 31) as f64
    };

    for _ in 0..n {
        for _ in 0..3 {
            // Box-Muller transform
            let u1 = lcg_next(&mut rng_state).max(1e-15);
            let u2 = lcg_next(&mut rng_state);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            velocities.push(z * sigma);
        }
    }

    // Remove center-of-mass velocity
    let mut vx_sum = 0.0;
    let mut vy_sum = 0.0;
    let mut vz_sum = 0.0;
    for i in 0..n {
        vx_sum += velocities[i * 3];
        vy_sum += velocities[i * 3 + 1];
        vz_sum += velocities[i * 3 + 2];
    }
    let n_f = n as f64;
    for i in 0..n {
        velocities[i * 3] -= vx_sum / n_f;
        velocities[i * 3 + 1] -= vy_sum / n_f;
        velocities[i * 3 + 2] -= vz_sum / n_f;
    }

    // Rescale to exact target temperature
    // KE = 0.5 * m * sum(v²), T = 2*KE/(3N) = m*sum(v²)/(3N)
    let mut v_sq_sum = 0.0;
    for v in &velocities {
        v_sq_sum += v * v;
    }
    let t_current = mass * v_sq_sum / (3.0 * n_f);
    let scale = (temperature / t_current).sqrt();
    for v in &mut velocities {
        *v *= scale;
    }

    velocities
}

/// Run the full GPU MD simulation
pub async fn run_simulation(
    config: &MdConfig,
) -> Result<MdSimulation, crate::error::HotSpringError> {
    let t_start = Instant::now();
    let n = config.n_particles;
    let box_side = config.box_side();
    let prefactor = config.force_prefactor();
    let temperature = config.temperature();
    let mass = config.reduced_mass();

    println!("  ── Initializing {n} particles ──");
    println!("    Box side: {box_side:.4} a_ws");
    println!(
        "    κ = {}, Γ = {}, T* = {:.6}",
        config.kappa, config.gamma, temperature
    );
    println!(
        "    rc = {} a_ws, dt* = {}, m* = {}",
        config.rc, config.dt, mass
    );
    println!("    Force prefactor: {prefactor:.4}");

    // Initialize particles
    let (positions, n_actual) = init_fcc_lattice(n, box_side);
    let n = n_actual.min(n); // might be slightly different due to FCC packing
    let velocities = init_velocities(n, temperature, mass, 42);

    println!("    Placed {n} particles on FCC lattice");

    // Initialize GPU
    let gpu = GpuF64::new().await?;
    if !gpu.has_f64 {
        return Err(crate::error::HotSpringError::NoShaderF64);
    }
    gpu.print_info();

    // Compile shaders — native f64 builtins (sqrt, exp, round, floor) are used
    // directly; no math_f64 preamble needed. See f64_builtin_test for validation.
    println!("  ── Compiling f64 WGSL shaders (native builtins) ──");
    let t_compile = Instant::now();

    let force_pipeline = gpu.create_pipeline(shaders::SHADER_YUKAWA_FORCE, "yukawa_force_f64");
    let kick_drift_pipeline =
        gpu.create_pipeline(shaders::SHADER_VV_KICK_DRIFT, "vv_kick_drift_f64");
    let half_kick_pipeline = gpu.create_pipeline(shaders::SHADER_VV_HALF_KICK, "vv_half_kick_f64");
    let berendsen_pipeline = gpu.create_pipeline(shaders::SHADER_BERENDSEN, "berendsen_f64");
    let ke_pipeline = gpu.create_pipeline(shaders::SHADER_KINETIC_ENERGY, "kinetic_energy_f64");

    println!(
        "    Compiled 5 shaders in {:.1}ms",
        t_compile.elapsed().as_secs_f64() * 1000.0
    );

    // Create GPU buffers
    let pos_buf = gpu.create_f64_output_buffer(n * 3, "positions");
    let vel_buf = gpu.create_f64_output_buffer(n * 3, "velocities");
    let force_buf = gpu.create_f64_output_buffer(n * 3, "forces");
    let pe_buf = gpu.create_f64_output_buffer(n, "pe_per_particle");
    let ke_buf = gpu.create_f64_output_buffer(n, "ke_per_particle");

    // Upload initial positions and velocities
    gpu.upload_f64(&pos_buf, &positions);
    gpu.upload_f64(&vel_buf, &velocities);

    // Force params: [n, kappa, prefactor, cutoff_sq, box_x, box_y, box_z, epsilon]
    let force_params = vec![
        n as f64,
        config.kappa,
        prefactor,
        config.rc * config.rc,
        box_side,
        box_side,
        box_side,
        0.0, // no softening
    ];
    let force_params_buf = gpu.create_f64_buffer(&force_params, "force_params");

    // VV params: [n, dt, mass, _, box_x, box_y, box_z, _]
    let vv_params = vec![
        n as f64, config.dt, mass, 0.0, box_side, box_side, box_side, 0.0,
    ];
    let vv_params_buf = gpu.create_f64_buffer(&vv_params, "vv_params");

    // Half-kick params: [n, dt, mass, _]
    let hk_params = vec![n as f64, config.dt, mass, 0.0];
    let hk_params_buf = gpu.create_f64_buffer(&hk_params, "hk_params");

    // KE params: [n, mass, _, _]
    let ke_params = vec![n as f64, mass, 0.0, 0.0];
    let ke_params_buf = gpu.create_f64_buffer(&ke_params, "ke_params");

    let workgroups = n.div_ceil(64) as u32;

    // ── Bind groups ──
    let force_bg = gpu.create_bind_group(
        &force_pipeline,
        &[&pos_buf, &force_buf, &pe_buf, &force_params_buf],
    );
    let kick_drift_bg = gpu.create_bind_group(
        &kick_drift_pipeline,
        &[&pos_buf, &vel_buf, &force_buf, &vv_params_buf],
    );
    let half_kick_bg =
        gpu.create_bind_group(&half_kick_pipeline, &[&vel_buf, &force_buf, &hk_params_buf]);
    let ke_bg = gpu.create_bind_group(&ke_pipeline, &[&vel_buf, &ke_buf, &ke_params_buf]);

    // ── Compute initial forces ──
    gpu.dispatch(&force_pipeline, &force_bg, workgroups);

    // ══════════════════════════════════════════════════════════════
    //  Equilibration
    // ══════════════════════════════════════════════════════════════
    println!("  ── Equilibration ({} steps) ──", config.equil_steps);
    let t_equil = Instant::now();

    for step in 0..config.equil_steps {
        // 1. Half-kick + drift + PBC
        gpu.dispatch(&kick_drift_pipeline, &kick_drift_bg, workgroups);
        // 2. Recompute forces
        gpu.dispatch(&force_pipeline, &force_bg, workgroups);
        // 3. Second half-kick
        gpu.dispatch(&half_kick_pipeline, &half_kick_bg, workgroups);

        // 4. Berendsen thermostat every step
        if step % 10 == 0 {
            // Compute current KE on GPU, read back to get temperature
            gpu.dispatch(&ke_pipeline, &ke_bg, workgroups);
            let ke_per_particle = gpu.read_back_f64(&ke_buf, n)?;
            let total_ke: f64 = ke_per_particle.iter().sum();
            // T* = 2 * KE / (3 * N) in reduced units (k_B = 1)
            let t_current = 2.0 * total_ke / (3.0 * n as f64);

            if t_current > 1e-30 {
                let ratio =
                    1.0 + (config.dt / config.berendsen_tau) * (temperature / t_current - 1.0);
                let scale = ratio.max(0.0).sqrt();

                // Upload Berendsen params and apply
                let beren_params = vec![n as f64, scale, 0.0, 0.0];
                let beren_params_buf = gpu.create_f64_buffer(&beren_params, "beren_params");
                let beren_bg =
                    gpu.create_bind_group(&berendsen_pipeline, &[&vel_buf, &beren_params_buf]);
                gpu.dispatch(&berendsen_pipeline, &beren_bg, workgroups);
            }
        }

        if step % 1000 == 0 || step == config.equil_steps - 1 {
            // Quick temperature check
            gpu.dispatch(&ke_pipeline, &ke_bg, workgroups);
            let ke_data = gpu.read_back_f64(&ke_buf, n)?;
            let total_ke: f64 = ke_data.iter().sum();
            let t_current = 2.0 * total_ke / (3.0 * n as f64);
            println!("    Step {step}: T* = {t_current:.6} (target {temperature:.6})");
        }
    }
    let equil_time = t_equil.elapsed().as_secs_f64();
    println!("    Equilibration complete in {equil_time:.2}s");

    // ══════════════════════════════════════════════════════════════
    //  Production
    // ══════════════════════════════════════════════════════════════
    println!("  ── Production ({} steps) ──", config.prod_steps);
    let t_prod = Instant::now();

    let mut energy_history = Vec::new();
    let mut positions_snapshots = Vec::new();
    let mut velocity_snapshots = Vec::new();

    for step in 0..config.prod_steps {
        // 1. Half-kick + drift + PBC
        gpu.dispatch(&kick_drift_pipeline, &kick_drift_bg, workgroups);
        // 2. Recompute forces
        gpu.dispatch(&force_pipeline, &force_bg, workgroups);
        // 3. Second half-kick
        gpu.dispatch(&half_kick_pipeline, &half_kick_bg, workgroups);

        // Dump at intervals
        if step % config.dump_step == 0 {
            // Read back KE and PE
            gpu.dispatch(&ke_pipeline, &ke_bg, workgroups);
            let ke_data = gpu.read_back_f64(&ke_buf, n)?;
            let pe_data = gpu.read_back_f64(&pe_buf, n)?;

            let total_ke: f64 = ke_data.iter().sum();
            let total_pe: f64 = pe_data.iter().sum();
            let total_e = total_ke + total_pe;
            let t_current = 2.0 * total_ke / (3.0 * n as f64);

            energy_history.push(EnergyRecord {
                step,
                ke: total_ke,
                pe: total_pe,
                total: total_e,
                temperature: t_current,
            });

            // Store position/velocity snapshots for observable computation
            // Only store every 100 dumps (every 1000 steps) to save memory
            if step % (config.dump_step * 100) == 0 {
                let pos_snap = gpu.read_back_f64(&pos_buf, n * 3)?;
                let vel_snap = gpu.read_back_f64(&vel_buf, n * 3)?;
                positions_snapshots.push(pos_snap);
                velocity_snapshots.push(vel_snap);
            }
        }

        if step % 5000 == 0 || step == config.prod_steps - 1 {
            if let Some(last) = energy_history.last() {
                println!(
                    "    Step {}: T*={:.6}, KE={:.4}, PE={:.4}, E={:.4}",
                    step, last.temperature, last.ke, last.pe, last.total
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
        rdf_histogram: Vec::new(), // computed in observables phase
        wall_time_s: total_time,
        steps_per_sec,
    })
}

// ═══════════════════════════════════════════════════════════════════
// Cell List Infrastructure
// ═══════════════════════════════════════════════════════════════════

/// Cell list metadata
pub struct CellList {
    pub n_cells: [usize; 3], // cells per dimension
    pub cell_size: [f64; 3], // cell side length
    pub n_cells_total: usize,
    pub cell_start: Vec<u32>,       // first particle index per cell
    pub cell_count: Vec<u32>,       // particle count per cell
    pub sorted_indices: Vec<usize>, // particle index in sorted order
}

impl CellList {
    /// Build cell list from positions on CPU
    pub fn build(positions: &[f64], n: usize, box_side: f64, rc: f64) -> Self {
        let n_cells_per_dim = (box_side / rc).floor() as usize;
        let n_cells_per_dim = n_cells_per_dim.max(3); // minimum 3 cells per dim
        let cell_size = box_side / n_cells_per_dim as f64;
        let n_cells_total = n_cells_per_dim * n_cells_per_dim * n_cells_per_dim;

        // Assign each particle to a cell
        let mut cell_ids = Vec::with_capacity(n);
        for i in 0..n {
            let cx = ((positions[i * 3] / cell_size) as usize).min(n_cells_per_dim - 1);
            let cy = ((positions[i * 3 + 1] / cell_size) as usize).min(n_cells_per_dim - 1);
            let cz = ((positions[i * 3 + 2] / cell_size) as usize).min(n_cells_per_dim - 1);
            let cell_id = cx + cy * n_cells_per_dim + cz * n_cells_per_dim * n_cells_per_dim;
            cell_ids.push(cell_id);
        }

        // Sort particles by cell index
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by_key(|&i| cell_ids[i]);

        // Build cell_start and cell_count
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

        CellList {
            n_cells: [n_cells_per_dim; 3],
            cell_size: [cell_size; 3],
            n_cells_total,
            cell_start,
            cell_count,
            sorted_indices: indices,
        }
    }

    /// Sort position/velocity arrays according to cell list order
    pub fn sort_array(&self, data: &[f64], stride: usize) -> Vec<f64> {
        let mut sorted = vec![0.0f64; data.len()];
        for (new_idx, &old_idx) in self.sorted_indices.iter().enumerate() {
            for s in 0..stride {
                sorted[new_idx * stride + s] = data[old_idx * stride + s];
            }
        }
        sorted
    }

    /// Unsort: map from sorted back to original order
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

/// Run simulation with cell list (for N > ~2000)
pub async fn run_simulation_celllist(
    config: &MdConfig,
) -> Result<MdSimulation, crate::error::HotSpringError> {
    let t_start = Instant::now();
    let n = config.n_particles;
    let box_side = config.box_side();
    let prefactor = config.force_prefactor();
    let temperature = config.temperature();
    let mass = config.reduced_mass();

    println!("  ── Initializing {n} particles (cell-list mode) ──");
    println!("    Box side: {box_side:.4} a_ws");
    println!(
        "    κ = {}, Γ = {}, T* = {:.6}",
        config.kappa, config.gamma, temperature
    );
    println!(
        "    rc = {} a_ws, dt* = {}, m* = {}",
        config.rc, config.dt, mass
    );

    // Initialize particles
    let (positions, n_actual) = init_fcc_lattice(n, box_side);
    let n = n_actual.min(n);
    let velocities = init_velocities(n, temperature, mass, 42);

    // Build initial cell list
    let cell_list = CellList::build(&positions, n, box_side, config.rc);
    println!(
        "    Cell list: {}×{}×{} = {} cells, cell_size={:.3} a_ws",
        cell_list.n_cells[0],
        cell_list.n_cells[1],
        cell_list.n_cells[2],
        cell_list.n_cells_total,
        cell_list.cell_size[0]
    );
    println!("    Placed {n} particles on FCC lattice");

    // Initialize GPU
    let gpu = GpuF64::new().await?;
    if !gpu.has_f64 {
        return Err(crate::error::HotSpringError::NoShaderF64);
    }
    gpu.print_info();

    // Compile shaders — native f64 builtins, no math_f64 preamble needed
    println!("  ── Compiling f64 WGSL shaders (cell-list, native builtins) ──");
    let t_compile = Instant::now();

    let force_pipeline_cl =
        gpu.create_pipeline(shaders::SHADER_YUKAWA_FORCE_CELLLIST, "yukawa_force_cl_f64");
    let kick_drift_pipeline =
        gpu.create_pipeline(shaders::SHADER_VV_KICK_DRIFT, "vv_kick_drift_f64");
    let half_kick_pipeline = gpu.create_pipeline(shaders::SHADER_VV_HALF_KICK, "vv_half_kick_f64");
    let berendsen_pipeline = gpu.create_pipeline(shaders::SHADER_BERENDSEN, "berendsen_f64");
    let ke_pipeline = gpu.create_pipeline(shaders::SHADER_KINETIC_ENERGY, "kinetic_energy_f64");

    println!(
        "    Compiled 5 shaders in {:.1}ms",
        t_compile.elapsed().as_secs_f64() * 1000.0
    );

    // Create GPU buffers
    let pos_buf = gpu.create_f64_output_buffer(n * 3, "positions");
    let vel_buf = gpu.create_f64_output_buffer(n * 3, "velocities");
    let force_buf = gpu.create_f64_output_buffer(n * 3, "forces");
    let pe_buf = gpu.create_f64_output_buffer(n, "pe_per_particle");
    let ke_buf = gpu.create_f64_output_buffer(n, "ke_per_particle");

    // Cell list buffers (u32)
    let cell_start_buf = gpu.create_u32_buffer(&cell_list.cell_start, "cell_start");
    let cell_count_buf = gpu.create_u32_buffer(&cell_list.cell_count, "cell_count");

    // Upload initial sorted positions and velocities
    let sorted_pos = cell_list.sort_array(&positions, 3);
    let sorted_vel = cell_list.sort_array(&velocities, 3);
    gpu.upload_f64(&pos_buf, &sorted_pos);
    gpu.upload_f64(&vel_buf, &sorted_vel);

    // Extended force params for cell list
    let force_params_cl = vec![
        n as f64,
        config.kappa,
        prefactor,
        config.rc * config.rc,
        box_side,
        box_side,
        box_side,
        0.0,
        cell_list.n_cells[0] as f64,
        cell_list.n_cells[1] as f64,
        cell_list.n_cells[2] as f64,
        cell_list.cell_size[0],
        cell_list.cell_size[1],
        cell_list.cell_size[2],
        cell_list.n_cells_total as f64,
        0.0, // padding
    ];
    let force_params_cl_buf = gpu.create_f64_buffer(&force_params_cl, "force_params_cl");

    let vv_params = vec![
        n as f64, config.dt, mass, 0.0, box_side, box_side, box_side, 0.0,
    ];
    let vv_params_buf = gpu.create_f64_buffer(&vv_params, "vv_params");

    let hk_params = vec![n as f64, config.dt, mass, 0.0];
    let hk_params_buf = gpu.create_f64_buffer(&hk_params, "hk_params");

    let ke_params = vec![n as f64, mass, 0.0, 0.0];
    let ke_params_buf = gpu.create_f64_buffer(&ke_params, "ke_params");

    let workgroups = n.div_ceil(64) as u32;

    // Bind groups
    let force_bg = gpu.create_bind_group(
        &force_pipeline_cl,
        &[
            &pos_buf,
            &force_buf,
            &pe_buf,
            &force_params_cl_buf,
            &cell_start_buf,
            &cell_count_buf,
        ],
    );
    let kick_drift_bg = gpu.create_bind_group(
        &kick_drift_pipeline,
        &[&pos_buf, &vel_buf, &force_buf, &vv_params_buf],
    );
    let half_kick_bg =
        gpu.create_bind_group(&half_kick_pipeline, &[&vel_buf, &force_buf, &hk_params_buf]);
    let ke_bg = gpu.create_bind_group(&ke_pipeline, &[&vel_buf, &ke_buf, &ke_params_buf]);

    // Compute initial forces
    gpu.dispatch(&force_pipeline_cl, &force_bg, workgroups);

    // Rebuild cell list every step to maintain correctness.
    // The CPU sort is O(N) and the upload is ~48KB for N=10k — fast enough.
    let rebuild_interval = 1;

    // ── Equilibration ──
    println!("  ── Equilibration ({} steps) ──", config.equil_steps);
    let t_equil = Instant::now();

    for step in 0..config.equil_steps {
        gpu.dispatch(&kick_drift_pipeline, &kick_drift_bg, workgroups);

        // Periodically rebuild cell list
        if step % rebuild_interval == 0 && step > 0 {
            // Read back positions, rebuild cell list, re-sort, re-upload
            let pos_data = gpu.read_back_f64(&pos_buf, n * 3)?;
            let vel_data = gpu.read_back_f64(&vel_buf, n * 3)?;
            let force_data = gpu.read_back_f64(&force_buf, n * 3)?;

            let new_cl = CellList::build(&pos_data, n, box_side, config.rc);

            let sorted_p = new_cl.sort_array(&pos_data, 3);
            let sorted_v = new_cl.sort_array(&vel_data, 3);
            let sorted_f = new_cl.sort_array(&force_data, 3);

            gpu.upload_f64(&pos_buf, &sorted_p);
            gpu.upload_f64(&vel_buf, &sorted_v);
            gpu.upload_f64(&force_buf, &sorted_f);

            // Update cell list buffers
            upload_u32(&gpu, &cell_start_buf, &new_cl.cell_start);
            upload_u32(&gpu, &cell_count_buf, &new_cl.cell_count);
        }

        gpu.dispatch(&force_pipeline_cl, &force_bg, workgroups);
        gpu.dispatch(&half_kick_pipeline, &half_kick_bg, workgroups);

        // Thermostat
        if step % 10 == 0 {
            gpu.dispatch(&ke_pipeline, &ke_bg, workgroups);
            let ke_data = gpu.read_back_f64(&ke_buf, n)?;
            let total_ke: f64 = ke_data.iter().sum();
            let t_current = 2.0 * total_ke / (3.0 * n as f64);

            if t_current > 1e-30 {
                let ratio =
                    1.0 + (config.dt / config.berendsen_tau) * (temperature / t_current - 1.0);
                let scale = ratio.max(0.0).sqrt();
                let beren_params = vec![n as f64, scale, 0.0, 0.0];
                let beren_params_buf = gpu.create_f64_buffer(&beren_params, "beren_params");
                let beren_bg =
                    gpu.create_bind_group(&berendsen_pipeline, &[&vel_buf, &beren_params_buf]);
                gpu.dispatch(&berendsen_pipeline, &beren_bg, workgroups);
            }
        }

        if step % 1000 == 0 || step == config.equil_steps - 1 {
            gpu.dispatch(&ke_pipeline, &ke_bg, workgroups);
            let ke_data = gpu.read_back_f64(&ke_buf, n)?;
            let total_ke: f64 = ke_data.iter().sum();
            let t_current = 2.0 * total_ke / (3.0 * n as f64);
            println!("    Step {step}: T* = {t_current:.6} (target {temperature:.6})");
        }
    }
    println!(
        "    Equilibration complete in {:.2}s",
        t_equil.elapsed().as_secs_f64()
    );

    // ── Production ──
    println!("  ── Production ({} steps) ──", config.prod_steps);
    let t_prod = Instant::now();

    let mut energy_history = Vec::new();
    let mut positions_snapshots = Vec::new();
    let mut velocity_snapshots = Vec::new();

    for step in 0..config.prod_steps {
        gpu.dispatch(&kick_drift_pipeline, &kick_drift_bg, workgroups);

        // Rebuild cell list periodically
        if step % rebuild_interval == 0 && step > 0 {
            let pos_data = gpu.read_back_f64(&pos_buf, n * 3)?;
            let vel_data = gpu.read_back_f64(&vel_buf, n * 3)?;
            let force_data = gpu.read_back_f64(&force_buf, n * 3)?;

            let new_cl = CellList::build(&pos_data, n, box_side, config.rc);

            let sorted_p = new_cl.sort_array(&pos_data, 3);
            let sorted_v = new_cl.sort_array(&vel_data, 3);
            let sorted_f = new_cl.sort_array(&force_data, 3);

            gpu.upload_f64(&pos_buf, &sorted_p);
            gpu.upload_f64(&vel_buf, &sorted_v);
            gpu.upload_f64(&force_buf, &sorted_f);

            upload_u32(&gpu, &cell_start_buf, &new_cl.cell_start);
            upload_u32(&gpu, &cell_count_buf, &new_cl.cell_count);
        }

        gpu.dispatch(&force_pipeline_cl, &force_bg, workgroups);
        gpu.dispatch(&half_kick_pipeline, &half_kick_bg, workgroups);

        // Dump
        if step % config.dump_step == 0 {
            gpu.dispatch(&ke_pipeline, &ke_bg, workgroups);
            let ke_data = gpu.read_back_f64(&ke_buf, n)?;
            let pe_data = gpu.read_back_f64(&pe_buf, n)?;

            let total_ke: f64 = ke_data.iter().sum();
            let total_pe: f64 = pe_data.iter().sum();
            let total_e = total_ke + total_pe;
            let t_current = 2.0 * total_ke / (3.0 * n as f64);

            energy_history.push(EnergyRecord {
                step,
                ke: total_ke,
                pe: total_pe,
                total: total_e,
                temperature: t_current,
            });

            if step % (config.dump_step * 100) == 0 {
                let pos_snap = gpu.read_back_f64(&pos_buf, n * 3)?;
                let vel_snap = gpu.read_back_f64(&vel_buf, n * 3)?;
                positions_snapshots.push(pos_snap);
                velocity_snapshots.push(vel_snap);
            }
        }

        if step % 5000 == 0 || step == config.prod_steps - 1 {
            if let Some(last) = energy_history.last() {
                println!(
                    "    Step {}: T*={:.6}, KE={:.4}, PE={:.4}, E={:.4}",
                    step, last.temperature, last.ke, last.pe, last.total
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

/// Upload u32 data to a GPU buffer
fn upload_u32(gpu: &GpuF64, buffer: &wgpu::Buffer, data: &[u32]) {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    gpu.queue.write_buffer(buffer, 0, &bytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::md::config;

    /// Pure: f64 byte parsing (matches read_back_f64 logic) — testable without GPU
    fn bytes_to_f64(data: &[u8]) -> Vec<f64> {
        data.chunks_exact(8)
            .map(|chunk| {
                let bytes: [u8; 8] = chunk.try_into().expect("8-byte f64 chunk");
                f64::from_le_bytes(bytes)
            })
            .collect()
    }

    #[test]
    fn read_back_f64_byte_conversion_roundtrip() {
        let original: Vec<f64> = vec![0.0, 1.0, -1.0, std::f64::consts::PI];
        let bytes: Vec<u8> = original.iter().flat_map(|v| v.to_le_bytes()).collect();
        let recovered = bytes_to_f64(&bytes);
        assert_eq!(original.len(), recovered.len());
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert_eq!(*a, *b);
        }
    }

    #[test]
    fn init_fcc_lattice_structure() {
        let (positions, n) = init_fcc_lattice(108, 10.0);
        assert!(n >= 108);
        assert_eq!(positions.len(), n * 3);
        let box_side = 10.0;
        for i in 0..n {
            assert!(positions[i * 3] >= 0.0 && positions[i * 3] <= box_side);
            assert!(positions[i * 3 + 1] >= 0.0 && positions[i * 3 + 1] <= box_side);
            assert!(positions[i * 3 + 2] >= 0.0 && positions[i * 3 + 2] <= box_side);
        }
    }

    #[test]
    fn init_fcc_lattice_4_atoms_per_cell() {
        let (_, n) = init_fcc_lattice(4, 2.0);
        assert_eq!(n, 4);
        let (_, n) = init_fcc_lattice(32, 4.0);
        assert_eq!(n, 32);
    }

    #[test]
    fn init_velocities_center_of_mass_zero() {
        let v = init_velocities(100, 1.0, 1.0, 42);
        let mut vx = 0.0;
        let mut vy = 0.0;
        let mut vz = 0.0;
        for i in 0..100 {
            vx += v[i * 3];
            vy += v[i * 3 + 1];
            vz += v[i * 3 + 2];
        }
        assert!(vx.abs() < 1e-10);
        assert!(vy.abs() < 1e-10);
        assert!(vz.abs() < 1e-10);
    }

    #[test]
    fn init_velocities_temperature() {
        let n = 500;
        let t_target = 2.0;
        let mass = 3.0;
        let v = init_velocities(n, t_target, mass, 12345);
        let ke: f64 = v.iter().map(|x| x * x).sum::<f64>() * 0.5 * mass;
        let t_actual = 2.0 * ke / (3.0 * n as f64);
        assert!(
            (t_actual - t_target).abs() < 0.05,
            "T* = {t_actual} (target {t_target})"
        );
    }

    #[test]
    fn cell_list_build() {
        let (positions, n) = init_fcc_lattice(64, 8.0);
        let box_side = 8.0;
        let rc = 2.0;
        let cl = CellList::build(&positions, n, box_side, rc);
        assert!(cl.n_cells_total > 0);
        assert_eq!(cl.sorted_indices.len(), n);
        let total: u32 = cl.cell_count.iter().sum();
        assert_eq!(total, n as u32);
    }

    #[test]
    fn energy_record_total() {
        let rec = EnergyRecord {
            step: 0,
            ke: 100.0,
            pe: -150.0,
            total: -50.0,
            temperature: 1.0,
        };
        assert!((rec.ke + rec.pe - rec.total).abs() < 1e-10);
    }

    #[test]
    fn config_validation_box_side() {
        let config = config::quick_test_case(256);
        let l = config.box_side();
        let expected = (4.0 * std::f64::consts::PI * 256.0 / 3.0).cbrt();
        assert!((l - expected).abs() < 1e-10);
    }

    #[test]
    #[ignore = "requires GPU"]
    fn run_simulation_gpu() {
        // Placeholder: run_simulation requires GPU
        let config = config::quick_test_case(64);
        let _ = config;
    }
}
