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

use barracuda::shaders::precision::ShaderTemplate;

use std::f64::consts::PI;

/// Patch the math_f64 preamble for WGSL/Naga compatibility:
/// 1. Strip recursive gamma_f64 (WGSL doesn't support recursion)
/// 2. Replace 1e308 literals (overflows f32 during AbstractFloat → concrete)
/// 3. Replace -1e308 similarly
fn patch_math_f64_preamble(preamble: &str) -> String {
    let mut result = String::with_capacity(preamble.len());
    let mut skip = false;
    for line in preamble.lines() {
        // Strip gamma_f64 function entirely (recursive, not needed for MD)
        if line.contains("GAMMA FUNCTION") || line.starts_with("fn gamma_f64") {
            skip = true;
        }
        if skip {
            if line == "}" {
                skip = false;
                continue;
            }
            continue;
        }
        // Fix 1e308 / -1e308 literals: Naga parses AbstractFloat → f32 first,
        // and 1e308 overflows f32 (max ~3.4e38). For MD physics, 1e38 is more
        // than sufficient as an "infinity" sentinel.
        let patched = line
            .replace("f64_const(x, 1e308)", "f64_const(x, 1e38)")
            .replace("f64_const(x, -1e308)", "f64_const(x, -1e38)");
        result.push_str(&patched);
        result.push('\n');
    }
    result
}
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
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
    for v in velocities.iter() {
        v_sq_sum += v * v;
    }
    let t_current = mass * v_sq_sum / (3.0 * n_f);
    let scale = (temperature / t_current).sqrt();
    for v in velocities.iter_mut() {
        *v *= scale;
    }

    velocities
}

/// Run the full GPU MD simulation
pub async fn run_simulation(config: &MdConfig) -> Result<MdSimulation, String> {
    let t_start = Instant::now();
    let n = config.n_particles;
    let box_side = config.box_side();
    let prefactor = config.force_prefactor();
    let temperature = config.temperature();
    let mass = config.reduced_mass();

    println!("  ── Initializing {} particles ──", n);
    println!("    Box side: {:.4} a_ws", box_side);
    println!("    κ = {}, Γ = {}, T* = {:.6}", config.kappa, config.gamma, temperature);
    println!("    rc = {} a_ws, dt* = {}, m* = {}", config.rc, config.dt, mass);
    println!("    Force prefactor: {:.4}", prefactor);

    // Initialize particles
    let (mut positions, n_actual) = init_fcc_lattice(n, box_side);
    let n = n_actual.min(n); // might be slightly different due to FCC packing
    let velocities = init_velocities(n, temperature, mass, 42);

    println!("    Placed {} particles on FCC lattice", n);

    // Initialize GPU
    let gpu = GpuF64::new().await?;
    if !gpu.has_f64 {
        return Err("GPU does not support SHADER_F64 — cannot run f64 MD".into());
    }
    gpu.print_info();

    // Compile shaders (prepend math_f64 library, excluding recursive gamma_f64
    // which WGSL/Naga does not support)
    println!("  ── Compiling f64 WGSL shaders ──");
    let t_compile = Instant::now();

    let md_math = patch_math_f64_preamble(&ShaderTemplate::math_f64_preamble());
    let prepend = |body: &str| -> String { format!("{}\n\n{}", md_math, body) };

    let force_shader = prepend(shaders::SHADER_YUKAWA_FORCE);
    let kick_drift_shader = prepend(shaders::SHADER_VV_KICK_DRIFT);
    let half_kick_shader = prepend(shaders::SHADER_VV_HALF_KICK);
    let berendsen_shader = prepend(shaders::SHADER_BERENDSEN);
    let ke_shader = prepend(shaders::SHADER_KINETIC_ENERGY);

    let force_pipeline = gpu.create_pipeline(&force_shader, "yukawa_force_f64");
    let kick_drift_pipeline = gpu.create_pipeline(&kick_drift_shader, "vv_kick_drift_f64");
    let half_kick_pipeline = gpu.create_pipeline(&half_kick_shader, "vv_half_kick_f64");
    let berendsen_pipeline = gpu.create_pipeline(&berendsen_shader, "berendsen_f64");
    let ke_pipeline = gpu.create_pipeline(&ke_shader, "kinetic_energy_f64");

    println!("    Compiled 5 shaders in {:.1}ms", t_compile.elapsed().as_secs_f64() * 1000.0);

    // Create GPU buffers
    let pos_buf = gpu.create_f64_output_buffer(n * 3, "positions");
    let vel_buf = gpu.create_f64_output_buffer(n * 3, "velocities");
    let force_buf = gpu.create_f64_output_buffer(n * 3, "forces");
    let pe_buf = gpu.create_f64_output_buffer(n, "pe_per_particle");
    let ke_buf = gpu.create_f64_output_buffer(n, "ke_per_particle");

    // Upload initial positions and velocities
    upload_f64(&gpu, &pos_buf, &positions);
    upload_f64(&gpu, &vel_buf, &velocities);

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
        n as f64, config.dt, mass, 0.0,
        box_side, box_side, box_side, 0.0,
    ];
    let vv_params_buf = gpu.create_f64_buffer(&vv_params, "vv_params");

    // Half-kick params: [n, dt, mass, _]
    let hk_params = vec![n as f64, config.dt, mass, 0.0];
    let hk_params_buf = gpu.create_f64_buffer(&hk_params, "hk_params");

    // KE params: [n, mass, _, _]
    let ke_params = vec![n as f64, mass, 0.0, 0.0];
    let ke_params_buf = gpu.create_f64_buffer(&ke_params, "ke_params");

    let workgroups = ((n + 63) / 64) as u32;

    // ── Bind groups ──
    let force_bg = create_bind_group(
        &gpu,
        &force_pipeline,
        &[&pos_buf, &force_buf, &pe_buf, &force_params_buf],
    );
    let kick_drift_bg = create_bind_group(
        &gpu,
        &kick_drift_pipeline,
        &[&pos_buf, &vel_buf, &force_buf, &vv_params_buf],
    );
    let half_kick_bg = create_bind_group(
        &gpu,
        &half_kick_pipeline,
        &[&vel_buf, &force_buf, &hk_params_buf],
    );
    let ke_bg = create_bind_group(
        &gpu,
        &ke_pipeline,
        &[&vel_buf, &ke_buf, &ke_params_buf],
    );

    // ── Compute initial forces ──
    dispatch(&gpu, &force_pipeline, &force_bg, workgroups);

    // ══════════════════════════════════════════════════════════════
    //  Equilibration
    // ══════════════════════════════════════════════════════════════
    println!("  ── Equilibration ({} steps) ──", config.equil_steps);
    let t_equil = Instant::now();

    for step in 0..config.equil_steps {
        // 1. Half-kick + drift + PBC
        dispatch(&gpu, &kick_drift_pipeline, &kick_drift_bg, workgroups);
        // 2. Recompute forces
        dispatch(&gpu, &force_pipeline, &force_bg, workgroups);
        // 3. Second half-kick
        dispatch(&gpu, &half_kick_pipeline, &half_kick_bg, workgroups);

        // 4. Berendsen thermostat every step
        if step % 10 == 0 {
            // Compute current KE on GPU, read back to get temperature
            dispatch(&gpu, &ke_pipeline, &ke_bg, workgroups);
            let ke_per_particle = read_back_f64(&gpu, &ke_buf, n);
            let total_ke: f64 = ke_per_particle.iter().sum();
            // T* = 2 * KE / (3 * N) in reduced units (k_B = 1)
            let t_current = 2.0 * total_ke / (3.0 * n as f64);

            if t_current > 1e-30 {
                let ratio = 1.0 + (config.dt / config.berendsen_tau) * (temperature / t_current - 1.0);
                let scale = ratio.max(0.0).sqrt();

                // Upload Berendsen params and apply
                let beren_params = vec![n as f64, scale, 0.0, 0.0];
                let beren_params_buf = gpu.create_f64_buffer(&beren_params, "beren_params");
                let beren_bg = create_bind_group(
                    &gpu,
                    &berendsen_pipeline,
                    &[&vel_buf, &beren_params_buf],
                );
                dispatch(&gpu, &berendsen_pipeline, &beren_bg, workgroups);
            }
        }

        if step % 1000 == 0 || step == config.equil_steps - 1 {
            // Quick temperature check
            dispatch(&gpu, &ke_pipeline, &ke_bg, workgroups);
            let ke_data = read_back_f64(&gpu, &ke_buf, n);
            let total_ke: f64 = ke_data.iter().sum();
            let t_current = 2.0 * total_ke / (3.0 * n as f64);
            println!("    Step {}: T* = {:.6} (target {:.6})", step, t_current, temperature);
        }
    }
    let equil_time = t_equil.elapsed().as_secs_f64();
    println!("    Equilibration complete in {:.2}s", equil_time);

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
        dispatch(&gpu, &kick_drift_pipeline, &kick_drift_bg, workgroups);
        // 2. Recompute forces
        dispatch(&gpu, &force_pipeline, &force_bg, workgroups);
        // 3. Second half-kick
        dispatch(&gpu, &half_kick_pipeline, &half_kick_bg, workgroups);

        // Dump at intervals
        if step % config.dump_step == 0 {
            // Read back KE and PE
            dispatch(&gpu, &ke_pipeline, &ke_bg, workgroups);
            let ke_data = read_back_f64(&gpu, &ke_buf, n);
            let pe_data = read_back_f64(&gpu, &pe_buf, n);

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
                let pos_snap = read_back_f64(&gpu, &pos_buf, n * 3);
                let vel_snap = read_back_f64(&gpu, &vel_buf, n * 3);
                positions_snapshots.push(pos_snap);
                velocity_snapshots.push(vel_snap);
            }
        }

        if step % 5000 == 0 || step == config.prod_steps - 1 {
            let last = energy_history.last().unwrap();
            println!(
                "    Step {}: T*={:.6}, KE={:.4}, PE={:.4}, E={:.4}",
                step, last.temperature, last.ke, last.pe, last.total
            );
        }
    }

    let prod_time = t_prod.elapsed().as_secs_f64();
    let total_time = t_start.elapsed().as_secs_f64();
    let total_steps = config.equil_steps + config.prod_steps;
    let steps_per_sec = total_steps as f64 / total_time;

    println!("    Production complete in {:.2}s", prod_time);
    println!("    Total: {:.2}s ({:.1} steps/s)", total_time, steps_per_sec);

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
// GPU Helper Functions
// ═══════════════════════════════════════════════════════════════════

/// Upload f64 data to a GPU buffer
fn upload_f64(gpu: &GpuF64, buffer: &wgpu::Buffer, data: &[f64]) {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    gpu.queue.write_buffer(buffer, 0, &bytes);
}

/// Read back f64 data from GPU buffer
fn read_back_f64(gpu: &GpuF64, buffer: &wgpu::Buffer, count: usize) -> Vec<f64> {
    let staging = gpu.create_staging_buffer(count * 8, "readback");

    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, (count * 8) as u64);
    gpu.queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    gpu.device.poll(wgpu::Maintain::Wait);
    receiver.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let result: Vec<f64> = data
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    drop(data);
    staging.unmap();
    result
}

/// Dispatch a compute pipeline (fire-and-forget within queue)
fn dispatch(
    gpu: &GpuF64,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroups: u32,
) {
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("md_dispatch"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("md_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    gpu.queue.submit(std::iter::once(encoder.finish()));
}

/// Create a bind group from a pipeline and buffer slice
fn create_bind_group(
    gpu: &GpuF64,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    let layout = pipeline.get_bind_group_layout(0);
    let entries: Vec<wgpu::BindGroupEntry> = buffers
        .iter()
        .enumerate()
        .map(|(i, buf)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: buf.as_entire_binding(),
        })
        .collect();

    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("md_bind_group"),
        layout: &layout,
        entries: &entries,
    })
}

// ═══════════════════════════════════════════════════════════════════
// Cell List Infrastructure
// ═══════════════════════════════════════════════════════════════════

/// Cell list metadata
pub struct CellList {
    pub n_cells: [usize; 3],     // cells per dimension
    pub cell_size: [f64; 3],     // cell side length
    pub n_cells_total: usize,
    pub cell_start: Vec<u32>,    // first particle index per cell
    pub cell_count: Vec<u32>,    // particle count per cell
    pub sorted_indices: Vec<usize>, // particle index in sorted order
}

impl CellList {
    /// Build cell list from positions on CPU
    pub fn build(
        positions: &[f64],
        n: usize,
        box_side: f64,
        rc: f64,
    ) -> Self {
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
pub async fn run_simulation_celllist(config: &MdConfig) -> Result<MdSimulation, String> {
    let t_start = Instant::now();
    let n = config.n_particles;
    let box_side = config.box_side();
    let prefactor = config.force_prefactor();
    let temperature = config.temperature();
    let mass = config.reduced_mass();

    println!("  ── Initializing {} particles (cell-list mode) ──", n);
    println!("    Box side: {:.4} a_ws", box_side);
    println!("    κ = {}, Γ = {}, T* = {:.6}", config.kappa, config.gamma, temperature);
    println!("    rc = {} a_ws, dt* = {}, m* = {}", config.rc, config.dt, mass);

    // Initialize particles
    let (positions, n_actual) = init_fcc_lattice(n, box_side);
    let n = n_actual.min(n);
    let velocities = init_velocities(n, temperature, mass, 42);

    // Build initial cell list
    let cell_list = CellList::build(&positions, n, box_side, config.rc);
    println!(
        "    Cell list: {}×{}×{} = {} cells, cell_size={:.3} a_ws",
        cell_list.n_cells[0], cell_list.n_cells[1], cell_list.n_cells[2],
        cell_list.n_cells_total, cell_list.cell_size[0]
    );
    println!("    Placed {} particles on FCC lattice", n);

    // Initialize GPU
    let gpu = GpuF64::new().await?;
    if !gpu.has_f64 {
        return Err("GPU does not support SHADER_F64".into());
    }
    gpu.print_info();

    // Compile shaders
    println!("  ── Compiling f64 WGSL shaders (cell-list) ──");
    let t_compile = Instant::now();

    let md_math = patch_math_f64_preamble(&ShaderTemplate::math_f64_preamble());
    let prepend = |body: &str| -> String { format!("{}\n\n{}", md_math, body) };

    let force_shader_cl = prepend(shaders::SHADER_YUKAWA_FORCE_CELLLIST);
    let kick_drift_shader = prepend(shaders::SHADER_VV_KICK_DRIFT);
    let half_kick_shader = prepend(shaders::SHADER_VV_HALF_KICK);
    let berendsen_shader = prepend(shaders::SHADER_BERENDSEN);
    let ke_shader = prepend(shaders::SHADER_KINETIC_ENERGY);

    let force_pipeline_cl = gpu.create_pipeline(&force_shader_cl, "yukawa_force_cl_f64");
    let kick_drift_pipeline = gpu.create_pipeline(&kick_drift_shader, "vv_kick_drift_f64");
    let half_kick_pipeline = gpu.create_pipeline(&half_kick_shader, "vv_half_kick_f64");
    let berendsen_pipeline = gpu.create_pipeline(&berendsen_shader, "berendsen_f64");
    let ke_pipeline = gpu.create_pipeline(&ke_shader, "kinetic_energy_f64");

    println!("    Compiled 5 shaders in {:.1}ms", t_compile.elapsed().as_secs_f64() * 1000.0);

    // Create GPU buffers
    let pos_buf = gpu.create_f64_output_buffer(n * 3, "positions");
    let vel_buf = gpu.create_f64_output_buffer(n * 3, "velocities");
    let force_buf = gpu.create_f64_output_buffer(n * 3, "forces");
    let pe_buf = gpu.create_f64_output_buffer(n, "pe_per_particle");
    let ke_buf = gpu.create_f64_output_buffer(n, "ke_per_particle");

    // Cell list buffers (u32)
    let cell_start_buf = create_u32_buffer(&gpu, &cell_list.cell_start, "cell_start");
    let cell_count_buf = create_u32_buffer(&gpu, &cell_list.cell_count, "cell_count");

    // Upload initial sorted positions and velocities
    let sorted_pos = cell_list.sort_array(&positions, 3);
    let sorted_vel = cell_list.sort_array(&velocities, 3);
    upload_f64(&gpu, &pos_buf, &sorted_pos);
    upload_f64(&gpu, &vel_buf, &sorted_vel);

    // Extended force params for cell list
    let force_params_cl = vec![
        n as f64,
        config.kappa,
        prefactor,
        config.rc * config.rc,
        box_side, box_side, box_side,
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
        n as f64, config.dt, mass, 0.0,
        box_side, box_side, box_side, 0.0,
    ];
    let vv_params_buf = gpu.create_f64_buffer(&vv_params, "vv_params");

    let hk_params = vec![n as f64, config.dt, mass, 0.0];
    let hk_params_buf = gpu.create_f64_buffer(&hk_params, "hk_params");

    let ke_params = vec![n as f64, mass, 0.0, 0.0];
    let ke_params_buf = gpu.create_f64_buffer(&ke_params, "ke_params");

    let workgroups = ((n + 63) / 64) as u32;

    // Bind groups
    let force_bg = create_bind_group(
        &gpu,
        &force_pipeline_cl,
        &[&pos_buf, &force_buf, &pe_buf, &force_params_cl_buf, &cell_start_buf, &cell_count_buf],
    );
    let kick_drift_bg = create_bind_group(
        &gpu,
        &kick_drift_pipeline,
        &[&pos_buf, &vel_buf, &force_buf, &vv_params_buf],
    );
    let half_kick_bg = create_bind_group(
        &gpu,
        &half_kick_pipeline,
        &[&vel_buf, &force_buf, &hk_params_buf],
    );
    let ke_bg = create_bind_group(
        &gpu,
        &ke_pipeline,
        &[&vel_buf, &ke_buf, &ke_params_buf],
    );

    // Compute initial forces
    dispatch(&gpu, &force_pipeline_cl, &force_bg, workgroups);

    // Rebuild cell list every step to maintain correctness.
    // The CPU sort is O(N) and the upload is ~48KB for N=10k — fast enough.
    let rebuild_interval = 1;

    // ── Equilibration ──
    println!("  ── Equilibration ({} steps) ──", config.equil_steps);
    let t_equil = Instant::now();

    for step in 0..config.equil_steps {
        dispatch(&gpu, &kick_drift_pipeline, &kick_drift_bg, workgroups);

        // Periodically rebuild cell list
        if step % rebuild_interval == 0 && step > 0 {
            // Read back positions, rebuild cell list, re-sort, re-upload
            let pos_data = read_back_f64(&gpu, &pos_buf, n * 3);
            let vel_data = read_back_f64(&gpu, &vel_buf, n * 3);
            let force_data = read_back_f64(&gpu, &force_buf, n * 3);

            let new_cl = CellList::build(&pos_data, n, box_side, config.rc);

            let sorted_p = new_cl.sort_array(&pos_data, 3);
            let sorted_v = new_cl.sort_array(&vel_data, 3);
            let sorted_f = new_cl.sort_array(&force_data, 3);

            upload_f64(&gpu, &pos_buf, &sorted_p);
            upload_f64(&gpu, &vel_buf, &sorted_v);
            upload_f64(&gpu, &force_buf, &sorted_f);

            // Update cell list buffers
            upload_u32(&gpu, &cell_start_buf, &new_cl.cell_start);
            upload_u32(&gpu, &cell_count_buf, &new_cl.cell_count);
        }

        dispatch(&gpu, &force_pipeline_cl, &force_bg, workgroups);
        dispatch(&gpu, &half_kick_pipeline, &half_kick_bg, workgroups);

        // Thermostat
        if step % 10 == 0 {
            dispatch(&gpu, &ke_pipeline, &ke_bg, workgroups);
            let ke_data = read_back_f64(&gpu, &ke_buf, n);
            let total_ke: f64 = ke_data.iter().sum();
            let t_current = 2.0 * total_ke / (3.0 * n as f64);

            if t_current > 1e-30 {
                let ratio = 1.0 + (config.dt / config.berendsen_tau) * (temperature / t_current - 1.0);
                let scale = ratio.max(0.0).sqrt();
                let beren_params = vec![n as f64, scale, 0.0, 0.0];
                let beren_params_buf = gpu.create_f64_buffer(&beren_params, "beren_params");
                let beren_bg = create_bind_group(&gpu, &berendsen_pipeline, &[&vel_buf, &beren_params_buf]);
                dispatch(&gpu, &berendsen_pipeline, &beren_bg, workgroups);
            }
        }

        if step % 1000 == 0 || step == config.equil_steps - 1 {
            dispatch(&gpu, &ke_pipeline, &ke_bg, workgroups);
            let ke_data = read_back_f64(&gpu, &ke_buf, n);
            let total_ke: f64 = ke_data.iter().sum();
            let t_current = 2.0 * total_ke / (3.0 * n as f64);
            println!("    Step {}: T* = {:.6} (target {:.6})", step, t_current, temperature);
        }
    }
    println!("    Equilibration complete in {:.2}s", t_equil.elapsed().as_secs_f64());

    // ── Production ──
    println!("  ── Production ({} steps) ──", config.prod_steps);
    let t_prod = Instant::now();

    let mut energy_history = Vec::new();
    let mut positions_snapshots = Vec::new();
    let mut velocity_snapshots = Vec::new();

    for step in 0..config.prod_steps {
        dispatch(&gpu, &kick_drift_pipeline, &kick_drift_bg, workgroups);

        // Rebuild cell list periodically
        if step % rebuild_interval == 0 && step > 0 {
            let pos_data = read_back_f64(&gpu, &pos_buf, n * 3);
            let vel_data = read_back_f64(&gpu, &vel_buf, n * 3);
            let force_data = read_back_f64(&gpu, &force_buf, n * 3);

            let new_cl = CellList::build(&pos_data, n, box_side, config.rc);

            let sorted_p = new_cl.sort_array(&pos_data, 3);
            let sorted_v = new_cl.sort_array(&vel_data, 3);
            let sorted_f = new_cl.sort_array(&force_data, 3);

            upload_f64(&gpu, &pos_buf, &sorted_p);
            upload_f64(&gpu, &vel_buf, &sorted_v);
            upload_f64(&gpu, &force_buf, &sorted_f);

            upload_u32(&gpu, &cell_start_buf, &new_cl.cell_start);
            upload_u32(&gpu, &cell_count_buf, &new_cl.cell_count);
        }

        dispatch(&gpu, &force_pipeline_cl, &force_bg, workgroups);
        dispatch(&gpu, &half_kick_pipeline, &half_kick_bg, workgroups);

        // Dump
        if step % config.dump_step == 0 {
            dispatch(&gpu, &ke_pipeline, &ke_bg, workgroups);
            let ke_data = read_back_f64(&gpu, &ke_buf, n);
            let pe_data = read_back_f64(&gpu, &pe_buf, n);

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
                let pos_snap = read_back_f64(&gpu, &pos_buf, n * 3);
                let vel_snap = read_back_f64(&gpu, &vel_buf, n * 3);
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

    println!("    Production complete in {:.2}s", prod_time);
    println!("    Total: {:.2}s ({:.1} steps/s)", total_time, steps_per_sec);

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

/// Create a u32 storage buffer
fn create_u32_buffer(gpu: &GpuF64, data: &[u32], label: &str) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: &bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    })
}

/// Upload u32 data to a GPU buffer
fn upload_u32(gpu: &GpuF64, buffer: &wgpu::Buffer, data: &[u32]) {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    gpu.queue.write_buffer(buffer, 0, &bytes);
}
