//! Cell-List Force Diagnostic
//!
//! Compares all-pairs vs cell-list Yukawa force kernels on identical particle
//! data at multiple N values. This isolates whether the cell-list energy
//! conservation bug is in the shader physics or the sort/rebuild infrastructure.
//!
//! Experiment 002: Cell-List Force Kernel Investigation
//!
//! Usage:
//!   cargo run --release --bin celllist_diag

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::md::config::MdConfig;
use hotspring_barracuda::md::shaders;
use hotspring_barracuda::md::simulation::{init_fcc_lattice, init_velocities, CellList};

use barracuda::shaders::precision::ShaderTemplate;

use std::time::Instant;

/// Patch the math_f64 preamble (same as simulation.rs)
fn patch_math_f64_preamble(preamble: &str) -> String {
    let mut result = String::with_capacity(preamble.len());
    let mut skip = false;
    for line in preamble.lines() {
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
        let patched = line
            .replace("f64_const(x, 1e308)", "f64_const(x, 1e38)")
            .replace("f64_const(x, -1e308)", "f64_const(x, -1e38)");
        result.push_str(&patched);
        result.push('\n');
    }
    result
}

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

/// Dispatch a compute pipeline
fn dispatch(
    gpu: &GpuF64,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroups: u32,
) {
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("diag_dispatch"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("diag_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    gpu.queue.submit(std::iter::once(encoder.finish()));
}

/// Create a bind group from a pipeline and buffers
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
        label: Some("diag_bind_group"),
        layout: &layout,
        entries: &entries,
    })
}

/// Run force comparison at a given N
async fn compare_forces(gpu: &GpuF64, n: usize) {
    let kappa = 2.0;
    let gamma = 158.0;
    let rc = 6.5;
    let prefactor = 1.0; // reduced units for OCP

    // Box side from number density = 1 in reduced units
    let box_side = (n as f64).cbrt() * (4.0 * std::f64::consts::PI / 3.0_f64).cbrt();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  N = {}  |  box = {:.3}  |  rc = {}  |  pairs = {}",
        n, box_side, rc, n * (n - 1) / 2);

    let cells_per_dim = (box_side / rc).floor() as usize;
    let cells_per_dim = cells_per_dim.max(3);
    let cell_size = box_side / cells_per_dim as f64;
    println!("  cells/dim = {}  |  cell_size = {:.4}  |  cell_size >= rc? {}",
        cells_per_dim, cell_size, if cell_size >= rc { "YES" } else { "NO ← BUG" });
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Initialize positions on FCC lattice (deterministic)
    let (positions, n_actual) = init_fcc_lattice(n, box_side);
    let n = n_actual.min(n);
    println!("  Placed {} particles on FCC lattice", n);

    // Prepare shaders
    let md_math = patch_math_f64_preamble(&ShaderTemplate::math_f64_preamble());
    let prepend = |body: &str| -> String { format!("{}\n\n{}", md_math, body) };

    let ap_source = prepend(shaders::SHADER_YUKAWA_FORCE);
    let cl_source = prepend(shaders::SHADER_YUKAWA_FORCE_CELLLIST);
    let cl_v2_source = prepend(shaders::SHADER_YUKAWA_FORCE_CELLLIST_V2);

    // V4: Cell-list with f64 cell arrays instead of u32
    let cl_v4_src = format!("{}\n\n{}", md_math, r#"
@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read_write> forces: array<f64>;
@group(0) @binding(2) var<storage, read_write> pe_buf: array<f64>;
@group(0) @binding(3) var<storage, read> params: array<f64>;
@group(0) @binding(4) var<storage, read> cell_data: array<f64>;

fn pbc_v4(delta: f64, box_size: f64) -> f64 {
    var d = delta;
    let half = box_size / 2.0;
    if (d > half)  { d = d - box_size; }
    if (d < -half) { d = d + box_size; }
    return d;
}

fn wrap_cell_v4(c: i32, n: i32) -> i32 {
    return ((c % n) + n) % n;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n_part = u32(params[0]);
    if (idx >= n_part) { return; }

    let kappa      = params[1];
    let prefactor  = params[2];
    let cutoff_sq  = params[3];
    let box_x      = params[4];
    let box_y      = params[5];
    let box_z      = params[6];
    let softening  = params[7];
    let ncx        = i32(params[8]);
    let ncy        = i32(params[9]);
    let ncz        = i32(params[10]);
    let csx        = params[11];
    let csy        = params[12];
    let csz        = params[13];

    let px = positions[idx * 3u];
    let py = positions[idx * 3u + 1u];
    let pz = positions[idx * 3u + 2u];

    let my_cx = i32(px / csx);
    let my_cy = i32(py / csy);
    let my_cz = i32(pz / csz);

    var acc_fx = px - px;
    var acc_fy = px - px;
    var acc_fz = px - px;
    var acc_pe = px - px;

    // cell_data layout: [start_0, count_0, start_1, count_1, ...]
    // All stored as f64, cast to u32 when needed
    for (var neigh = 0u; neigh < 27u; neigh = neigh + 1u) {
        let off_x = i32(neigh % 3u) - 1;
        let off_y = i32((neigh / 3u) % 3u) - 1;
        let off_z = i32(neigh / 9u) - 1;

        let nb_cx = wrap_cell_v4(my_cx + off_x, ncx);
        let nb_cy = wrap_cell_v4(my_cy + off_y, ncy);
        let nb_cz = wrap_cell_v4(my_cz + off_z, ncz);
        let cell_linear = u32(nb_cx + nb_cy * ncx + nb_cz * ncx * ncy);

        let start = u32(cell_data[cell_linear * 2u]);
        let cnt   = u32(cell_data[cell_linear * 2u + 1u]);

        for (var k = 0u; k < cnt; k = k + 1u) {
            let j = start + k;
            if (idx == j) { continue; }

            let qx = positions[j * 3u];
            let qy = positions[j * 3u + 1u];
            let qz = positions[j * 3u + 2u];

            let rx = pbc_v4(qx - px, box_x);
            let ry = pbc_v4(qy - py, box_y);
            let rz = pbc_v4(qz - pz, box_z);

            let r_sq = rx * rx + ry * ry + rz * rz;
            if (r_sq > cutoff_sq) { continue; }

            let r = sqrt_f64(r_sq + softening);
            let scr = exp_f64(-kappa * r);
            let fmag = prefactor * scr * (1.0 + kappa * r) / r_sq;
            let ir = 1.0 / r;

            acc_fx = acc_fx - fmag * rx * ir;
            acc_fy = acc_fy - fmag * ry * ir;
            acc_fz = acc_fz - fmag * rz * ir;
            acc_pe = acc_pe + 0.5 * prefactor * scr * ir;
        }
    }

    forces[idx * 3u]      = acc_fx;
    forces[idx * 3u + 1u] = acc_fy;
    forces[idx * 3u + 2u] = acc_fz;
    pe_buf[idx] = acc_pe;
}
"#);

    // V3: No-cutoff cell-list — test if force formula gives different results
    let cl_v3_src = format!("{}\n\n{}", md_math, r#"
@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read_write> forces: array<f64>;
@group(0) @binding(2) var<storage, read_write> pe_buf: array<f64>;
@group(0) @binding(3) var<storage, read> params: array<f64>;
@group(0) @binding(4) var<storage, read> cell_start_arr: array<u32>;
@group(0) @binding(5) var<storage, read> cell_count_arr: array<u32>;

fn pbc_v3(delta: f64, box_size: f64) -> f64 {
    var d = delta;
    let half = box_size / 2.0;
    if (d > half)  { d = d - box_size; }
    if (d < -half) { d = d + box_size; }
    return d;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n_part = u32(params[0]);
    if (idx >= n_part) { return; }

    let kappa      = params[1];
    let prefactor  = params[2];
    let box_x      = params[4];
    let box_y      = params[5];
    let box_z      = params[6];
    let softening  = params[7];

    // Touch cell buffers to keep bindings alive
    let _cs = cell_start_arr[0u];
    let _cc = cell_count_arr[0u];

    let px = positions[idx * 3u];
    let py = positions[idx * 3u + 1u];
    let pz = positions[idx * 3u + 2u];

    var acc_fx = px - px;
    var acc_fy = px - px;
    var acc_fz = px - px;
    var acc_pe = px - px;

    // ALL-PAIRS loop (same as hybrid) but with cell-list bindings
    // NO cutoff check — accumulate ALL pairs
    for (var j = 0u; j < n_part; j = j + 1u) {
        if (idx == j) { continue; }

        let qx = positions[j * 3u];
        let qy = positions[j * 3u + 1u];
        let qz = positions[j * 3u + 2u];

        let rx = pbc_v3(qx - px, box_x);
        let ry = pbc_v3(qy - py, box_y);
        let rz = pbc_v3(qz - pz, box_z);

        let r_sq = rx * rx + ry * ry + rz * rz;
        // NO cutoff check!

        let r = sqrt_f64(r_sq + softening);
        let scr = exp_f64(-kappa * r);
        let fmag = prefactor * scr * (1.0 + kappa * r) / r_sq;
        let ir = 1.0 / r;

        acc_fx = acc_fx - fmag * rx * ir;
        acc_fy = acc_fy - fmag * ry * ir;
        acc_fz = acc_fz - fmag * rz * ir;
        acc_pe = acc_pe + 0.5 * prefactor * scr * ir;
    }

    forces[idx * 3u]      = acc_fx;
    forces[idx * 3u + 1u] = acc_fy;
    forces[idx * 3u + 2u] = acc_fz;
    pe_buf[idx] = acc_pe;
}
"#);

    let ap_pipeline = gpu.create_pipeline(&ap_source, "yukawa_force_allpairs");
    let cl_pipeline = gpu.create_pipeline(&cl_source, "yukawa_force_celllist");

    let workgroups = ((n + 63) / 64) as u32;

    // ── All-pairs force computation ──
    let pos_buf_ap = gpu.create_f64_output_buffer(n * 3, "pos_ap");
    let force_buf_ap = gpu.create_f64_output_buffer(n * 3, "force_ap");
    let pe_buf_ap = gpu.create_f64_output_buffer(n, "pe_ap");

    upload_f64(gpu, &pos_buf_ap, &positions);

    let force_params = vec![
        n as f64, kappa, prefactor, rc * rc,
        box_side, box_side, box_side, 0.0,
    ];
    let force_params_buf = gpu.create_f64_buffer(&force_params, "force_params_ap");

    let ap_bg = create_bind_group(gpu, &ap_pipeline,
        &[&pos_buf_ap, &force_buf_ap, &pe_buf_ap, &force_params_buf]);

    let t0 = Instant::now();
    dispatch(gpu, &ap_pipeline, &ap_bg, workgroups);
    let forces_ap = read_back_f64(gpu, &force_buf_ap, n * 3);
    let pe_ap = read_back_f64(gpu, &pe_buf_ap, n);
    let ap_time = t0.elapsed().as_secs_f64();

    let total_pe_ap: f64 = pe_ap.iter().sum();
    println!("  All-pairs: PE = {:.6}, computed in {:.3}ms", total_pe_ap, ap_time * 1000.0);

    // ── Cell-list force computation ──
    // Build cell list from SAME positions (unsorted)
    let cell_list = CellList::build(&positions, n, box_side, rc);
    println!("  Cell list: {}×{}×{} = {} cells",
        cell_list.n_cells[0], cell_list.n_cells[1], cell_list.n_cells[2],
        cell_list.n_cells_total);

    // Sort positions for cell-list kernel
    let sorted_pos = cell_list.sort_array(&positions, 3);

    let pos_buf_cl = gpu.create_f64_output_buffer(n * 3, "pos_cl");
    let force_buf_cl = gpu.create_f64_output_buffer(n * 3, "force_cl");
    let pe_buf_cl = gpu.create_f64_output_buffer(n, "pe_cl");

    upload_f64(gpu, &pos_buf_cl, &sorted_pos);

    let cell_start_buf = create_u32_buffer(gpu, &cell_list.cell_start, "cell_start");
    let cell_count_buf = create_u32_buffer(gpu, &cell_list.cell_count, "cell_count");

    let force_params_cl = vec![
        n as f64, kappa, prefactor, rc * rc,
        box_side, box_side, box_side, 0.0,
        cell_list.n_cells[0] as f64,
        cell_list.n_cells[1] as f64,
        cell_list.n_cells[2] as f64,
        cell_list.cell_size[0],
        cell_list.cell_size[1],
        cell_list.cell_size[2],
        cell_list.n_cells_total as f64,
        0.0,
    ];
    let force_params_cl_buf = gpu.create_f64_buffer(&force_params_cl, "force_params_cl");

    let cl_bg = create_bind_group(gpu, &cl_pipeline,
        &[&pos_buf_cl, &force_buf_cl, &pe_buf_cl, &force_params_cl_buf,
          &cell_start_buf, &cell_count_buf]);

    let t0 = Instant::now();
    dispatch(gpu, &cl_pipeline, &cl_bg, workgroups);
    let forces_cl_sorted = read_back_f64(gpu, &force_buf_cl, n * 3);
    let pe_cl_sorted = read_back_f64(gpu, &pe_buf_cl, n);
    let cl_time = t0.elapsed().as_secs_f64();

    // Unsort cell-list forces back to original order for comparison
    let forces_cl = cell_list.unsort_array(&forces_cl_sorted, 3);
    let pe_cl_unsorted: Vec<f64> = {
        let mut unsorted = vec![0.0f64; n];
        for (new_idx, &old_idx) in cell_list.sorted_indices.iter().enumerate() {
            unsorted[old_idx] = pe_cl_sorted[new_idx];
        }
        unsorted
    };

    let total_pe_cl: f64 = pe_cl_unsorted.iter().sum();
    println!("  Cell-list v1: PE = {:.6}, computed in {:.3}ms", total_pe_cl, cl_time * 1000.0);

    // ── Cell-list v2 (flat loop) ──
    let cl_v2_pipeline = gpu.create_pipeline(&cl_v2_source, "yukawa_force_celllist_v2");

    let pos_buf_v2 = gpu.create_f64_output_buffer(n * 3, "pos_v2");
    let force_buf_v2 = gpu.create_f64_output_buffer(n * 3, "force_v2");
    let pe_buf_v2 = gpu.create_f64_output_buffer(n, "pe_v2");
    upload_f64(gpu, &pos_buf_v2, &sorted_pos);

    let cell_start_buf_v2 = create_u32_buffer(gpu, &cell_list.cell_start, "cell_start_v2");
    let cell_count_buf_v2 = create_u32_buffer(gpu, &cell_list.cell_count, "cell_count_v2");

    let v2_bg = create_bind_group(gpu, &cl_v2_pipeline,
        &[&pos_buf_v2, &force_buf_v2, &pe_buf_v2, &force_params_cl_buf,
          &cell_start_buf_v2, &cell_count_buf_v2]);

    let t0 = Instant::now();
    dispatch(gpu, &cl_v2_pipeline, &v2_bg, workgroups);
    let forces_v2_sorted = read_back_f64(gpu, &force_buf_v2, n * 3);
    let pe_v2_sorted = read_back_f64(gpu, &pe_buf_v2, n);
    let v2_time = t0.elapsed().as_secs_f64();

    let forces_v2 = cell_list.unsort_array(&forces_v2_sorted, 3);
    let pe_v2_unsorted: Vec<f64> = {
        let mut unsorted = vec![0.0f64; n];
        for (new_idx, &old_idx) in cell_list.sorted_indices.iter().enumerate() {
            unsorted[old_idx] = pe_v2_sorted[new_idx];
        }
        unsorted
    };

    let total_pe_v2: f64 = pe_v2_unsorted.iter().sum();
    println!("  Cell-list v2 (flat): PE = {:.6}, computed in {:.3}ms", total_pe_v2, v2_time * 1000.0);

    // ── V4: Cell-list with f64 cell arrays ──
    let cl_v4_pipeline = gpu.create_pipeline(&cl_v4_src, "celllist_f64_data");

    let pos_buf_v4 = gpu.create_f64_output_buffer(n * 3, "pos_v4");
    let force_buf_v4 = gpu.create_f64_output_buffer(n * 3, "force_v4");
    let pe_buf_v4 = gpu.create_f64_output_buffer(n, "pe_v4");
    upload_f64(gpu, &pos_buf_v4, &sorted_pos);

    // Pack cell_start and cell_count into f64 array: [start0, count0, start1, count1, ...]
    let cell_data_f64: Vec<f64> = cell_list.cell_start.iter().zip(cell_list.cell_count.iter())
        .flat_map(|(&s, &c)| vec![s as f64, c as f64])
        .collect();
    let cell_data_buf = gpu.create_f64_buffer(&cell_data_f64, "cell_data_f64");

    let v4_bg = create_bind_group(gpu, &cl_v4_pipeline,
        &[&pos_buf_v4, &force_buf_v4, &pe_buf_v4, &force_params_cl_buf, &cell_data_buf]);

    dispatch(gpu, &cl_v4_pipeline, &v4_bg, workgroups);
    let forces_v4_sorted = read_back_f64(gpu, &force_buf_v4, n * 3);
    let pe_v4_sorted = read_back_f64(gpu, &pe_buf_v4, n);

    let forces_v4 = cell_list.unsort_array(&forces_v4_sorted, 3);
    let pe_v4_unsorted: Vec<f64> = {
        let mut u = vec![0.0f64; n];
        for (new_idx, &old_idx) in cell_list.sorted_indices.iter().enumerate() {
            u[old_idx] = pe_v4_sorted[new_idx];
        }
        u
    };
    let total_pe_v4: f64 = pe_v4_unsorted.iter().sum();
    // ── V5: Cell-list enumeration WITHOUT cutoff (f64 cell data) ──
    let cl_v5_src = format!("{}\n\n{}", md_math, r#"
@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read_write> forces: array<f64>;
@group(0) @binding(2) var<storage, read_write> pe_buf: array<f64>;
@group(0) @binding(3) var<storage, read> params: array<f64>;
@group(0) @binding(4) var<storage, read> cell_data: array<f64>;

fn pbc_v5(delta: f64, box_size: f64) -> f64 {
    var d = delta;
    let half = box_size / 2.0;
    if (d > half)  { d = d - box_size; }
    if (d < -half) { d = d + box_size; }
    return d;
}

fn wrap_cell_v5(c: i32, n: i32) -> i32 {
    return ((c % n) + n) % n;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n_part = u32(params[0]);
    if (idx >= n_part) { return; }

    let kappa      = params[1];
    let prefactor  = params[2];
    let box_x      = params[4];
    let box_y      = params[5];
    let box_z      = params[6];
    let softening  = params[7];
    let ncx        = i32(params[8]);
    let ncy        = i32(params[9]);
    let ncz        = i32(params[10]);
    let csx        = params[11];
    let csy        = params[12];
    let csz        = params[13];

    let px = positions[idx * 3u];
    let py = positions[idx * 3u + 1u];
    let pz = positions[idx * 3u + 2u];

    let my_cx = i32(px / csx);
    let my_cy = i32(py / csy);
    let my_cz = i32(pz / csz);

    var acc_pe = px - px;

    for (var neigh = 0u; neigh < 27u; neigh = neigh + 1u) {
        let off_x = i32(neigh % 3u) - 1;
        let off_y = i32((neigh / 3u) % 3u) - 1;
        let off_z = i32(neigh / 9u) - 1;

        let nb_cx = wrap_cell_v5(my_cx + off_x, ncx);
        let nb_cy = wrap_cell_v5(my_cy + off_y, ncy);
        let nb_cz = wrap_cell_v5(my_cz + off_z, ncz);
        let cell_linear = u32(nb_cx + nb_cy * ncx + nb_cz * ncx * ncy);

        let start = u32(cell_data[cell_linear * 2u]);
        let cnt   = u32(cell_data[cell_linear * 2u + 1u]);

        for (var k = 0u; k < cnt; k = k + 1u) {
            let j = start + k;
            if (idx == j) { continue; }

            let qx = positions[j * 3u];
            let qy = positions[j * 3u + 1u];
            let qz = positions[j * 3u + 2u];

            let rx = pbc_v5(qx - px, box_x);
            let ry = pbc_v5(qy - py, box_y);
            let rz = pbc_v5(qz - pz, box_z);

            let r_sq = rx * rx + ry * ry + rz * rz;
            // NO CUTOFF CHECK — take all pairs

            let r = sqrt_f64(r_sq + softening);
            let scr = exp_f64(-kappa * r);
            let ir = 1.0 / r;
            acc_pe = acc_pe + 0.5 * prefactor * scr * ir;
        }
    }

    forces[idx * 3u]      = px - px;
    forces[idx * 3u + 1u] = px - px;
    forces[idx * 3u + 2u] = px - px;
    pe_buf[idx] = acc_pe;
}
"#);

    let cl_v5_pipeline = gpu.create_pipeline(&cl_v5_src, "celllist_no_cutoff");
    let pos_buf_v5 = gpu.create_f64_output_buffer(n * 3, "pos_v5");
    let force_buf_v5 = gpu.create_f64_output_buffer(n * 3, "force_v5");
    let pe_buf_v5_gpu = gpu.create_f64_output_buffer(n, "pe_v5");
    upload_f64(gpu, &pos_buf_v5, &sorted_pos);
    let cell_data_buf_v5 = gpu.create_f64_buffer(&cell_data_f64, "cell_data_v5");
    let v5_bg = create_bind_group(gpu, &cl_v5_pipeline,
        &[&pos_buf_v5, &force_buf_v5, &pe_buf_v5_gpu, &force_params_cl_buf, &cell_data_buf_v5]);
    dispatch(gpu, &cl_v5_pipeline, &v5_bg, workgroups);
    let pe_v5 = read_back_f64(gpu, &pe_buf_v5_gpu, n);
    let pe_v5_unsorted: Vec<f64> = {
        let mut u = vec![0.0f64; n];
        for (ni, &oi) in cell_list.sorted_indices.iter().enumerate() { u[oi] = pe_v5[ni]; }
        u
    };
    let total_pe_v5: f64 = pe_v5_unsorted.iter().sum();
    println!("  V5 (cell-list enum, NO cutoff, f64): PE = {:.6}", total_pe_v5);

    // ── V6: Debug trace — record ALL j indices visited by particle 0 ──
    if n <= 500 {
        let cl_v6_src = format!("{}\n\n{}", md_math, r#"
@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read_write> j_trace: array<f64>;
@group(0) @binding(2) var<storage, read_write> pe_buf: array<f64>;
@group(0) @binding(3) var<storage, read> params: array<f64>;
@group(0) @binding(4) var<storage, read> cell_data: array<f64>;

fn wrap_v6(c: i32, n: i32) -> i32 {
    return ((c % n) + n) % n;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_part = u32(params[0]);
    let ncx = i32(params[8]);
    let ncy = i32(params[9]);
    let ncz = i32(params[10]);
    let csx = params[11];
    let csy = params[12];
    let csz = params[13];

    let px = positions[0u];
    let py = positions[1u];
    let pz = positions[2u];

    let my_cx = i32(px / csx);
    let my_cy = i32(py / csy);
    let my_cz = i32(pz / csz);

    var out_idx = 0u;

    // Write header: [my_cx, my_cy, my_cz, n_part]
    j_trace[0u] = f64(my_cx);
    j_trace[1u] = f64(my_cy);
    j_trace[2u] = f64(my_cz);
    j_trace[3u] = f64(n_part);
    out_idx = 4u;

    for (var neigh = 0u; neigh < 27u; neigh = neigh + 1u) {
        let off_x = i32(neigh % 3u) - 1;
        let off_y = i32((neigh / 3u) % 3u) - 1;
        let off_z = i32(neigh / 9u) - 1;

        let nb_cx = wrap_v6(my_cx + off_x, ncx);
        let nb_cy = wrap_v6(my_cy + off_y, ncy);
        let nb_cz = wrap_v6(my_cz + off_z, ncz);
        let cell_linear = u32(nb_cx + nb_cy * ncx + nb_cz * ncx * ncy);

        let start = u32(cell_data[cell_linear * 2u]);
        let cnt   = u32(cell_data[cell_linear * 2u + 1u]);

        // Record: [neigh, off_x, off_y, off_z, cell_linear, start, cnt]
        j_trace[out_idx]      = f64(neigh);
        j_trace[out_idx + 1u] = f64(off_x);
        j_trace[out_idx + 2u] = f64(off_y);
        j_trace[out_idx + 3u] = f64(off_z);
        j_trace[out_idx + 4u] = f64(cell_linear);
        j_trace[out_idx + 5u] = f64(start);
        j_trace[out_idx + 6u] = f64(cnt);
        out_idx = out_idx + 7u;

        for (var k = 0u; k < cnt; k = k + 1u) {
            let j = start + k;
            // Record each j visited
            j_trace[out_idx] = f64(j);
            out_idx = out_idx + 1u;
        }
    }
    // Final marker
    j_trace[out_idx] = px - px - f64(999);
    pe_buf[0u] = f64(out_idx);
}
"#);
        let v6_pipeline = gpu.create_pipeline(&cl_v6_src, "j_trace_debug");
        // Max entries: 4 header + 27*(7 + max_per_cell) + 1 marker
        let trace_size = 4 + 27 * (7 + 100) + 1;
        let pos_buf_v6 = gpu.create_f64_output_buffer(n * 3, "pos_v6");
        let trace_buf = gpu.create_f64_output_buffer(trace_size, "trace");
        let pe_buf_v6 = gpu.create_f64_output_buffer(n, "pe_v6");
        upload_f64(gpu, &pos_buf_v6, &sorted_pos);
        let cell_data_v6 = gpu.create_f64_buffer(&cell_data_f64, "cd_v6");
        let v6_bg = create_bind_group(gpu, &v6_pipeline,
            &[&pos_buf_v6, &trace_buf, &pe_buf_v6, &force_params_cl_buf, &cell_data_v6]);
        dispatch(gpu, &v6_pipeline, &v6_bg, 1);
        let trace_data = read_back_f64(gpu, &trace_buf, trace_size);
        let pe_v6 = read_back_f64(gpu, &pe_buf_v6, 1);

        println!("  V6 j-trace for sorted particle 0:");
        println!("    Cell: ({},{},{}), N={}", trace_data[0] as i32, trace_data[1] as i32, trace_data[2] as i32, trace_data[3] as u32);

        let mut idx = 4usize;
        let mut all_j: Vec<u32> = Vec::new();
        for _neigh in 0..27 {
            if idx >= trace_size { break; }
            let neigh_id = trace_data[idx] as u32;
            let off_x = trace_data[idx+1] as i32;
            let off_y = trace_data[idx+2] as i32;
            let off_z = trace_data[idx+3] as i32;
            let cl = trace_data[idx+4] as u32;
            let start = trace_data[idx+5] as u32;
            let cnt = trace_data[idx+6] as u32;
            idx += 7;

            let mut js: Vec<u32> = Vec::new();
            for _ in 0..cnt {
                if idx >= trace_size { break; }
                let j = trace_data[idx] as u32;
                js.push(j);
                all_j.push(j);
                idx += 1;
            }
            println!("    neigh {:>2}: off=({:+},{:+},{:+}) cell={:>2} start={:>3} cnt={} j={:?}",
                neigh_id, off_x, off_y, off_z, cl, start, cnt, js);
        }

        // Check for duplicates
        all_j.sort();
        let total_before = all_j.len();
        all_j.dedup();
        let total_after = all_j.len();
        println!("    Total j visits: {} unique: {} duplicates: {}",
            total_before, total_after, total_before - total_after);
        if total_before != total_after {
            println!("    ⚠ DUPLICATE J-INDICES DETECTED! This is the bug!");
        }
        println!();
    }

    let net_fx_v4: f64 = (0..n).map(|i| forces_v4[i*3]).sum();
    let net_fy_v4: f64 = (0..n).map(|i| forces_v4[i*3+1]).sum();
    let net_fz_v4: f64 = (0..n).map(|i| forces_v4[i*3+2]).sum();
    let net_f_v4 = (net_fx_v4*net_fx_v4 + net_fy_v4*net_fy_v4 + net_fz_v4*net_fz_v4).sqrt();
    let pe_diff_v4 = (total_pe_ap - total_pe_v4).abs();
    let pe_rel_v4 = if total_pe_ap.abs() > 1e-30 { pe_diff_v4 / total_pe_ap.abs() } else { pe_diff_v4 };
    println!("  Cell-list v4 (f64 cell data): PE = {:.6}, net force = {:.2e}, PE diff = {:.2e} (rel {:.2e})",
        total_pe_v4, net_f_v4, pe_diff_v4, pe_rel_v4);
    if pe_rel_v4 < 1e-6 {
        println!("  ✓ V4 PASS: f64 cell data fixes the bug! array<u32> was the problem.");
    } else {
        println!("  ✗ V4 FAIL: f64 cell data doesn't fix it either.");
    }

    // ── V3: No-cutoff all-pairs with cell-list bindings (sorted positions) ──
    let cl_v3_pipeline = gpu.create_pipeline(&cl_v3_src, "no_cutoff_test");
    let pos_buf_v3 = gpu.create_f64_output_buffer(n * 3, "pos_v3");
    let force_buf_v3 = gpu.create_f64_output_buffer(n * 3, "force_v3");
    let pe_buf_v3_gpu = gpu.create_f64_output_buffer(n, "pe_v3");
    upload_f64(gpu, &pos_buf_v3, &sorted_pos);
    let cs_v3 = create_u32_buffer(gpu, &cell_list.cell_start, "cs_v3");
    let cc_v3 = create_u32_buffer(gpu, &cell_list.cell_count, "cc_v3");
    let v3_bg = create_bind_group(gpu, &cl_v3_pipeline,
        &[&pos_buf_v3, &force_buf_v3, &pe_buf_v3_gpu, &force_params_cl_buf, &cs_v3, &cc_v3]);
    dispatch(gpu, &cl_v3_pipeline, &v3_bg, workgroups);
    let pe_v3_sorted = read_back_f64(gpu, &pe_buf_v3_gpu, n);
    let pe_v3_unsorted: Vec<f64> = {
        let mut u = vec![0.0f64; n];
        for (new_idx, &old_idx) in cell_list.sorted_indices.iter().enumerate() {
            u[old_idx] = pe_v3_sorted[new_idx];
        }
        u
    };
    let total_pe_v3: f64 = pe_v3_unsorted.iter().sum();
    println!("  V3 (no cutoff, AP loop, sorted): PE = {:.6}", total_pe_v3);

    // Also compute AP PE without cutoff on ORIGINAL positions for comparison
    let pos_buf_v3_orig = gpu.create_f64_output_buffer(n * 3, "pos_v3_orig");
    let force_buf_v3_orig = gpu.create_f64_output_buffer(n * 3, "force_v3_orig");
    let pe_buf_v3_orig = gpu.create_f64_output_buffer(n, "pe_v3_orig");
    upload_f64(gpu, &pos_buf_v3_orig, &positions);
    let cs_v3_orig = create_u32_buffer(gpu, &cell_list.cell_start, "cs_v3_orig");
    let cc_v3_orig = create_u32_buffer(gpu, &cell_list.cell_count, "cc_v3_orig");
    let v3_orig_bg = create_bind_group(gpu, &cl_v3_pipeline,
        &[&pos_buf_v3_orig, &force_buf_v3_orig, &pe_buf_v3_orig, &force_params_cl_buf, &cs_v3_orig, &cc_v3_orig]);
    dispatch(gpu, &cl_v3_pipeline, &v3_orig_bg, workgroups);
    let pe_v3_orig_data = read_back_f64(gpu, &pe_buf_v3_orig, n);
    let total_pe_v3_orig: f64 = pe_v3_orig_data.iter().sum();
    println!("  V3 (no cutoff, AP loop, original): PE = {:.6}", total_pe_v3_orig);
    println!("  AP (with cutoff): PE = {:.6}", total_pe_ap);

    // ── Compare ──
    let pe_diff_v2 = (total_pe_ap - total_pe_v2).abs();
    let pe_rel_v2 = if total_pe_ap.abs() > 1e-30 { pe_diff_v2 / total_pe_ap.abs() } else { pe_diff_v2 };
    let net_fx_v2: f64 = (0..n).map(|i| forces_v2[i * 3]).sum();
    let net_fy_v2: f64 = (0..n).map(|i| forces_v2[i * 3 + 1]).sum();
    let net_fz_v2: f64 = (0..n).map(|i| forces_v2[i * 3 + 2]).sum();
    let net_f_v2 = (net_fx_v2*net_fx_v2 + net_fy_v2*net_fy_v2 + net_fz_v2*net_fz_v2).sqrt();
    println!("  Cell-list v2: PE diff from AP = {:.2e} (relative {:.2e}), net force = {:.2e}",
        pe_diff_v2, pe_rel_v2, net_f_v2);
    if pe_rel_v2 < 1e-6 && net_f_v2 < 1e-4 {
        println!("  ✓ V2 PASS: flat loop cell-list matches all-pairs!");
    } else {
        println!("  ✗ V2 FAIL: flat loop still broken");
    }
    println!();

    let pe_diff = (total_pe_ap - total_pe_cl).abs();
    let pe_rel = if total_pe_ap.abs() > 1e-30 { pe_diff / total_pe_ap.abs() } else { pe_diff };
    println!();
    println!("  ── Comparison ──");
    println!("  Total PE: all-pairs = {:.10}, cell-list = {:.10}", total_pe_ap, total_pe_cl);
    println!("  PE difference: {:.2e} (relative: {:.2e})", pe_diff, pe_rel);

    // Per-particle force comparison
    let mut max_force_diff = 0.0f64;
    let mut max_force_particle = 0usize;
    let mut rms_diff = 0.0f64;
    let mut total_force_mag_ap = 0.0f64;
    let mut n_mismatched = 0usize;
    let threshold = 1e-8; // relative difference threshold

    for i in 0..n {
        let fx_ap = forces_ap[i * 3];
        let fy_ap = forces_ap[i * 3 + 1];
        let fz_ap = forces_ap[i * 3 + 2];
        let fx_cl = forces_cl[i * 3];
        let fy_cl = forces_cl[i * 3 + 1];
        let fz_cl = forces_cl[i * 3 + 2];

        let mag_ap = (fx_ap * fx_ap + fy_ap * fy_ap + fz_ap * fz_ap).sqrt();
        let diff = ((fx_ap - fx_cl).powi(2) + (fy_ap - fy_cl).powi(2) + (fz_ap - fz_cl).powi(2)).sqrt();

        total_force_mag_ap += mag_ap;
        rms_diff += diff * diff;

        let rel = if mag_ap > 1e-30 { diff / mag_ap } else { diff };
        if rel > threshold {
            n_mismatched += 1;
        }
        if diff > max_force_diff {
            max_force_diff = diff;
            max_force_particle = i;
        }
    }

    rms_diff = (rms_diff / n as f64).sqrt();
    let avg_force = total_force_mag_ap / n as f64;
    let rms_rel = if avg_force > 1e-30 { rms_diff / avg_force } else { rms_diff };
    let max_rel = if avg_force > 1e-30 { max_force_diff / avg_force } else { max_force_diff };

    println!("  Force RMS diff: {:.4e} (relative: {:.4e})", rms_diff, rms_rel);
    println!("  Force max diff: {:.4e} at particle {} (relative: {:.4e})",
        max_force_diff, max_force_particle, max_rel);
    println!("  Particles with |ΔF/F| > {:.0e}: {}/{}", threshold, n_mismatched, n);
    println!("  Average |F|: {:.6}", avg_force);

    // Net force check (should be ~0 for both kernels if Newton's 3rd law holds)
    let net_fx_ap: f64 = (0..n).map(|i| forces_ap[i * 3]).sum();
    let net_fy_ap: f64 = (0..n).map(|i| forces_ap[i * 3 + 1]).sum();
    let net_fz_ap: f64 = (0..n).map(|i| forces_ap[i * 3 + 2]).sum();
    let net_f_ap = (net_fx_ap * net_fx_ap + net_fy_ap * net_fy_ap + net_fz_ap * net_fz_ap).sqrt();

    let net_fx_cl: f64 = (0..n).map(|i| forces_cl[i * 3]).sum();
    let net_fy_cl: f64 = (0..n).map(|i| forces_cl[i * 3 + 1]).sum();
    let net_fz_cl: f64 = (0..n).map(|i| forces_cl[i * 3 + 2]).sum();
    let net_f_cl = (net_fx_cl * net_fx_cl + net_fy_cl * net_fy_cl + net_fz_cl * net_fz_cl).sqrt();

    println!("  Net force (all-pairs): ({:.4e}, {:.4e}, {:.4e}) |F|={:.4e}",
        net_fx_ap, net_fy_ap, net_fz_ap, net_f_ap);
    println!("  Net force (cell-list): ({:.4e}, {:.4e}, {:.4e}) |F|={:.4e}",
        net_fx_cl, net_fy_cl, net_fz_cl, net_f_cl);

    // Summary
    let pe_ok = pe_rel < 1e-6;
    let force_ok = rms_rel < 1e-6;
    let net_ok = net_f_cl < 1e-4 * avg_force * n as f64;

    if pe_ok && force_ok && net_ok {
        println!("  ✓ PASS: cell-list matches all-pairs at N={}", n);
    } else {
        println!("  ✗ FAIL at N={}:", n);
        if !pe_ok { println!("    - PE mismatch: {:.2e} relative", pe_rel); }
        if !force_ok { println!("    - Force mismatch: {:.2e} RMS relative", rms_rel); }
        if !net_ok { println!("    - Net force too large: {:.2e}", net_f_cl); }
    }

    // Show first few mismatched particles for debugging
    if n_mismatched > 0 && n_mismatched <= 20 {
        println!();
        println!("  Mismatched particles (first up to 10):");
        let mut shown = 0;
        for i in 0..n {
            if shown >= 10 { break; }
            let fx_ap = forces_ap[i * 3];
            let fy_ap = forces_ap[i * 3 + 1];
            let fz_ap = forces_ap[i * 3 + 2];
            let fx_cl = forces_cl[i * 3];
            let fy_cl = forces_cl[i * 3 + 1];
            let fz_cl = forces_cl[i * 3 + 2];
            let mag_ap = (fx_ap * fx_ap + fy_ap * fy_ap + fz_ap * fz_ap).sqrt();
            let diff = ((fx_ap - fx_cl).powi(2) + (fy_ap - fy_cl).powi(2) + (fz_ap - fz_cl).powi(2)).sqrt();
            let rel = if mag_ap > 1e-30 { diff / mag_ap } else { diff };
            if rel > threshold {
                println!("    particle {:>5}: AP=({:+.6e},{:+.6e},{:+.6e}) CL=({:+.6e},{:+.6e},{:+.6e}) Δ={:.2e}",
                    i, fx_ap, fy_ap, fz_ap, fx_cl, fy_cl, fz_cl, diff);
                shown += 1;
            }
        }
    } else if n_mismatched > 20 {
        // Show worst 5
        println!();
        println!("  Worst 5 mismatches:");
        let mut diffs: Vec<(usize, f64)> = (0..n).map(|i| {
            let diff = ((forces_ap[i*3] - forces_cl[i*3]).powi(2)
                + (forces_ap[i*3+1] - forces_cl[i*3+1]).powi(2)
                + (forces_ap[i*3+2] - forces_cl[i*3+2]).powi(2)).sqrt();
            (i, diff)
        }).collect();
        diffs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for &(i, diff) in diffs.iter().take(5) {
            let pos = &positions[i*3..i*3+3];
            let cx = (pos[0] / cell_size) as usize;
            let cy = (pos[1] / cell_size) as usize;
            let cz = (pos[2] / cell_size) as usize;
            println!("    particle {:>5}: pos=({:.3},{:.3},{:.3}) cell=({},{},{}) Δ|F|={:.4e}  AP=({:+.4e},{:+.4e},{:+.4e})  CL=({:+.4e},{:+.4e},{:+.4e})",
                i, pos[0], pos[1], pos[2], cx, cy, cz, diff,
                forces_ap[i*3], forces_ap[i*3+1], forces_ap[i*3+2],
                forces_cl[i*3], forces_cl[i*3+1], forces_cl[i*3+2]);
        }
    }

    println!();
}

/// Hybrid test: uses cell-list BINDINGS but all-pairs LOOP
/// This isolates whether the bug is in cell enumeration or buffer infrastructure
async fn test_hybrid(gpu: &GpuF64, n: usize) {
    let kappa = 2.0;
    let rc = 6.5;
    let prefactor = 1.0;
    let box_side = (n as f64).cbrt() * (4.0 * std::f64::consts::PI / 3.0_f64).cbrt();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  HYBRID TEST: N = {} — all-pairs loop, cell-list params", n);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let (positions, n_actual) = init_fcc_lattice(n, box_side);
    let n = n_actual.min(n);
    let cell_list = CellList::build(&positions, n, box_side, rc);
    let sorted_pos = cell_list.sort_array(&positions, 3);

    let md_math = patch_math_f64_preamble(&ShaderTemplate::math_f64_preamble());

    // Hybrid kernel: same bindings as cell-list, but loops over all j
    let hybrid_shader_src = format!("{}\n\n{}", md_math, r#"
@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read_write> forces: array<f64>;
@group(0) @binding(2) var<storage, read_write> pe_buf: array<f64>;
@group(0) @binding(3) var<storage, read> params: array<f64>;
@group(0) @binding(4) var<storage, read> cell_start: array<u32>;
@group(0) @binding(5) var<storage, read> cell_count: array<u32>;

fn pbc_delta_h(delta: f64, box_size: f64) -> f64 {
    return delta - box_size * round_f64(delta / box_size);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(params[0]);
    if (i >= n) { return; }

    let kappa     = params[1];
    let prefactor = params[2];
    let cutoff_sq = params[3];
    let box_x     = params[4];
    let box_y     = params[5];
    let box_z     = params[6];
    let eps       = params[7];

    let xi = positions[i * 3u];
    let yi = positions[i * 3u + 1u];
    let zi = positions[i * 3u + 2u];

    var fx = xi - xi;
    var fy = xi - xi;
    var fz = xi - xi;
    var pe = xi - xi;

    // Touch cell_start/cell_count to keep bindings alive (compiler optimization workaround)
    let _cs0 = cell_start[0u];
    let _cc0 = cell_count[0u];

    // ALL-PAIRS loop (ignoring cell_start/cell_count)
    for (var j = 0u; j < n; j = j + 1u) {
        if (i == j) { continue; }
        let xj = positions[j * 3u];
        let yj = positions[j * 3u + 1u];
        let zj = positions[j * 3u + 2u];

        var ddx = pbc_delta_h(xj - xi, box_x);
        var ddy = pbc_delta_h(yj - yi, box_y);
        var ddz = pbc_delta_h(zj - zi, box_z);

        let r_sq = ddx * ddx + ddy * ddy + ddz * ddz;
        if (r_sq > cutoff_sq) { continue; }

        let r = sqrt_f64(r_sq + eps);
        let screening = exp_f64(-kappa * r);
        let force_mag = prefactor * screening * (1.0 + kappa * r) / r_sq;
        let inv_r = 1.0 / r;

        fx = fx - force_mag * ddx * inv_r;
        fy = fy - force_mag * ddy * inv_r;
        fz = fz - force_mag * ddz * inv_r;
        pe = pe + 0.5 * prefactor * screening * inv_r;
    }

    forces[i * 3u]      = fx;
    forces[i * 3u + 1u] = fy;
    forces[i * 3u + 2u] = fz;
    pe_buf[i] = pe;
}
"#);

    let hybrid_pipeline = gpu.create_pipeline(&hybrid_shader_src, "hybrid_ap_cl_bindings");

    // Create buffers with cell-list layout (SORTED positions)
    let pos_buf = gpu.create_f64_output_buffer(n * 3, "pos_hybrid");
    let force_buf = gpu.create_f64_output_buffer(n * 3, "force_hybrid");
    let pe_buf_gpu = gpu.create_f64_output_buffer(n, "pe_hybrid");
    upload_f64(gpu, &pos_buf, &sorted_pos);

    let force_params_cl = vec![
        n as f64, kappa, prefactor, rc * rc,
        box_side, box_side, box_side, 0.0,
        cell_list.n_cells[0] as f64, cell_list.n_cells[1] as f64, cell_list.n_cells[2] as f64,
        cell_list.cell_size[0], cell_list.cell_size[1], cell_list.cell_size[2],
        cell_list.n_cells_total as f64, 0.0,
    ];
    let params_buf = gpu.create_f64_buffer(&force_params_cl, "params_hybrid");
    let cs_buf = create_u32_buffer(gpu, &cell_list.cell_start, "cs_hybrid");
    let cc_buf = create_u32_buffer(gpu, &cell_list.cell_count, "cc_hybrid");

    let bg = create_bind_group(gpu, &hybrid_pipeline,
        &[&pos_buf, &force_buf, &pe_buf_gpu, &params_buf, &cs_buf, &cc_buf]);
    let workgroups = ((n + 63) / 64) as u32;

    dispatch(gpu, &hybrid_pipeline, &bg, workgroups);
    let forces_h = read_back_f64(gpu, &force_buf, n * 3);
    let pe_h = read_back_f64(gpu, &pe_buf_gpu, n);

    // Unsort back to original order
    let forces_unsorted = cell_list.unsort_array(&forces_h, 3);
    let pe_unsorted: Vec<f64> = {
        let mut u = vec![0.0f64; n];
        for (new_idx, &old_idx) in cell_list.sorted_indices.iter().enumerate() {
            u[old_idx] = pe_h[new_idx];
        }
        u
    };

    let total_pe: f64 = pe_unsorted.iter().sum();
    let net_fx: f64 = (0..n).map(|i| forces_unsorted[i*3]).sum();
    let net_fy: f64 = (0..n).map(|i| forces_unsorted[i*3+1]).sum();
    let net_fz: f64 = (0..n).map(|i| forces_unsorted[i*3+2]).sum();
    let net_f = (net_fx*net_fx + net_fy*net_fy + net_fz*net_fz).sqrt();

    println!("  Hybrid (all-pairs loop, cell-list bindings, sorted positions):");
    println!("    PE = {:.10}", total_pe);
    println!("    Net force: ({:.4e}, {:.4e}, {:.4e}) |F|={:.4e}", net_fx, net_fy, net_fz, net_f);

    if net_f < 1e-6 {
        println!("  ✓ HYBRID PASS: all-pairs on sorted positions is correct");
        println!("    → Bug is in the CELL ENUMERATION LOOP, not in params/bindings/sorting");
    } else {
        println!("  ✗ HYBRID FAIL: even all-pairs gives wrong results with sorted positions!");
        println!("    → Bug is in BUFFER INFRASTRUCTURE (params, sorting, or GPU buffer layout)");
    }
    println!();
}

#[tokio::main]
async fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Cell-List Force Diagnostic                                 ║");
    println!("║  Experiment 002: All-Pairs vs Cell-List comparison          ║");
    println!("║  Same positions, same params → forces must match            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let gpu = GpuF64::new().await.expect("Failed to init GPU");
    gpu.print_info();
    println!();

    // Test at multiple N values to see how the bug scales
    // N=108 (3³×4 FCC): tiny, 3 cells/dim → cell-list degenerates to all-pairs
    // N=500: small, 1 cell/dim (forced to 3) → effectively all-pairs
    // N=2048: medium, 3 cells/dim → first real cell-list test
    // N=4000: medium, 4 cells/dim → 4×4×4 cells
    // N=8788: large, 5 cells/dim → matches the broken N=10k threshold
    // N=10976: actual N=10k FCC → the exact failure case

    let sizes = [108, 500, 2048, 4000, 8788, 10976];

    for &n in &sizes {
        compare_forces(&gpu, n).await;
    }

    // ── Phase 1b: Verify cell_start/cell_count arrays for N=108 ──
    {
        let n_test = 108;
        let box_s = (n_test as f64).cbrt() * (4.0 * std::f64::consts::PI / 3.0_f64).cbrt();
        let (pos, n_a) = init_fcc_lattice(n_test, box_s);
        let n_test = n_a.min(n_test);
        let cl = CellList::build(&pos, n_test, box_s, 6.5);

        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║  Phase 1b: Cell list integrity check (N=108)                ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!("  cells/dim = {}, total cells = {}", cl.n_cells[0], cl.n_cells_total);
        let total_count: u32 = cl.cell_count.iter().sum();
        println!("  sum(cell_count) = {} (should be {})", total_count, n_test);

        // Check sorted_indices is a valid permutation
        let mut seen = vec![false; n_test];
        for &idx in &cl.sorted_indices {
            assert!(idx < n_test, "sorted_indices out of range: {}", idx);
            assert!(!seen[idx], "duplicate in sorted_indices: {}", idx);
            seen[idx] = true;
        }
        println!("  sorted_indices: valid permutation of 0..{} ✓", n_test);

        // Print cell contents
        println!("  Cell contents:");
        for c in 0..cl.n_cells_total {
            let start = cl.cell_start[c] as usize;
            let count = cl.cell_count[c] as usize;
            if count > 0 {
                let cx = c % cl.n_cells[0];
                let cy = (c / cl.n_cells[0]) % cl.n_cells[1];
                let cz = c / (cl.n_cells[0] * cl.n_cells[1]);
                println!("    cell ({},{},{}) = idx {}: start={}, count={}", cx, cy, cz, c, start, count);
            }
        }

        // For particle 0 in sorted order, check which cells would be searched
        let sp = cl.sort_array(&pos, 3);
        let xi = sp[0]; let yi = sp[1]; let zi = sp[2];
        let cell_sx = cl.cell_size[0];
        let ci_x = (xi / cell_sx) as i32;
        let ci_y = (yi / cl.cell_size[1]) as i32;
        let ci_z = (zi / cl.cell_size[2]) as i32;
        let nx = cl.n_cells[0] as i32;
        println!("  Sorted particle 0: pos=({:.4},{:.4},{:.4}), cell=({},{},{})",
            xi, yi, zi, ci_x, ci_y, ci_z);

        // Simulate the 3x3x3 search
        let mut total_checked = 0usize;
        let mut cells_visited = Vec::new();
        for dz in -1..=1i32 {
            for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    let wx = ((ci_x + dx) % nx + nx) % nx;
                    let wy = ((ci_y + dy) % nx + nx) % nx;
                    let wz = ((ci_z + dz) % nx + nx) % nx;
                    let c_idx = (wx + wy * nx + wz * nx * nx) as usize;
                    let count = cl.cell_count[c_idx] as usize;
                    total_checked += count;
                    cells_visited.push((dx, dy, dz, wx, wy, wz, c_idx, count));
                }
            }
        }

        // Check for duplicate cells
        let mut cell_ids: Vec<usize> = cells_visited.iter().map(|v| v.6).collect();
        let n_unique_before = cell_ids.len();
        cell_ids.sort();
        cell_ids.dedup();
        let n_unique = cell_ids.len();
        println!("  27 neighbor offsets map to {} unique cells (expect 27 for nx=3)", n_unique);
        if n_unique < n_unique_before {
            println!("  ⚠ DUPLICATE CELLS DETECTED! {} visits but {} unique", n_unique_before, n_unique);
            for v in &cells_visited {
                println!("    offset ({:+},{:+},{:+}) → wrapped ({},{},{}) = cell {}, count={}",
                    v.0, v.1, v.2, v.3, v.4, v.5, v.6, v.7);
            }
        }
        println!("  Total particles checked by sorted particle 0: {} (expect {} = N-1 for all cells)",
            total_checked, n_test - 1);
        println!();
    }

    // ── Phase 1c: GPU pair-counting diagnostic ──
    {
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║  Phase 1c: GPU pair count comparison (N=108)                ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!();

        let n_test = 108usize;
        let rc = 6.5;
        let box_s = (n_test as f64).cbrt() * (4.0 * std::f64::consts::PI / 3.0_f64).cbrt();
        let (pos, n_a) = init_fcc_lattice(n_test, box_s);
        let n_test = n_a.min(n_test);
        let cl = CellList::build(&pos, n_test, box_s, rc);
        let sorted_pos = cl.sort_array(&pos, 3);

        let md_math = patch_math_f64_preamble(&ShaderTemplate::math_f64_preamble());

        // Kernel that counts pairs and outputs per-particle cell assignments & debug info
        let count_shader_src = format!("{}\n\n{}", md_math, r#"
@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read_write> debug_out: array<f64>;
@group(0) @binding(2) var<storage, read> params: array<f64>;
@group(0) @binding(3) var<storage, read> cell_start_buf: array<u32>;
@group(0) @binding(4) var<storage, read> cell_count_buf: array<u32>;

fn cell_idx_d(cx: i32, cy: i32, cz: i32, nx: i32, ny: i32, nz: i32) -> u32 {
    let wx = ((cx % nx) + nx) % nx;
    let wy = ((cy % ny) + ny) % ny;
    let wz = ((cz % nz) + nz) % nz;
    return u32(wx + wy * nx + wz * nx * ny);
}

fn pbc_delta_d(delta: f64, box_size: f64) -> f64 {
    return delta - box_size * round_f64(delta / box_size);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(params[0]);
    if (i >= n) { return; }

    let cutoff_sq = params[3];
    let box_x     = params[4];
    let box_y     = params[5];
    let box_z     = params[6];
    let nx        = i32(params[8]);
    let ny        = i32(params[9]);
    let nz        = i32(params[10]);
    let cell_sx   = params[11];
    let cell_sy   = params[12];
    let cell_sz   = params[13];

    let xi = positions[i * 3u];
    let yi = positions[i * 3u + 1u];
    let zi = positions[i * 3u + 2u];

    let ci_x = i32(xi / cell_sx);
    let ci_y = i32(yi / cell_sy);
    let ci_z = i32(zi / cell_sz);

    var pair_count = xi - xi;  // 0.0 as f64
    var total_checked = xi - xi;
    var cells_visited = xi - xi;

    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let c_idx = cell_idx_d(ci_x + dx, ci_y + dy, ci_z + dz, nx, ny, nz);
                let start = cell_start_buf[c_idx];
                let count = cell_count_buf[c_idx];
                cells_visited = cells_visited + 1.0;
                total_checked = total_checked + f64(count);

                for (var jj = 0u; jj < count; jj = jj + 1u) {
                    let j = start + jj;
                    if (i == j) { continue; }

                    let xj = positions[j * 3u];
                    let yj = positions[j * 3u + 1u];
                    let zj = positions[j * 3u + 2u];

                    var ddx = pbc_delta_d(xj - xi, box_x);
                    var ddy = pbc_delta_d(yj - yi, box_y);
                    var ddz = pbc_delta_d(zj - zi, box_z);

                    let r_sq = ddx * ddx + ddy * ddy + ddz * ddz;
                    if (r_sq > cutoff_sq) { continue; }
                    pair_count = pair_count + 1.0;
                }
            }
        }
    }

    // Output: [cell_x, cell_y, cell_z, cells_visited, total_checked, pair_count, nx_read, cell_sx_read]
    debug_out[i * 8u]      = f64(ci_x);
    debug_out[i * 8u + 1u] = f64(ci_y);
    debug_out[i * 8u + 2u] = f64(ci_z);
    debug_out[i * 8u + 3u] = cells_visited;
    debug_out[i * 8u + 4u] = total_checked;
    debug_out[i * 8u + 5u] = pair_count;
    debug_out[i * 8u + 6u] = f64(nx);
    debug_out[i * 8u + 7u] = cell_sx;
}
"#);

        let count_pipeline = gpu.create_pipeline(&count_shader_src, "pair_count_diag");

        let pos_buf = gpu.create_f64_output_buffer(n_test * 3, "pos_count");
        let debug_buf = gpu.create_f64_output_buffer(n_test * 8, "debug_count");
        upload_f64(&gpu, &pos_buf, &sorted_pos);

        let params_cl = vec![
            n_test as f64, 2.0, 1.0, rc * rc,
            box_s, box_s, box_s, 0.0,
            cl.n_cells[0] as f64, cl.n_cells[1] as f64, cl.n_cells[2] as f64,
            cl.cell_size[0], cl.cell_size[1], cl.cell_size[2],
            cl.n_cells_total as f64, 0.0,
        ];
        let params_buf = gpu.create_f64_buffer(&params_cl, "params_count");
        let cs_buf = create_u32_buffer(&gpu, &cl.cell_start, "cs_count");
        let cc_buf = create_u32_buffer(&gpu, &cl.cell_count, "cc_count");

        let bg = create_bind_group(&gpu, &count_pipeline,
            &[&pos_buf, &debug_buf, &params_buf, &cs_buf, &cc_buf]);
        let wg = ((n_test + 63) / 64) as u32;

        dispatch(&gpu, &count_pipeline, &bg, wg);
        let debug_data = read_back_f64(&gpu, &debug_buf, n_test * 8);

        println!("  GPU pair-count results (first 10 sorted particles):");
        println!("  {:>4} {:>8} {:>8} {:>8} {:>6} {:>8} {:>6} {:>4} {:>8}",
            "idx", "pos_x", "pos_y", "pos_z", "cell", "checked", "pairs", "nx", "cell_sx");
        for i in 0..10.min(n_test) {
            let cx = debug_data[i*8] as i32;
            let cy = debug_data[i*8+1] as i32;
            let cz = debug_data[i*8+2] as i32;
            let cells = debug_data[i*8+3] as i32;
            let checked = debug_data[i*8+4] as i32;
            let pairs = debug_data[i*8+5] as i32;
            let nx_r = debug_data[i*8+6] as i32;
            let csx = debug_data[i*8+7];
            println!("  {:>4} {:>8.3} {:>8.3} {:>8.3} ({},{},{}) {:>8} {:>6} {:>4} {:>8.4}",
                i, sorted_pos[i*3], sorted_pos[i*3+1], sorted_pos[i*3+2],
                cx, cy, cz, checked, pairs, nx_r, csx);
        }

        // Check: does GPU read nx correctly?
        let nx_gpu = debug_data[6] as i32;
        let csx_gpu = debug_data[7];
        println!();
        println!("  GPU reads: nx={} (expect {}), cell_sx={:.4} (expect {:.4})",
            nx_gpu, cl.n_cells[0], csx_gpu, cl.cell_size[0]);

        // Compare pair counts
        let total_pairs_cl: f64 = (0..n_test).map(|i| debug_data[i*8+5]).sum();
        println!("  Total pairs found (cell-list): {:.0}", total_pairs_cl);
        println!();

        // Also count pairs via all-pairs for comparison
        let ap_count_src = format!("{}\n\n{}", md_math, r#"
@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read_write> counts: array<f64>;
@group(0) @binding(2) var<storage, read> params: array<f64>;

fn pbc_delta_ac(delta: f64, box_size: f64) -> f64 {
    return delta - box_size * round_f64(delta / box_size);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(params[0]);
    if (i >= n) { return; }

    let cutoff_sq = params[3];
    let box_x = params[4];
    let box_y = params[5];
    let box_z = params[6];

    let xi = positions[i * 3u];
    let yi = positions[i * 3u + 1u];
    let zi = positions[i * 3u + 2u];

    var pc = xi - xi;  // 0.0
    var pe = xi - xi;

    for (var j = 0u; j < n; j = j + 1u) {
        if (i == j) { continue; }
        let xj = positions[j * 3u];
        let yj = positions[j * 3u + 1u];
        let zj = positions[j * 3u + 2u];
        var ddx = pbc_delta_ac(xj - xi, box_x);
        var ddy = pbc_delta_ac(yj - yi, box_y);
        var ddz = pbc_delta_ac(zj - zi, box_z);
        let r_sq = ddx * ddx + ddy * ddy + ddz * ddz;
        if (r_sq > cutoff_sq) { continue; }
        pc = pc + 1.0;
        let r = sqrt_f64(r_sq);
        let kappa = params[1];
        let prefactor = params[2];
        pe = pe + 0.5 * prefactor * exp_f64(-kappa * r) / r;
    }

    counts[i * 2u] = pc;
    counts[i * 2u + 1u] = pe;
}
"#);
        let ap_count_pipeline = gpu.create_pipeline(&ap_count_src, "ap_count_diag");
        let ap_pos_buf = gpu.create_f64_output_buffer(n_test * 3, "ap_pos_count");
        let ap_counts_buf = gpu.create_f64_output_buffer(n_test * 2, "ap_counts");
        upload_f64(&gpu, &ap_pos_buf, &sorted_pos);  // SAME sorted positions!
        let ap_params = vec![n_test as f64, 2.0, 1.0, rc * rc, box_s, box_s, box_s, 0.0];
        let ap_params_buf = gpu.create_f64_buffer(&ap_params, "ap_params_count");
        let ap_bg = create_bind_group(&gpu, &ap_count_pipeline,
            &[&ap_pos_buf, &ap_counts_buf, &ap_params_buf]);
        dispatch(&gpu, &ap_count_pipeline, &ap_bg, ((n_test + 63) / 64) as u32);
        let ap_count_data = read_back_f64(&gpu, &ap_counts_buf, n_test * 2);

        println!("  All-pairs pair counts (same sorted positions, first 10):");
        let mut total_ap_pairs = 0.0f64;
        let mut total_ap_pe = 0.0f64;
        for i in 0..10.min(n_test) {
            let ap_pc = ap_count_data[i*2];
            let ap_pe = ap_count_data[i*2+1];
            let cl_pc = debug_data[i*8+5];
            total_ap_pairs += ap_pc;
            total_ap_pe += ap_pe;
            let match_str = if (ap_pc - cl_pc).abs() < 0.5 { "✓" } else { "✗ DIFF" };
            println!("    particle {:>3}: AP pairs={:.0}, CL pairs={:.0} {}  AP_PE={:.6}, CL_PE=?",
                i, ap_pc, cl_pc, match_str, ap_pe);
        }
        for i in 10..n_test {
            total_ap_pairs += ap_count_data[i*2];
            total_ap_pe += ap_count_data[i*2+1];
        }
        println!("  Total AP pairs: {:.0}, Total CL pairs: {:.0}", total_ap_pairs, total_pairs_cl);
        println!("  Total AP PE: {:.6}", total_ap_pe);
        println!();

        // Now verify u32 buffer reads: GPU shader that reads cell_start/cell_count
        // and outputs them to a debug buffer for comparison
        let verify_u32_src = format!("{}\n\n{}", md_math, r#"
@group(0) @binding(0) var<storage, read> cell_start_in: array<u32>;
@group(0) @binding(1) var<storage, read> cell_count_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f64>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Read first 27 cell_start and cell_count values, output as f64
    for (var c = 0u; c < 27u; c = c + 1u) {
        out[c * 2u]      = f64(cell_start_in[c]);
        out[c * 2u + 1u] = f64(cell_count_in[c]);
    }
}
"#);
        let verify_pipeline = gpu.create_pipeline(&verify_u32_src, "verify_u32");
        let out_buf = gpu.create_f64_output_buffer(27 * 2, "verify_out");
        let verify_bg = create_bind_group(&gpu, &verify_pipeline,
            &[&cs_buf, &cc_buf, &out_buf]);
        dispatch(&gpu, &verify_pipeline, &verify_bg, 1);
        let verify_data = read_back_f64(&gpu, &out_buf, 27 * 2);

        println!("  u32 buffer verification (GPU reads vs CPU values):");
        println!("  {:>4} {:>10} {:>10} {:>10} {:>10} {:>5}",
            "cell", "GPU_start", "CPU_start", "GPU_count", "CPU_count", "match");
        let mut all_match = true;
        for c in 0..27 {
            let gpu_start = verify_data[c * 2] as u32;
            let gpu_count = verify_data[c * 2 + 1] as u32;
            let cpu_start = cl.cell_start[c];
            let cpu_count = cl.cell_count[c];
            let ok = gpu_start == cpu_start && gpu_count == cpu_count;
            if !ok { all_match = false; }
            println!("  {:>4} {:>10} {:>10} {:>10} {:>10} {:>5}",
                c, gpu_start, cpu_start, gpu_count, cpu_count,
                if ok { "✓" } else { "✗ MISMATCH" });
        }
        if all_match {
            println!("  ✓ All u32 buffer reads match CPU values");
        } else {
            println!("  ✗ u32 BUFFER READ MISMATCH — this is the bug source!");
        }
        println!();
    }

    // ── Phase 2: Hybrid test (all-pairs loop, cell-list bindings) ──
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Phase 2: Hybrid Isolation Test                             ║");
    println!("║  All-pairs loop + cell-list bindings → isolate bug location ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    for &n in &[500, 2048, 10976] {
        test_hybrid(&gpu, n).await;
    }

    println!("════════════════════════════════════════════════════════════════");
    println!("  Diagnostic complete. See experiment 002 journal for analysis.");
    println!("════════════════════════════════════════════════════════════════");
}
