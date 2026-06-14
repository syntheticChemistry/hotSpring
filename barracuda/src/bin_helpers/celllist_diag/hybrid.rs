// SPDX-License-Identifier: AGPL-3.0-or-later

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::md::diag::net_force;
use hotspring_barracuda::md::shaders::patch_math_f64_preamble;
use hotspring_barracuda::md::simulation::{CellList, init_fcc_lattice};
use hotspring_barracuda::validation::ValidationHarness;

use barracuda::shaders::precision::ShaderTemplate;

/// Hybrid test: uses cell-list BINDINGS but all-pairs LOOP
pub fn test_hybrid(gpu: &GpuF64, n: usize, harness: &mut ValidationHarness) {
    let kappa = 2.0;
    let rc = 6.5;
    let prefactor = 1.0;
    let box_side = (n as f64).cbrt() * (4.0 * std::f64::consts::PI / 3.0_f64).cbrt();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  HYBRID TEST: N = {n} — all-pairs loop, cell-list params");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let (positions, n_actual) = init_fcc_lattice(n, box_side);
    let n = n_actual.min(n);
    let cell_list = CellList::build(&positions, n, box_side, rc);
    let sorted_pos = cell_list.sort_array(&positions, 3);

    let md_math = patch_math_f64_preamble(&ShaderTemplate::math_f64_preamble());

    // Hybrid kernel: same bindings as cell-list, but loops over all j
    let hybrid_shader_src = format!(
        "{}\n\n{}",
        md_math,
        include_str!("../../bin/shaders/celllist_diag/hybrid_allpairs_f64.wgsl"),
    );

    let hybrid_pipeline = gpu.create_pipeline(&hybrid_shader_src, "hybrid_ap_cl_bindings");

    // Create buffers with cell-list layout (SORTED positions)
    let pos_buf = gpu.create_f64_output_buffer(n * 3, "pos_hybrid");
    let force_buf = gpu.create_f64_output_buffer(n * 3, "force_hybrid");
    let pe_buf_gpu = gpu.create_f64_output_buffer(n, "pe_hybrid");
    gpu.upload_f64(&pos_buf, &sorted_pos);

    let force_params_cl = vec![
        n as f64,
        kappa,
        prefactor,
        rc * rc,
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
        0.0,
    ];
    let params_buf = gpu.create_f64_buffer(&force_params_cl, "params_hybrid");
    let cs_buf = gpu.create_u32_buffer(&cell_list.cell_start, "cs_hybrid");
    let cc_buf = gpu.create_u32_buffer(&cell_list.cell_count, "cc_hybrid");

    let bg = gpu.create_bind_group(
        &hybrid_pipeline,
        &[
            &pos_buf,
            &force_buf,
            &pe_buf_gpu,
            &params_buf,
            &cs_buf,
            &cc_buf,
        ],
    );
    let workgroups = n.div_ceil(64) as u32;

    gpu.dispatch(&hybrid_pipeline, &bg, workgroups);
    let forces_h = gpu.read_back_f64(&force_buf, n * 3).expect("GPU readback");
    let pe_h = gpu.read_back_f64(&pe_buf_gpu, n).expect("GPU readback");

    let forces_unsorted = cell_list.unsort_array(&forces_h, 3);
    let pe_unsorted = cell_list.unsort_array(&pe_h, 1);
    let total_pe: f64 = pe_unsorted.iter().sum();
    let (net_fx, net_fy, net_fz, net_f) = net_force(&forces_unsorted, n);

    println!("  Hybrid (all-pairs loop, cell-list bindings, sorted positions):");
    println!("    PE = {total_pe:.10}");
    println!("    Net force: ({net_fx:.4e}, {net_fy:.4e}, {net_fz:.4e}) |F|={net_f:.4e}");

    harness.check_upper(&format!("N{n}_hybrid_net_force"), net_f, 1e-6);
    if net_f < 1e-6 {
        println!("  ✓ HYBRID PASS: all-pairs on sorted positions is correct");
        println!("    → Bug is in the CELL ENUMERATION LOOP, not in params/bindings/sorting");
    } else {
        println!("  ✗ HYBRID FAIL: even all-pairs gives wrong results with sorted positions!");
        println!("    → Bug is in BUFFER INFRASTRUCTURE (params, sorting, or GPU buffer layout)");
    }
    println!();
}
