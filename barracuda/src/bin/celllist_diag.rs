// SPDX-License-Identifier: AGPL-3.0-only

//! Cell-List Force Diagnostic
//!
//! Compares all-pairs vs cell-list Yukawa force kernels on identical particle
//! data at multiple N values. This isolates whether the cell-list energy
//! conservation bug is in the shader physics or the sort/rebuild infrastructure.
//!
//! Experiment 002: Cell-List Force Kernel Investigation
//!
//! Usage:
//!   cargo run --release --bin `celllist_diag`

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::md::diag::{force_comparison_stats, net_force, print_force_mismatches};
use hotspring_barracuda::md::shaders;
use hotspring_barracuda::md::simulation::{init_fcc_lattice, CellList};
use hotspring_barracuda::validation::ValidationHarness;

use barracuda::shaders::precision::ShaderTemplate;
use hotspring_barracuda::md::shaders::patch_math_f64_preamble;

use std::time::Instant;

/// Run force comparison at a given N
fn compare_forces(gpu: &GpuF64, n: usize, harness: &mut ValidationHarness) {
    let kappa = 2.0;
    let rc = 6.5;
    let prefactor = 1.0; // reduced units for OCP

    // Box side from number density = 1 in reduced units
    let box_side = (n as f64).cbrt() * (4.0 * std::f64::consts::PI / 3.0_f64).cbrt();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "  N = {}  |  box = {:.3}  |  rc = {}  |  pairs = {}",
        n,
        box_side,
        rc,
        n * (n - 1) / 2
    );

    let cells_per_dim = (box_side / rc).floor() as usize;
    let cells_per_dim = cells_per_dim.max(3);
    let cell_size = box_side / cells_per_dim as f64;
    println!(
        "  cells/dim = {}  |  cell_size = {:.4}  |  cell_size >= rc? {}",
        cells_per_dim,
        cell_size,
        if cell_size >= rc { "YES" } else { "NO ← BUG" }
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Initialize positions on FCC lattice (deterministic)
    let (positions, n_actual) = init_fcc_lattice(n, box_side);
    let n = n_actual.min(n);
    println!("  Placed {n} particles on FCC lattice");

    // Prepare shaders
    let md_math = patch_math_f64_preamble(&ShaderTemplate::math_f64_preamble());
    let prepend = |body: &str| -> String { format!("{md_math}\n\n{body}") };

    let ap_source = prepend(shaders::SHADER_YUKAWA_FORCE);
    let cl_source = prepend(shaders::SHADER_YUKAWA_FORCE_CELLLIST);
    let cl_v2_source = prepend(shaders::SHADER_YUKAWA_FORCE_CELLLIST_V2);

    // V4: Cell-list with f64 cell arrays instead of u32
    let cl_v4_src = format!(
        "{}\n\n{}",
        md_math,
        include_str!("shaders/celllist_diag/celllist_v4_f64.wgsl"),
    );

    // V3: No-cutoff cell-list — test if force formula gives different results
    let cl_v3_src = format!(
        "{}\n\n{}",
        md_math,
        include_str!("shaders/celllist_diag/allpairs_nocutoff_f64.wgsl"),
    );

    let ap_pipeline = gpu.create_pipeline(&ap_source, "yukawa_force_allpairs");
    let cl_pipeline = gpu.create_pipeline(&cl_source, "yukawa_force_celllist");

    let workgroups = n.div_ceil(64) as u32;

    // ── All-pairs force computation ──
    let pos_buf_ap = gpu.create_f64_output_buffer(n * 3, "pos_ap");
    let force_buf_ap = gpu.create_f64_output_buffer(n * 3, "force_ap");
    let pe_buf_ap = gpu.create_f64_output_buffer(n, "pe_ap");

    gpu.upload_f64(&pos_buf_ap, &positions);

    let force_params = vec![
        n as f64,
        kappa,
        prefactor,
        rc * rc,
        box_side,
        box_side,
        box_side,
        0.0,
    ];
    let force_params_buf = gpu.create_f64_buffer(&force_params, "force_params_ap");

    let ap_bg = gpu.create_bind_group(
        &ap_pipeline,
        &[&pos_buf_ap, &force_buf_ap, &pe_buf_ap, &force_params_buf],
    );

    let t0 = Instant::now();
    gpu.dispatch(&ap_pipeline, &ap_bg, workgroups);
    let forces_ap = gpu
        .read_back_f64(&force_buf_ap, n * 3)
        .expect("GPU readback");
    let pe_ap = gpu.read_back_f64(&pe_buf_ap, n).expect("GPU readback");
    let ap_time = t0.elapsed().as_secs_f64();

    let total_pe_ap: f64 = pe_ap.iter().sum();
    println!(
        "  All-pairs: PE = {:.6}, computed in {:.3}ms",
        total_pe_ap,
        ap_time * 1000.0
    );

    // ── Cell-list force computation ──
    // Build cell list from SAME positions (unsorted)
    let cell_list = CellList::build(&positions, n, box_side, rc);
    println!(
        "  Cell list: {}×{}×{} = {} cells",
        cell_list.n_cells[0], cell_list.n_cells[1], cell_list.n_cells[2], cell_list.n_cells_total
    );

    // Sort positions for cell-list kernel
    let sorted_pos = cell_list.sort_array(&positions, 3);

    let pos_buf_cl = gpu.create_f64_output_buffer(n * 3, "pos_cl");
    let force_buf_cl = gpu.create_f64_output_buffer(n * 3, "force_cl");
    let pe_buf_cl = gpu.create_f64_output_buffer(n, "pe_cl");

    gpu.upload_f64(&pos_buf_cl, &sorted_pos);

    let cell_start_buf = gpu.create_u32_buffer(&cell_list.cell_start, "cell_start");
    let cell_count_buf = gpu.create_u32_buffer(&cell_list.cell_count, "cell_count");

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
    let force_params_cl_buf = gpu.create_f64_buffer(&force_params_cl, "force_params_cl");

    let cl_bg = gpu.create_bind_group(
        &cl_pipeline,
        &[
            &pos_buf_cl,
            &force_buf_cl,
            &pe_buf_cl,
            &force_params_cl_buf,
            &cell_start_buf,
            &cell_count_buf,
        ],
    );

    let t0 = Instant::now();
    gpu.dispatch(&cl_pipeline, &cl_bg, workgroups);
    let forces_cl_sorted = gpu
        .read_back_f64(&force_buf_cl, n * 3)
        .expect("GPU readback");
    let pe_cl_sorted = gpu.read_back_f64(&pe_buf_cl, n).expect("GPU readback");
    let cl_time = t0.elapsed().as_secs_f64();

    let forces_cl = cell_list.unsort_array(&forces_cl_sorted, 3);
    let pe_cl_unsorted = cell_list.unsort_array(&pe_cl_sorted, 1);

    let total_pe_cl: f64 = pe_cl_unsorted.iter().sum();
    println!(
        "  Cell-list v1: PE = {:.6}, computed in {:.3}ms",
        total_pe_cl,
        cl_time * 1000.0
    );

    // ── Cell-list v2 (flat loop) ──
    let cl_v2_pipeline = gpu.create_pipeline(&cl_v2_source, "yukawa_force_celllist_v2");

    let pos_buf_v2 = gpu.create_f64_output_buffer(n * 3, "pos_v2");
    let force_buf_v2 = gpu.create_f64_output_buffer(n * 3, "force_v2");
    let pe_buf_v2 = gpu.create_f64_output_buffer(n, "pe_v2");
    gpu.upload_f64(&pos_buf_v2, &sorted_pos);

    let cell_start_buf_v2 = gpu.create_u32_buffer(&cell_list.cell_start, "cell_start_v2");
    let cell_count_buf_v2 = gpu.create_u32_buffer(&cell_list.cell_count, "cell_count_v2");

    let v2_bg = gpu.create_bind_group(
        &cl_v2_pipeline,
        &[
            &pos_buf_v2,
            &force_buf_v2,
            &pe_buf_v2,
            &force_params_cl_buf,
            &cell_start_buf_v2,
            &cell_count_buf_v2,
        ],
    );

    let t0 = Instant::now();
    gpu.dispatch(&cl_v2_pipeline, &v2_bg, workgroups);
    let forces_v2_sorted = gpu
        .read_back_f64(&force_buf_v2, n * 3)
        .expect("GPU readback");
    let pe_v2_sorted = gpu.read_back_f64(&pe_buf_v2, n).expect("GPU readback");
    let v2_time = t0.elapsed().as_secs_f64();

    let forces_v2 = cell_list.unsort_array(&forces_v2_sorted, 3);
    let pe_v2_unsorted = cell_list.unsort_array(&pe_v2_sorted, 1);

    let total_pe_v2: f64 = pe_v2_unsorted.iter().sum();
    println!(
        "  Cell-list v2 (flat): PE = {:.6}, computed in {:.3}ms",
        total_pe_v2,
        v2_time * 1000.0
    );

    // ── V4: Cell-list with f64 cell arrays ──
    let cl_v4_pipeline = gpu.create_pipeline(&cl_v4_src, "celllist_f64_data");

    let pos_buf_v4 = gpu.create_f64_output_buffer(n * 3, "pos_v4");
    let force_buf_v4 = gpu.create_f64_output_buffer(n * 3, "force_v4");
    let pe_buf_v4 = gpu.create_f64_output_buffer(n, "pe_v4");
    gpu.upload_f64(&pos_buf_v4, &sorted_pos);

    // Pack cell_start and cell_count into f64 array: [start0, count0, start1, count1, ...]
    let cell_data_f64: Vec<f64> = cell_list
        .cell_start
        .iter()
        .zip(cell_list.cell_count.iter())
        .flat_map(|(&s, &c)| vec![f64::from(s), f64::from(c)])
        .collect();
    let cell_data_buf = gpu.create_f64_buffer(&cell_data_f64, "cell_data_f64");

    let v4_bg = gpu.create_bind_group(
        &cl_v4_pipeline,
        &[
            &pos_buf_v4,
            &force_buf_v4,
            &pe_buf_v4,
            &force_params_cl_buf,
            &cell_data_buf,
        ],
    );

    gpu.dispatch(&cl_v4_pipeline, &v4_bg, workgroups);
    let forces_v4_sorted = gpu
        .read_back_f64(&force_buf_v4, n * 3)
        .expect("GPU readback");
    let pe_v4_sorted = gpu.read_back_f64(&pe_buf_v4, n).expect("GPU readback");

    let forces_v4 = cell_list.unsort_array(&forces_v4_sorted, 3);
    let pe_v4_unsorted = cell_list.unsort_array(&pe_v4_sorted, 1);
    let total_pe_v4: f64 = pe_v4_unsorted.iter().sum();
    // ── V5: Cell-list enumeration WITHOUT cutoff (f64 cell data) ──
    let cl_v5_src = format!(
        "{}\n\n{}",
        md_math,
        include_str!("shaders/celllist_diag/celllist_v5_pe_f64.wgsl"),
    );

    let cl_v5_pipeline = gpu.create_pipeline(&cl_v5_src, "celllist_no_cutoff");
    let pos_buf_v5 = gpu.create_f64_output_buffer(n * 3, "pos_v5");
    let force_buf_v5 = gpu.create_f64_output_buffer(n * 3, "force_v5");
    let pe_buf_v5_gpu = gpu.create_f64_output_buffer(n, "pe_v5");
    gpu.upload_f64(&pos_buf_v5, &sorted_pos);
    let cell_data_buf_v5 = gpu.create_f64_buffer(&cell_data_f64, "cell_data_v5");
    let v5_bg = gpu.create_bind_group(
        &cl_v5_pipeline,
        &[
            &pos_buf_v5,
            &force_buf_v5,
            &pe_buf_v5_gpu,
            &force_params_cl_buf,
            &cell_data_buf_v5,
        ],
    );
    gpu.dispatch(&cl_v5_pipeline, &v5_bg, workgroups);
    let pe_v5 = gpu.read_back_f64(&pe_buf_v5_gpu, n).expect("GPU readback");
    let pe_v5_unsorted = cell_list.unsort_array(&pe_v5, 1);
    let total_pe_v5: f64 = pe_v5_unsorted.iter().sum();
    println!("  V5 (cell-list enum, NO cutoff, f64): PE = {total_pe_v5:.6}");

    // ── V6: Debug trace — record ALL j indices visited by particle 0 ──
    if n <= 500 {
        let cl_v6_src = format!(
            "{}\n\n{}",
            md_math,
            include_str!("shaders/celllist_diag/celllist_v6_debug_f64.wgsl"),
        );
        let v6_pipeline = gpu.create_pipeline(&cl_v6_src, "j_trace_debug");
        // Max entries: 4 header + 27*(7 + max_per_cell) + 1 marker
        let trace_size = 4 + 27 * (7 + 100) + 1;
        let pos_buf_v6 = gpu.create_f64_output_buffer(n * 3, "pos_v6");
        let trace_buf = gpu.create_f64_output_buffer(trace_size, "trace");
        let pe_buf_v6 = gpu.create_f64_output_buffer(n, "pe_v6");
        gpu.upload_f64(&pos_buf_v6, &sorted_pos);
        let cell_data_v6 = gpu.create_f64_buffer(&cell_data_f64, "cd_v6");
        let v6_bg = gpu.create_bind_group(
            &v6_pipeline,
            &[
                &pos_buf_v6,
                &trace_buf,
                &pe_buf_v6,
                &force_params_cl_buf,
                &cell_data_v6,
            ],
        );
        gpu.dispatch(&v6_pipeline, &v6_bg, 1);
        let trace_data = gpu
            .read_back_f64(&trace_buf, trace_size)
            .expect("GPU readback");
        let _pe_v6 = gpu.read_back_f64(&pe_buf_v6, 1).expect("GPU readback");

        println!("  V6 j-trace for sorted particle 0:");
        println!(
            "    Cell: ({},{},{}), N={}",
            trace_data[0] as i32, trace_data[1] as i32, trace_data[2] as i32, trace_data[3] as u32
        );

        let mut idx = 4usize;
        let mut all_j: Vec<u32> = Vec::new();
        for _neigh in 0..27 {
            if idx >= trace_size {
                break;
            }
            let neigh_id = trace_data[idx] as u32;
            let off_x = trace_data[idx + 1] as i32;
            let off_y = trace_data[idx + 2] as i32;
            let off_z = trace_data[idx + 3] as i32;
            let cl = trace_data[idx + 4] as u32;
            let start = trace_data[idx + 5] as u32;
            let cnt = trace_data[idx + 6] as u32;
            idx += 7;

            let mut js: Vec<u32> = Vec::new();
            for _ in 0..cnt {
                if idx >= trace_size {
                    break;
                }
                let j = trace_data[idx] as u32;
                js.push(j);
                all_j.push(j);
                idx += 1;
            }
            println!(
                "    neigh {neigh_id:>2}: off=({off_x:+},{off_y:+},{off_z:+}) cell={cl:>2} start={start:>3} cnt={cnt} j={js:?}"
            );
        }

        // Check for duplicates
        all_j.sort_unstable();
        let total_before = all_j.len();
        all_j.dedup();
        let total_after = all_j.len();
        println!(
            "    Total j visits: {} unique: {} duplicates: {}",
            total_before,
            total_after,
            total_before - total_after
        );
        if total_before != total_after {
            println!("    ⚠ DUPLICATE J-INDICES DETECTED! This is the bug!");
        }
        println!();
    }

    let (_, _, _, net_f_v4) = net_force(&forces_v4, n);
    let pe_diff_v4 = (total_pe_ap - total_pe_v4).abs();
    let pe_rel_v4 = if total_pe_ap.abs() > 1e-30 {
        pe_diff_v4 / total_pe_ap.abs()
    } else {
        pe_diff_v4
    };
    println!("  Cell-list v4 (f64 cell data): PE = {total_pe_v4:.6}, net force = {net_f_v4:.2e}, PE diff = {pe_diff_v4:.2e} (rel {pe_rel_v4:.2e})");
    harness.check_upper(&format!("N{n}_V4_PE_rel_diff"), pe_rel_v4, 1e-6);
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
    gpu.upload_f64(&pos_buf_v3, &sorted_pos);
    let cs_v3 = gpu.create_u32_buffer(&cell_list.cell_start, "cs_v3");
    let cc_v3 = gpu.create_u32_buffer(&cell_list.cell_count, "cc_v3");
    let v3_bg = gpu.create_bind_group(
        &cl_v3_pipeline,
        &[
            &pos_buf_v3,
            &force_buf_v3,
            &pe_buf_v3_gpu,
            &force_params_cl_buf,
            &cs_v3,
            &cc_v3,
        ],
    );
    gpu.dispatch(&cl_v3_pipeline, &v3_bg, workgroups);
    let pe_v3_sorted = gpu.read_back_f64(&pe_buf_v3_gpu, n).expect("GPU readback");
    let pe_v3_unsorted = cell_list.unsort_array(&pe_v3_sorted, 1);
    let total_pe_v3: f64 = pe_v3_unsorted.iter().sum();
    println!("  V3 (no cutoff, AP loop, sorted): PE = {total_pe_v3:.6}");

    // Also compute AP PE without cutoff on ORIGINAL positions for comparison
    let pos_buf_v3_orig = gpu.create_f64_output_buffer(n * 3, "pos_v3_orig");
    let force_buf_v3_orig = gpu.create_f64_output_buffer(n * 3, "force_v3_orig");
    let pe_buf_v3_orig = gpu.create_f64_output_buffer(n, "pe_v3_orig");
    gpu.upload_f64(&pos_buf_v3_orig, &positions);
    let cs_v3_orig = gpu.create_u32_buffer(&cell_list.cell_start, "cs_v3_orig");
    let cc_v3_orig = gpu.create_u32_buffer(&cell_list.cell_count, "cc_v3_orig");
    let v3_orig_bg = gpu.create_bind_group(
        &cl_v3_pipeline,
        &[
            &pos_buf_v3_orig,
            &force_buf_v3_orig,
            &pe_buf_v3_orig,
            &force_params_cl_buf,
            &cs_v3_orig,
            &cc_v3_orig,
        ],
    );
    gpu.dispatch(&cl_v3_pipeline, &v3_orig_bg, workgroups);
    let pe_v3_orig_data = gpu.read_back_f64(&pe_buf_v3_orig, n).expect("GPU readback");
    let total_pe_v3_orig: f64 = pe_v3_orig_data.iter().sum();
    println!("  V3 (no cutoff, AP loop, original): PE = {total_pe_v3_orig:.6}");
    println!("  AP (with cutoff): PE = {total_pe_ap:.6}");

    // ── Compare ──
    let pe_diff_v2 = (total_pe_ap - total_pe_v2).abs();
    let pe_rel_v2 = if total_pe_ap.abs() > 1e-30 {
        pe_diff_v2 / total_pe_ap.abs()
    } else {
        pe_diff_v2
    };
    let (_, _, _, net_f_v2) = net_force(&forces_v2, n);
    println!(
        "  Cell-list v2: PE diff from AP = {pe_diff_v2:.2e} (relative {pe_rel_v2:.2e}), net force = {net_f_v2:.2e}"
    );
    harness.check_upper(&format!("N{n}_V2_PE_rel_diff"), pe_rel_v2, 1e-6);
    harness.check_upper(&format!("N{n}_V2_net_force"), net_f_v2, 1e-4);
    if pe_rel_v2 < 1e-6 && net_f_v2 < 1e-4 {
        println!("  ✓ V2 PASS: flat loop cell-list matches all-pairs!");
    } else {
        println!("  ✗ V2 FAIL: flat loop still broken");
    }
    println!();

    let pe_diff = (total_pe_ap - total_pe_cl).abs();
    let pe_rel = if total_pe_ap.abs() > 1e-30 {
        pe_diff / total_pe_ap.abs()
    } else {
        pe_diff
    };
    println!();
    println!("  ── Comparison ──");
    println!("  Total PE: all-pairs = {total_pe_ap:.10}, cell-list = {total_pe_cl:.10}");
    println!("  PE difference: {pe_diff:.2e} (relative: {pe_rel:.2e})");

    let threshold = 1e-8;
    let (max_force_diff, max_force_particle, rms_diff, rms_rel, max_rel, avg_force, n_mismatched) =
        force_comparison_stats(&forces_ap, &forces_cl, n, threshold);

    println!("  Force RMS diff: {rms_diff:.4e} (relative: {rms_rel:.4e})");
    println!(
        "  Force max diff: {max_force_diff:.4e} at particle {max_force_particle} (relative: {max_rel:.4e})"
    );
    println!("  Particles with |ΔF/F| > {threshold:.0e}: {n_mismatched}/{n}");
    println!("  Average |F|: {avg_force:.6}");

    let (net_fx_ap, net_fy_ap, net_fz_ap, net_f_ap) = net_force(&forces_ap, n);
    let (net_fx_cl, net_fy_cl, net_fz_cl, net_f_cl) = net_force(&forces_cl, n);

    println!(
        "  Net force (all-pairs): ({net_fx_ap:.4e}, {net_fy_ap:.4e}, {net_fz_ap:.4e}) |F|={net_f_ap:.4e}"
    );
    println!(
        "  Net force (cell-list): ({net_fx_cl:.4e}, {net_fy_cl:.4e}, {net_fz_cl:.4e}) |F|={net_f_cl:.4e}"
    );

    // Summary
    let pe_ok = pe_rel < 1e-6;
    let force_ok = rms_rel < 1e-6;
    let net_ok = net_f_cl < 1e-4 * avg_force * n as f64;
    let net_threshold = 1e-4 * avg_force * n as f64;

    harness.check_upper(&format!("N{n}_celllist_PE_rel"), pe_rel, 1e-6);
    harness.check_upper(&format!("N{n}_celllist_force_RMS_rel"), rms_rel, 1e-6);
    harness.check_upper(&format!("N{n}_celllist_net_force"), net_f_cl, net_threshold);

    if pe_ok && force_ok && net_ok {
        println!("  ✓ PASS: cell-list matches all-pairs at N={n}");
    } else {
        println!("  ✗ FAIL at N={n}:");
        if !pe_ok {
            println!("    - PE mismatch: {pe_rel:.2e} relative");
        }
        if !force_ok {
            println!("    - Force mismatch: {rms_rel:.2e} RMS relative");
        }
        if !net_ok {
            println!("    - Net force too large: {net_f_cl:.2e}");
        }
    }

    print_force_mismatches(&forces_ap, &forces_cl, &positions, cell_size, n, threshold);
    println!();
}

fn run_phase1b_integrity(harness: &mut ValidationHarness) {
    let n_test = 108;
    let box_s = (n_test as f64).cbrt() * (4.0 * std::f64::consts::PI / 3.0_f64).cbrt();
    let (pos, n_a) = init_fcc_lattice(n_test, box_s);
    let n_test = n_a.min(n_test);
    let cl = CellList::build(&pos, n_test, box_s, 6.5);
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Phase 1b: Cell list integrity check (N=108)                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!(
        "  cells/dim = {}, total cells = {}",
        cl.n_cells[0], cl.n_cells_total
    );
    let total_count: u32 = cl.cell_count.iter().sum();
    println!("  sum(cell_count) = {total_count} (should be {n_test})");
    harness.check_abs(
        "phase1b_cell_count_sum",
        f64::from(total_count),
        n_test as f64,
        0.5,
    );
    let mut seen = vec![false; n_test];
    for &idx in &cl.sorted_indices {
        assert!(idx < n_test, "sorted_indices out of range: {idx}");
        assert!(!seen[idx], "duplicate in sorted_indices: {idx}");
        seen[idx] = true;
    }
    println!("  sorted_indices: valid permutation of 0..{n_test} ✓");
    println!("  Cell contents:");
    for c in 0..cl.n_cells_total {
        let start = cl.cell_start[c] as usize;
        let count = cl.cell_count[c] as usize;
        if count > 0 {
            let cx = c % cl.n_cells[0];
            let cy = (c / cl.n_cells[0]) % cl.n_cells[1];
            let cz = c / (cl.n_cells[0] * cl.n_cells[1]);
            println!("    cell ({cx},{cy},{cz}) = idx {c}: start={start}, count={count}");
        }
    }
    let sp = cl.sort_array(&pos, 3);
    let (xi, yi, zi) = (sp[0], sp[1], sp[2]);
    let (ci_x, ci_y, ci_z) = (
        (xi / cl.cell_size[0]) as i32,
        (yi / cl.cell_size[1]) as i32,
        (zi / cl.cell_size[2]) as i32,
    );
    let nx = cl.n_cells[0] as i32;
    println!("  Sorted particle 0: pos=({xi:.4},{yi:.4},{zi:.4}), cell=({ci_x},{ci_y},{ci_z})");
    let mut total_checked = 0usize;
    let mut cells_visited = Vec::new();
    for dz in -1..=1i32 {
        for dy in -1..=1i32 {
            for dx in -1..=1i32 {
                let (wx, wy, wz) = (
                    ((ci_x + dx) % nx + nx) % nx,
                    ((ci_y + dy) % nx + nx) % nx,
                    ((ci_z + dz) % nx + nx) % nx,
                );
                let c_idx = (wx + wy * nx + wz * nx * nx) as usize;
                let count = cl.cell_count[c_idx] as usize;
                total_checked += count;
                cells_visited.push((dx, dy, dz, wx, wy, wz, c_idx, count));
            }
        }
    }
    let mut cell_ids: Vec<usize> = cells_visited.iter().map(|v| v.6).collect();
    let n_unique_before = cell_ids.len();
    cell_ids.sort_unstable();
    cell_ids.dedup();
    let n_unique = cell_ids.len();
    println!("  27 neighbor offsets map to {n_unique} unique cells (expect 27 for nx=3)");
    if n_unique < n_unique_before {
        println!("  ⚠ DUPLICATE CELLS DETECTED! {n_unique_before} visits but {n_unique} unique");
        for v in &cells_visited {
            println!(
                "    offset ({:+},{:+},{:+}) → wrapped ({},{},{}) = cell {}, count={}",
                v.0, v.1, v.2, v.3, v.4, v.5, v.6, v.7
            );
        }
    }
    println!(
        "  Total particles checked by sorted particle 0: {} (expect {} = N-1 for all cells)",
        total_checked,
        n_test - 1
    );
    println!();
}

fn run_phase1c_pair_count(gpu: &GpuF64, harness: &mut ValidationHarness) {
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
    let count_shader_src = format!(
        "{}\n\n{}",
        md_math,
        include_str!("shaders/celllist_diag/paircount_celllist_f64.wgsl")
    );
    let count_pipeline = gpu.create_pipeline(&count_shader_src, "pair_count_diag");
    let pos_buf = gpu.create_f64_output_buffer(n_test * 3, "pos_count");
    let debug_buf = gpu.create_f64_output_buffer(n_test * 8, "debug_count");
    gpu.upload_f64(&pos_buf, &sorted_pos);
    let params_cl = vec![
        n_test as f64,
        2.0,
        1.0,
        rc * rc,
        box_s,
        box_s,
        box_s,
        0.0,
        cl.n_cells[0] as f64,
        cl.n_cells[1] as f64,
        cl.n_cells[2] as f64,
        cl.cell_size[0],
        cl.cell_size[1],
        cl.cell_size[2],
        cl.n_cells_total as f64,
        0.0,
    ];
    let params_buf = gpu.create_f64_buffer(&params_cl, "params_count");
    let cs_buf = gpu.create_u32_buffer(&cl.cell_start, "cs_count");
    let cc_buf = gpu.create_u32_buffer(&cl.cell_count, "cc_count");
    let bg = gpu.create_bind_group(
        &count_pipeline,
        &[&pos_buf, &debug_buf, &params_buf, &cs_buf, &cc_buf],
    );
    gpu.dispatch(&count_pipeline, &bg, n_test.div_ceil(64) as u32);
    let debug_data = gpu
        .read_back_f64(&debug_buf, n_test * 8)
        .expect("GPU readback");
    println!("  GPU pair-count results (first 10 sorted particles):");
    println!(
        "  {:>4} {:>8} {:>8} {:>8} {:>6} {:>8} {:>6} {:>4} {:>8}",
        "idx", "pos_x", "pos_y", "pos_z", "cell", "checked", "pairs", "nx", "cell_sx"
    );
    for i in 0..10.min(n_test) {
        let (cx, cy, cz) = (
            debug_data[i * 8] as i32,
            debug_data[i * 8 + 1] as i32,
            debug_data[i * 8 + 2] as i32,
        );
        let (checked, pairs, nx_r, csx) = (
            debug_data[i * 8 + 4] as i32,
            debug_data[i * 8 + 5] as i32,
            debug_data[i * 8 + 6] as i32,
            debug_data[i * 8 + 7],
        );
        println!(
            "  {:>4} {:>8.3} {:>8.3} {:>8.3} ({},{},{}) {:>8} {:>6} {:>4} {:>8.4}",
            i,
            sorted_pos[i * 3],
            sorted_pos[i * 3 + 1],
            sorted_pos[i * 3 + 2],
            cx,
            cy,
            cz,
            checked,
            pairs,
            nx_r,
            csx
        );
    }
    let (nx_gpu, csx_gpu) = (debug_data[6] as i32, debug_data[7]);
    println!();
    println!(
        "  GPU reads: nx={} (expect {}), cell_sx={:.4} (expect {:.4})",
        nx_gpu, cl.n_cells[0], csx_gpu, cl.cell_size[0]
    );
    let total_pairs_cl: f64 = (0..n_test).map(|i| debug_data[i * 8 + 5]).sum();
    println!("  Total pairs found (cell-list): {total_pairs_cl:.0}");
    println!();
    let ap_count_src = format!(
        "{}\n\n{}",
        md_math,
        include_str!("shaders/celllist_diag/paircount_allpairs_f64.wgsl")
    );
    let ap_count_pipeline = gpu.create_pipeline(&ap_count_src, "ap_count_diag");
    let ap_pos_buf = gpu.create_f64_output_buffer(n_test * 3, "ap_pos_count");
    let ap_counts_buf = gpu.create_f64_output_buffer(n_test * 2, "ap_counts");
    gpu.upload_f64(&ap_pos_buf, &sorted_pos);
    let ap_params = vec![n_test as f64, 2.0, 1.0, rc * rc, box_s, box_s, box_s, 0.0];
    let ap_params_buf = gpu.create_f64_buffer(&ap_params, "ap_params_count");
    let ap_bg = gpu.create_bind_group(
        &ap_count_pipeline,
        &[&ap_pos_buf, &ap_counts_buf, &ap_params_buf],
    );
    gpu.dispatch(&ap_count_pipeline, &ap_bg, n_test.div_ceil(64) as u32);
    let ap_count_data = gpu
        .read_back_f64(&ap_counts_buf, n_test * 2)
        .expect("GPU readback");
    println!("  All-pairs pair counts (same sorted positions, first 10):");
    let mut total_ap_pairs = 0.0f64;
    let mut total_ap_pe = 0.0f64;
    for i in 0..10.min(n_test) {
        let (ap_pc, ap_pe, cl_pc) = (
            ap_count_data[i * 2],
            ap_count_data[i * 2 + 1],
            debug_data[i * 8 + 5],
        );
        total_ap_pairs += ap_pc;
        total_ap_pe += ap_pe;
        let m = if (ap_pc - cl_pc).abs() < 0.5 {
            "✓"
        } else {
            "✗ DIFF"
        };
        println!("    particle {i:>3}: AP pairs={ap_pc:.0}, CL pairs={cl_pc:.0} {m}  AP_PE={ap_pe:.6}, CL_PE=?");
    }
    for i in 10..n_test {
        total_ap_pairs += ap_count_data[i * 2];
        total_ap_pe += ap_count_data[i * 2 + 1];
    }
    println!("  Total AP pairs: {total_ap_pairs:.0}, Total CL pairs: {total_pairs_cl:.0}");
    harness.check_abs(
        "phase1c_pair_count_AP_vs_CL",
        total_ap_pairs,
        total_pairs_cl,
        0.5,
    );
    println!("  Total AP PE: {total_ap_pe:.6}");
    println!();
    let verify_u32_src = format!(
        "{}\n\n{}",
        md_math,
        include_str!("shaders/celllist_diag/verify_u32_f64.wgsl")
    );
    let verify_pipeline = gpu.create_pipeline(&verify_u32_src, "verify_u32");
    let out_buf = gpu.create_f64_output_buffer(27 * 2, "verify_out");
    let verify_bg = gpu.create_bind_group(&verify_pipeline, &[&cs_buf, &cc_buf, &out_buf]);
    gpu.dispatch(&verify_pipeline, &verify_bg, 1);
    let verify_data = gpu.read_back_f64(&out_buf, 27 * 2).expect("GPU readback");
    println!("  u32 buffer verification (GPU reads vs CPU values):");
    println!(
        "  {:>4} {:>10} {:>10} {:>10} {:>10} {:>5}",
        "cell", "GPU_start", "CPU_start", "GPU_count", "CPU_count", "match"
    );
    let mut all_match = true;
    for c in 0..27 {
        let (gpu_start, gpu_count) = (verify_data[c * 2] as u32, verify_data[c * 2 + 1] as u32);
        let (cpu_start, cpu_count) = (cl.cell_start[c], cl.cell_count[c]);
        let ok = gpu_start == cpu_start && gpu_count == cpu_count;
        if !ok {
            all_match = false;
        }
        println!(
            "  {:>4} {:>10} {:>10} {:>10} {:>10} {:>5}",
            c,
            gpu_start,
            cpu_start,
            gpu_count,
            cpu_count,
            if ok { "✓" } else { "✗ MISMATCH" }
        );
    }
    harness.check_bool("phase1c_u32_buffer_match", all_match);
    println!(
        "  {}",
        if all_match {
            "✓ All u32 buffer reads match CPU values"
        } else {
            "✗ u32 BUFFER READ MISMATCH — this is the bug source!"
        }
    );
    println!();
}

/// Hybrid test: uses cell-list BINDINGS but all-pairs LOOP
fn test_hybrid(gpu: &GpuF64, n: usize, harness: &mut ValidationHarness) {
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
        include_str!("shaders/celllist_diag/hybrid_allpairs_f64.wgsl"),
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

#[tokio::main]
async fn main() {
    let mut harness = ValidationHarness::new("celllist_diag");
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
        compare_forces(&gpu, n, &mut harness);
    }

    run_phase1b_integrity(&mut harness);
    run_phase1c_pair_count(&gpu, &mut harness);

    // ── Phase 2: Hybrid test ──
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Phase 2: Hybrid Isolation Test                             ║");
    println!("║  All-pairs loop + cell-list bindings → isolate bug location ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    for &n in &[500, 2048, 10976] {
        test_hybrid(&gpu, n, &mut harness);
    }

    println!("════════════════════════════════════════════════════════════════");
    println!("  Diagnostic complete. See experiment 002 journal for analysis.");
    println!("════════════════════════════════════════════════════════════════");
    harness.finish();
}
