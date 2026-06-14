// SPDX-License-Identifier: AGPL-3.0-or-later

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::md::shaders::patch_math_f64_preamble;
use hotspring_barracuda::md::simulation::{CellList, init_fcc_lattice};
use hotspring_barracuda::validation::ValidationHarness;

use barracuda::shaders::precision::ShaderTemplate;

pub fn run_phase1c_pair_count(gpu: &GpuF64, harness: &mut ValidationHarness) {
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
        include_str!("../../bin/shaders/celllist_diag/paircount_celllist_f64.wgsl")
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
        include_str!("../../bin/shaders/celllist_diag/paircount_allpairs_f64.wgsl")
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
        println!(
            "    particle {i:>3}: AP pairs={ap_pc:.0}, CL pairs={cl_pc:.0} {m}  AP_PE={ap_pe:.6}, CL_PE=?"
        );
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
        include_str!("../../bin/shaders/celllist_diag/verify_u32_f64.wgsl")
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
