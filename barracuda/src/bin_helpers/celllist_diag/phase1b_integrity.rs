// SPDX-License-Identifier: AGPL-3.0-or-later

use hotspring_barracuda::md::simulation::{CellList, init_fcc_lattice};
use hotspring_barracuda::validation::ValidationHarness;

pub fn run_phase1b_integrity(harness: &mut ValidationHarness) {
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
