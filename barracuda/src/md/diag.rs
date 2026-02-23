// SPDX-License-Identifier: AGPL-3.0-only

//! Diagnostic helpers for cell-list force validation.
//!
//! Used by `celllist_diag` and other MD diagnostic binaries to compare
//! all-pairs vs cell-list force results.

/// Net force (sum over all particles) and magnitude.
#[must_use]
pub fn net_force(forces: &[f64], n: usize) -> (f64, f64, f64, f64) {
    let fx: f64 = (0..n).map(|i| forces[i * 3]).sum();
    let fy: f64 = (0..n).map(|i| forces[i * 3 + 1]).sum();
    let fz: f64 = (0..n).map(|i| forces[i * 3 + 2]).sum();
    let mag = fx.mul_add(fx, fy.mul_add(fy, fz * fz)).sqrt();
    (fx, fy, fz, mag)
}

/// Force comparison statistics: max diff, RMS, mismatches.
#[must_use]
pub fn force_comparison_stats(
    forces_ap: &[f64],
    forces_cl: &[f64],
    n: usize,
    threshold: f64,
) -> (f64, usize, f64, f64, f64, f64, usize) {
    let mut max_force_diff = 0.0f64;
    let mut max_force_particle = 0usize;
    let mut rms_diff = 0.0f64;
    let mut total_force_mag_ap = 0.0f64;
    let mut n_mismatched = 0usize;

    for i in 0..n {
        let fx_ap = forces_ap[i * 3];
        let fy_ap = forces_ap[i * 3 + 1];
        let fz_ap = forces_ap[i * 3 + 2];
        let fx_cl = forces_cl[i * 3];
        let fy_cl = forces_cl[i * 3 + 1];
        let fz_cl = forces_cl[i * 3 + 2];

        let mag_ap = (fx_ap * fx_ap + fy_ap * fy_ap + fz_ap * fz_ap).sqrt();
        let diff = (fz_ap - fz_cl)
            .mul_add(
                fz_ap - fz_cl,
                (fy_ap - fy_cl).mul_add(fy_ap - fy_cl, (fx_ap - fx_cl).powi(2)),
            )
            .sqrt();

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
    let rms_rel = if avg_force > 1e-30 {
        rms_diff / avg_force
    } else {
        rms_diff
    };
    let max_rel = if avg_force > 1e-30 {
        max_force_diff / avg_force
    } else {
        max_force_diff
    };

    (
        max_force_diff,
        max_force_particle,
        rms_diff,
        rms_rel,
        max_rel,
        avg_force,
        n_mismatched,
    )
}

/// Print mismatched particles (first 10 or worst 5).
pub fn print_force_mismatches(
    forces_ap: &[f64],
    forces_cl: &[f64],
    positions: &[f64],
    cell_size: f64,
    n: usize,
    threshold: f64,
) {
    let n_mismatched = (0..n)
        .filter(|&i| {
            let mag_ap = (forces_ap[i * 3].powi(2)
                + forces_ap[i * 3 + 1].powi(2)
                + forces_ap[i * 3 + 2].powi(2))
            .sqrt();
            let diff = (forces_ap[i * 3 + 2] - forces_cl[i * 3 + 2])
                .mul_add(
                    forces_ap[i * 3 + 2] - forces_cl[i * 3 + 2],
                    (forces_ap[i * 3 + 1] - forces_cl[i * 3 + 1]).mul_add(
                        forces_ap[i * 3 + 1] - forces_cl[i * 3 + 1],
                        (forces_ap[i * 3] - forces_cl[i * 3]).powi(2),
                    ),
                )
                .sqrt();
            let rel = if mag_ap > 1e-30 { diff / mag_ap } else { diff };
            rel > threshold
        })
        .count();

    if n_mismatched > 0 && n_mismatched <= 20 {
        println!();
        println!("  Mismatched particles (first up to 10):");
        let mut shown = 0;
        for i in 0..n {
            if shown >= 10 {
                break;
            }
            let (fx_ap, fy_ap, fz_ap) =
                (forces_ap[i * 3], forces_ap[i * 3 + 1], forces_ap[i * 3 + 2]);
            let (fx_cl, fy_cl, fz_cl) =
                (forces_cl[i * 3], forces_cl[i * 3 + 1], forces_cl[i * 3 + 2]);
            let mag_ap = (fx_ap * fx_ap + fy_ap * fy_ap + fz_ap * fz_ap).sqrt();
            let diff = (fz_ap - fz_cl)
                .mul_add(
                    fz_ap - fz_cl,
                    (fy_ap - fy_cl).mul_add(fy_ap - fy_cl, (fx_ap - fx_cl).powi(2)),
                )
                .sqrt();
            let rel = if mag_ap > 1e-30 { diff / mag_ap } else { diff };
            if rel > threshold {
                println!(
                    "    particle {i:>5}: AP=({fx_ap:+.6e},{fy_ap:+.6e},{fz_ap:+.6e}) CL=({fx_cl:+.6e},{fy_cl:+.6e},{fz_cl:+.6e}) Δ={diff:.2e}"
                );
                shown += 1;
            }
        }
    } else if n_mismatched > 20 {
        println!();
        println!("  Worst 5 mismatches:");
        let mut diffs: Vec<(usize, f64)> = (0..n)
            .map(|i| {
                let diff = (forces_ap[i * 3 + 2] - forces_cl[i * 3 + 2])
                    .mul_add(
                        forces_ap[i * 3 + 2] - forces_cl[i * 3 + 2],
                        (forces_ap[i * 3 + 1] - forces_cl[i * 3 + 1]).mul_add(
                            forces_ap[i * 3 + 1] - forces_cl[i * 3 + 1],
                            (forces_ap[i * 3] - forces_cl[i * 3]).powi(2),
                        ),
                    )
                    .sqrt();
                (i, diff)
            })
            .collect();
        diffs.sort_by(|a, b| b.1.total_cmp(&a.1));
        for &(i, diff) in diffs.iter().take(5) {
            let pos = &positions[i * 3..i * 3 + 3];
            let (cx, cy, cz) = (
                (pos[0] / cell_size) as usize,
                (pos[1] / cell_size) as usize,
                (pos[2] / cell_size) as usize,
            );
            println!(
                "    particle {:>5}: pos=({:.3},{:.3},{:.3}) cell=({},{},{}) Δ|F|={:.4e}  AP=({:+.4e},{:+.4e},{:+.4e})  CL=({:+.4e},{:+.4e},{:+.4e})",
                i,
                pos[0],
                pos[1],
                pos[2],
                cx,
                cy,
                cz,
                diff,
                forces_ap[i * 3],
                forces_ap[i * 3 + 1],
                forces_ap[i * 3 + 2],
                forces_cl[i * 3],
                forces_cl[i * 3 + 1],
                forces_cl[i * 3 + 2]
            );
        }
    }
}
