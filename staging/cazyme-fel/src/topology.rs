// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::constants::{BASIN_1C4_MIN_DEG, BASIN_4C1_MAX_DEG};
use crate::types::{Barrier, Basin, FesResult};

/// Identify basins (local minima) in the 1D FEL.
pub fn find_basins(fes: &FesResult) -> Vec<Basin> {
    let mut basins = Vec::new();
    let n = fes.nbins;

    for i in 1..n - 1 {
        if fes.free_energy[i] < fes.free_energy[i - 1]
            && fes.free_energy[i] < fes.free_energy[i + 1]
        {
            let theta_rad = fes.grid[i];
            let theta_deg = theta_rad.to_degrees();
            let label = if theta_deg < BASIN_4C1_MAX_DEG {
                "4C1 chair"
            } else if theta_deg > BASIN_1C4_MIN_DEG {
                "1C4 chair"
            } else {
                "boat/skew-boat"
            };
            basins.push(Basin {
                theta_rad,
                theta_deg,
                energy_kjmol: fes.free_energy[i],
                label: label.to_string(),
            });
        }
    }

    // Check endpoints
    if n > 1 && fes.free_energy[0] < fes.free_energy[1] {
        let theta_deg = fes.grid[0].to_degrees();
        let label = if theta_deg < BASIN_4C1_MAX_DEG { "4C1 chair" } else { "1C4 chair" };
        basins.insert(0, Basin {
            theta_rad: fes.grid[0],
            theta_deg,
            energy_kjmol: fes.free_energy[0],
            label: label.to_string(),
        });
    }
    if n > 1 && fes.free_energy[n - 1] < fes.free_energy[n - 2] {
        let theta_deg = fes.grid[n - 1].to_degrees();
        let label = if theta_deg > BASIN_1C4_MIN_DEG { "1C4 chair" } else { "4C1 chair" };
        basins.push(Basin {
            theta_rad: fes.grid[n - 1],
            theta_deg,
            energy_kjmol: fes.free_energy[n - 1],
            label: label.to_string(),
        });
    }

    basins
}

/// Find barriers between adjacent basins.
pub fn find_barriers(fes: &FesResult, basins: &[Basin]) -> Vec<Barrier> {
    let mut barriers = Vec::new();
    let mut sorted: Vec<&Basin> = basins.iter().collect();
    sorted.sort_by(|a, b| a.theta_rad.partial_cmp(&b.theta_rad).unwrap());

    for pair in sorted.windows(2) {
        let b1 = pair[0];
        let b2 = pair[1];

        // Find grid indices for this pair
        let i1 = fes.grid.iter().position(|&x| (x - b1.theta_rad).abs() < 1e-6)
            .unwrap_or(0);
        let i2 = fes.grid.iter().position(|&x| (x - b2.theta_rad).abs() < 1e-6)
            .unwrap_or(fes.nbins - 1);

        if i1 >= i2 {
            continue;
        }

        let (max_idx, max_e) = fes.free_energy[i1..=i2]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, &e)| (i1 + i, e))
            .unwrap_or((i1, 0.0));

        let ref_energy = b1.energy_kjmol.min(b2.energy_kjmol);
        barriers.push(Barrier {
            from_label: b1.label.clone(),
            to_label: b2.label.clone(),
            theta_deg: fes.grid[max_idx].to_degrees(),
            height_kjmol: max_e - ref_energy,
        });
    }

    barriers
}
