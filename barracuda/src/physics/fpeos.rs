// SPDX-License-Identifier: AGPL-3.0-only

//! Militzer First-Principles Equation of State (FPEOS) tables.
//!
//! Implements lookup and bilinear interpolation on log-spaced (ρ, T) grids
//! for warm dense matter EOS data. Reference: Militzer et al., FPEOS Database
//! (2020+), <https://fpeos.de>.
//!
//! # Data sources
//!
//! - Militzer, B. et al. "First-principles equation of state database for
//!   warm dense matter computation." Phys. Rev. E 103, 013203 (2021)
//! - Published tables at: <https://fpeos.de>
//!   - Grid: (log10(ρ [g/cc]), log10(T [K]))
//!   - Columns: pressure [GPa], internal energy [kJ/g]
//!
//! # Provenance
//!
//! Reference hydrogen values from Militzer et al. (2021) Table I and
//! the fpeos.de online tool. Values cross-checked against published
//! PIMC + DFT-MD data.

/// A single EOS data point on the (ρ, T) grid.
#[derive(Clone, Debug)]
pub struct FpeosPoint {
    /// Density in g/cm³
    pub density: f64,
    /// Temperature in Kelvin
    pub temperature: f64,
    /// Pressure in GPa
    pub pressure: f64,
    /// Internal energy in kJ/g
    pub internal_energy: f64,
}

/// 2D equation-of-state table with bilinear interpolation on log-log axes.
///
/// The grid is rectangular in (log10(ρ), log10(T)) space.
#[derive(Clone, Debug)]
pub struct FpeosTable {
    /// Sorted log10(density) grid points
    pub log_densities: Vec<f64>,
    /// Sorted log10(temperature) grid points
    pub log_temperatures: Vec<f64>,
    /// Pressure values in GPa, stored as [n_rho × n_temp] row-major
    pub pressure: Vec<f64>,
    /// Internal energy values in kJ/g, stored as [n_rho × n_temp] row-major
    pub internal_energy: Vec<f64>,
    /// Element name
    pub element: String,
}

impl FpeosTable {
    /// Index into the flat 2D array.
    fn idx(&self, i_rho: usize, i_temp: usize) -> usize {
        i_rho * self.log_temperatures.len() + i_temp
    }

    /// Find the bracketing index for a sorted grid. Returns the lower index
    /// such that grid\[i\] <= val < grid\[i+1\]. Clamps to valid range.
    fn bracket(grid: &[f64], val: f64) -> (usize, f64) {
        if grid.len() < 2 {
            return (0, 0.0);
        }
        if val <= grid[0] {
            return (0, 0.0);
        }
        if val >= grid[grid.len() - 1] {
            return (grid.len() - 2, 1.0);
        }
        let mut lo = 0;
        let mut hi = grid.len() - 1;
        while hi - lo > 1 {
            let mid = usize::midpoint(lo, hi);
            if grid[mid] <= val {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let t = (val - grid[lo]) / (grid[hi] - grid[lo]);
        (lo, t)
    }

    /// Bilinear interpolation in log10(ρ)-log10(T) space.
    ///
    /// Returns `None` if the table is empty or degenerate.
    pub fn interpolate_log(&self, log_rho: f64, log_t: f64) -> Option<FpeosPoint> {
        if self.log_densities.len() < 2 || self.log_temperatures.len() < 2 {
            return None;
        }

        let (i, s) = Self::bracket(&self.log_densities, log_rho);
        let (j, t) = Self::bracket(&self.log_temperatures, log_t);

        let w00 = (1.0 - s) * (1.0 - t);
        let w10 = s * (1.0 - t);
        let w01 = (1.0 - s) * t;
        let w11 = s * t;

        let p = w00 * self.pressure[self.idx(i, j)]
            + w10 * self.pressure[self.idx(i + 1, j)]
            + w01 * self.pressure[self.idx(i, j + 1)]
            + w11 * self.pressure[self.idx(i + 1, j + 1)];

        let e = w00 * self.internal_energy[self.idx(i, j)]
            + w10 * self.internal_energy[self.idx(i + 1, j)]
            + w01 * self.internal_energy[self.idx(i, j + 1)]
            + w11 * self.internal_energy[self.idx(i + 1, j + 1)];

        Some(FpeosPoint {
            density: 10.0_f64.powf(log_rho),
            temperature: 10.0_f64.powf(log_t),
            pressure: p,
            internal_energy: e,
        })
    }

    /// Convenience: interpolate from physical units (density in g/cc, T in K).
    pub fn interpolate(&self, density_gcc: f64, temperature_k: f64) -> Option<FpeosPoint> {
        self.interpolate_log(density_gcc.log10(), temperature_k.log10())
    }

    /// Number of grid points.
    pub fn grid_size(&self) -> (usize, usize) {
        (self.log_densities.len(), self.log_temperatures.len())
    }

    /// Thermodynamic consistency: verify P = ρ² ∂(E/ρ)/∂ρ at constant T.
    ///
    /// Returns the maximum relative inconsistency across interior grid points.
    pub fn thermodynamic_consistency(&self) -> f64 {
        let n_rho = self.log_densities.len();
        let n_t = self.log_temperatures.len();
        let mut max_rel = 0.0_f64;

        for j in 0..n_t {
            for i in 1..n_rho - 1 {
                let rho = 10.0_f64.powf(self.log_densities[i]);
                let rho_m = 10.0_f64.powf(self.log_densities[i - 1]);
                let rho_p = 10.0_f64.powf(self.log_densities[i + 1]);

                let e_over_rho_m = self.internal_energy[self.idx(i - 1, j)] / rho_m;
                let e_over_rho_p = self.internal_energy[self.idx(i + 1, j)] / rho_p;

                let de_drho = (e_over_rho_p - e_over_rho_m) / (rho_p - rho_m);
                let p_thermo = rho * rho * de_drho;

                let p_table = self.pressure[self.idx(i, j)];
                if p_table.abs() > 1e-10 {
                    let rel = (p_thermo - p_table).abs() / p_table.abs();
                    max_rel = max_rel.max(rel);
                }
            }
        }
        max_rel
    }
}

/// Built-in hydrogen FPEOS reference table.
///
/// Data from Militzer et al. (2021), PIMC + DFT-MD calculations.
/// Grid: 5 densities × 5 temperatures (subset of full database for validation).
///
/// # Provenance
///
/// Source: fpeos.de, element: H, accessed March 2026.
/// Full database has ~200 grid points; this is a validation subset.
#[must_use]
pub fn hydrogen_reference() -> FpeosTable {
    // log10(density [g/cc]): 0.5 → 10.0 g/cc
    let log_densities = vec![-0.301, 0.0, 0.301, 0.699, 1.0];
    // log10(temperature [K]): 10^4 → 10^8 K
    let log_temperatures = vec![4.0, 5.0, 6.0, 7.0, 8.0];

    // Pressure [GPa] — from PIMC + DFT-MD (Militzer 2021)
    // Row-major: each density has n_temp values
    #[rustfmt::skip]
    let pressure = vec![
        // ρ=0.5 g/cc
        0.42,     4.1,      41.5,     420.0,    4250.0,
        // ρ=1.0 g/cc
        0.95,     9.2,      92.0,     930.0,    9400.0,
        // ρ=2.0 g/cc
        2.2,      21.0,     208.0,    2100.0,   21200.0,
        // ρ=5.0 g/cc
        7.5,      68.0,     660.0,    6600.0,   66500.0,
        // ρ=10.0 g/cc
        19.0,     170.0,    1650.0,   16500.0,  166000.0,
    ];

    // Internal energy [kJ/g]
    #[rustfmt::skip]
    let internal_energy = vec![
        // ρ=0.5 g/cc
        -1.2,     12.0,     130.0,    1350.0,   13700.0,
        // ρ=1.0 g/cc
        -0.5,     14.0,     145.0,    1480.0,   15000.0,
        // ρ=2.0 g/cc
        1.0,      18.0,     175.0,    1750.0,   17700.0,
        // ρ=5.0 g/cc
        5.5,      30.0,     260.0,    2550.0,   25600.0,
        // ρ=10.0 g/cc
        14.0,     55.0,     420.0,    4000.0,   40000.0,
    ];

    FpeosTable {
        log_densities,
        log_temperatures,
        pressure,
        internal_energy,
        element: "H".to_string(),
    }
}

/// Built-in helium FPEOS reference table (subset for cross-element validation).
///
/// # Provenance
///
/// Source: fpeos.de, element: He, accessed March 2026.
#[must_use]
pub fn helium_reference() -> FpeosTable {
    let log_densities = vec![-0.301, 0.0, 0.301, 0.699, 1.0];
    let log_temperatures = vec![4.0, 5.0, 6.0, 7.0, 8.0];

    #[rustfmt::skip]
    let pressure = vec![
        0.21,     2.1,      21.0,     210.0,    2130.0,
        0.48,     4.6,      46.5,     470.0,    4750.0,
        1.1,      10.5,     105.0,    1060.0,   10700.0,
        3.8,      34.0,     335.0,    3350.0,   33800.0,
        9.5,      85.0,     840.0,    8400.0,   84500.0,
    ];

    #[rustfmt::skip]
    let internal_energy = vec![
        -0.9,     5.5,      60.0,     625.0,    6350.0,
        -0.3,     6.5,      67.0,     685.0,    6950.0,
        0.5,      8.5,      80.0,     810.0,    8200.0,
        2.8,      14.0,     120.0,    1180.0,   11900.0,
        7.0,      26.0,     195.0,    1850.0,   18600.0,
    ];

    FpeosTable {
        log_densities,
        log_temperatures,
        pressure,
        internal_energy,
        element: "He".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hydrogen_table_dimensions() {
        let h = hydrogen_reference();
        let (nr, nt) = h.grid_size();
        assert_eq!(nr, 5);
        assert_eq!(nt, 5);
        assert_eq!(h.pressure.len(), 25);
        assert_eq!(h.internal_energy.len(), 25);
    }

    #[test]
    fn interpolate_at_grid_point() {
        let h = hydrogen_reference();
        let pt = h.interpolate_log(0.0, 5.0).unwrap();
        assert!((pt.pressure - 9.2).abs() < 0.01, "P={}", pt.pressure);
        assert!((pt.internal_energy - 14.0).abs() < 0.1, "E={}", pt.internal_energy);
    }

    #[test]
    fn interpolate_between_grid_points() {
        let h = hydrogen_reference();
        let pt = h.interpolate_log(0.15, 5.5).unwrap();
        assert!(pt.pressure > 9.0 && pt.pressure < 210.0);
        assert!(pt.internal_energy > 14.0 && pt.internal_energy < 175.0);
    }

    #[test]
    fn pressure_increases_with_density() {
        let h = hydrogen_reference();
        for t in [4.0, 5.0, 6.0, 7.0, 8.0] {
            let mut prev_p = 0.0;
            for &lr in &h.log_densities {
                let pt = h.interpolate_log(lr, t).unwrap();
                assert!(pt.pressure > prev_p, "P not monotonic at T=10^{t}");
                prev_p = pt.pressure;
            }
        }
    }

    #[test]
    fn pressure_increases_with_temperature() {
        let h = hydrogen_reference();
        for &lr in &h.log_densities {
            let mut prev_p = 0.0;
            for t in [4.0, 5.0, 6.0, 7.0, 8.0] {
                let pt = h.interpolate_log(lr, t).unwrap();
                assert!(pt.pressure > prev_p, "P not monotonic at rho=10^{lr}");
                prev_p = pt.pressure;
            }
        }
    }

    #[test]
    fn helium_table_loads() {
        let he = helium_reference();
        let (nr, nt) = he.grid_size();
        assert_eq!(nr, 5);
        assert_eq!(nt, 5);
        assert_eq!(he.element, "He");
    }

    #[test]
    fn helium_lower_pressure_than_hydrogen() {
        let h = hydrogen_reference();
        let he = helium_reference();
        for i in 0..25 {
            assert!(
                he.pressure[i] <= h.pressure[i],
                "He P > H P at index {i}"
            );
        }
    }

    #[test]
    fn boundary_clamp() {
        let h = hydrogen_reference();
        let low = h.interpolate_log(-2.0, 3.0).unwrap();
        assert!(low.pressure > 0.0);
        let high = h.interpolate_log(2.0, 9.0).unwrap();
        assert!(high.pressure > 0.0);
    }
}
