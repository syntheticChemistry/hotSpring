// SPDX-License-Identifier: AGPL-3.0-only

//! Energy validation for MD simulations.
//!
//! Checks conservation and drift from energy history records.

use crate::md::config::MdConfig;
use crate::md::simulation::EnergyRecord;
use crate::tolerances::{DIVISION_GUARD, ENERGY_DRIFT_PCT};

/// Energy validation result
#[derive(Clone, Debug)]
pub struct EnergyValidation {
    pub mean_total: f64,
    pub std_total: f64,
    pub drift_pct: f64,
    pub mean_temperature: f64,
    pub std_temperature: f64,
    pub passed: bool,
}

/// Validate energy conservation
#[must_use]
pub fn validate_energy(history: &[EnergyRecord], _config: &MdConfig) -> EnergyValidation {
    if history.is_empty() {
        return EnergyValidation {
            mean_total: 0.0,
            std_total: 0.0,
            drift_pct: 0.0,
            mean_temperature: 0.0,
            std_temperature: 0.0,
            passed: false,
        };
    }

    // Skip first 10% of production for transient effects
    let skip = history.len() / 10;
    let stable = &history[skip..];

    let mean_e: f64 = stable.iter().map(|e| e.total).sum::<f64>() / stable.len() as f64;
    let var_e: f64 = stable
        .iter()
        .map(|e| (e.total - mean_e).powi(2))
        .sum::<f64>()
        / stable.len() as f64;
    let std_e = var_e.sqrt();

    // Drift: (E_final - E_initial) / |E_mean|
    // stable is non-empty: early return if history.is_empty(); with len≥1, skip=len/10 < len
    let (e_initial, e_final) = match (stable.first(), stable.last()) {
        (Some(first), Some(last)) => (first.total, last.total),
        _ => {
            return EnergyValidation {
                mean_total: mean_e,
                std_total: std_e,
                drift_pct: 0.0,
                mean_temperature: 0.0,
                std_temperature: 0.0,
                passed: false,
            }
        }
    };
    let drift_pct = if mean_e.abs() > DIVISION_GUARD {
        ((e_final - e_initial) / mean_e.abs()).abs() * 100.0
    } else {
        0.0
    };

    let mean_t: f64 = stable.iter().map(|e| e.temperature).sum::<f64>() / stable.len() as f64;
    let var_t: f64 = stable
        .iter()
        .map(|e| (e.temperature - mean_t).powi(2))
        .sum::<f64>()
        / stable.len() as f64;
    let std_t = var_t.sqrt();

    let passed = drift_pct < ENERGY_DRIFT_PCT;

    EnergyValidation {
        mean_total: mean_e,
        std_total: std_e,
        drift_pct,
        mean_temperature: mean_t,
        std_temperature: std_t,
        passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)] // exact known values (0.0)
    fn validate_energy_empty_history() {
        let config = crate::md::config::quick_test_case(500);
        let result = validate_energy(&[], &config);
        assert!(!result.passed, "empty history should fail");
        assert_eq!(result.mean_total, 0.0);
        assert_eq!(result.std_total, 0.0);
        assert_eq!(result.drift_pct, 0.0);
    }

    #[test]
    fn validate_energy_constant_energy() {
        let config = crate::md::config::quick_test_case(500);
        let history: Vec<EnergyRecord> = (0..100)
            .map(|i| EnergyRecord {
                step: i,
                ke: 50.0,
                pe: -100.0,
                total: -50.0,
                temperature: 0.006,
            })
            .collect();
        let result = validate_energy(&history, &config);
        assert!(result.passed, "constant energy should pass");
        assert!(result.drift_pct < 0.001, "drift should be ~0");
    }

    #[test]
    fn validate_energy_large_drift_fails() {
        let config = crate::md::config::quick_test_case(500);
        let history: Vec<EnergyRecord> = (0..100)
            .map(|i| EnergyRecord {
                step: i,
                ke: 50.0,
                pe: -100.0 + i as f64, // drifting PE
                total: -50.0 + i as f64,
                temperature: 0.006,
            })
            .collect();
        let result = validate_energy(&history, &config);
        assert!(!result.passed, "large drift should fail");
    }

    #[test]
    fn validate_energy_diverging_energy_fails() {
        let config = crate::md::config::quick_test_case(500);
        let history: Vec<EnergyRecord> = (0..50)
            .map(|i| EnergyRecord {
                step: i,
                ke: 50.0,
                pe: -100.0 - i as f64 * 2.0, // diverging
                total: -50.0 - i as f64 * 2.0,
                temperature: 0.006,
            })
            .collect();
        let result = validate_energy(&history, &config);
        assert!(!result.passed);
    }

    #[test]
    fn energy_record_construction() {
        let rec = EnergyRecord {
            step: 100,
            ke: 25.0,
            pe: -75.0,
            total: -50.0,
            temperature: 0.005,
        };
        assert_eq!(rec.step, 100);
        assert!((rec.ke - 25.0).abs() < 1e-10);
        assert!((rec.pe - (-75.0)).abs() < 1e-10);
        assert!((rec.total - (-50.0)).abs() < 1e-10);
        assert!((rec.temperature - 0.005).abs() < 1e-10);
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact known value (0.0)
    fn validate_energy_mean_near_zero_drift_safe() {
        let config = crate::md::config::quick_test_case(500);
        let history: Vec<EnergyRecord> = (0..20)
            .map(|i| EnergyRecord {
                step: i,
                ke: 1e-20,
                pe: -1e-20,
                total: 0.0,
                temperature: 1e-20,
            })
            .collect();
        let result = validate_energy(&history, &config);
        assert_eq!(
            result.drift_pct, 0.0,
            "|mean_e| < DIVISION_GUARD => drift_pct = 0"
        );
    }

    #[test]
    fn energy_validation_struct_fields() {
        let config = crate::md::config::quick_test_case(500);
        let history: Vec<EnergyRecord> = (0..50)
            .map(|i| EnergyRecord {
                step: i,
                ke: 10.0,
                pe: -20.0,
                total: -10.0,
                temperature: 0.05,
            })
            .collect();
        let val = validate_energy(&history, &config);
        assert!(val.mean_total < 0.0);
        assert!(val.std_total >= 0.0);
        assert!(val.mean_temperature > 0.0);
        assert!(val.std_temperature >= 0.0);
    }
}
