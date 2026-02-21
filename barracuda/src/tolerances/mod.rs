// SPDX-License-Identifier: AGPL-3.0-only

//! Centralized validation tolerances with physical justification.
//!
//! Every tolerance threshold used in validation binaries is defined here
//! with documentation of its origin and rationale. No ad-hoc magic numbers.
//!
//! # Tolerance categories
//!
//! | Category | Basis | Example |
//! |----------|-------|---------|
//! | Machine precision | IEEE 754 f64 | 1e-10 for exact arithmetic |
//! | Numerical method | Algorithm convergence | 1e-6 for iterative solvers |
//! | Physical model | Model limitations | 0.5% for energy drift |
//! | Literature | Published uncertainty | NMP 2σ bounds |
//!
//! See `whitePaper/METHODOLOGY.md` for the full acceptance protocol.
//!
//! ## Control JSON Policy
//!
//! Validation binaries use hardcoded constants derived from control JSON files
//! in `control/`. The JSON files are the authoritative source; constants are
//! frozen at a specific commit (see `provenance.rs` for commit hashes).
//! Binaries do NOT load JSON at runtime — this ensures deterministic validation
//! independent of filesystem state. To update baselines, re-run the Python
//! control scripts and update both the JSON files and the Rust constants.

pub mod core;
pub mod lattice;
pub mod md;
pub mod npu;
pub mod physics;

pub use core::{
    ASSOC_LEGENDRE_TOLERANCE, BESSEL_NEAR_ZERO_ABS, BESSEL_TOLERANCE, BETA_VIA_LNGAMMA_TOLERANCE,
    BFGS_GTOL, BFGS_TOLERANCE, BISECT_CONVERGENCE_TOL, BRENT_TOLERANCE, CHI2_CDF_TOLERANCE,
    DIGAMMA_FD_TOLERANCE, EIGH_TOLERANCE, ERF_TOLERANCE, EXACT_F64, FACTORIAL_TOLERANCE,
    GAMMA_TOLERANCE, GPU_VS_CPU_F64, INCOMPLETE_GAMMA_TOLERANCE, ITERATIVE_F64, LAGUERRE_TOLERANCE,
    LU_TOLERANCE, NELDER_MEAD_FTOL, NELDER_MEAD_TOLERANCE, NORMAL_CDF_TOLERANCE,
    NORMAL_PPF_TOLERANCE, QR_TOLERANCE, RK45_ATOL, RK45_RTOL, RK45_TOLERANCE, SOBOL_TOLERANCE,
    SVD_TOLERANCE, TRIDIAG_TOLERANCE,
};

pub use lattice::{
    ANDERSON_1D_LYAPUNOV_TOLERANCE, ANDERSON_EIGENVALUE_ABS, GOE_DEVIATION_TOLERANCE, GOE_MEAN_R,
    HOFSTADTER_SYMMETRY_TOLERANCE, HOTQCD_CONSISTENCY, HOTQCD_MAX_VIOLATIONS,
    LANCZOS_BREAKDOWN_THRESHOLD, LANCZOS_EIGENVALUE_GPU_PARITY, LANCZOS_EXTREMAL_REL,
    LANCZOS_TRIDIAG_EIGENVALUE_ABS, LATTICE_CG_RESIDUAL, LATTICE_CG_VERIFY_RESIDUAL,
    LATTICE_COLD_ACTION_ABS, LATTICE_COLD_PLAQUETTE_ABS, LATTICE_DIRAC_COLD_PARITY,
    LATTICE_DIRAC_HOT_PARITY, LATTICE_DIRAC_ZERO_INPUT_ABS, LATTICE_GPU_CG_COLD_PARITY,
    LATTICE_GPU_CG_HOT_PARITY, LATTICE_HMC_ACCEPTANCE_MIN, NAK_EIGENSOLVE_PARITY,
    NAK_EIGENSOLVE_REGRESSION, NAK_EIGENSOLVE_VS_CPU_REL, POISSON_DEVIATION_TOLERANCE,
    POISSON_MEAN_R, SPMV_GPU_VS_CPU_ABS, SPMV_IDENTITY_ABS, SPMV_ITERATED_ABS, U1_COLD_ACTION_ABS,
    U1_COLD_PLAQUETTE_ABS, U1_CONDENSATE_MONOTONICITY, U1_HIGGS_CONDENSATE_BOUNDARY,
    U1_HMC_ACCEPTANCE_MIN, U1_PYTHON_RUST_PARITY, U1_STRONG_COUPLING_PLAQ_MAX,
    U1_WEAK_COUPLING_PLAQ_MIN,
};

pub use md::{
    BCS_CHEMICAL_POTENTIAL_REL, BCS_PARTICLE_NUMBER_ABS, CELLLIST_NETFORCE_TOLERANCE,
    CELLLIST_PE_TOLERANCE, DALIGAULT_FIT_RMSE, DALIGAULT_FIT_VS_CALIBRATION, ENERGY_DRIFT_PCT,
    GPU_EIGENSOLVE_REL, GPU_EIGENVECTOR_ORTHO, GPU_NATIVE_VS_SOFTWARE_F64, HFB_RUST_VS_PYTHON_REL,
    MD_ABSOLUTE_FLOOR, MD_EQUILIBRIUM_FORCE_ABS, MD_FORCE_TOLERANCE, NEWTON_3RD_LAW_ABS,
    PARITY_D_STAR_REL, PARITY_ENERGY_DIFF, PARITY_T_DIFF, PPPM_MADELUNG_REL,
    PPPM_MULTI_PARTICLE_NET_FORCE, PPPM_NEWTON_3RD_ABS, RDF_TAIL_TOLERANCE,
    TRANSPORT_D_STAR_CPU_GPU_PARITY, TRANSPORT_D_STAR_VS_FIT, TRANSPORT_D_STAR_VS_FIT_LITE,
    TRANSPORT_D_STAR_VS_SARKAS, TRANSPORT_MSD_VACF_AGREEMENT, TRANSPORT_T_STABILITY,
    TRANSPORT_VISCOSITY_VS_SARKAS,
};

pub use npu::{
    BCS_DEGENERACY_PARTICLE_NUMBER_ABS, CLASSIFIER_EPOCHS, CLASSIFIER_LEARNING_RATE,
    CLASSIFIER_VARIANCE_GUARD, ESN_D_STAR_REL, ESN_F32_CLASSIFICATION_AGREEMENT,
    ESN_F32_LATTICE_LOOSE_PARITY, ESN_F32_LATTICE_PARITY, ESN_INT4_PREDICTION_PARITY,
    ESN_MONITORING_OVERHEAD_PCT, ESN_PHASE_ACCURACY_MIN, ESN_TRAINING_LOSS_MAX, ESN_VACF_R2_MIN,
    NPU_BATCH_SPEEDUP_MIN, NPU_F32_PARITY, NPU_FC_DEPTH_OVERHEAD, NPU_INT4_FULL_QUANTIZATION,
    NPU_INT4_QUANTIZATION, NPU_INT8_QUANTIZATION, NPU_MULTI_OUTPUT_OVERHEAD,
    NPU_WEIGHT_MUTATION_LINEARITY, PHASE_BOUNDARY_BETA_C_ERROR,
};

pub use physics::{
    sigma_theo, BCS_DENSITY_SKIP, COULOMB_R_MIN, DEFORMATION_GUESS_GENERIC, DEFORMATION_GUESS_SD,
    DEFORMATION_GUESS_WEAK, DEFORMED_COULOMB_R_MIN, DENSITY_FLOOR, DIVISION_GUARD,
    GPU_JACOBI_CONVERGENCE, HFB_RUST_VS_EXP_REL, L1_CHI2_THRESHOLD, L1_PROXY_THRESHOLD,
    L2_CHI2_THRESHOLD, NEAR_ZERO_EXPECTED, NMP_N_SIGMA, PAIRING_GAP_THRESHOLD, RHO_POWF_GUARD,
    SCF_ENERGY_TOLERANCE, SCREENED_CRITICAL_VS_LITERATURE, SCREENED_HYDROGEN_VS_EXACT,
    SCREENED_ION_SPHERE_SQRT3_ABS, SCREENED_PYTHON_RUST_PARITY, SCREENED_SP_TO_IS_LIMIT,
    SHARP_FILLING_THRESHOLD, SIGMA_THEO_FLOOR_MEV, SIGMA_THEO_FRACTION, SPIN_ORBIT_R_MIN,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::assertions_on_constants)] // constants sanity check
    fn tolerance_ordering() {
        assert!(EXACT_F64 < ITERATIVE_F64);
        assert!(ITERATIVE_F64 < GPU_VS_CPU_F64);
        assert!(GPU_VS_CPU_F64 < MD_FORCE_TOLERANCE);
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact known values
    fn sigma_theo_floor() {
        assert_eq!(sigma_theo(100.0), 2.0);
        assert_eq!(sigma_theo(500.0), 5.0);
        assert_eq!(sigma_theo(0.0), 2.0);
    }

    #[test]
    #[allow(clippy::assertions_on_constants)] // constants sanity check
    fn energy_drift_is_generous() {
        // Measured drift is 0.000-0.002%; threshold 0.5% is 250× above worst case.
        // Ensure we retain margin above measured (0.01% = 5× measured).
        assert!(ENERGY_DRIFT_PCT > 0.01);
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact known value
    fn nmp_sigma_is_two() {
        assert!((NMP_N_SIGMA - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    #[allow(clippy::assertions_on_constants)] // constants sanity check
    fn physics_guards_are_positive() {
        assert!(DENSITY_FLOOR > 0.0);
        assert!(SPIN_ORBIT_R_MIN > 0.0);
        assert!(COULOMB_R_MIN > 0.0);
        assert!(DIVISION_GUARD > 0.0);
        assert!(PAIRING_GAP_THRESHOLD > 0.0);
        assert!(RHO_POWF_GUARD > 0.0);
        assert!(GPU_JACOBI_CONVERGENCE > 0.0);
        assert!(BCS_DENSITY_SKIP > 0.0);
    }

    #[test]
    #[allow(clippy::assertions_on_constants)] // constants sanity check
    fn guard_hierarchy() {
        // RHO_POWF_GUARD < DENSITY_FLOOR < SPIN_ORBIT_R_MIN
        assert!(RHO_POWF_GUARD < DENSITY_FLOOR);
        assert!(DIVISION_GUARD < RHO_POWF_GUARD);
    }

    #[test]
    #[allow(clippy::assertions_on_constants)] // constants sanity check
    fn deformation_guesses_ordered() {
        assert!(DEFORMATION_GUESS_WEAK < DEFORMATION_GUESS_GENERIC);
        assert!(DEFORMATION_GUESS_GENERIC < DEFORMATION_GUESS_SD);
    }

    #[test]
    #[allow(clippy::assertions_on_constants)] // constants sanity check
    fn all_tolerances_are_positive() {
        let tols = [
            EXACT_F64,
            ITERATIVE_F64,
            GPU_VS_CPU_F64,
            LU_TOLERANCE,
            QR_TOLERANCE,
            SVD_TOLERANCE,
            TRIDIAG_TOLERANCE,
            EIGH_TOLERANCE,
            GAMMA_TOLERANCE,
            ERF_TOLERANCE,
            BESSEL_TOLERANCE,
            MD_FORCE_TOLERANCE,
            MD_ABSOLUTE_FLOOR,
            NEWTON_3RD_LAW_ABS,
            NELDER_MEAD_TOLERANCE,
            BFGS_TOLERANCE,
            BRENT_TOLERANCE,
            RK45_TOLERANCE,
            SOBOL_TOLERANCE,
            SCF_ENERGY_TOLERANCE,
            SHARP_FILLING_THRESHOLD,
            TRANSPORT_D_STAR_CPU_GPU_PARITY,
            TRANSPORT_T_STABILITY,
            TRANSPORT_D_STAR_VS_FIT_LITE,
            TRANSPORT_MSD_VACF_AGREEMENT,
            PARITY_D_STAR_REL,
            PARITY_T_DIFF,
            PARITY_ENERGY_DIFF,
            LATTICE_COLD_PLAQUETTE_ABS,
            LATTICE_COLD_ACTION_ABS,
            LATTICE_HMC_ACCEPTANCE_MIN,
            LATTICE_CG_RESIDUAL,
            HOTQCD_CONSISTENCY,
            NAK_EIGENSOLVE_VS_CPU_REL,
            NAK_EIGENSOLVE_PARITY,
            NAK_EIGENSOLVE_REGRESSION,
            PPPM_MULTI_PARTICLE_NET_FORCE,
            SCREENED_HYDROGEN_VS_EXACT,
            SCREENED_CRITICAL_VS_LITERATURE,
            SCREENED_PYTHON_RUST_PARITY,
            U1_COLD_PLAQUETTE_ABS,
            U1_COLD_ACTION_ABS,
            U1_HMC_ACCEPTANCE_MIN,
            U1_WEAK_COUPLING_PLAQ_MIN,
            U1_PYTHON_RUST_PARITY,
            LATTICE_GPU_CG_COLD_PARITY,
            LATTICE_GPU_CG_HOT_PARITY,
            LATTICE_CG_VERIFY_RESIDUAL,
            LATTICE_DIRAC_ZERO_INPUT_ABS,
            LATTICE_DIRAC_COLD_PARITY,
            LATTICE_DIRAC_HOT_PARITY,
            SPMV_IDENTITY_ABS,
            SPMV_GPU_VS_CPU_ABS,
            SPMV_ITERATED_ABS,
            LANCZOS_BREAKDOWN_THRESHOLD,
            LANCZOS_EIGENVALUE_GPU_PARITY,
            SCREENED_ION_SPHERE_SQRT3_ABS,
            BFGS_GTOL,
            NELDER_MEAD_FTOL,
            BISECT_CONVERGENCE_TOL,
            RK45_ATOL,
            RK45_RTOL,
            NORMAL_CDF_TOLERANCE,
            NORMAL_PPF_TOLERANCE,
            NPU_F32_PARITY,
            NPU_INT8_QUANTIZATION,
            NPU_INT4_QUANTIZATION,
            NPU_INT4_FULL_QUANTIZATION,
            NPU_FC_DEPTH_OVERHEAD,
            NPU_BATCH_SPEEDUP_MIN,
            NPU_MULTI_OUTPUT_OVERHEAD,
            NPU_WEIGHT_MUTATION_LINEARITY,
            // Spectral theory tolerances
            LANCZOS_TRIDIAG_EIGENVALUE_ABS,
            LANCZOS_EXTREMAL_REL,
            ANDERSON_EIGENVALUE_ABS,
            GOE_MEAN_R,
            GOE_DEVIATION_TOLERANCE,
            POISSON_MEAN_R,
            POISSON_DEVIATION_TOLERANCE,
            ANDERSON_1D_LYAPUNOV_TOLERANCE,
            HOFSTADTER_SYMMETRY_TOLERANCE,
            SCREENED_SP_TO_IS_LIMIT,
            // U(1) Abelian Higgs
            U1_STRONG_COUPLING_PLAQ_MAX,
            U1_HIGGS_CONDENSATE_BOUNDARY,
            U1_CONDENSATE_MONOTONICITY,
            // ESN reservoir
            ESN_VACF_R2_MIN,
            ESN_D_STAR_REL,
            ESN_TRAINING_LOSS_MAX,
            // HFB vs experiment
            HFB_RUST_VS_EXP_REL,
            // ESN heterogeneous pipeline
            ESN_F32_LATTICE_PARITY,
            ESN_F32_CLASSIFICATION_AGREEMENT,
            ESN_F32_LATTICE_LOOSE_PARITY,
            ESN_INT4_PREDICTION_PARITY,
            ESN_PHASE_ACCURACY_MIN,
            ESN_MONITORING_OVERHEAD_PCT,
            PHASE_BOUNDARY_BETA_C_ERROR,
            BCS_DEGENERACY_PARTICLE_NUMBER_ABS,
        ];
        for (i, &t) in tols.iter().enumerate() {
            assert!(t > 0.0, "tolerance index {i} must be positive, got {t}");
        }
    }

    #[test]
    #[allow(clippy::assertions_on_constants)] // constants sanity check
    fn hotqcd_max_violations_nonzero() {
        assert!(HOTQCD_MAX_VIOLATIONS > 0);
    }

    #[test]
    #[allow(clippy::assertions_on_constants)] // constants sanity check
    fn spectral_goe_poisson_separation() {
        assert!(GOE_MEAN_R > POISSON_MEAN_R);
        assert!(
            GOE_MEAN_R - POISSON_MEAN_R > GOE_DEVIATION_TOLERANCE + POISSON_DEVIATION_TOLERANCE
        );
    }
}
