// SPDX-License-Identifier: AGPL-3.0-only

//! NPU and reservoir computing tolerances: quantization, hardware probing,
//! ESN transport validation, and heterogeneous pipeline acceptance.

// ═══════════════════════════════════════════════════════════════════
// NPU quantization tolerances (metalForge AKD1000 validation)
// ═══════════════════════════════════════════════════════════════════

/// ESN f64 → f32 prediction parity: relative error.
///
/// f32 has ~7.2 significant digits; ESN forward pass with 50-dim reservoir
/// and 100-frame sequences accumulates ~O(50*100) = 5000 FP operations.
/// Measured: <0.001% mean error across 6 test cases (Python control).
pub const NPU_F32_PARITY: f64 = 0.001;

/// ESN f64 → int8 quantized prediction: relative error.
///
/// Symmetric uniform quantization of weights to 8-bit integers introduces
/// quantization noise proportional to max_abs(w) / 127. For W_in in [-0.5, 0.5]
/// and W_res with spectral_radius=0.95, measured mean error is ~0.34%.
/// 5% threshold is conservative.
///
/// Source: control/metalforge_npu/scripts/npu_quantization_parity.py
pub const NPU_INT8_QUANTIZATION: f64 = 0.05;

/// ESN f64 → int4 quantized prediction: relative error.
///
/// 4-bit quantization maps weights to [-7, 7] integers. The dynamic range
/// reduction from 15.9 significant digits (f64) to 4 bits (3.9 significant
/// digits) causes ~5.5% mean error and up to ~14% worst-case for predictions
/// near the weight matrix's null space.
/// 30% threshold accommodates worst-case phase diagram corners.
///
/// Source: control/metalforge_npu/scripts/npu_quantization_parity.py
pub const NPU_INT4_QUANTIZATION: f64 = 0.30;

/// ESN f64 → int4 weights + int4 activations: relative error.
///
/// When both weights AND activations are quantized to 4-bit (matching AKD1000
/// hardware), the error compounds through the reservoir update loop. The tanh
/// activation's non-linearity partially mitigates quantization noise (clamping
/// to [-1,1]) but the iterative state update amplifies errors over 100 frames.
/// Measured: ~8.9% mean, ~24% worst-case.
/// 50% threshold is generous for full-hardware simulation.
///
/// Source: control/metalforge_npu/scripts/npu_quantization_parity.py
pub const NPU_INT4_FULL_QUANTIZATION: f64 = 0.50;

// ═══════════════════════════════════════════════════════════════════
// NPU beyond-SDK tolerances (metalForge AKD1000 hardware probing)
// ═══════════════════════════════════════════════════════════════════

/// FC depth overhead: latency increase from depth=1 to depth=7.
///
/// All FC layers merge into a single hardware sequence via SkipDMA.
/// Measured: ~7% overhead for 7 extra layers. 30% is generous.
///
/// Source: control/metalforge_npu/scripts/npu_beyond_sdk.py
pub const NPU_FC_DEPTH_OVERHEAD: f64 = 0.30;

/// Batch inference speedup: batch=8 vs batch=1 throughput ratio.
///
/// PCIe round-trip amortizes across batch. Measured: 2.35×.
/// 1.5× is the minimum acceptable amortization.
///
/// Source: control/metalforge_npu/scripts/npu_beyond_sdk.py
pub const NPU_BATCH_SPEEDUP_MIN: f64 = 1.5;

/// Multi-output overhead: latency increase from 1→10 outputs.
///
/// The NP mesh parallelism handles multiple outputs simultaneously.
/// Measured: 4.5% overhead. 30% is generous.
///
/// Source: control/metalforge_npu/scripts/npu_beyond_sdk.py
pub const NPU_MULTI_OUTPUT_OVERHEAD: f64 = 0.30;

/// Weight mutation linearity: max error for w×k producing output×k.
///
/// Changing FC weights via set_variable() must produce proportional
/// output changes. Measured: 0.0000 error.
///
/// Source: control/metalforge_npu/scripts/npu_beyond_sdk.py
pub const NPU_WEIGHT_MUTATION_LINEARITY: f64 = 0.01;

// ═══════════════════════════════════════════════════════════════════
// Reservoir computing (ESN) transport tolerances
// ═══════════════════════════════════════════════════════════════════

/// ESN VACF prediction: R² correlation threshold.
///
/// The echo state network trained on MD VACF data should achieve R² > 0.50
/// (capturing at least half the variance) for the normalized VACF decay.
/// This is a minimum-quality gate, not a precision target.
pub const ESN_VACF_R2_MIN: f64 = 0.50;

/// ESN D* prediction: relative error vs MD reference.
///
/// The ESN-predicted D* (from integrating the predicted VACF) should
/// agree with the MD-computed D* to within 80%. The ESN is a surrogate
/// model, not a precise calculator — 80% captures the expected surrogate
/// approximation error for short training sequences.
pub const ESN_D_STAR_REL: f64 = 0.80;

/// ESN training loss convergence: minimum improvement.
///
/// After ridge regression, training MSE should be < 0.05. Higher loss
/// indicates reservoir hyperparameters are misconfigured.
pub const ESN_TRAINING_LOSS_MAX: f64 = 0.05;

// ═══════════════════════════════════════════════════════════════════
// ESN heterogeneous pipeline validation tolerances
// ═══════════════════════════════════════════════════════════════════

/// ESN f64 vs f32 prediction parity: absolute error with real lattice data.
///
/// Unlike `NPU_F32_PARITY` (controlled Python validation on synthetic data),
/// this tolerance covers ESN predictions driven by real HMC observables
/// (plaquette, Polyakov loop) where input noise amplifies FP differences.
/// 0.01 accommodates the 30-dim reservoir state accumulation.
pub const ESN_F32_LATTICE_PARITY: f64 = 0.01;

/// ESN f64 vs f32 classification agreement: minimum fraction.
///
/// On a 4^4 lattice phase scan, CPU f64 and NpuSimulator f32 predictions
/// must classify the same phase (confined vs deconfined) for > 90% of
/// test points. Disagreement below 90% indicates a quantization or
/// numerical issue beyond expected f32 noise.
pub const ESN_F32_CLASSIFICATION_AGREEMENT: f64 = 0.90;

/// ESN f64 vs f32 prediction parity: absolute error (lattice NPU binary).
///
/// More generous than `ESN_F32_LATTICE_PARITY` because the lattice_npu
/// binary drives predictions through real HMC configurations (not
/// synthetic) with higher variance in observables. 0.1 captures the
/// worst-case divergence for 30-dim reservoir + 10-frame sequences.
pub const ESN_F32_LATTICE_LOOSE_PARITY: f64 = 0.1;

/// ESN int4 quantized vs f64: absolute prediction error.
///
/// 4-bit quantization of readout weights maps W_out to [-7, 7] integers.
/// Reservoir state is kept at f32. Measured max error: ~0.3 on phase
/// classification predictions spanning [0, 1]. 0.5 is the acceptance
/// threshold (larger errors indicate readout quantization noise dominates).
pub const ESN_INT4_PREDICTION_PARITY: f64 = 0.5;

/// ESN phase classification accuracy: minimum for ESN-on-lattice pipeline.
///
/// The ESN trained on synthetic plaquette/Polyakov data must achieve > 80%
/// phase accuracy on the test split. Below 80%, the ESN reservoir is
/// misconfigured or the synthetic training data is unrealistic.
pub const ESN_PHASE_ACCURACY_MIN: f64 = 0.80;

/// ESN monitoring overhead: prediction time as % of simulation time.
///
/// The heterogeneous pipeline must add < 5% overhead to the simulation
/// (HMC trajectory time). At 5%, the ESN prediction (< 100 μs) is
/// negligible relative to HMC (~5 ms per trajectory on a 4^4 lattice).
pub const ESN_MONITORING_OVERHEAD_PCT: f64 = 5.0;

/// Phase boundary detection: β_c error on 4^4 SU(3) lattice.
///
/// The known β_c ≈ 5.692 for SU(3) on 4^4. With limited statistics and
/// a small lattice, the ESN-detected crossover can be off by ~0.3-0.4.
/// 0.5 accommodates finite-size effects and ESN surrogate uncertainty.
///
/// Source: Wilson (1974), Creutz (1980).
pub const PHASE_BOUNDARY_BETA_C_ERROR: f64 = 0.5;

/// BCS with degeneracy: particle number absolute error.
///
/// BCS bisection for degenerate levels (e.g., O-16 with 3 proton levels,
/// 2j+1 degeneracies) converges more slowly than non-degenerate BCS due
/// to the discrete shell structure. With n_levels=3 and large Δ=12/√A,
/// GPU bisection converges within 0.04 particles of the target.
/// 0.05 = max observed + 0.01 margin.
pub const BCS_DEGENERACY_PARTICLE_NUMBER_ABS: f64 = 0.05;

/// Normalization variance guard for the pre-screening classifier.
///
/// During feature normalization, any feature with variance below this
/// threshold is treated as constant (std clamped to this floor).
/// At 1e-10, features varying by less than ~1e-5 of their mean are
/// effectively constant and would cause numerical blow-up if divided by
/// their true standard deviation.
pub const CLASSIFIER_VARIANCE_GUARD: f64 = 1e-10;

/// Learning rate for the pre-screening logistic regression classifier.
///
/// Standard mini-batch logistic regression learning rate. The classifier
/// is a simple 10→1 linear model; 0.01 converges reliably in 200 epochs
/// for the Skyrme parameter space without oscillation.
pub const CLASSIFIER_LEARNING_RATE: f64 = 0.01;

/// Training epochs for the pre-screening logistic regression classifier.
///
/// 200 epochs is sufficient for convergence of a 10-parameter logistic
/// regression on the typical ~100–1000 sample training sets accumulated
/// during L1/L2 sweeps. Loss plateaus well before 200 epochs.
pub const CLASSIFIER_EPOCHS: u32 = 200;
