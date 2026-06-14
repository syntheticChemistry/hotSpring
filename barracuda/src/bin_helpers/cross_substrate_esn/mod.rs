// SPDX-License-Identifier: AGPL-3.0-or-later

//! Helpers extracted from `cross_substrate_esn_benchmark`: per-experiment runners,
//! CPU-only fallback, and campaign summary/report.

pub mod cpu_only;
pub mod exp1_timing_matrix;
pub mod exp2_gpu_dispatch;
pub mod exp3_npu_envelope;
pub mod exp4_scaling_crossover;
pub mod exp5_gpu_esn;
pub mod exp6_qcd_workload;
pub mod summary;

pub const SEQUENCE_LENGTH: usize = 50;
pub const N_WARMUP: usize = 3;
pub const N_REPS: usize = 20;
pub const INPUT_SIZE: usize = 8;
