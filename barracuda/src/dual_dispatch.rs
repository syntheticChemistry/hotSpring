// SPDX-License-Identifier: AGPL-3.0-or-later

//! Dual-dispatch executor for heterogeneous GPU workloads.
//!
//! Executes workloads according to a `WorkloadAssignment`:
//! - **PreciseOnly / ThroughputOnly**: dispatch to one GPU
//! - **Split**: partition data, dispatch in parallel, merge results
//! - **Redundant**: same data to both, cross-validate results

use crate::device_pair::DevicePair;
use crate::error::HotSpringError;
use crate::gpu::GpuF64;
use crate::workload_planner::WorkloadAssignment;

/// Result of a dual-dispatch operation.
#[derive(Debug)]
pub enum DualResult {
    /// Result from a single GPU.
    Single(Vec<f64>),
    /// Merged result from both GPUs (split workload).
    Merged(Vec<f64>),
    /// Results from both GPUs for cross-validation.
    Redundant {
        /// Result from the precise card.
        precise: Vec<f64>,
        /// Result from the throughput card.
        throughput: Vec<f64>,
    },
}

impl DualResult {
    /// Get the final result as a single slice (merges or picks primary).
    #[must_use]
    pub fn values(&self) -> &[f64] {
        match self {
            Self::Single(v) | Self::Merged(v) => v,
            Self::Redundant { precise, .. } => precise,
        }
    }

    /// For Redundant results, compute the max relative error between cards.
    #[must_use]
    pub fn cross_validation_error(&self) -> Option<f64> {
        match self {
            Self::Redundant {
                precise,
                throughput,
            } => {
                let max_rel = precise
                    .iter()
                    .zip(throughput.iter())
                    .map(|(&p, &t)| {
                        let denom = p.abs().max(t.abs()).max(1e-30);
                        (p - t).abs() / denom
                    })
                    .fold(0.0_f64, f64::max);
                Some(max_rel)
            }
            _ => None,
        }
    }
}

/// A GPU kernel that can be dispatched on a single device.
///
/// Implementations upload data, run the compute shader, and read back results.
pub trait GpuKernel: Send + Sync {
    /// Execute the kernel on the given GPU with the provided data slice.
    ///
    /// # Errors
    ///
    /// Returns error if GPU dispatch, upload, or readback fails.
    fn execute(&self, gpu: &GpuF64, data: &[f64]) -> Result<Vec<f64>, HotSpringError>;
}

/// Execute a kernel according to the workload assignment.
///
/// # Errors
///
/// Returns error if any GPU dispatch fails.
pub fn dispatch_dual(
    pair: &DevicePair,
    assignment: &WorkloadAssignment,
    kernel: &dyn GpuKernel,
    data: &[f64],
) -> Result<DualResult, HotSpringError> {
    match assignment {
        WorkloadAssignment::PreciseOnly => {
            let result = kernel.execute(&pair.precise, data)?;
            Ok(DualResult::Single(result))
        }

        WorkloadAssignment::ThroughputOnly => {
            let result = kernel.execute(&pair.throughput, data)?;
            Ok(DualResult::Single(result))
        }

        WorkloadAssignment::Split { precise_fraction } => {
            let split_idx = ((data.len() as f64) * precise_fraction) as usize;
            let split_idx = split_idx.min(data.len());

            let (data_precise, data_throughput) = data.split_at(split_idx);

            let precise_gpu = &pair.precise;
            let throughput_gpu = &pair.throughput;

            let precise_result = kernel.execute(precise_gpu, data_precise)?;
            let throughput_result = kernel.execute(throughput_gpu, data_throughput)?;

            let mut merged = precise_result;
            merged.extend(throughput_result);
            Ok(DualResult::Merged(merged))
        }

        WorkloadAssignment::Redundant => {
            let precise = kernel.execute(&pair.precise, data)?;
            let throughput = kernel.execute(&pair.throughput, data)?;

            Ok(DualResult::Redundant {
                precise,
                throughput,
            })
        }
    }
}

#[cfg(test)]
#[allow(dead_code, clippy::expect_used)]
mod tests {
    use super::*;

    struct DoubleKernel;

    impl GpuKernel for DoubleKernel {
        fn execute(&self, _gpu: &GpuF64, data: &[f64]) -> Result<Vec<f64>, HotSpringError> {
            Ok(data.iter().map(|x| x * 2.0).collect())
        }
    }

    #[test]
    fn dual_result_values_single() {
        let r = DualResult::Single(vec![1.0, 2.0, 3.0]);
        assert_eq!(r.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn dual_result_values_merged() {
        let r = DualResult::Merged(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(r.values(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn dual_result_cross_validation_exact() {
        let r = DualResult::Redundant {
            precise: vec![1.0, 2.0, 3.0],
            throughput: vec![1.0, 2.0, 3.0],
        };
        let err = r.cross_validation_error().expect("should have error");
        assert!(err < 1e-15, "exact match should give ~0 error, got {err}");
    }

    #[test]
    fn dual_result_cross_validation_small_diff() {
        let r = DualResult::Redundant {
            precise: vec![1.0, 2.0, 3.0],
            throughput: vec![1.0 + 1e-10, 2.0 - 1e-10, 3.0 + 1e-10],
        };
        let err = r.cross_validation_error().expect("should have error");
        assert!(err < 1e-8, "small diff should give small error, got {err}");
    }

    #[test]
    fn dual_result_no_cross_validation_for_single() {
        let r = DualResult::Single(vec![1.0]);
        assert!(r.cross_validation_error().is_none());
    }
}
