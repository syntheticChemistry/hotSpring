// SPDX-License-Identifier: AGPL-3.0-only

//! Typed errors for hotSpring GPU and simulation operations.
//!
//! Replaces `Result<_, String>` in public APIs with a proper enum so callers
//! can pattern-match on failure modes (no adapter, missing feature, device
//! creation) rather than parsing opaque strings.

use std::fmt;

/// Errors arising from GPU initialization, simulation, or data loading.
#[derive(Debug)]
pub enum HotSpringError {
    /// No compatible GPU adapter was found by wgpu.
    NoAdapter,

    /// GPU device creation failed (wraps the underlying wgpu error message).
    DeviceCreation(String),

    /// GPU lacks the `SHADER_F64` feature required for f64 compute.
    NoShaderF64,

    /// Data file loading failed (path, underlying IO or parse error).
    DataLoad(String),

    /// GPU compute operation failed (buffer map, dispatch, readback).
    GpuCompute(String),

    /// Invalid operation (e.g. predict before train).
    InvalidOperation(String),

    /// Propagated from barracuda primitives (ReduceScalarPipeline, etc.)
    Barracuda(barracuda::error::BarracudaError),
}

impl fmt::Display for HotSpringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "No GPU adapter found"),
            Self::DeviceCreation(e) => write!(f, "Failed to create GPU device: {e}"),
            Self::NoShaderF64 => {
                write!(
                    f,
                    "GPU does not support SHADER_F64 — cannot run f64 computation"
                )
            }
            Self::DataLoad(msg) => write!(f, "Data loading failed: {msg}"),
            Self::GpuCompute(msg) => write!(f, "GPU compute failed: {msg}"),
            Self::InvalidOperation(msg) => write!(f, "Invalid operation: {msg}"),
            Self::Barracuda(e) => write!(f, "BarraCUDA error: {e}"),
        }
    }
}

impl std::error::Error for HotSpringError {}

impl From<barracuda::error::BarracudaError> for HotSpringError {
    fn from(e: barracuda::error::BarracudaError) -> Self {
        Self::Barracuda(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_no_adapter() {
        let err = HotSpringError::NoAdapter;
        assert_eq!(err.to_string(), "No GPU adapter found");
    }

    #[test]
    fn display_device_creation() {
        let err = HotSpringError::DeviceCreation("wgpu error".into());
        assert_eq!(err.to_string(), "Failed to create GPU device: wgpu error");
    }

    #[test]
    fn display_no_shader_f64() {
        let err = HotSpringError::NoShaderF64;
        assert!(err.to_string().contains("SHADER_F64"));
        assert!(err.to_string().contains("f64"));
    }

    #[test]
    fn error_trait_works() {
        let err = HotSpringError::NoAdapter;
        let dyn_err: &dyn std::error::Error = &err;
        assert_eq!(dyn_err.to_string(), "No GPU adapter found");
    }

    #[test]
    fn error_conversion_via_result() {
        let err = HotSpringError::NoShaderF64;
        let msg = err.to_string();
        assert!(msg.contains("SHADER_F64"));
    }

    #[test]
    fn display_data_load() {
        let err = HotSpringError::DataLoad("experimental data: file not found".into());
        let s = err.to_string();
        assert!(s.contains("Data loading failed"));
        assert!(s.contains("experimental data: file not found"));
    }

    #[test]
    fn display_barracuda() {
        let barracuda_err = barracuda::error::BarracudaError::device("wgpu timeout");
        let err = HotSpringError::Barracuda(barracuda_err);
        let s = err.to_string();
        assert!(s.contains("BarraCUDA error"));
        assert!(s.contains("Device error"));
        assert!(s.contains("wgpu timeout"));
    }

    #[test]
    fn from_barracuda_error_conversion() {
        let barracuda_err = barracuda::error::BarracudaError::gpu("buffer overflow");
        let hotspring_err: HotSpringError = barracuda_err.into();
        let s = hotspring_err.to_string();
        assert!(s.contains("BarraCUDA error"));
        assert!(s.contains("GPU error"));
        assert!(s.contains("buffer overflow"));
    }

    #[test]
    fn display_gpu_compute() {
        let err = HotSpringError::GpuCompute("eigenvalue readback: buffer map failed".into());
        let s = err.to_string();
        assert!(s.contains("GPU compute failed"));
        assert!(s.contains("eigenvalue readback"));
    }

    #[test]
    fn display_invalid_operation() {
        let err = HotSpringError::InvalidOperation("ESN not trained".into());
        let s = err.to_string();
        assert!(s.contains("Invalid operation"));
        assert!(s.contains("ESN not trained"));
    }
}
