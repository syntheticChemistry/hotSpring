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
        }
    }
}

impl std::error::Error for HotSpringError {}

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
}
