// SPDX-License-Identifier: AGPL-3.0-or-later

//! Chuna validation helpers for Papers 43, 44, 45 and GPU parity checks.

mod gpu_parity;
mod paper_43;
mod paper_44;
mod paper_45;

pub use gpu_parity::gpu_substrate_validation;
pub use paper_43::{paper_43_gradient_flow, CpuReferenceValues};
pub use paper_44::paper_44_dielectric;
pub use paper_45::paper_45_kinetic_fluid;
