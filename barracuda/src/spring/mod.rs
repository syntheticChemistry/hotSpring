// SPDX-License-Identifier: AGPL-3.0-or-later

//! Spring-specific (legitimately local) QCD infrastructure.
//!
//! These modules are NOT candidates for upstream absorption — they represent
//! hotSpring's unique value: production campaign scheduling, physics validation
//! thresholds, and provenance/attestation workflows.

pub mod campaign;
pub mod provenance;
pub mod tolerances;
pub mod validation;
