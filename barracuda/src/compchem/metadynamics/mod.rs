// SPDX-License-Identifier: AGPL-3.0-or-later
//! Metadynamics bias layer for CAZyme conformational free energy landscapes.
//!
//! Components:
//! - **Cremer-Pople CVs** — ring puckering coordinates (θ, φ) from atomic positions
//! - **Gaussian hill deposition** — well-tempered metadynamics bias potential
//! - **Bias evaluation** — V(s) computation for force augmentation

pub mod cremer_pople;
pub mod hills;

pub use cremer_pople::{CremerPople, PuckeringCoords};
pub use hills::{GaussianHill, MetadynamicsBias};
