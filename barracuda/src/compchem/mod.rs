// SPDX-License-Identifier: AGPL-3.0-or-later
//! Computational chemistry modules for sovereign biomolecular simulation.
//!
//! Bridges GROMACS topology data → barraCuda GPU force shaders.
//! Includes the braid pipeline for lithoSpore → sporePrint provenance.

pub mod braid_pipeline;
pub mod metadynamics;
pub mod parity;
pub mod topology;
