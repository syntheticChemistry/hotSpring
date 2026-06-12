// SPDX-License-Identifier: AGPL-3.0-or-later
//! GROMACS topology parser and bonded-term extractor.
//!
//! Parses `.top` / `.itp` files and produces typed bonded interaction
//! lists that map directly to barraCuda GPU shader parameter structs.

pub mod gromacs;
pub mod types;

pub use gromacs::{GmxTopology, TopologyParseError};
pub use types::*;
