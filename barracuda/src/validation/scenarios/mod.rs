// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validation scenarios — absorbed experiment patterns.
//!
//! Each scenario is a self-contained validation function that exercises
//! a specific hotSpring physics domain. Scenarios evolved from the
//! prokaryotic experiment binary era and were absorbed into the library
//! at the interstadial transition.
//!
//! # Usage
//!
//! ```rust,no_run
//! use hotspring_barracuda::validation::scenarios::{ScenarioRegistry, Tier};
//!
//! let registry = build_registry();
//! for scenario in registry.all() {
//!     println!("{}: {}", scenario.meta.id, scenario.meta.track);
//! }
//! ```

pub mod registry;

pub use registry::{Scenario, ScenarioMeta, ScenarioRegistry, Tier, Track};

pub mod s_composition_health;
#[cfg(feature = "barracuda-local")]
pub mod s_dielectric;
#[cfg(feature = "barracuda-local")]
pub mod s_gradient_flow;
pub mod s_lattice_plaquette;
pub mod s_ltee_anderson;
pub mod s_md_yukawa;
pub mod s_sarkas_yukawa_md;
pub mod s_screened_coulomb;
pub mod s_semf_parity;
#[cfg(feature = "barracuda-local")]
pub mod s_spectral_lanczos;
pub mod s_tolerance_ordering;
pub mod s_transport;

/// Build the canonical scenario registry with all absorbed scenarios.
#[must_use]
pub fn build_registry() -> ScenarioRegistry {
    let mut r = ScenarioRegistry::new();
    r.register(s_semf_parity::SCENARIO);
    r.register(s_lattice_plaquette::SCENARIO);
    r.register(s_screened_coulomb::SCENARIO);
    #[cfg(feature = "barracuda-local")]
    r.register(s_gradient_flow::SCENARIO);
    #[cfg(feature = "barracuda-local")]
    r.register(s_dielectric::SCENARIO);
    r.register(s_transport::SCENARIO);
    #[cfg(feature = "barracuda-local")]
    r.register(s_spectral_lanczos::SCENARIO);
    r.register(s_md_yukawa::SCENARIO);
    r.register(s_sarkas_yukawa_md::SCENARIO);
    r.register(s_composition_health::SCENARIO);
    r.register(s_tolerance_ordering::SCENARIO);
    r.register(s_ltee_anderson::SCENARIO);
    r
}
