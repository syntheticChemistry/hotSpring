//! hotSpring Nuclear EOS — Shared physics, data, and reference math modules
//!
//! Validates BarraCUDA library modules against Python control experiments.
//! Physics implementations are direct Rust ports of:
//!   - `control/surrogate/nuclear-eos/wrapper/skyrme_hf.py`
//!   - `control/surrogate/nuclear-eos/wrapper/skyrme_hfb.py`
//!   - `control/surrogate/nuclear-eos/wrapper/objective.py`
//!
//! Reference math modules (for BarraCUDA team to implement in library):
//!   - `surrogate` — LOO-CV auto-smoothing, penalty filtering, round-based NM
//!   - `stats` — Chi-squared decomposition, bootstrap CI, convergence diagnostics

pub mod data;
pub mod physics;
pub mod prescreen;
pub mod stats;
pub mod surrogate;

