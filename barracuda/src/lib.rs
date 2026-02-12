//! hotSpring Nuclear EOS — Shared physics and data modules
//!
//! Validates BarraCUDA library modules against Python control experiments.
//! Physics implementations are direct Rust ports of:
//!   - `control/surrogate/nuclear-eos/wrapper/skyrme_hf.py`
//!   - `control/surrogate/nuclear-eos/wrapper/skyrme_hfb.py`
//!   - `control/surrogate/nuclear-eos/wrapper/objective.py`

pub mod data;
pub mod physics;
pub mod prescreen;

