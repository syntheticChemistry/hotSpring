// SPDX-License-Identifier: AGPL-3.0-or-later

//! Silicon budget calculator helpers: GPU specs and compound ceiling math.

pub mod ceiling;
pub mod specs;

pub use ceiling::print_compound_budget;
pub use specs::{
    classify_vendor, lookup_silicon_specs, print_budget, print_precision_tier_analysis,
    print_working_set_analysis, GpuSiliconBudget, GpuVendor,
};
