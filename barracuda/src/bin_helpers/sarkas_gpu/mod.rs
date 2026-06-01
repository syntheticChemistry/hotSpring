// SPDX-License-Identifier: AGPL-3.0-or-later

//! Helpers extracted from `sarkas_gpu`: validation sweeps, scaling, and brain experiments.

pub mod brain;
pub mod paper_parity;
pub mod quick;
pub mod scaling;

pub use brain::{run_brain_nscale, run_brain_skin_sweep, run_brain_sweep};
pub use paper_parity::run_paper_parity;
pub use quick::{run_full_sweep, run_long_sweep, run_quick_validation};
pub use scaling::{run_n_scaling, run_scaling_test};

use hotspring_barracuda::bench::BenchReport;
use hotspring_barracuda::md::config::MdConfig;
use hotspring_barracuda::md::sarkas_harness::run_single_case;
use hotspring_barracuda::validation::ValidationHarness;

pub async fn run_for_each_case(
    report: &mut BenchReport,
    harness: &mut ValidationHarness,
    cases: &[MdConfig],
    harness_prefix: &str,
    format_case_header: impl Fn(usize, usize, &MdConfig) -> String,
    results_banner: &str,
) -> (usize, usize) {
    let mut passed = 0;
    let total = cases.len();

    for (i, case) in cases.iter().enumerate() {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("{}", format_case_header(i + 1, total, case));
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();

        let case_passed = run_single_case(case, report).await;
        if case_passed {
            passed += 1;
        }
        harness.check_bool(&format!("{harness_prefix}_{}", i + 1), case_passed);
        println!();
    }

    if !results_banner.is_empty() {
        println!("═══════════════════════════════════════════════════════════");
        println!("  {results_banner}: {passed}/{total} cases passed");
        println!("═══════════════════════════════════════════════════════════");
    }

    (passed, total)
}
