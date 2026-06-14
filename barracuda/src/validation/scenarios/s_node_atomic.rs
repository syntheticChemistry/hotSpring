// SPDX-License-Identifier: AGPL-3.0-or-later

//! Scenario: Node Atomic (proton) — structural validation of the compute
//! dispatch atomic. Validates domain composition, science parity baselines,
//! and standalone behavior without requiring live primals.

use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "node-atomic",
        track: Track::CompositionParity,
        tier: Tier::Rust,
        provenance_crate: "validate_nucleus_node",
        provenance_date: "2026-05-13",
        description: "Node atomic (proton): compute dispatch composition + science baselines",
    },
    run,
};

pub fn run(v: &mut ValidationHarness) {
    use crate::composition::AtomicType;
    use crate::primal_bridge::NucleusContext;
    use std::collections::HashMap;

    let node_domains = AtomicType::Node.required_domains();
    v.check_bool("node:domain_count", node_domains.len() == 5);
    v.check_bool("node:has_crypto", node_domains.contains(&"crypto"));
    v.check_bool("node:has_discovery", node_domains.contains(&"discovery"));
    v.check_bool("node:has_compute", node_domains.contains(&"compute"));
    v.check_bool("node:has_math", node_domains.contains(&"math"));
    v.check_bool("node:has_shader", node_domains.contains(&"shader"));

    let tower_domains = AtomicType::Tower.required_domains();
    let tower_subset = tower_domains.iter().all(|d| node_domains.contains(d));
    v.check_bool("node:superset_of_tower", tower_subset);

    let node_only_count = node_domains
        .iter()
        .filter(|d| !tower_domains.contains(d))
        .count();
    v.check_bool("node:adds_compute_math_shader", node_only_count == 3);

    let label = AtomicType::Node.label();
    v.check_bool("node:label_proton", label.contains("proton"));

    let empty_ctx = NucleusContext {
        discovered: HashMap::new(),
        family_id: "node-atomic-scenario".into(),
    };
    let standalone = crate::composition::validate_atomic(&empty_ctx, AtomicType::Node, v);
    v.check_bool("node:standalone_not_passed", !standalone.passed);
    v.check_bool("node:standalone_zero_alive", standalone.primals_alive == 0);
    v.check_bool(
        "node:standalone_missing_all",
        standalone.primals_missing.len() == 5,
    );

    let z = 82_usize;
    let n = 126_usize;
    let be = crate::physics::semf_binding_energy(z, n, &crate::provenance::SLY4_PARAMS);
    v.check_bool("node:semf_pb208_finite", be.is_finite());
    v.check_upper("node:semf_pb208_range", (be - 1636.0).abs(), 10.0);

    {
        use crate::lattice::wilson::Lattice;
        let dims = [4_usize, 4, 4, 4];
        let beta = 6.0;
        let lat = Lattice::hot_start(dims, beta, 42);
        let plaq = lat.average_plaquette();
        v.check_bool("node:plaquette_finite", plaq.is_finite());
        v.check_upper("node:plaquette_range", plaq, 1.0);
        v.check_bool("node:plaquette_positive", plaq > 0.0);
    }
}
