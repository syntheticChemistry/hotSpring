// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: Mixed Hardware — validates forge dispatch routing,
//! pipeline topology construction, NUCLEUS atomic composition,
//! and biomeOS graph coordination without requiring live hardware.

use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "mixed-hardware",
        track: Track::GpuCompute,
        tier: Tier::Rust,
        provenance_crate: "validate_mixed_substrate",
        provenance_date: "2026-05-17",
        description: "Forge dispatch routing, pipeline topology, NUCLEUS atomics, biomeOS graph",
    },
    run,
};

pub fn run(v: &mut ValidationHarness) {
    check_dispatch_routing(v);
    check_pipeline_topologies(v);
    check_nucleus_atomics(v);
    check_biome_graph(v);
}

fn check_dispatch_routing(v: &mut ValidationHarness) {
    use hotspring_forge::dispatch::{Reason, Workload, profiles, route};
    use hotspring_forge::substrate::{Capability, Identity, Properties, Substrate, SubstrateKind};

    let gpu = Substrate {
        kind: SubstrateKind::Gpu,
        identity: Identity::named("Test GPU"),
        properties: Properties::default(),
        capabilities: vec![
            Capability::F64Compute,
            Capability::ScalarReduce,
            Capability::ShaderDispatch,
        ],
    };
    let cpu = Substrate {
        kind: SubstrateKind::Cpu,
        identity: Identity::named("Test CPU"),
        properties: Properties::default(),
        capabilities: vec![
            Capability::F64Compute,
            Capability::F32Compute,
            Capability::ConjugateGradient,
            Capability::SparseSpMV,
            Capability::Eigensolve,
        ],
    };
    let subs = [gpu, cpu];

    let md = profiles::md_force();
    let d = route(&md, &subs);
    v.check_bool("mixed:md_routes_to_gpu", d.is_some_and(|d| d.substrate.kind == SubstrateKind::Gpu));

    let val = profiles::cpu_validation();
    let d2 = route(&val, &subs);
    v.check_bool(
        "mixed:validation_prefers_cpu",
        d2.is_some_and(|d| d.reason == Reason::Preferred),
    );

    let work = Workload::new("fallback test", vec![Capability::F64Compute]);
    let d3 = route(&work, &subs);
    v.check_bool(
        "mixed:f64_best_avail_gpu",
        d3.is_some_and(|d| d.substrate.kind == SubstrateKind::Gpu),
    );
}

fn check_pipeline_topologies(v: &mut ValidationHarness) {
    use hotspring_forge::pipeline::{ChannelKind, topologies};
    use hotspring_forge::substrate::SubstrateKind;

    let oracle = topologies::qcd_gpu_npu_oracle();
    v.check_bool("mixed:oracle_4_stages", oracle.stages().len() == 4);
    v.check_bool("mixed:oracle_3_edges", oracle.edges().len() == 3);

    let ordered = oracle.ordered_stages();
    v.check_bool("mixed:oracle_ordered_4", ordered.len() == 4);

    let counts = oracle.substrate_counts();
    v.check_bool("mixed:oracle_has_gpu", counts.contains_key(&SubstrateKind::Gpu));
    v.check_bool("mixed:oracle_has_npu", counts.contains_key(&SubstrateKind::Npu));

    let direct = topologies::mixed_pcie_direct();
    let has_p2p = direct.edges().iter().any(|e| e.channel == ChannelKind::PcieDirect);
    v.check_bool("mixed:direct_has_p2p", has_p2p);

    let nucleus = topologies::nucleus_atomic();
    v.check_bool("mixed:nucleus_3_stages", nucleus.stages().len() == 3);

    let baseline = topologies::qcd_cpu_baseline();
    v.check_bool(
        "mixed:baseline_all_local",
        baseline.edges().iter().all(|e| e.channel == ChannelKind::Local),
    );
}

fn check_nucleus_atomics(v: &mut ValidationHarness) {
    use hotspring_forge::nucleus::{AtomicBinding, AtomicType};
    use hotspring_forge::substrate::{Identity, Properties, Substrate, SubstrateKind};

    let gpu = Substrate {
        kind: SubstrateKind::Gpu,
        identity: Identity::named("Test GPU"),
        properties: Properties::default(),
        capabilities: vec![],
    };
    let cpu = Substrate {
        kind: SubstrateKind::Cpu,
        identity: Identity::named("Test CPU"),
        properties: Properties::default(),
        capabilities: vec![],
    };

    v.check_bool(
        "mixed:tower_binds_cpu",
        AtomicBinding::bind(AtomicType::Tower, &cpu).is_some(),
    );
    v.check_bool(
        "mixed:tower_binds_gpu",
        AtomicBinding::bind(AtomicType::Tower, &gpu).is_some(),
    );
    v.check_bool(
        "mixed:node_binds_gpu",
        AtomicBinding::bind(AtomicType::Node, &gpu).is_some(),
    );
    v.check_bool(
        "mixed:nest_rejects_gpu",
        AtomicBinding::bind(AtomicType::Nest, &gpu).is_none(),
    );
    v.check_bool(
        "mixed:nest_binds_cpu",
        AtomicBinding::bind(AtomicType::Nest, &cpu).is_some(),
    );
    v.check_bool(
        "mixed:tower_subset_node",
        AtomicType::Tower.is_subset_of(&AtomicType::Node),
    );
    v.check_bool(
        "mixed:tower_subset_full",
        AtomicType::Tower.is_subset_of(&AtomicType::FullNucleus),
    );
    v.check_bool(
        "mixed:node_not_subset_tower",
        !AtomicType::Node.is_subset_of(&AtomicType::Tower),
    );

    let domains = AtomicType::FullNucleus.required_domains();
    v.check_bool("mixed:full_has_crypto", domains.contains(&"crypto"));
    v.check_bool("mixed:full_has_compute", domains.contains(&"compute"));
    v.check_bool("mixed:full_has_dag", domains.contains(&"dag"));
}

fn check_biome_graph(v: &mut ValidationHarness) {
    use hotspring_forge::biome_graph::{pcie_direct_nucleus_graph, standard_nucleus_graph};
    use hotspring_forge::nucleus::AtomicType;
    use hotspring_forge::pipeline::ChannelKind;
    use hotspring_forge::substrate::SubstrateKind;

    let g = standard_nucleus_graph();
    v.check_bool("mixed:graph_3_nodes", g.nodes().len() == 3);
    v.check_bool("mixed:graph_3_edges", g.edges().len() == 3);

    let tower_id = g.nodes_by_atomic(AtomicType::Tower)[0].id;
    let nest_id = g.nodes_by_atomic(AtomicType::Nest)[0].id;
    let path = g.shortest_path(tower_id, nest_id);
    v.check_bool("mixed:tower_reaches_nest", path.is_some());

    let reachable = g.reachable_from(nest_id);
    v.check_bool("mixed:nest_from_cpu", reachable.contains(&SubstrateKind::Cpu));
    v.check_bool("mixed:nest_from_gpu", reachable.contains(&SubstrateKind::Gpu));

    let gd = pcie_direct_nucleus_graph();
    v.check_bool("mixed:direct_4_nodes", gd.nodes().len() == 4);
    let p2p_count = gd
        .edges()
        .iter()
        .filter(|e| e.channel == ChannelKind::PcieDirect)
        .count();
    v.check_bool("mixed:direct_1_p2p", p2p_count == 1);

    let gpu_id = gd.nodes_by_substrate(SubstrateKind::Gpu)[0].id;
    let nest_d = gd.nodes_by_atomic(AtomicType::Nest)[0].id;
    let dpath = gd.shortest_path(gpu_id, nest_d);
    v.check_bool("mixed:gpu_reaches_nest", dpath.is_some());
    if let Some(ref dp) = dpath {
        v.check_bool("mixed:path_has_p2p", gd.pcie_direct_hops(dp) == 1);
    }
}
