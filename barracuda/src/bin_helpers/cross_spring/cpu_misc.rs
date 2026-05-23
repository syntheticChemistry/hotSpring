// SPDX-License-Identifier: AGPL-3.0-or-later

use std::time::Instant;

pub fn bench_neighbor_precompute() {
    println!("═══ Phase 1c: Neighbor Table Precompute ═══");
    println!(
        "  Provenance: hotSpring build_neighbors (HMC) -> toadStool S80 NeighborMode::precompute_periodic_4d"
    );
    println!("  Note: hotSpring idx = t*V3 + x*V2 + y*Nz + z (z fastest)");
    println!("        toadStool idx = t*V3 + z*V2 + y*Nx + x (x fastest)");
    println!();

    use barracuda::ops::lattice::NeighborMode;

    for &dims in &[
        [4u32, 4, 4, 4],
        [8, 8, 8, 8],
        [12, 12, 12, 12],
        [16, 16, 16, 16],
    ] {
        let vol = dims.iter().product::<u32>() as usize;
        let t = Instant::now();
        let mode = NeighborMode::precompute_periodic_4d(dims);
        let us = t.elapsed().as_micros();

        let table_kb = match &mode {
            NeighborMode::PrecomputedBuffer(v) => v.len() * 4 / 1024,
            NeighborMode::OnTheFly => 0,
        };

        let t_hs = Instant::now();
        let lat = hotspring_barracuda::lattice::wilson::Lattice::cold_start(
            [
                dims[0] as usize,
                dims[1] as usize,
                dims[2] as usize,
                dims[3] as usize,
            ],
            6.0,
        );
        let hs_table = hotspring_barracuda::lattice::gpu_hmc::build_neighbors(&lat);
        let us_hs = t_hs.elapsed().as_micros();

        println!(
            "  {}^4 (vol={vol:>6}): toadStool={us:>5}us, hotSpring={us_hs:>5}us, table={table_kb}KB, entries={}",
            dims[0],
            hs_table.len()
        );
    }

    // Verify hotSpring table self-consistency (fwd(bwd(s)) == s)
    let lat = hotspring_barracuda::lattice::wilson::Lattice::cold_start([4, 4, 4, 4], 6.0);
    let nbr = hotspring_barracuda::lattice::gpu_hmc::build_neighbors(&lat);
    let vol = lat.volume();
    let mut ok = true;
    for s in 0..vol {
        for mu in 0..4 {
            let fwd = nbr[s * 8 + mu * 2] as usize;
            let bwd = nbr[s * 8 + mu * 2 + 1] as usize;
            if nbr[fwd * 8 + mu * 2 + 1] as usize != s {
                ok = false;
            }
            if nbr[bwd * 8 + mu * 2] as usize != s {
                ok = false;
            }
        }
    }
    println!(
        "  Inverse consistency (4^4 hotSpring): {}",
        if ok { "PASS" } else { "FAIL" }
    );
    println!();
}

pub fn bench_fma_precision_routing() {
    println!("═══ Phase 1d: FMA Policy + Precision Tier Routing ═══");
    println!("  Provenance: hotSpring v0.6.25 precision brain → barraCuda Sprint 2");
    println!("  coralReef Iter 30: FmaPolicy::Separate splits fma→mul+add for bit-exact QCD");
    println!("  Cross-spring: all springs benefit from domain-aware precision routing");
    println!();

    use barracuda::device::fma_policy::{FmaPolicy, domain_requires_separate_fma};
    use barracuda::device::precision_tier::{PhysicsDomain, PrecisionTier};

    let domains = [
        (PhysicsDomain::LatticeQcd, "LatticeQcd"),
        (PhysicsDomain::GradientFlow, "GradientFlow"),
        (PhysicsDomain::NuclearEos, "NuclearEOS"),
        (PhysicsDomain::MolecularDynamics, "MolecularDynamics"),
        (PhysicsDomain::Dielectric, "Dielectric"),
        (PhysicsDomain::KineticFluid, "KineticFluid"),
        (PhysicsDomain::Statistics, "Statistics"),
        (PhysicsDomain::Bioinformatics, "Bioinformatics"),
    ];

    for (domain, label) in &domains {
        let needs_separate = domain_requires_separate_fma(domain);
        let policy = if needs_separate {
            FmaPolicy::Separate
        } else {
            FmaPolicy::Contract
        };
        println!("  {label:20} → FMA={policy}, separate_required={needs_separate}");
    }

    println!();

    let tiers = [
        PrecisionTier::F32,
        PrecisionTier::DF64,
        PrecisionTier::F64,
        PrecisionTier::F64Precise,
    ];

    for tier in &tiers {
        println!(
            "  {tier:12} → {bits} mantissa bits",
            bits = tier.mantissa_bits()
        );
    }

    println!();
}
