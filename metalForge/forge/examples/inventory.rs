// SPDX-License-Identifier: AGPL-3.0-only

//! Discover and print all compute substrates on this machine.
//!
//! GPU discovery uses the same wgpu path that toadstool/barracuda uses.
//! NPU and CPU discovery are local probes.

fn main() {
    let substrates = hotspring_forge::inventory::discover();
    hotspring_forge::inventory::print_inventory(&substrates);

    println!();
    println!("Dispatch examples:");
    println!();

    use hotspring_forge::dispatch::{self, Workload};
    use hotspring_forge::substrate::Capability;

    let workloads = [
        Workload::new(
            "MD force kernel",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        ),
        Workload::new(
            "CG solver",
            vec![Capability::F64Compute, Capability::ConjugateGradient],
        ),
        Workload::new(
            "Phase classifier",
            vec![Capability::QuantizedInference { bits: 8 }],
        ),
        Workload::new(
            "Eigensolve",
            vec![Capability::F64Compute, Capability::Eigensolve],
        ),
        Workload::new(
            "Validation",
            vec![Capability::F64Compute, Capability::SimdVector],
        ),
    ];

    for work in &workloads {
        match dispatch::route(work, &substrates) {
            Some(d) => println!("  {:25} → {} ({:?})", work.name, d.substrate, d.reason),
            None => println!("  {:25} → NO CAPABLE SUBSTRATE", work.name),
        }
    }
}
