// SPDX-License-Identifier: AGPL-3.0-only

//! Hardware inventory — assemble probed substrates.
//!
//! Runs GPU probing (via wgpu/barracuda), CPU probing (procfs), and
//! NPU probing (local device nodes). Returns every substrate discovered
//! on this machine. If it's not in the inventory, it doesn't exist to us.

use crate::probe;
use crate::substrate::{Substrate, SubstrateKind};

/// Discover all compute substrates on this machine.
///
/// Always returns at least one entry (CPU). GPUs come from wgpu adapter
/// enumeration. NPUs come from device node probing.
#[must_use]
pub fn discover() -> Vec<Substrate> {
    let mut substrates = Vec::new();

    substrates.push(probe::probe_cpu());
    substrates.extend(probe::probe_gpus());
    substrates.extend(probe::probe_npus());

    substrates
}

/// Print a human-readable inventory to stdout.
pub fn print_inventory(substrates: &[Substrate]) {
    println!("┌──────────────────────────────────────────────┐");
    println!("│  hotSpring Forge — Hardware Inventory         │");
    println!("├──────────────────────────────────────────────┤");

    for (i, s) in substrates.iter().enumerate() {
        println!("│ {i}: {s}");

        if let Some(ref backend) = s.identity.backend {
            println!("│    backend: {backend}");
        }
        if let Some(idx) = s.identity.adapter_index {
            println!("│    adapter: {idx}");
        }
        if let Some(ref node) = s.identity.device_node {
            println!("│    device:  {node}");
        }
        if let Some(cores) = s.properties.core_count {
            let threads = s.properties.thread_count.unwrap_or(cores);
            println!("│    cores:   {cores} ({threads} threads)");
        }
        if s.properties.has_f64 {
            println!("│    SHADER_F64: YES");
        }

        println!("│    caps:    {}", s.capability_summary());
        println!("│");
    }

    let gpu_count = substrates
        .iter()
        .filter(|s| s.kind == SubstrateKind::Gpu)
        .count();
    let npu_count = substrates
        .iter()
        .filter(|s| s.kind == SubstrateKind::Npu)
        .count();
    println!("│  Total: 1 CPU, {gpu_count} GPU(s), {npu_count} NPU(s)");
    println!("└──────────────────────────────────────────────┘");
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn discover_always_has_cpu() {
        let subs = discover();
        assert!(subs.iter().any(|s| s.kind == SubstrateKind::Cpu));
    }

    #[test]
    fn discover_finds_gpus_via_wgpu() {
        let subs = discover();
        let gpus: Vec<_> = subs
            .iter()
            .filter(|s| s.kind == SubstrateKind::Gpu)
            .collect();
        for gpu in &gpus {
            assert!(gpu.has(&crate::substrate::Capability::ShaderDispatch));
            assert!(gpu.identity.adapter_index.is_some());
        }
    }

    #[test]
    fn inventory_has_one_cpu() {
        let subs = discover();
        let cpu_count = subs.iter().filter(|s| s.kind == SubstrateKind::Cpu).count();
        assert_eq!(cpu_count, 1);
    }
}
