// SPDX-License-Identifier: AGPL-3.0-or-later

//! Lockup defense catalog — crash vectors discovered in Exp 229/232.
//!
//! Each vector has a confirmed kill mechanism and a proven defense in the
//! diesel engine (toadStool). This module provides constants and types for
//! validation binaries to verify defenses are active at runtime.

/// Crash vector categories from Exp 232 reprofile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrashCategory {
    ConfirmedKill,
    ConfirmedHang,
    Potential,
    CrossGen,
}

/// A crash vector with its defense mechanism.
#[derive(Debug, Clone)]
pub struct CrashVector {
    pub id: &'static str,
    pub category: CrashCategory,
    pub description: &'static str,
    pub defense: &'static str,
    pub rpc_probe: &'static str,
}

pub const A1_IRQ_STORM: CrashVector = CrashVector {
    id: "A1",
    category: CrashCategory::ConfirmedKill,
    description: "IRQ storm from INTR_EN quench to read-only register",
    defense: "InterruptProfile.disable_offset() — generation-aware INTR_EN disable",
    rpc_probe: "sovereign.defense_status",
};

pub const A2_NVIDIA_CLOSE_INTR: CrashVector = CrashVector {
    id: "A2",
    category: CrashCategory::ConfirmedKill,
    description: "nvidia_close re-enables INTR_EN after quench",
    defense: "pmc::quench_interrupts() + pmc::intx_disable() in post_exit_quench",
    rpc_probe: "sovereign.defense_status",
};

pub const A3_NV_DEV_FREE_STACKS: CrashVector = CrashVector {
    id: "A3",
    category: CrashCategory::ConfirmedKill,
    description: "nv_dev_free_stacks use-after-free",
    defense: "nv_close_device RetAtEntry NOP patch",
    rpc_probe: "sovereign.patch_status",
};

pub const A4_NV_PCI_REMOVE_HANG: CrashVector = CrashVector {
    id: "A4",
    category: CrashCategory::ConfirmedKill,
    description: "nv_pci_remove os_delay hang (infinite sleep)",
    defense: "nv_pci_remove RetAtEntry NOP patch",
    rpc_probe: "sovereign.patch_status",
};

pub const A5_PCI_LOCK_DEADLOCK: CrashVector = CrashVector {
    id: "A5",
    category: CrashCategory::ConfirmedKill,
    description: "pci_lock deadlock during bridge keepalive",
    defense: "HandoffExclusionGuard — mutual exclusion with keepalive thread",
    rpc_probe: "sovereign.defense_status",
};

pub const A6_IRQ_DOMAIN_REMOVE: CrashVector = CrashVector {
    id: "A6",
    category: CrashCategory::ConfirmedKill,
    description: "irq_domain_remove crash from cleanup_module NOP (REVERTED)",
    defense: "DO NOT NOP cleanup_module — let it run with post-exit interrupt quench",
    rpc_probe: "sovereign.defense_status",
};

pub const B1_VFIO_ANCHOR_LEAK: CrashVector = CrashVector {
    id: "B1",
    category: CrashCategory::ConfirmedHang,
    description: "VFIO anchor session leak prevents re-bind",
    defense: "Ember immortal fd + preflight anchor probe",
    rpc_probe: "ember.status",
};

pub const B2_NVSOV_ZOMBIE: CrashVector = CrashVector {
    id: "B2",
    category: CrashCategory::ConfirmedHang,
    description: "nvsov module zombie (refcount -1)",
    defense: "Preflight halt on zombie detection",
    rpc_probe: "sovereign.preflight",
};

pub const B3_DRIVER_OVERRIDE_HANG: CrashVector = CrashVector {
    id: "B3",
    category: CrashCategory::ConfirmedHang,
    description: "driver_override sysfs write hangs in D-state GPU",
    defense: "fire-and-poll unbind with 330s deadline",
    rpc_probe: "sovereign.defense_status",
};

pub const ALL_VECTORS: &[&CrashVector] = &[
    &A1_IRQ_STORM,
    &A2_NVIDIA_CLOSE_INTR,
    &A3_NV_DEV_FREE_STACKS,
    &A4_NV_PCI_REMOVE_HANG,
    &A5_PCI_LOCK_DEADLOCK,
    &A6_IRQ_DOMAIN_REMOVE,
    &B1_VFIO_ANCHOR_LEAK,
    &B2_NVSOV_ZOMBIE,
    &B3_DRIVER_OVERRIDE_HANG,
];

pub const DEFENSE_MECHANISMS: &[&str] = &[
    "interrupt_quench",
    "post_exit_quench",
    "exclusion_guard",
    "fire_and_poll_unbind",
    "kernel_sentinel",
];
