// SPDX-License-Identifier: AGPL-3.0-or-later

//! NUCLEUS atomic types for metalForge — Tower, Node, Nest.
//!
//! Mirrors the atomic composition from `barracuda::composition` but
//! owned by forge for substrate-level coordination. biomeOS graphs
//! coordinate these atomics across substrates; forge maps them to
//! local hardware.
//!
//! # Hierarchy
//!
//! - **Tower** (BearDog + Songbird): trust boundary + service discovery
//! - **Node** (Tower + ToadStool + barraCuda + coralReef): compute dispatch
//! - **Nest** (Tower + NestGate + rhizoCrypt + loamSpine + sweetGrass): storage + provenance
//! - **FullNucleus**: Tower + Node + Nest (all 9 primals)

use std::fmt;

/// NUCLEUS atomic type — the smallest deployable unit of the ecosystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtomicType {
    /// Trust boundary + discovery (BearDog + Songbird).
    Tower,
    /// Compute dispatch (Tower + ToadStool + barraCuda + coralReef).
    Node,
    /// Storage + provenance (Tower + NestGate + rhizoCrypt + loamSpine + sweetGrass).
    Nest,
    /// Full composition: Tower + Node + Nest.
    FullNucleus,
}

impl AtomicType {
    /// Domains required by this atomic type.
    #[must_use]
    pub const fn required_domains(&self) -> &'static [&'static str] {
        match self {
            Self::Tower => &["crypto", "discovery"],
            Self::Node => &["crypto", "discovery", "compute", "math", "shader"],
            Self::Nest => &[
                "crypto",
                "discovery",
                "storage",
                "dag",
                "ledger",
                "attribution",
            ],
            Self::FullNucleus => &[
                "crypto",
                "discovery",
                "compute",
                "math",
                "shader",
                "storage",
                "dag",
                "ledger",
                "attribution",
            ],
        }
    }

    /// Which substrate kinds can host this atomic.
    ///
    /// Tower can run anywhere (crypto + discovery are lightweight).
    /// Node needs GPU or CPU compute capability.
    /// Nest needs storage I/O (CPU or dedicated storage substrate).
    #[must_use]
    pub fn compatible_substrates(&self) -> &'static [crate::substrate::SubstrateKind] {
        use crate::substrate::SubstrateKind;
        match self {
            Self::Tower => &[SubstrateKind::Cpu, SubstrateKind::Gpu, SubstrateKind::Npu],
            Self::Node => &[SubstrateKind::Gpu, SubstrateKind::Cpu],
            Self::Nest => &[SubstrateKind::Cpu],
            Self::FullNucleus => &[SubstrateKind::Cpu],
        }
    }

    /// True if this atomic's required domains are a subset of `other`'s.
    #[must_use]
    pub fn is_subset_of(&self, other: &Self) -> bool {
        let other_domains = other.required_domains();
        self.required_domains()
            .iter()
            .all(|d| other_domains.contains(d))
    }
}

impl fmt::Display for AtomicType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tower => write!(f, "Tower (electron: trust + discovery)"),
            Self::Node => write!(f, "Node (proton: compute dispatch)"),
            Self::Nest => write!(f, "Nest (neutron: storage + provenance)"),
            Self::FullNucleus => write!(f, "FullNucleus (Tower + Node + Nest)"),
        }
    }
}

/// Mapping from an atomic type to a substrate for deployment.
#[derive(Debug, Clone)]
pub struct AtomicBinding<'a> {
    /// Which atomic is being bound.
    pub atomic: AtomicType,
    /// Which substrate hosts it.
    pub substrate: &'a crate::substrate::Substrate,
    /// Domains satisfied by this binding.
    pub domains_satisfied: Vec<&'static str>,
}

impl<'a> AtomicBinding<'a> {
    /// Create a binding, checking compatibility.
    ///
    /// Returns `None` if the substrate kind is not compatible with
    /// the atomic type.
    #[must_use]
    pub fn bind(atomic: AtomicType, substrate: &'a crate::substrate::Substrate) -> Option<Self> {
        if !atomic.compatible_substrates().contains(&substrate.kind) {
            return None;
        }
        Some(Self {
            atomic,
            substrate,
            domains_satisfied: atomic.required_domains().to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::substrate::{Identity, Properties, Substrate, SubstrateKind};

    fn test_gpu() -> Substrate {
        Substrate {
            kind: SubstrateKind::Gpu,
            identity: Identity::named("Test GPU"),
            properties: Properties::default(),
            capabilities: vec![],
        }
    }

    fn test_cpu() -> Substrate {
        Substrate {
            kind: SubstrateKind::Cpu,
            identity: Identity::named("Test CPU"),
            properties: Properties::default(),
            capabilities: vec![],
        }
    }

    #[test]
    fn tower_domains() {
        assert_eq!(AtomicType::Tower.required_domains(), &["crypto", "discovery"]);
    }

    #[test]
    fn tower_subset_of_node() {
        assert!(AtomicType::Tower.is_subset_of(&AtomicType::Node));
        assert!(AtomicType::Tower.is_subset_of(&AtomicType::FullNucleus));
    }

    #[test]
    fn node_not_subset_of_tower() {
        assert!(!AtomicType::Node.is_subset_of(&AtomicType::Tower));
    }

    #[test]
    fn node_binds_to_gpu() {
        let gpu = test_gpu();
        assert!(AtomicBinding::bind(AtomicType::Node, &gpu).is_some());
    }

    #[test]
    fn nest_rejects_gpu() {
        let gpu = test_gpu();
        assert!(AtomicBinding::bind(AtomicType::Nest, &gpu).is_none());
    }

    #[test]
    fn nest_binds_to_cpu() {
        let cpu = test_cpu();
        assert!(AtomicBinding::bind(AtomicType::Nest, &cpu).is_some());
    }

    #[test]
    fn full_nucleus_contains_all() {
        assert!(AtomicType::Tower.is_subset_of(&AtomicType::FullNucleus));
        assert!(AtomicType::Node.is_subset_of(&AtomicType::FullNucleus));
        assert!(AtomicType::Nest.is_subset_of(&AtomicType::FullNucleus));
    }
}
