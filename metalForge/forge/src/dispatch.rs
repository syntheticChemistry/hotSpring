// SPDX-License-Identifier: AGPL-3.0-only

//! Dispatch routing — route workloads to capable substrates.
//!
//! The dispatcher examines the inventory and routes each workload to the
//! substrate that best matches its requirements. This is capability-based:
//! we ask "who can do f64 + CG?" not "send to GPU #0".

use crate::substrate::{Capability, Substrate, SubstrateKind};

/// A workload that needs to be dispatched to a substrate.
#[derive(Debug)]
pub struct Workload {
    pub name: String,
    pub required: Vec<Capability>,
    pub preferred_substrate: Option<SubstrateKind>,
}

/// Dispatch decision — which substrate was chosen and why.
#[derive(Debug)]
pub struct Decision<'a> {
    pub substrate: &'a Substrate,
    pub reason: Reason,
}

/// Why a particular substrate was chosen.
#[derive(Debug, PartialEq, Eq)]
pub enum Reason {
    /// The workload's preferred substrate had all capabilities.
    Preferred,
    /// Best capable substrate by priority (GPU > NPU > CPU).
    BestAvailable,
}

impl Workload {
    /// Create a workload with name and required capabilities.
    #[must_use]
    pub fn new(name: impl Into<String>, required: Vec<Capability>) -> Self {
        Self {
            name: name.into(),
            required,
            preferred_substrate: None,
        }
    }

    /// Set the preferred substrate kind.
    #[must_use]
    pub const fn prefer(mut self, kind: SubstrateKind) -> Self {
        self.preferred_substrate = Some(kind);
        self
    }
}

/// Route a workload to the best matching substrate.
///
/// Selection priority:
/// 1. Preferred substrate (if specified and capable)
/// 2. GPU (for compute-heavy work)
/// 3. NPU (for inference)
/// 4. CPU (fallback)
#[must_use]
pub fn route<'a>(workload: &Workload, substrates: &'a [Substrate]) -> Option<Decision<'a>> {
    let capable: Vec<&Substrate> = substrates
        .iter()
        .filter(|s| workload.required.iter().all(|req| s.has(req)))
        .collect();

    if capable.is_empty() {
        return None;
    }

    if let Some(pref) = workload.preferred_substrate {
        if let Some(s) = capable.iter().find(|s| s.kind == pref) {
            return Some(Decision {
                substrate: s,
                reason: Reason::Preferred,
            });
        }
    }

    let best = capable
        .iter()
        .find(|s| s.kind == SubstrateKind::Gpu)
        .or_else(|| capable.iter().find(|s| s.kind == SubstrateKind::Npu))
        .or_else(|| capable.iter().find(|s| s.kind == SubstrateKind::Cpu))?;

    Some(Decision {
        substrate: best,
        reason: Reason::BestAvailable,
    })
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::substrate::{Identity, Properties};

    fn make_gpu(name: &str, caps: Vec<Capability>) -> Substrate {
        Substrate {
            kind: SubstrateKind::Gpu,
            identity: Identity::named(name),
            properties: Properties::default(),
            capabilities: caps,
        }
    }

    fn make_cpu() -> Substrate {
        Substrate {
            kind: SubstrateKind::Cpu,
            identity: Identity::named("CPU"),
            properties: Properties::default(),
            capabilities: vec![
                Capability::F64Compute,
                Capability::F32Compute,
                Capability::SparseSpMV,
                Capability::Eigensolve,
                Capability::ConjugateGradient,
            ],
        }
    }

    #[test]
    fn routes_to_gpu_for_f64() {
        let gpu = make_gpu(
            "RTX 4070",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        );
        let cpu = make_cpu();
        let subs = [gpu, cpu];
        let work = Workload::new("MD force", vec![Capability::F64Compute]);

        let d = route(&work, &subs).expect("should route");
        assert_eq!(d.substrate.kind, SubstrateKind::Gpu);
        assert_eq!(d.reason, Reason::BestAvailable);
    }

    #[test]
    fn falls_back_to_cpu() {
        let subs = [make_cpu()];
        let work = Workload::new("validation", vec![Capability::F64Compute]);

        let d = route(&work, &subs).expect("should route to CPU");
        assert_eq!(d.substrate.kind, SubstrateKind::Cpu);
    }

    #[test]
    fn no_route_if_incapable() {
        let subs = [make_cpu()];
        let work = Workload::new(
            "NPU inference",
            vec![Capability::QuantizedInference { bits: 4 }],
        );

        assert!(route(&work, &subs).is_none());
    }

    #[test]
    fn respects_preference() {
        let gpu = make_gpu("GPU", vec![Capability::F64Compute]);
        let cpu = make_cpu();
        let subs = [gpu, cpu];
        let work =
            Workload::new("steering", vec![Capability::F64Compute]).prefer(SubstrateKind::Cpu);

        let d = route(&work, &subs).expect("should route");
        assert_eq!(d.substrate.kind, SubstrateKind::Cpu);
        assert_eq!(d.reason, Reason::Preferred);
    }

    #[test]
    fn preference_ignored_if_incapable() {
        let gpu = make_gpu(
            "GPU",
            vec![Capability::F64Compute, Capability::ConjugateGradient],
        );
        let cpu = make_cpu();
        let subs = [gpu, cpu];
        let work = Workload::new(
            "CG solve",
            vec![Capability::F64Compute, Capability::ConjugateGradient],
        )
        .prefer(SubstrateKind::Npu);

        let d = route(&work, &subs).expect("should route");
        assert_eq!(d.substrate.kind, SubstrateKind::Gpu);
        assert_eq!(d.reason, Reason::BestAvailable);
    }
}
