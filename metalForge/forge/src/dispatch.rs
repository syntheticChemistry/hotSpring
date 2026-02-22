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

/// Predefined workload profiles for hotSpring physics.
///
/// These encode the capability requirements for each physics domain.
/// When toadstool absorbs forge's dispatch, these profiles document
/// what each biome needs from the substrate layer.
pub mod profiles {
    use super::Workload;
    use crate::substrate::{Capability, SubstrateKind};

    /// Yukawa OCP molecular dynamics (GPU f64 force + reduce + cell-list).
    #[must_use]
    pub fn md_force() -> Workload {
        Workload::new(
            "MD force kernel",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
    }

    /// HFB nuclear structure (GPU batched eigensolve).
    #[must_use]
    pub fn hfb_eigensolve() -> Workload {
        Workload::new(
            "HFB eigensolve",
            vec![Capability::F64Compute, Capability::Eigensolve],
        )
    }

    /// Lattice QCD CG solver (GPU Dirac + dot product + axpy).
    #[must_use]
    pub fn lattice_cg() -> Workload {
        Workload::new(
            "Lattice CG solver",
            vec![Capability::F64Compute, Capability::ConjugateGradient],
        )
    }

    /// ESN transport prediction on NPU (quantized inference).
    #[must_use]
    pub fn esn_npu_inference() -> Workload {
        Workload::new(
            "ESN NPU inference",
            vec![Capability::QuantizedInference { bits: 8 }],
        )
        .prefer(SubstrateKind::Npu)
    }

    /// ESN transport prediction on GPU (f32 reservoir + readout).
    #[must_use]
    pub fn esn_gpu_inference() -> Workload {
        Workload::new(
            "ESN GPU inference",
            vec![Capability::F32Compute, Capability::ShaderDispatch],
        )
        .prefer(SubstrateKind::Gpu)
    }

    /// CPU validation (f64 reference, no GPU required).
    #[must_use]
    pub fn cpu_validation() -> Workload {
        Workload::new("CPU validation", vec![Capability::F64Compute]).prefer(SubstrateKind::Cpu)
    }

    /// SpMV spectral theory (GPU sparse matrix-vector).
    #[must_use]
    pub fn spectral_spmv() -> Workload {
        Workload::new(
            "Spectral SpMV",
            vec![Capability::F64Compute, Capability::SparseSpMV],
        )
    }

    /// Heterogeneous pipeline: GPU compute + NPU inference + CPU steering.
    /// Returns the NPU workload — GPU and CPU are handled by separate routes.
    #[must_use]
    pub fn hetero_npu_phase_classifier() -> Workload {
        Workload::new(
            "Phase classifier (hetero)",
            vec![Capability::QuantizedInference { bits: 4 }],
        )
        .prefer(SubstrateKind::Npu)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::profiles;
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
    fn profile_md_force_routes_to_gpu() {
        let gpu = make_gpu(
            "RTX 4070",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        );
        let cpu = make_cpu();
        let subs = [gpu, cpu];
        let work = profiles::md_force();
        let d = route(&work, &subs).expect("MD force should route");
        assert_eq!(d.substrate.kind, SubstrateKind::Gpu);
    }

    #[test]
    fn profile_cpu_validation_prefers_cpu() {
        let gpu = make_gpu("GPU", vec![Capability::F64Compute]);
        let cpu = make_cpu();
        let subs = [gpu, cpu];
        let work = profiles::cpu_validation();
        let d = route(&work, &subs).expect("validation should route");
        assert_eq!(d.substrate.kind, SubstrateKind::Cpu);
        assert_eq!(d.reason, Reason::Preferred);
    }

    #[test]
    fn profile_esn_npu_falls_back_to_cpu_without_npu() {
        let subs = [make_cpu()];
        let work = profiles::esn_npu_inference();
        assert!(route(&work, &subs).is_none(), "no NPU capability on CPU");
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
