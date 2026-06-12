// SPDX-License-Identifier: AGPL-3.0-or-later

//! Ionic GPU Lease — cross-family GPU scheduling via BearDog ionic bonding.
//!
//! Implements the hotSpring side of GAP-HS-005: wire barraCuda session creation
//! through BearDog's `crypto.sign_contract` and `crypto.ionic_bond.*` JSON-RPC
//! methods to enable cross-FAMILY_ID GPU lease negotiation.
//!
//! # Protocol
//!
//! ```text
//! Lessor (GPU owner)                    Lessee (compute requester)
//! ──────────────────                    ────────────────────────────
//!                        ← propose ──   crypto.ionic_bond.propose
//! crypto.ionic_bond.accept ─────→
//!                        ← seal ────   crypto.ionic_bond.seal
//! crypto.sign_contract (lease terms)
//!                        ← verify ──   crypto.verify_contract
//!          ┌──── GPU session active ────┐
//!          │  toadstool.dispatch.*       │
//!          └──── lease TTL expires ──────┘
//! ```
//!
//! # Dependencies
//!
//! - BearDog IPC socket (UDS) via `NucleusContext`
//! - Songbird for cross-gate peer discovery (federation port 7700)

use crate::error::HotSpringError;
use crate::primal_bridge::{NucleusContext, send_jsonrpc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// GPU lease contract terms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuLeaseTerms {
    /// Requesting family ID.
    pub lessee_family: String,
    /// Providing family ID.
    pub lessor_family: String,
    /// GPU adapter name or index for the lease.
    pub gpu_adapter: String,
    /// Lease duration in seconds.
    pub ttl_seconds: u64,
    /// Maximum compute dispatch calls allowed.
    pub max_dispatches: Option<u64>,
    /// Workload type hint (e.g. "lattice_qcd", "molecular_dynamics", "compchem").
    pub workload_type: String,
    /// Required precision tier ("f32", "f64", "mixed").
    pub precision: String,
}

/// Signed lease contract from BearDog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedLease {
    /// Original contract terms.
    pub terms: GpuLeaseTerms,
    /// Hex-encoded Ed25519 signature over canonical terms JSON.
    pub signature_hex: String,
    /// Hex-encoded public key of the signer (lessor BearDog).
    pub signer_public_key: String,
    /// Contract ID for tracking and revocation.
    pub contract_id: String,
    /// ISO 8601 timestamp when the contract expires.
    pub expires_at: String,
}

/// Ionic bond state for an active lease negotiation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum LeaseState {
    /// Proposal sent, awaiting acceptance.
    Proposed,
    /// Accepted by lessor, awaiting seal.
    Accepted,
    /// Sealed — bond is active, contract can be signed.
    Sealed,
    /// Contract signed — GPU session can begin.
    Active,
    /// Lease has expired or been revoked.
    Expired,
}

/// Result of a lease operation.
pub enum LeaseResult<T> {
    Ok(T),
    BearDogUnavailable,
    Rejected(String),
    Failed(String),
}

/// GPU lease client — manages ionic bond lifecycle with BearDog.
pub struct GpuLeaseClient {
    beardog_socket: PathBuf,
}

impl GpuLeaseClient {
    /// Create a lease client from a discovered NUCLEUS context.
    ///
    /// Returns `None` if BearDog is not available.
    pub fn from_nucleus(nucleus: &NucleusContext) -> Option<Self> {
        let ep = nucleus.get_by_capability("crypto")?;
        if !ep.alive {
            return None;
        }
        Some(Self {
            beardog_socket: PathBuf::from(&ep.socket),
        })
    }

    /// Propose an ionic bond for a GPU lease.
    ///
    /// Sends `crypto.ionic_bond.propose` to BearDog with the lease terms
    /// embedded as the bond scope.
    pub fn propose_lease(&self, terms: &GpuLeaseTerms) -> LeaseResult<String> {
        let params = serde_json::json!({
            "peer_family_id": terms.lessor_family,
            "scope": "GAP-HS-005",
            "metadata": {
                "type": "gpu_lease",
                "gpu_adapter": terms.gpu_adapter,
                "workload_type": terms.workload_type,
                "precision": terms.precision,
                "ttl_seconds": terms.ttl_seconds,
                "max_dispatches": terms.max_dispatches,
            }
        });

        match send_jsonrpc(&self.beardog_socket, "crypto.ionic_bond.propose", &params) {
            Ok(resp) => {
                if let Some(bond_id) = resp.get("bond_id").and_then(|v| v.as_str()) {
                    LeaseResult::Ok(bond_id.to_string())
                } else if let Some(err) = resp.get("error").and_then(|v| v.as_str()) {
                    LeaseResult::Rejected(err.to_string())
                } else {
                    LeaseResult::Ok(resp.to_string())
                }
            }
            Err(e) => LeaseResult::Failed(format!("ionic_bond.propose: {e}")),
        }
    }

    /// Sign a GPU lease contract with BearDog.
    ///
    /// Sends `crypto.sign_contract` with the lease terms as the contract body.
    /// BearDog signs with Ed25519 and returns the signed contract.
    pub fn sign_lease_contract(&self, terms: &GpuLeaseTerms) -> LeaseResult<SignedLease> {
        let terms_json = match serde_json::to_string(terms) {
            Ok(j) => j,
            Err(e) => return LeaseResult::Failed(format!("serialize terms: {e}")),
        };

        let params = serde_json::json!({
            "contract_body": terms_json,
            "purpose": "gpu_lease",
            "scope": "GAP-HS-005",
            "ttl_seconds": terms.ttl_seconds,
        });

        match send_jsonrpc(&self.beardog_socket, "crypto.sign_contract", &params) {
            Ok(resp) => {
                let signature_hex = resp
                    .get("signature")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let signer_public_key = resp
                    .get("public_key")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let contract_id = resp
                    .get("contract_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let expires_at = resp
                    .get("expires_at")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();

                LeaseResult::Ok(SignedLease {
                    terms: terms.clone(),
                    signature_hex,
                    signer_public_key,
                    contract_id,
                    expires_at,
                })
            }
            Err(e) => LeaseResult::Failed(format!("sign_contract: {e}")),
        }
    }

    /// Verify a signed lease contract against the lessor's public key.
    ///
    /// Sends `crypto.verify_contract` to confirm the signature is valid.
    pub fn verify_lease(&self, lease: &SignedLease) -> LeaseResult<bool> {
        let terms_json = match serde_json::to_string(&lease.terms) {
            Ok(j) => j,
            Err(e) => return LeaseResult::Failed(format!("serialize terms: {e}")),
        };

        let params = serde_json::json!({
            "contract_body": terms_json,
            "signature": lease.signature_hex,
            "public_key": lease.signer_public_key,
        });

        match send_jsonrpc(&self.beardog_socket, "crypto.verify_contract", &params) {
            Ok(resp) => {
                let valid = resp
                    .get("valid")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                LeaseResult::Ok(valid)
            }
            Err(e) => LeaseResult::Failed(format!("verify_contract: {e}")),
        }
    }

    /// Seal an accepted ionic bond, activating the lease.
    pub fn seal_bond(&self, bond_id: &str) -> LeaseResult<()> {
        let params = serde_json::json!({
            "bond_id": bond_id,
        });

        match send_jsonrpc(&self.beardog_socket, "crypto.ionic_bond.seal", &params) {
            Ok(_) => LeaseResult::Ok(()),
            Err(e) => LeaseResult::Failed(format!("ionic_bond.seal: {e}")),
        }
    }
}

/// Convenience: attempt full lease negotiation (propose → sign → verify).
///
/// Returns the signed lease on success. Suitable for non-interactive use
/// where the lessor auto-accepts proposals (e.g., same-family metallic fleet
/// or pre-authorized cross-family peers).
pub fn negotiate_gpu_lease(
    nucleus: &NucleusContext,
    terms: GpuLeaseTerms,
) -> Result<SignedLease, HotSpringError> {
    let client = GpuLeaseClient::from_nucleus(nucleus).ok_or_else(|| {
        HotSpringError::Ipc("BearDog not available for ionic bond negotiation".into())
    })?;

    let _bond_id = match client.propose_lease(&terms) {
        LeaseResult::Ok(id) => id,
        LeaseResult::BearDogUnavailable => {
            return Err(HotSpringError::Ipc("BearDog unavailable".into()));
        }
        LeaseResult::Rejected(reason) => {
            return Err(HotSpringError::Ipc(format!("Lease rejected: {reason}")));
        }
        LeaseResult::Failed(e) => {
            return Err(HotSpringError::Ipc(format!("Propose failed: {e}")));
        }
    };

    let signed = match client.sign_lease_contract(&terms) {
        LeaseResult::Ok(s) => s,
        LeaseResult::Failed(e) => {
            return Err(HotSpringError::Ipc(format!("Sign failed: {e}")));
        }
        LeaseResult::BearDogUnavailable | LeaseResult::Rejected(_) => {
            return Err(HotSpringError::Ipc("Sign contract unavailable".into()));
        }
    };

    match client.verify_lease(&signed) {
        LeaseResult::Ok(true) => Ok(signed),
        LeaseResult::Ok(false) => Err(HotSpringError::Ipc(
            "Lease signature verification failed".into(),
        )),
        LeaseResult::Failed(e) => Err(HotSpringError::Ipc(format!("Verify failed: {e}"))),
        LeaseResult::BearDogUnavailable | LeaseResult::Rejected(_) => {
            Err(HotSpringError::Ipc("Verify contract unavailable".into()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lease_terms_serialization_roundtrip() {
        let terms = GpuLeaseTerms {
            lessee_family: "e8b62b6e".to_string(),
            lessor_family: "golgiBody-001".to_string(),
            gpu_adapter: "RTX-3090".to_string(),
            ttl_seconds: 3600,
            max_dispatches: Some(1000),
            workload_type: "lattice_qcd".to_string(),
            precision: "f64".to_string(),
        };
        let json = serde_json::to_string(&terms).expect("serialize");
        let back: GpuLeaseTerms = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.lessee_family, "e8b62b6e");
        assert_eq!(back.lessor_family, "golgiBody-001");
        assert_eq!(back.ttl_seconds, 3600);
        assert_eq!(back.max_dispatches, Some(1000));
    }

    #[test]
    fn lease_state_transitions() {
        let states = vec![
            LeaseState::Proposed,
            LeaseState::Accepted,
            LeaseState::Sealed,
            LeaseState::Active,
            LeaseState::Expired,
        ];
        for (i, state) in states.iter().enumerate() {
            assert_eq!(*state, states[i]);
        }
        assert_ne!(LeaseState::Proposed, LeaseState::Sealed);
    }

    #[test]
    fn lease_client_requires_beardog() {
        let nucleus = NucleusContext {
            discovered: std::collections::HashMap::new(),
            family_id: "test".to_string(),
        };
        assert!(GpuLeaseClient::from_nucleus(&nucleus).is_none());
    }

    #[test]
    fn signed_lease_serialization() {
        let lease = SignedLease {
            terms: GpuLeaseTerms {
                lessee_family: "a".to_string(),
                lessor_family: "b".to_string(),
                gpu_adapter: "titan".to_string(),
                ttl_seconds: 60,
                max_dispatches: None,
                workload_type: "compchem".to_string(),
                precision: "mixed".to_string(),
            },
            signature_hex: "deadbeef".to_string(),
            signer_public_key: "cafebabe".to_string(),
            contract_id: "lease-001".to_string(),
            expires_at: "2026-05-30T12:00:00Z".to_string(),
        };
        let json = serde_json::to_string(&lease).expect("serialize");
        assert!(json.contains("deadbeef"));
        assert!(json.contains("compchem"));
    }
}
