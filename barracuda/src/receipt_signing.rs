// SPDX-License-Identifier: AGPL-3.0-or-later

//! Ed25519 receipt signing via bearDog `crypto.sign` JSON-RPC.
//!
//! When bearDog is available (detected via [`NucleusContext`]), signs
//! the JSON receipt with Ed25519 and writes a detached `.sig` file.
//! When bearDog is absent, the receipt is written unsigned.

use crate::primal_bridge::NucleusContext;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Signature metadata embedded in a signed receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiptSignature {
    /// Hex-encoded Ed25519 detached signature over the canonical receipt JSON.
    pub signature_hex: String,
    /// Hex-encoded Ed25519 public key of the signer.
    pub signer_public_key: String,
    /// Algorithm identifier (`"Ed25519"`).
    pub algorithm: String,
}

/// Result of a signing attempt.
pub enum SignResult {
    Signed(ReceiptSignature),
    Unavailable,
    Failed(String),
}

/// Sign `receipt_json` via bearDog and write detached sig alongside `receipt_path`.
///
/// Returns [`SignResult::Unavailable`] if bearDog is not in the NUCLEUS context.
/// Returns [`SignResult::Failed`] on IPC error (non-fatal — receipt is still valid).
pub fn sign_receipt(
    nucleus: &NucleusContext,
    receipt_json: &str,
    receipt_path: &Path,
) -> SignResult {
    match nucleus.get_by_capability("crypto") {
        Some(ep) if ep.alive => {}
        _ => return SignResult::Unavailable,
    }

    let params = serde_json::json!({
        "payload": receipt_json,
        "encoding": "utf8",
        "algorithm": "Ed25519",
    });

    let resp = match nucleus.call_by_capability("crypto", "crypto.sign", params) {
        Ok(r) => r,
        Err(e) => return SignResult::Failed(format!("bearDog crypto.sign: {e}")),
    };

    let Some(result) = resp.get("result") else {
        let err = resp
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("no result");
        return SignResult::Failed(format!("bearDog error: {err}"));
    };

    let sig_hex = result
        .get("signature")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();
    let pubkey_hex = result
        .get("public_key")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();

    if sig_hex.is_empty() || pubkey_hex.is_empty() {
        return SignResult::Failed("bearDog returned empty signature or key".into());
    }

    let sig_path = receipt_path.with_extension("sig");
    if let Err(e) = std::fs::write(&sig_path, &sig_hex) {
        eprintln!("  warning: could not write {}: {e}", sig_path.display());
    }

    SignResult::Signed(ReceiptSignature {
        signature_hex: sig_hex,
        signer_public_key: pubkey_hex,
        algorithm: "Ed25519".to_string(),
    })
}

/// Convenience: attempt signing and embed the signature into a `serde_json::Value` receipt.
///
/// If signing succeeds, adds `"signature"`, `"signer_public_key"`, and
/// `"signature_algorithm"` fields to the receipt value. Prints status.
pub fn sign_and_embed(
    nucleus: &NucleusContext,
    receipt: &mut serde_json::Value,
    receipt_path: &Path,
) {
    let receipt_str = serde_json::to_string_pretty(receipt).unwrap_or_default();

    match sign_receipt(nucleus, &receipt_str, receipt_path) {
        SignResult::Signed(sig) => {
            if let Some(obj) = receipt.as_object_mut() {
                obj.insert(
                    "signature".to_string(),
                    serde_json::Value::String(sig.signature_hex),
                );
                obj.insert(
                    "signer_public_key".to_string(),
                    serde_json::Value::String(sig.signer_public_key),
                );
                obj.insert(
                    "signature_algorithm".to_string(),
                    serde_json::Value::String(sig.algorithm),
                );
            }
            println!("  bearDog: receipt signed (Ed25519)");
        }
        SignResult::Unavailable => {}
        SignResult::Failed(e) => {
            eprintln!("  bearDog: signing failed — {e}");
        }
    }
}
