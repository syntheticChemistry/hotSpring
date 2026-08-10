// SPDX-License-Identifier: AGPL-3.0-or-later

//! pseudoSpore signer — requests Ed25519 signature from bearDog.
//!
//! This is strandGate's side of the signature workflow:
//!   1. Compute BLAKE3 root hash of receipts/checksums.blake3
//!   2. Connect to bearDog via IPC (UDS or TCP via biomeOS)
//!   3. Request Ed25519 signature over the root hash
//!   4. Write signature to provenance/signature.ed25519
//!   5. Write public key reference for validators
//!
//! Until bearDog's crypto.sign endpoint is deployed, this binary
//! operates in --dry-run mode (generates the request payload without
//! actually contacting bearDog).
//!
//! Usage:
//!   pseudospore_sign [--dry-run] [--bundle-dir path]
//!
//! Convergence pattern:
//!   strandGate (this binary) → bearDog IPC → signed artifact
//!   The same pattern is used by ANY gate/spring needing a signature.

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

fn default_bundle_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("hotspring/pseudospore_hotspring-qcd-sun_v1.0.0-rung1")
}

#[derive(Serialize)]
struct SignRequest {
    method: String,
    payload_blake3: String,
    signer: String,
    artifact_name: String,
    artifact_version: String,
    timestamp: String,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct SignResponse {
    signature: String,
    public_key: String,
    timestamp: String,
}

#[derive(Serialize)]
struct SignatureRecord {
    root_hash: String,
    signature: String,
    public_key: String,
    signer: String,
    timestamp: String,
    algorithm: String,
}

fn compute_root_hash(bundle_dir: &PathBuf) -> Option<String> {
    let checksums_path = bundle_dir.join("receipts/checksums.blake3");
    std::fs::read(&checksums_path).ok().map(|data| blake3::hash(&data).to_hex().to_string())
}

fn current_timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    format!("{}Z", now)
}

fn request_signature_ipc(request: &SignRequest) -> Result<SignResponse, String> {
    // bearDog protocol: JSON-RPC over UDS
    // Method: "crypto.sign_ed25519"
    // Params: { "message": base64(blake3_hash_bytes), "key_id": "spine", "purpose": "pseudospore" }
    // Response: { "signature": base64, "algorithm": "Ed25519", "public_key": base64 }

    let socket_path = std::path::Path::new("/run/beardog/beardog.sock");
    if !socket_path.exists() {
        return Err("bearDog socket not found at /run/beardog/beardog.sock — use --dry-run".to_string());
    }

    use std::io::{Read, Write};
    use std::os::unix::net::UnixStream;

    let message_bytes = hex::decode(&request.payload_blake3)
        .map_err(|e| format!("Failed to decode BLAKE3 hex: {}", e))?;
    let message_b64 = base64::engine::general_purpose::STANDARD.encode(&message_bytes);

    let rpc_request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "crypto.sign_ed25519",
        "params": {
            "message": message_b64,
            "key_id": "spine",
            "purpose": "pseudospore"
        }
    });

    let mut stream = UnixStream::connect(socket_path)
        .map_err(|e| format!("Cannot connect to bearDog: {}", e))?;
    stream.set_read_timeout(Some(std::time::Duration::from_secs(5))).ok();

    let payload = serde_json::to_vec(&rpc_request).unwrap();
    stream.write_all(&payload)
        .map_err(|e| format!("Write to bearDog failed: {}", e))?;
    stream.shutdown(std::net::Shutdown::Write).ok();

    let mut response_buf = Vec::new();
    stream.read_to_end(&mut response_buf)
        .map_err(|e| format!("Read from bearDog failed: {}", e))?;

    let rpc_response: serde_json::Value = serde_json::from_slice(&response_buf)
        .map_err(|e| format!("Invalid JSON from bearDog: {}", e))?;

    if let Some(error) = rpc_response.get("error") {
        return Err(format!("bearDog error: {}", error));
    }

    let result = rpc_response.get("result")
        .ok_or("No result in bearDog response")?;

    Ok(SignResponse {
        signature: result.get("signature").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        public_key: result.get("public_key").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        timestamp: request.timestamp.clone(),
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dry_run = args.iter().any(|a| a == "--dry-run");

    let bundle_dir = args.iter().position(|a| a == "--bundle-dir")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(default_bundle_dir);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  pseudoSpore Signer — bearDog Ed25519                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    if dry_run {
        println!("  MODE: dry-run (generates request payload, no IPC)");
    } else {
        println!("  MODE: live (will contact bearDog)");
    }
    println!("  Bundle: {:?}", bundle_dir);
    println!();

    // Compute root hash
    let root_hash = match compute_root_hash(&bundle_dir) {
        Some(h) => h,
        None => {
            eprintln!("  ERROR: Cannot read receipts/checksums.blake3");
            eprintln!("  Run pseudospore_manifest first.");
            std::process::exit(1);
        }
    };

    println!("  Root BLAKE3: {}", &root_hash[..32]);
    println!();

    // Build request
    let request = SignRequest {
        method: "crypto.sign".to_string(),
        payload_blake3: root_hash.clone(),
        signer: "strandGate".to_string(),
        artifact_name: "hotspring-qcd-sun".to_string(),
        artifact_version: "1.0.0-rung1".to_string(),
        timestamp: current_timestamp(),
    };

    if dry_run {
        // Write the request payload for upstream to see the interface
        let request_json = serde_json::to_string_pretty(&request).unwrap();
        let request_path = bundle_dir.join("provenance/sign_request.json");
        std::fs::create_dir_all(bundle_dir.join("provenance")).unwrap();
        std::fs::write(&request_path, &request_json).unwrap();

        println!("  Request payload written to: provenance/sign_request.json");
        println!();
        println!("  ┌─ crypto.sign request ─────────────────────────────────┐");
        for line in request_json.lines() {
            println!("  │ {:<55} │", line);
        }
        println!("  └────────────────────────────────────────────────────────┘");
        println!();
        println!("  Awaiting bearDog crypto.sign endpoint deployment.");
        println!("  When ready: pseudospore_sign --bundle-dir {:?}", bundle_dir);
        return;
    }

    // Live mode: contact bearDog
    match request_signature_ipc(&request) {
        Ok(response) => {
            let record = SignatureRecord {
                root_hash,
                signature: response.signature.clone(),
                public_key: response.public_key.clone(),
                signer: "strandGate".to_string(),
                timestamp: response.timestamp,
                algorithm: "Ed25519".to_string(),
            };

            let sig_path = bundle_dir.join("provenance/signature.ed25519");
            std::fs::create_dir_all(bundle_dir.join("provenance")).unwrap();
            std::fs::write(&sig_path, serde_json::to_string_pretty(&record).unwrap()).unwrap();

            println!("  ✓ Signature obtained from bearDog");
            println!("  ✓ Written to: provenance/signature.ed25519");
            println!();
            println!("  Next: pseudospore_bundle --include-signature");
            println!("  Then: ironGate nft.register + westGate content.ingest");
        }
        Err(e) => {
            eprintln!("  ERROR: {}", e);
            eprintln!();
            eprintln!("  bearDog is not reachable. Options:");
            eprintln!("    1. Deploy bearDog crypto.sign endpoint");
            eprintln!("    2. Use --dry-run to generate request payload");
            std::process::exit(1);
        }
    }
}
