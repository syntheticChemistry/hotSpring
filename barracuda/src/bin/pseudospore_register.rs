// SPDX-License-Identifier: AGPL-3.0-or-later

//! pseudoSpore registrar — pushes signed bundle to ironGate + westGate.
//!
//! Final pipeline stage (strandGate's side):
//!   1. Ingest .tar.gz into westGate CAS (content.ingest)
//!   2. Register NFT on ironGate (nft.register) with CAS address + signature
//!   3. Report public URL for sporePrint to serve
//!
//! Until ironGate's nft.register endpoint is deployed, this binary
//! operates in --dry-run mode (shows what would be sent).
//!
//! Usage:
//!   pseudospore_register [--dry-run] [--bundle path.tar.gz]
//!
//! Convergence pattern:
//!   strandGate (this binary) → westGate CAS + ironGate NFT → sporePrint
//!   This is the abstract "publish" step for ANY spring producing data.

use serde::Serialize;
use std::path::PathBuf;

fn default_bundle_path() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("hotspring/pseudospore_hotspring-qcd-sun_v1.0.0-rung1.tar.gz")
}

fn default_manifest_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("hotspring/pseudospore_hotspring-qcd-sun_v1.0.0-rung1")
}

#[derive(Serialize)]
struct ContentIngestRequest {
    method: String,
    path: String,
    metadata: IngestMetadata,
}

#[derive(Serialize)]
struct IngestMetadata {
    artifact_type: String,
    name: String,
    version: String,
    origin_gate: String,
}

#[derive(Serialize)]
struct NftRegisterRequest {
    method: String,
    artifact_type: String,
    name: String,
    version: String,
    content_blake3: String,
    scope_blake3: String,
    signature: String,
    cas_address: String,
    origin_gate: String,
}

fn blake3_file(path: &std::path::Path) -> Option<String> {
    std::fs::read(path).ok().map(|data| blake3::hash(&data).to_hex().to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dry_run = args.iter().any(|a| a == "--dry-run");

    let bundle_path = args.iter().position(|a| a == "--bundle")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(default_bundle_path);

    let manifest_dir = default_manifest_dir();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  pseudoSpore Registrar — ironGate + westGate                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    if dry_run {
        println!("  MODE: dry-run (generates request payloads, no IPC)");
    } else {
        println!("  MODE: live (will contact westGate + ironGate)");
    }
    println!("  Bundle: {:?}", bundle_path);
    println!();

    // Check prerequisites
    if !bundle_path.exists() {
        eprintln!("  ERROR: Bundle not found: {:?}", bundle_path);
        eprintln!("  Run pseudospore_bundle first.");
        std::process::exit(1);
    }

    let signature_path = manifest_dir.join("provenance/signature.ed25519");
    let has_signature = signature_path.exists();

    // Compute hashes
    let bundle_blake3 = blake3_file(&bundle_path).unwrap_or_else(|| "ERROR".to_string());
    let scope_blake3 = blake3_file(&manifest_dir.join("scope.toml")).unwrap_or_else(|| "ERROR".to_string());

    println!("  Bundle BLAKE3:  {}...", &bundle_blake3[..24]);
    println!("  Scope BLAKE3:   {}...", &scope_blake3[..24]);
    println!("  Signature:      {}", if has_signature { "present" } else { "MISSING (unsigned)" });
    println!();

    // Load signature if present
    let signature_b64 = if has_signature {
        std::fs::read_to_string(&signature_path)
            .ok()
            .and_then(|s| {
                serde_json::from_str::<serde_json::Value>(&s).ok()
                    .and_then(|v| v.get("signature").and_then(|s| s.as_str().map(String::from)))
            })
            .unwrap_or_else(|| "unsigned".to_string())
    } else {
        "unsigned".to_string()
    };

    // Stage 1: westGate content.ingest
    println!("  ═══ Stage 1: westGate CAS Ingest ═══");
    let ingest_request = ContentIngestRequest {
        method: "content.ingest".to_string(),
        path: bundle_path.to_string_lossy().to_string(),
        metadata: IngestMetadata {
            artifact_type: "pseudoSpore".to_string(),
            name: "hotspring-qcd-sun".to_string(),
            version: "1.0.0-rung1".to_string(),
            origin_gate: "strandGate".to_string(),
        },
    };

    let ingest_json = serde_json::to_string_pretty(&ingest_request).unwrap();

    if dry_run {
        println!("  Would send to westGate (TCP :7800 or biomeOS capability.call):");
        println!();
        for line in ingest_json.lines() {
            println!("    {}", line);
        }
        println!();
        println!("  Expected response: cas_hash = blake3:{}", &bundle_blake3[..32]);
    } else {
        // TODO: Wire to westGate IPC
        // Protocol options (in preference order):
        //   1. biomeOS capability.call("content.ingest", payload) — via UDS
        //   2. Direct TCP to westGate :7800 (songBird mesh address)
        //   3. Fallback: rsync + trigger (legacy, not preferred)
        eprintln!("  westGate content.ingest IPC not yet wired from strandGate.");
        eprintln!("  Use --dry-run or wire via biomeOS capability.call.");
    }

    println!();

    // Stage 2: ironGate nft.register
    println!("  ═══ Stage 2: ironGate NFT Registration ═══");
    let nft_request = NftRegisterRequest {
        method: "nft.register".to_string(),
        artifact_type: "pseudoSpore".to_string(),
        name: "hotspring-qcd-sun".to_string(),
        version: "1.0.0-rung1".to_string(),
        content_blake3: bundle_blake3.clone(),
        scope_blake3,
        signature: signature_b64,
        cas_address: format!("westgate://content/blake3:{}", &bundle_blake3[..32]),
        origin_gate: "strandGate".to_string(),
    };

    let nft_json = serde_json::to_string_pretty(&nft_request).unwrap();

    if dry_run {
        println!("  Would send to ironGate (nft.register endpoint):");
        println!();
        for line in nft_json.lines() {
            println!("    {}", line);
        }
        println!();
        println!("  Expected response:");
        println!("    nft_id: <uuid>");
        println!("    verification_url: https://nestgate.io/pseudospore/{}", &bundle_blake3[..16]);
    } else {
        eprintln!("  ironGate nft.register endpoint not yet deployed.");
        eprintln!("  Use --dry-run or wait for ironGate team.");
    }

    println!();
    println!("  ═══════════════════════════════════════════════════════════");

    if dry_run {
        // Write request payloads for upstream reference
        let requests_dir = manifest_dir.join("provenance/registration_requests");
        std::fs::create_dir_all(&requests_dir).unwrap();
        std::fs::write(requests_dir.join("westgate_content_ingest.json"), &ingest_json).unwrap();
        std::fs::write(requests_dir.join("irongate_nft_register.json"), &nft_json).unwrap();

        let cas_address = format!("westgate://content/blake3:{}", &bundle_blake3[..32]);
        hotspring_barracuda::gossip::pseudospore_registered(
            "hotspring-qcd-sun",
            "1.0.0-rung1",
            "dry-run-pending",
            &cas_address,
        );

        println!("  Request payloads written to: provenance/registration_requests/");
        println!("  Share with ironGate + westGate teams for endpoint wiring.");
        println!("  [gossip: pseudospore.registered (dry-run)]");
    }

    println!();
    println!("  Pipeline complete. Full live path:");
    println!("    compute → manifest → bundle → sign → register → publish");
    println!("  ═══════════════════════════════════════════════════════════");
}
