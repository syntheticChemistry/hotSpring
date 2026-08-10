// SPDX-License-Identifier: AGPL-3.0-or-later

//! pseudoSpore validator — verifies BLAKE3 integrity of a bundle.
//!
//! This binary ships INSIDE the pseudoSpore bundle. A reviewer runs it to:
//!   1. Verify all data files match their BLAKE3 checksums
//!   2. Verify the scope.toml is well-formed
//!   3. Verify the environment.toml is present
//!   4. Report any integrity failures
//!
//! Future: Ed25519 signature verification (requires bearDog public key)
//!
//! Usage:
//!   pseudospore_validate [path_to_bundle_dir]
//!   pseudospore_validate --check-only  (exit 0 if valid, 1 if not)

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn blake3_file(path: &Path) -> Option<String> {
    std::fs::read(path).ok().map(|data| blake3::hash(&data).to_hex().to_string())
}

fn load_checksums(path: &Path) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    if let Ok(content) = std::fs::read_to_string(path) {
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((hash, file)) = line.split_once("  ") {
                map.insert(file.to_string(), hash.to_string());
            }
        }
    }
    map
}

struct ValidationResult {
    total_files: usize,
    verified: usize,
    missing: Vec<String>,
    corrupted: Vec<(String, String, String)>,
    scope_valid: bool,
    environment_valid: bool,
    validation_json_valid: bool,
}

fn validate_bundle(bundle_dir: &Path, data_dir: Option<&Path>) -> ValidationResult {
    let checksums_path = bundle_dir.join("receipts/checksums.blake3");
    let checksums = load_checksums(&checksums_path);

    let actual_data_dir = data_dir.unwrap_or_else(|| {
        // Look for data in sibling production_v2 or in bundle/data/
        if bundle_dir.join("data").exists() {
            return bundle_dir.join("data").leak();
        }
        let prod = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("hotspring/production_v2");
        prod.leak()
    });

    let mut verified = 0;
    let mut missing = Vec::new();
    let mut corrupted = Vec::new();

    for (file, expected_hash) in &checksums {
        let file_path = actual_data_dir.join(file);
        if !file_path.exists() {
            missing.push(file.clone());
            continue;
        }
        match blake3_file(&file_path) {
            Some(actual_hash) => {
                if actual_hash == *expected_hash {
                    verified += 1;
                } else {
                    corrupted.push((file.clone(), expected_hash.clone(), actual_hash));
                }
            }
            None => missing.push(file.clone()),
        }
    }

    let scope_valid = bundle_dir.join("scope.toml").exists()
        && std::fs::read_to_string(bundle_dir.join("scope.toml"))
            .map(|s| s.contains("[artifact]") && s.contains("pseudoSpore"))
            .unwrap_or(false);

    let environment_valid = bundle_dir.join("receipts/environment.toml").exists();
    let validation_json_valid = bundle_dir.join("validation.json").exists();

    ValidationResult {
        total_files: checksums.len(),
        verified,
        missing,
        corrupted,
        scope_valid,
        environment_valid,
        validation_json_valid,
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let check_only = args.iter().any(|a| a == "--check-only");

    let bundle_dir = args.iter()
        .find(|a| !a.starts_with('-') && *a != &args[0])
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("hotspring/pseudospore_hotspring-qcd-sun_v1.0.0-rung1")
        });

    if !check_only {
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║  pseudoSpore Validator — hotspring-qcd-sun                  ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!();
        println!("  Bundle: {:?}", bundle_dir);
        println!();
    }

    if !bundle_dir.exists() {
        if !check_only {
            eprintln!("  ERROR: Bundle directory does not exist: {:?}", bundle_dir);
        }
        return ExitCode::from(1);
    }

    let result = validate_bundle(&bundle_dir, None);

    if check_only {
        if result.corrupted.is_empty() && result.missing.is_empty()
            && result.scope_valid && result.environment_valid
        {
            return ExitCode::SUCCESS;
        } else {
            return ExitCode::from(1);
        }
    }

    // Structure checks
    println!("  Structure:");
    let s = |b: bool| if b { "✓" } else { "✗" };
    println!("    [{}] scope.toml", s(result.scope_valid));
    println!("    [{}] receipts/environment.toml", s(result.environment_valid));
    println!("    [{}] validation.json", s(result.validation_json_valid));
    println!("    [{}] receipts/checksums.blake3 ({} entries)",
             s(result.total_files > 0), result.total_files);
    println!();

    // Integrity checks
    println!("  BLAKE3 Integrity:");
    println!("    Verified: {}/{}", result.verified, result.total_files);

    if !result.missing.is_empty() {
        println!("    Missing ({}):", result.missing.len());
        for f in &result.missing[..result.missing.len().min(10)] {
            println!("      - {}", f);
        }
        if result.missing.len() > 10 {
            println!("      ... and {} more", result.missing.len() - 10);
        }
    }

    if !result.corrupted.is_empty() {
        println!("    CORRUPTED ({}):", result.corrupted.len());
        for (file, expected, actual) in &result.corrupted {
            println!("      - {}", file);
            println!("        expected: {}", &expected[..16]);
            println!("        actual:   {}", &actual[..16]);
        }
    }

    println!();

    let all_pass = result.corrupted.is_empty()
        && result.missing.is_empty()
        && result.scope_valid
        && result.environment_valid;

    if all_pass {
        println!("  ═══ RESULT: PASS ═══");
        println!("  All {} data files verified. Provenance intact.", result.verified);
        ExitCode::SUCCESS
    } else {
        println!("  ═══ RESULT: FAIL ═══");
        if !result.corrupted.is_empty() {
            println!("  {} files corrupted — data integrity compromised.", result.corrupted.len());
        }
        if !result.missing.is_empty() {
            println!("  {} files missing — bundle incomplete.", result.missing.len());
        }
        ExitCode::from(1)
    }
}
