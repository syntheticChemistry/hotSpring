// SPDX-License-Identifier: AGPL-3.0-or-later

//! pseudoSpore bundler — packages production data + manifest into a .tar.gz
//!
//! Creates the final distributable artifact:
//!   pseudospore_hotspring-qcd-sun_v1.0.0-rung1.tar.gz
//!
//! Contents:
//!   pseudospore_hotspring-qcd-sun_v1.0.0-rung1/
//!   ├── scope.toml
//!   ├── validation.json
//!   ├── README.md
//!   ├── receipts/
//!   │   ├── checksums.blake3
//!   │   ├── environment.toml
//!   │   └── compute_log.toml
//!   ├── provenance/
//!   │   └── ferment_transcript.json
//!   ├── data/
//!   │   └── *.json (production time series)
//!   └── configs/
//!       └── *.lat (final lattice configurations)
//!
//! The .lat files go in `configs/` (large, external verification).
//! The .json files go in `data/` (the actual science).
//!
//! Usage:
//!   pseudospore_bundle [--include-configs] [--output path.tar.gz]

use std::io::Write;
use std::path::PathBuf;

fn production_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("hotspring/production_v2")
}

fn manifest_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("hotspring/pseudospore_hotspring-qcd-sun_v1.0.0-rung1")
}

fn default_output() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("hotspring/pseudospore_hotspring-qcd-sun_v1.0.0-rung1.tar.gz")
}

const BUNDLE_PREFIX: &str = "pseudospore_hotspring-qcd-sun_v1.0.0-rung1";

fn generate_readme() -> String {
    format!(r#"# pseudoSpore: hotspring-qcd-sun v1.0.0-rung1

## What This Is

SU(3) pure gauge lattice QCD production data generated on consumer GPUs
using WebGPU/Vulkan with DF64 (emulated double precision on FP32 hardware).

## Verification

Run the validator to check BLAKE3 integrity of all data files:

    cargo run --bin pseudospore_validate -- .

Or directly:

    ./pseudospore_validate .

Exit code 0 = all files verified. Exit code 1 = integrity failure.

## Contents

- `scope.toml` — artifact metadata (what, when, how)
- `validation.json` — machine-readable validation results
- `receipts/checksums.blake3` — BLAKE3 hashes of all data files
- `receipts/environment.toml` — hardware and software environment
- `data/*.json` — production time series (plaquette, Polyakov, Wilson loops)
- `configs/*.lat` — final lattice configurations (binary, SU(3) link fields)

## Protocol

- Cold start (U = identity matrix)
- 500 warmup trajectories (volume-adaptive staged dt)
- 200 production trajectories (measured every step)
- Omelyan 2MN integrator, dt=0.01, n_md=20, tau=0.2
- 5 independent seeds per (beta, volume) grid point
- Grid: 16^4, 24^4, 32^4 × beta=5.9, 6.0, 6.2

## Reproduction

Build and run the production campaign binary:

    cd springs/hotSpring/barracuda
    CAMPAIGN_GPU=AMD cargo run --release --bin arxiv_production_campaign --features barracuda-local

Then generate the analysis:

    cargo run --release --bin arxiv_analysis -- --markdown

## License

AGPL-3.0-or-later. Data: CC-BY-SA-4.0.

## Provenance

This artifact is signed with Ed25519 (bearDog). The signature covers the
BLAKE3 root hash of all files listed in receipts/checksums.blake3.

Origin: ecoPrimals/springs/hotSpring
Gate: strandGate
GPU: AMD Radeon RX 6950 XT (NAVI21)
"#)
}

fn generate_compute_log(json_count: usize, lat_count: usize) -> String {
    format!(r#"# Compute log for hotspring-qcd-sun v1.0.0-rung1

[summary]
total_trajectories = {}
warmup_per_config = 500
production_per_config = 200
total_configs = {}
json_files = {}
lat_files = {}

[timing]
# Approximate wall-clock times (AMD RX 6950 XT)
ms_per_trajectory_16x4 = 270
ms_per_trajectory_24x4 = 1470
ms_per_trajectory_32x4 = 4600
estimated_total_hours = 8.5

[gpu]
device = "AMD Radeon RX 6950 XT (RADV NAVI21)"
vram_used_max_mb = 5800
precision = "DF64 (emulated FP64 on FP32 CUs)"
api = "WebGPU (wgpu 28.0, Vulkan backend)"
"#,
        json_count * 700,
        json_count,
        json_count,
        lat_count,
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let include_configs = args.iter().any(|a| a == "--include-configs");
    let output_path = args.iter().position(|a| a == "--output")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(default_output);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  pseudoSpore Bundler — hotspring-qcd-sun                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let prod_dir = production_dir();
    let mfst_dir = manifest_dir();

    if !mfst_dir.join("scope.toml").exists() {
        eprintln!("  ERROR: Manifest not found. Run pseudospore_manifest first.");
        std::process::exit(1);
    }

    // Collect files
    let json_files: Vec<PathBuf> = std::fs::read_dir(&prod_dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "json").unwrap_or(false))
        .collect();

    let lat_files: Vec<PathBuf> = std::fs::read_dir(&prod_dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "lat").unwrap_or(false))
        .collect();

    println!("  Data files: {} JSON, {} .lat configs", json_files.len(), lat_files.len());
    println!("  Include .lat configs: {}", include_configs);
    println!("  Output: {:?}", output_path);
    println!();

    // Create tarball
    let tar_file = std::fs::File::create(&output_path).unwrap();
    let enc = flate2::write::GzEncoder::new(tar_file, flate2::Compression::default());
    let mut tar = tar::Builder::new(enc);

    // Add manifest files
    let manifest_files = ["scope.toml", "validation.json"];
    for name in &manifest_files {
        let path = mfst_dir.join(name);
        if path.exists() {
            let data = std::fs::read(&path).unwrap();
            add_bytes_to_tar(&mut tar, &format!("{}/{}", BUNDLE_PREFIX, name), &data);
            println!("  + {}", name);
        }
    }

    // Add receipts/
    for name in ["checksums.blake3", "environment.toml"] {
        let path = mfst_dir.join("receipts").join(name);
        if path.exists() {
            let data = std::fs::read(&path).unwrap();
            add_bytes_to_tar(&mut tar, &format!("{}/receipts/{}", BUNDLE_PREFIX, name), &data);
            println!("  + receipts/{}", name);
        }
    }

    // Generate and add compute_log.toml
    let compute_log = generate_compute_log(json_files.len(), lat_files.len());
    add_bytes_to_tar(&mut tar, &format!("{}/receipts/compute_log.toml", BUNDLE_PREFIX), compute_log.as_bytes());
    println!("  + receipts/compute_log.toml");

    // Generate and add README.md
    let readme = generate_readme();
    add_bytes_to_tar(&mut tar, &format!("{}/README.md", BUNDLE_PREFIX), readme.as_bytes());
    println!("  + README.md");

    // Add data/*.json
    let mut data_size: u64 = 0;
    for path in &json_files {
        let fname = path.file_name().unwrap().to_string_lossy();
        let data = std::fs::read(path).unwrap();
        data_size += data.len() as u64;
        add_bytes_to_tar(&mut tar, &format!("{}/data/{}", BUNDLE_PREFIX, fname), &data);
    }
    println!("  + data/ ({} files, {:.1} MB)", json_files.len(), data_size as f64 / 1e6);

    // Optionally add configs/*.lat
    if include_configs {
        let mut lat_size: u64 = 0;
        for path in &lat_files {
            let fname = path.file_name().unwrap().to_string_lossy();
            let data = std::fs::read(path).unwrap();
            lat_size += data.len() as u64;
            add_bytes_to_tar(&mut tar, &format!("{}/configs/{}", BUNDLE_PREFIX, fname), &data);
        }
        println!("  + configs/ ({} files, {:.1} MB)", lat_files.len(), lat_size as f64 / 1e6);
    }

    // Finalize
    let enc = tar.into_inner().unwrap();
    enc.finish().unwrap();

    let final_size = std::fs::metadata(&output_path).unwrap().len();

    hotspring_barracuda::gossip::pseudospore_bundled(
        "hotspring-qcd-sun",
        "1.0.0-rung1",
        final_size,
    );

    println!();
    println!("  ═══════════════════════════════════════════════════════════");
    println!("  Bundle complete: {:.1} MB compressed [gossip: pseudospore.bundled]", final_size as f64 / 1e6);
    println!("  Path: {:?}", output_path);
    println!();
    println!("  Next steps:");
    println!("    1. bearDog sign (Ed25519 over BLAKE3 root hash)");
    println!("    2. ironGate register (NFT endpoint)");
    println!("    3. sporePrint serve (public download URL)");
    println!("  ═══════════════════════════════════════════════════════════");
}

fn add_bytes_to_tar<W: Write>(tar: &mut tar::Builder<W>, path: &str, data: &[u8]) {
    let mut header = tar::Header::new_gnu();
    header.set_size(data.len() as u64);
    header.set_mode(0o644);
    header.set_mtime(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    );
    header.set_cksum();
    tar.append_data(&mut header, path, data).unwrap();
}
