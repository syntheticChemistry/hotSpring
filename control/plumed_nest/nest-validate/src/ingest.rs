// SPDX-License-Identifier: AGPL-3.0-or-later

//! PLUMED-NEST ingestion — download, extract, and validate archives.
//!
//! Replaces ingest.sh with proper error handling, no zombie processes,
//! and structured progress reporting.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone)]
pub struct NestTarget {
    pub id: &'static str,
    pub plum_id: &'static str,
    pub url: &'static str,
    pub local_dir: &'static str,
    pub description: &'static str,
}

pub const TARGETS: &[NestTarget] = &[
    NestTarget {
        id: "01",
        plum_id: "19.009",
        url: "https://github.com/plumed/masterclass-21-4/archive/refs/heads/main.tar.gz",
        local_dir: "target_01_alanine_dipeptide",
        description: "Alanine dipeptide well-tempered metadynamics",
    },
    NestTarget {
        id: "02",
        plum_id: "24.029",
        url: "https://zenodo.org/records/12735917/files/OPES-Rew-HLDA.tar.gz",
        local_dir: "target_02_chignolin_opes",
        description: "Chignolin OPES folding (Ray & Rizzi 2024)",
    },
    NestTarget {
        id: "05",
        plum_id: "22.028",
        url: "https://zenodo.org/records/6614257/files/Rew_HILLS_COLVAR.tar.gz",
        local_dir: "target_05_glycan_pucker",
        description: "N-glycan Cremer-Pople puckering FEL",
    },
    NestTarget {
        id: "06",
        plum_id: "25.007",
        url: "https://zenodo.org/records/14745975/files/plumed-nest.tar.gz",
        local_dir: "target_06_cazyme_glycan",
        description: "CAZyme glycan landscape (REST-RECT)",
    },
];

pub fn ingest_all(root: &Path, filter: Option<&str>) {
    println!("\x1b[36m╔══════════════════════════════════════════════════════════════╗\x1b[0m");
    println!("\x1b[36m║  PLUMED-NEST Ingestion — Rust Native                        ║\x1b[0m");
    println!("\x1b[36m╚══════════════════════════════════════════════════════════════╝\x1b[0m");
    println!();

    for target in TARGETS {
        if let Some(f) = filter {
            if !target.id.contains(f) && !target.local_dir.contains(f) {
                continue;
            }
        }

        let target_dir = root.join(target.local_dir);
        let archive_dir = target_dir.join("archive");

        print!("  [{}/{}] {}... ", target.id, target.plum_id, target.description);

        if archive_dir.exists() && std::fs::read_dir(&archive_dir).map(|d| d.count() > 0).unwrap_or(false) {
            println!("\x1b[33mALREADY PRESENT\x1b[0m");
            continue;
        }

        std::fs::create_dir_all(&archive_dir).unwrap();

        match download_and_extract(target.url, &archive_dir) {
            Ok(n_files) => {
                println!("\x1b[32mOK\x1b[0m ({n_files} files)");
                validate_plumed_inputs(&target_dir);
            }
            Err(e) => {
                println!("\x1b[31mFAILED\x1b[0m: {e}");
            }
        }
    }
}

fn download_and_extract(url: &str, dest: &Path) -> Result<usize, String> {
    let resp = ureq::get(url)
        .call()
        .map_err(|e| format!("Download failed: {e}"))?;

    let mut body = Vec::new();
    resp.into_body()
        .into_reader()
        .read_to_end(&mut body)
        .map_err(|e| format!("Read failed: {e}"))?;

    let gz = flate2::read::GzDecoder::new(&body[..]);
    let mut archive = tar::Archive::new(gz);

    let mut count = 0;
    for entry in archive.entries().map_err(|e| format!("Tar error: {e}"))? {
        let mut entry = entry.map_err(|e| format!("Entry error: {e}"))?;
        entry.unpack_in(dest).map_err(|e| format!("Unpack error: {e}"))?;
        count += 1;
    }

    Ok(count)
}

fn validate_plumed_inputs(target_dir: &Path) {
    let plumed_dir = target_dir.join("plumed");
    if !plumed_dir.is_dir() {
        return;
    }

    let dat_files: Vec<PathBuf> = std::fs::read_dir(&plumed_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "dat").unwrap_or(false))
        .collect();

    for dat in &dat_files {
        let name = dat.file_name().unwrap().to_string_lossy();
        let result = Command::new("plumed")
            .args(["driver", "--natoms", "10000", "--parse-only", "--plumed"])
            .arg(dat)
            .current_dir(target_dir)
            .output();

        match result {
            Ok(output) if output.status.success() => {
                println!("    \x1b[32m✓\x1b[0m {name} parses cleanly");
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let first_err = stderr.lines()
                    .find(|l| l.contains("ERROR") || l.contains("not known"))
                    .unwrap_or("parse error");
                println!("    \x1b[33m⚠\x1b[0m {name}: {first_err}");
            }
            Err(e) => {
                println!("    \x1b[31m✗\x1b[0m {name}: failed to run plumed ({e})");
            }
        }
    }
}
