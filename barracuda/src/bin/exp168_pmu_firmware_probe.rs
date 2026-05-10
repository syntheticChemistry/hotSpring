// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 168: PMU firmware extraction probe for NVIDIA GV100 (Titan V).
//!
//! ## Context
//!
//! The Titan V Falcon v5 HS ROM blocks all boot paths because GV100 PMU
//! firmware is missing from `linux-firmware`. Without PMU firmware:
//!
//! - SEC2 ACR boot loader starts (`mb0 = 1`) but never completes
//! - WPR (Write-Protected Region) is never configured
//! - FECS ROM cannot load authenticated firmware → security trap at pc=0x1161
//!
//! The PMU firmware blob is embedded in the proprietary `nvidia-470` driver
//! (`nv-kernel.o_binary`) and potentially extractable from the `.run` installer.
//!
//! ## What this binary does
//!
//! 1. **ELF scan** (`--mode elf`): search `nv-kernel.o_binary` for firmware
//!    blobs by structural signature (PMU UC header + magic bytes).
//!
//! 2. **Squashfs scan** (`--mode squashfs`): extract and scan a `.run` NVIDIA
//!    installer squashfs for `nv_pmu*.bin`, `gpmu_ucode*.bin`, and similar.
//!
//! 3. **Directory scan** (`--mode dir`): walk a directory for any file matching
//!    known PMU firmware naming patterns and header signatures.
//!
//! 4. **Validate** (`--mode validate`): given a candidate blob, check the
//!    PMU UC header (magic, version, size) and print structured JSON.
//!
//! ## PMU firmware blob structure
//!
//! The PMU microcontroller uses Falcon v4 architecture (Volta GV100).
//! The Falcon UC header expected at offset 0:
//!
//! ```text
//! u32  magic       = 0x10DE0143  (NVIDIA Falcon UC magic for v4)
//! u32  version     (LSB = minor, upper = major)
//! u32  sig_size    (bytes, security signature)
//! u32  code_size   (bytes, PMU instruction memory)
//! u32  data_size   (bytes, PMU data memory)
//! u8   app_version[4]
//! ...  (platform-specific fields follow)
//! ```
//!
//! The blob is typically ~8-32 KB uncompressed (GV100 PMU is Falcon v4,
//! IMEM ≤ 256 KB but the init/boot firmware is much smaller).
//!
//! ## Usage
//!
//! ```text
//! # Scan nv-kernel.o_binary for PMU blobs (requires nvidia-470 installed)
//! cargo run --release --bin exp168_pmu_firmware_probe -- \
//!     --mode elf /usr/lib/x86_64-linux-gnu/libnvidia-glcore.so.470.256.02
//!
//! # Scan a .run installer (requires the file to exist; does NOT install it)
//! cargo run --release --bin exp168_pmu_firmware_probe -- \
//!     --mode squashfs /tmp/NVIDIA-Linux-x86_64-470.256.02.run
//!
//! # Validate a candidate blob
//! cargo run --release --bin exp168_pmu_firmware_probe -- \
//!     --mode validate /tmp/pmu_candidate.bin
//! ```
//!
//! ## Next steps after successful extraction
//!
//! See: `wateringHole/handoffs/HOTSPRING_CORALREEF_TITANV_WARM_DMATRF_HANDOFF_MAY07_2026.md`
//!
//! P0: Extract PMU FW blob (this binary)
//! P1: Feed blob to SEC2 ACR via `exp158_sec2_real_firmware.rs`
//! P2: ACR completes → WPR configured → FECS boots → compute dispatch

use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

// ── PMU blob signatures ───────────────────────────────────────────────────────

/// NVIDIA Falcon UC header magic for v4 (GV100 PMU uses Falcon v4).
const FALCON_UC_MAGIC_V4: u32 = 0x10DE_0143;

/// Alternate magic seen in some Volta firmware blobs.
const FALCON_UC_MAGIC_ALT: u32 = 0x10DE_0143 | 0x0001_0000;

/// Minimum plausible PMU blob size in bytes (header + some code).
const PMU_BLOB_MIN_BYTES: usize = 256;

/// Maximum plausible PMU blob size in bytes (full IMEM + DMEM for Volta).
const PMU_BLOB_MAX_BYTES: usize = 256 * 1024;

/// Alignment of firmware blobs within ELF binary (4-byte minimum, often 64-byte).
const SCAN_ALIGNMENT: usize = 4;

// ── Known PMU firmware filenames in NVIDIA installers ────────────────────────

const PMU_FILENAME_PATTERNS: &[&str] = &[
    "nv_pmu",
    "gpmu_ucode",
    "pmu_bl",
    "pmu_inst",
    "pmu_data",
    "gv100",
    "tu1", // Turing, but useful as reference
    "gp100",
    "gm200",
];

// ── Parsed PMU blob header ────────────────────────────────────────────────────

#[derive(Debug)]
struct FalconUcHeader {
    magic: u32,
    version: u32,
    sig_size: u32,
    code_size: u32,
    data_size: u32,
    offset_in_file: u64,
}

impl FalconUcHeader {
    fn total_size(&self) -> u64 {
        64 + u64::from(self.sig_size) + u64::from(self.code_size) + u64::from(self.data_size)
    }

    fn is_plausible(&self) -> bool {
        let total = self.total_size() as usize;
        (PMU_BLOB_MIN_BYTES..=PMU_BLOB_MAX_BYTES).contains(&total)
    }
}

/// Attempt to parse a Falcon UC header from a 16-byte slice.
fn try_parse_falcon_header(bytes: &[u8], offset: u64) -> Option<FalconUcHeader> {
    if bytes.len() < 20 {
        return None;
    }
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != FALCON_UC_MAGIC_V4 && magic != FALCON_UC_MAGIC_ALT {
        return None;
    }
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let sig_size = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    let code_size = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    let data_size = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    let hdr = FalconUcHeader {
        magic,
        version,
        sig_size,
        code_size,
        data_size,
        offset_in_file: offset,
    };
    if hdr.is_plausible() { Some(hdr) } else { None }
}

// ── ELF scan mode ─────────────────────────────────────────────────────────────

/// Scan a binary file (ELF or raw) for Falcon UC firmware blobs.
///
/// Uses a sliding window aligned to `SCAN_ALIGNMENT` bytes. Any match
/// whose header passes `is_plausible()` is reported and optionally extracted.
fn scan_file_for_blobs(path: &Path, output_dir: Option<&Path>) -> io::Result<Vec<FalconUcHeader>> {
    let mut file = std::fs::File::open(path)?;
    let file_len = file.metadata()?.len();

    eprintln!("Scanning: {} ({} bytes)", path.display(), file_len);

    const WINDOW: usize = 64 * 1024; // read 64 KiB at a time
    let mut found = Vec::new();
    let mut buf = vec![0u8; WINDOW + 20]; // overlap for cross-chunk detection
    let mut file_offset: u64 = 0;

    loop {
        let read_at = file_offset.min(file_len);
        file.seek(SeekFrom::Start(read_at))?;
        let n = file.read(&mut buf)?;
        if n < 20 {
            break;
        }

        let mut i = 0;
        while i + 20 <= n {
            if let Some(hdr) = try_parse_falcon_header(&buf[i..], file_offset + i as u64) {
                eprintln!(
                    "  PMU blob candidate at {:#010x}: magic={:#010x} ver={:#010x} \
                     sig={} code={} data={} total={}",
                    hdr.offset_in_file,
                    hdr.magic,
                    hdr.version,
                    hdr.sig_size,
                    hdr.code_size,
                    hdr.data_size,
                    hdr.total_size()
                );
                if let Some(outdir) = output_dir {
                    extract_blob(&mut file, &hdr, outdir)?;
                }
                found.push(hdr);
                i += SCAN_ALIGNMENT;
            } else {
                i += SCAN_ALIGNMENT;
            }
        }

        if file_offset + n as u64 >= file_len {
            break;
        }
        // Move forward, keeping 20-byte overlap for cross-chunk headers
        file_offset += (n - 20) as u64;
    }

    Ok(found)
}

/// Extract a firmware blob to `{outdir}/pmu_blob_{offset:#010x}.bin`.
fn extract_blob(file: &mut std::fs::File, hdr: &FalconUcHeader, outdir: &Path) -> io::Result<()> {
    let total = hdr.total_size() as usize;
    let mut blob = vec![0u8; total];
    file.seek(SeekFrom::Start(hdr.offset_in_file))?;
    let n = file.read(&mut blob)?;
    if n < total {
        eprintln!(
            "  WARNING: blob at {:#x} truncated ({n}/{total} bytes)",
            hdr.offset_in_file
        );
        blob.truncate(n);
    }

    std::fs::create_dir_all(outdir)?;
    let out_name = outdir.join(format!("pmu_blob_{:#010x}.bin", hdr.offset_in_file));
    std::fs::write(&out_name, &blob)?;
    eprintln!("  Extracted: {}", out_name.display());
    Ok(())
}

// ── Directory scan mode ───────────────────────────────────────────────────────

/// Walk a directory for files matching PMU firmware naming patterns.
///
/// Reports matches by name and validates their headers.
fn scan_directory(dir: &Path) -> io::Result<()> {
    eprintln!("Directory scan: {}", dir.display());
    scan_dir_recursive(dir, 0)
}

fn scan_dir_recursive(dir: &Path, depth: usize) -> io::Result<()> {
    if depth > 8 {
        return Ok(());
    }
    let entries = std::fs::read_dir(dir)?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let _ = scan_dir_recursive(&path, depth + 1);
        } else if path.is_file() {
            let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let fname_lower = fname.to_lowercase();
            let is_pmu_named = PMU_FILENAME_PATTERNS
                .iter()
                .any(|pat| fname_lower.contains(pat));
            let is_bin_ext = path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("bin"));
            if is_pmu_named || is_bin_ext {
                eprintln!("  Candidate: {}", path.display());
                match validate_blob(&path) {
                    Ok(hdr) => {
                        eprintln!(
                            "    ✓ Falcon UC header: magic={:#010x} ver={:#010x} \
                             code={} data={} total={}",
                            hdr.magic,
                            hdr.version,
                            hdr.code_size,
                            hdr.data_size,
                            hdr.total_size()
                        );
                    }
                    Err(e) => {
                        eprintln!("    ✗ No valid Falcon header: {e}");
                    }
                }
            }
        }
    }
    Ok(())
}

// ── Validation mode ───────────────────────────────────────────────────────────

/// Validate a candidate blob file and print structured JSON.
fn validate_blob(path: &Path) -> io::Result<FalconUcHeader> {
    let data = std::fs::read(path)?;
    if data.len() < 20 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "file too small for Falcon header",
        ));
    }
    try_parse_falcon_header(&data, 0).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "no valid Falcon UC header at offset 0",
        )
    })
}

fn print_blob_json(path: &Path, hdr: &FalconUcHeader) {
    println!(
        r#"{{
  "path": {path:?},
  "offset": {off},
  "magic": "{magic:#010x}",
  "version": "{ver:#010x}",
  "sig_size_bytes": {sig},
  "code_size_bytes": {code},
  "data_size_bytes": {data},
  "total_size_bytes": {total},
  "valid": true
}}"#,
        path = path.display().to_string(),
        off = hdr.offset_in_file,
        magic = hdr.magic,
        ver = hdr.version,
        sig = hdr.sig_size,
        code = hdr.code_size,
        data = hdr.data_size,
        total = hdr.total_size(),
    );
}

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Debug)]
enum Mode {
    Elf,
    Squashfs,
    Dir,
    Validate,
}

struct Args {
    mode: Mode,
    path: PathBuf,
    output_dir: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let args: Vec<String> = std::env::args().collect();
    let mut mode = Mode::Elf;
    let mut path = None;
    let mut output_dir = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" => {
                i += 1;
                mode = match args.get(i).map(String::as_str) {
                    Some("elf") => Mode::Elf,
                    Some("squashfs") => Mode::Squashfs,
                    Some("dir") => Mode::Dir,
                    Some("validate") => Mode::Validate,
                    other => return Err(format!("unknown mode: {other:?}")),
                };
            }
            "--output" | "-o" => {
                i += 1;
                if let Some(p) = args.get(i) {
                    output_dir = Some(PathBuf::from(p));
                }
            }
            arg if !arg.starts_with('-') => {
                path = Some(PathBuf::from(arg));
            }
            arg => return Err(format!("unknown argument: {arg}")),
        }
        i += 1;
    }

    let path = path.ok_or_else(|| "missing path argument".to_string())?;
    Ok(Args {
        mode,
        path,
        output_dir,
    })
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Exp 168: PMU Firmware Probe — Titan V (GV100) boot gate   ║");
    println!("║  P0 gate for sovereign Titan V compute dispatch             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("ERROR: {e}");
            eprintln!();
            eprintln!("Usage:");
            eprintln!(
                "  exp168_pmu_firmware_probe --mode elf     <nv-kernel.o_binary or libnvidia*.so>"
            );
            eprintln!("  exp168_pmu_firmware_probe --mode squashfs <NVIDIA-Linux-x86_64-470*.run>");
            eprintln!("  exp168_pmu_firmware_probe --mode dir     <directory>");
            eprintln!("  exp168_pmu_firmware_probe --mode validate <candidate.bin>");
            eprintln!();
            eprintln!("Options:");
            eprintln!("  --output <dir>   Extract found blobs to this directory");
            eprintln!();
            eprintln!("Context:");
            eprintln!("  GV100 PMU FW is NOT in linux-firmware. It is embedded in nvidia-470.");
            eprintln!(
                "  Without it, SEC2 ACR cannot complete → WPR never configured → Titan V stuck."
            );
            eprintln!(
                "  See: wateringHole/handoffs/HOTSPRING_CORALREEF_TITANV_WARM_DMATRF_HANDOFF_MAY07_2026.md"
            );
            std::process::exit(2);
        }
    };

    if !args.path.exists() {
        eprintln!("ERROR: path does not exist: {}", args.path.display());
        std::process::exit(1);
    }

    match args.mode {
        Mode::Validate => match validate_blob(&args.path) {
            Ok(hdr) => {
                println!("Falcon UC header VALID:");
                print_blob_json(&args.path, &hdr);
                std::process::exit(0);
            }
            Err(e) => {
                eprintln!("INVALID: {e}");
                std::process::exit(1);
            }
        },

        Mode::Dir => {
            if let Err(e) = scan_directory(&args.path) {
                eprintln!("ERROR: {e}");
                std::process::exit(1);
            }
        }

        Mode::Elf | Mode::Squashfs => {
            // Squashfs mode: the .run file is a self-extracting archive.
            // We attempt to locate the squashfs tail and scan it. If the tool
            // `unsquashfs` is available, prefer that. Otherwise, scan raw.
            if matches!(args.mode, Mode::Squashfs) {
                eprintln!("Mode: squashfs — checking for unsquashfs...");
                let has_unsquashfs = std::process::Command::new("which")
                    .arg("unsquashfs")
                    .output()
                    .is_ok_and(|o| o.status.success());

                if has_unsquashfs {
                    let outdir = args
                        .output_dir
                        .clone()
                        .unwrap_or_else(|| std::env::temp_dir().join("exp168_squashfs"));
                    eprintln!("  Extracting squashfs to {}...", outdir.display());
                    let status = std::process::Command::new("unsquashfs")
                        .args(["-d", outdir.to_str().unwrap_or("/tmp/exp168_squashfs")])
                        .arg(&args.path)
                        .status();
                    match status {
                        Ok(s) if s.success() => {
                            eprintln!("  Extraction complete. Scanning...");
                            let _ = scan_directory(&outdir);
                        }
                        Ok(s) => {
                            eprintln!(
                                "  unsquashfs exited with status {s}. Falling back to raw scan."
                            );
                            scan_and_report(&args.path, args.output_dir.as_deref());
                        }
                        Err(e) => {
                            eprintln!("  unsquashfs failed: {e}. Falling back to raw scan.");
                            scan_and_report(&args.path, args.output_dir.as_deref());
                        }
                    }
                    return;
                }
                eprintln!("  unsquashfs not found — scanning raw installer bytes.");
            }

            scan_and_report(&args.path, args.output_dir.as_deref());
        }
    }
}

fn scan_and_report(path: &Path, output_dir: Option<&Path>) {
    match scan_file_for_blobs(path, output_dir) {
        Ok(blobs) => {
            println!();
            println!("═══ Scan results ═══");
            if blobs.is_empty() {
                println!("  No Falcon UC firmware blobs found.");
                println!();
                println!("  Hints:");
                println!("  - Ensure you are scanning the correct file (nv-kernel.o_binary is");
                println!("    at /usr/lib/x86_64-linux-gnu/libnvidia-glcore.so or similar).");
                println!("  - Try --mode squashfs on the .run installer.");
                println!("  - Firmware may be compressed inside the ELF; consider binwalk.");
                std::process::exit(1);
            } else {
                println!("  Found {} Falcon UC blob(s):", blobs.len());
                for hdr in &blobs {
                    print_blob_json(path, hdr);
                }
                println!();
                println!(
                    "  Next: validate the extracted blob(s) and feed to exp158_sec2_real_firmware"
                );
                println!(
                    "  See: wateringHole/handoffs/HOTSPRING_CORALREEF_TITANV_WARM_DMATRF_HANDOFF_MAY07_2026.md"
                );
                std::process::exit(0);
            }
        }
        Err(e) => {
            eprintln!("ERROR: scan failed: {e}");
            std::process::exit(1);
        }
    }
}
