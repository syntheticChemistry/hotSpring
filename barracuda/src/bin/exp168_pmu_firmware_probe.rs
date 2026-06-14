// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 168: PMU firmware extraction probe for NVIDIA GPUs.
//!
//! ## Context
//!
//! PMU firmware is the gatekeeper for sovereign compute on both Titan V and K80:
//!
//! **Titan V (GV100):** The Falcon v5 HS ROM blocks all boot paths because
//! GV100 PMU firmware is missing from `linux-firmware`. Without PMU firmware:
//! - SEC2 ACR boot loader starts (`mb0 = 1`) but never completes
//! - WPR (Write-Protected Region) is never configured
//! - FECS ROM cannot load authenticated firmware → security trap at pc=0x1161
//!
//! **K80 (GK210):** GPCs remain power-gated because the PGOB PSW handshake
//! requires running PMU firmware. Without PMU, `pri_gpc_cnt=0` and all GPC
//! space reads return `0xbadf1002`. Additionally, upstream nouveau has no
//! chipset entry for chip ID `0xf2` (GK210), so nouveau cannot initialize
//! the GPU at all (see Exp 185).
//!
//! PMU firmware blobs are embedded in proprietary NVIDIA drivers (`nvidia-470`,
//! `nvidia-580`) and potentially extractable from `.run` installers or
//! installed kernel module objects.
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

/// Falcon UC header magic for v3 (Kepler GK110/GK210 PMU uses Falcon v3).
const FALCON_UC_MAGIC_V3: u32 = 0x10DE_0142;

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
    "pmu_init",
    "gv100",
    "gk110",
    "gk210",
    "tu1", // Turing, but useful as reference
    "gp100",
    "gm200",
    "falcon",
    "devinit",
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
    if magic != FALCON_UC_MAGIC_V4 && magic != FALCON_UC_MAGIC_ALT && magic != FALCON_UC_MAGIC_V3 {
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
    /// Scan an NVIDIA kernel module (.ko) for embedded firmware blobs.
    NvKo,
    /// Parse VBIOS ROM for BIT PMU tables (Kepler PMU firmware source).
    Vbios,
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
                    Some("nv-ko" | "ko") => Mode::NvKo,
                    Some("vbios" | "rom") => Mode::Vbios,
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
    println!("║  Exp 168: PMU Firmware Probe — Kepler + Volta boot gate    ║");
    println!("║  P0 gate for K80 GPC ungate + Titan V sovereign dispatch   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("ERROR: {e}");
            eprintln!();
            eprintln!("Usage:");
            eprintln!(
                "  exp168_pmu_firmware_probe --mode elf      <nv-kernel.o_binary or libnvidia*.so>"
            );
            eprintln!("  exp168_pmu_firmware_probe --mode squashfs <NVIDIA-Linux-x86_64-470*.run>");
            eprintln!("  exp168_pmu_firmware_probe --mode dir      <directory>");
            eprintln!("  exp168_pmu_firmware_probe --mode validate <candidate.bin>");
            eprintln!("  exp168_pmu_firmware_probe --mode nv-ko    <nvidia.ko or nvidia-drm.ko>");
            eprintln!("  exp168_pmu_firmware_probe --mode vbios    <gpu_vbios.rom>");
            eprintln!();
            eprintln!("Options:");
            eprintln!("  --output <dir>   Extract found blobs to this directory");
            eprintln!();
            eprintln!("Context:");
            eprintln!("  PMU FW is NOT in linux-firmware for GK210 or GV100.");
            eprintln!("  K80:     PMU needed for PGOB PSW → GPC ungate (pri_gpc_cnt=0 without it)");
            eprintln!("  Titan V: PMU needed for HBM2 training → SEC2 ACR → FECS unlock");
            eprintln!(
                "  Try scanning nvidia kernel objects, .run installers, or proprietary libs."
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

        Mode::NvKo => {
            eprintln!("Mode: nv-ko — scanning kernel module for embedded firmware...");
            scan_nvidia_ko(&args.path, args.output_dir.as_deref());
        }

        Mode::Vbios => {
            scan_vbios_for_pmu(&args.path);
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

/// Scan an nvidia kernel module (.ko) for embedded firmware blobs.
///
/// NVIDIA kernel modules embed firmware as ELF sections. The module may also
/// contain DEVINIT scripts and PMU microcode as data blobs. We scan both for
/// Falcon UC headers and for known byte patterns that indicate firmware data.
fn scan_nvidia_ko(path: &Path, output_dir: Option<&Path>) {
    eprintln!(
        "Scanning kernel module: {} ({} bytes)",
        path.display(),
        std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
    );

    // First, try ELF section extraction via `readelf` if available
    let readelf_result = std::process::Command::new("readelf")
        .args(["-S", "-W"])
        .arg(path)
        .output();

    if let Ok(output) = readelf_result {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let pmu_sections: Vec<&str> = stdout
                .lines()
                .filter(|l| {
                    let lower = l.to_lowercase();
                    lower.contains("pmu")
                        || lower.contains("falcon")
                        || lower.contains("devinit")
                        || lower.contains("firmware")
                        || lower.contains("ucode")
                })
                .collect();

            if pmu_sections.is_empty() {
                eprintln!("  No PMU/falcon/devinit ELF sections found by name.");
            } else {
                eprintln!("  PMU-related ELF sections:");
                for s in &pmu_sections {
                    eprintln!("    {s}");
                }
            }
        }
    }

    // Also try `objdump -h` for firmware data sections
    let objdump_result = std::process::Command::new("objdump")
        .args(["-h"])
        .arg(path)
        .output();

    if let Ok(output) = objdump_result {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let fw_sections: Vec<&str> = stdout
                .lines()
                .filter(|l| {
                    let lower = l.to_lowercase();
                    lower.contains(".nv_firmware")
                        || lower.contains("__firmware")
                        || lower.contains("pmu")
                        || lower.contains("devinit")
                })
                .collect();
            if !fw_sections.is_empty() {
                eprintln!("  Firmware data sections:");
                for s in &fw_sections {
                    eprintln!("    {s}");
                }
            }
        }
    }

    // Fall through to raw binary scan for Falcon UC headers
    eprintln!("  Raw binary scan for Falcon UC headers...");
    scan_and_report(path, output_dir);
}

/// Scan a VBIOS ROM for PMU-related structures.
///
/// Kepler PMU firmware comes from the VBIOS, not from separate firmware files.
/// The BIT (Binary Information Table) contains a PMU entry pointing to boot
/// code, code, and data segments. We search for the BIT header signature
/// and look for PMU-related entries.
fn scan_vbios_for_pmu(path: &Path) {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("ERROR reading VBIOS: {e}");
            return;
        }
    };

    eprintln!("VBIOS scan: {} ({} bytes)", path.display(), data.len());

    // Check for VBIOS signature (55 AA)
    if data.len() < 2 || data[0] != 0x55 || data[1] != 0xAA {
        eprintln!("  WARNING: file does not start with VBIOS signature (55 AA)");
    } else {
        eprintln!("  VBIOS signature OK (55 AA)");
    }

    // Search for BIT header: "BIT\0" at any aligned offset
    let mut bit_offsets = Vec::new();
    for i in 0..data.len().saturating_sub(4) {
        if data[i] == b'\xff' && data[i + 1] == b'\xff' {
            continue; // Skip empty regions
        }
        if &data[i..i + 4] == b"BIT\0" {
            bit_offsets.push(i);
        }
    }

    if bit_offsets.is_empty() {
        eprintln!("  No BIT (Binary Information Table) header found.");
        // Try BMP header (older format)
        for i in 0..data.len().saturating_sub(7) {
            if &data[i..i + 5] == b"\xff\x7fNV\0" {
                eprintln!("  Found BMP header at offset {i:#x} (older format)");
            }
        }
    } else {
        eprintln!("  Found {} BIT header(s):", bit_offsets.len());
        for &off in &bit_offsets {
            eprintln!("    BIT at offset {off:#x}");
            // BIT table structure:
            //   [0..3] = "BIT\0"
            //   [4]    = entry count or version
            //   [5]    = header size
            //   [6..7] = token/type entries follow
            // Each entry: type(1) + version(1) + size(2) + ptr(2)
            // PMU type = 'p' (0x70) or 'P' (0x50)
            if off + 8 <= data.len() {
                let hdr_size = data[off + 5] as usize;
                let start = off + hdr_size.max(6);
                let mut j = start;
                while j + 6 <= data.len() {
                    let entry_type = data[j];
                    let entry_version = data[j + 1];
                    let entry_data_size = u16::from_le_bytes([data[j + 2], data[j + 3]]) as usize;
                    let entry_ptr = u16::from_le_bytes([data[j + 4], data[j + 5]]) as usize;

                    if entry_type == 0xFF || (entry_type == 0 && entry_version == 0) {
                        break;
                    }

                    let type_char = if entry_type.is_ascii_graphic() {
                        format!("'{}' ({:#04x})", entry_type as char, entry_type)
                    } else {
                        format!("{entry_type:#04x}")
                    };

                    let is_pmu = entry_type == b'p' || entry_type == b'P';
                    let is_devinit = entry_type == b'I' || entry_type == b'i';
                    let marker = if is_pmu {
                        " ← PMU"
                    } else if is_devinit {
                        " ← DEVINIT"
                    } else {
                        ""
                    };

                    eprintln!(
                        "      BIT entry: type={type_char} ver={entry_version} \
                         size={entry_data_size} ptr={entry_ptr:#06x}{marker}"
                    );

                    if is_pmu && entry_ptr > 0 && entry_ptr + 8 <= data.len() {
                        eprintln!("      PMU table at {entry_ptr:#06x}:");
                        let pmu_bytes = &data[entry_ptr..];
                        if pmu_bytes.len() >= 8 {
                            let boot_addr = u16::from_le_bytes([pmu_bytes[0], pmu_bytes[1]]);
                            let boot_size = u16::from_le_bytes([pmu_bytes[2], pmu_bytes[3]]);
                            let code_addr = u16::from_le_bytes([pmu_bytes[4], pmu_bytes[5]]);
                            let code_size = u16::from_le_bytes([pmu_bytes[6], pmu_bytes[7]]);
                            eprintln!("        boot: addr={boot_addr:#06x} size={boot_size:#06x}");
                            eprintln!("        code: addr={code_addr:#06x} size={code_size:#06x}");
                            if pmu_bytes.len() >= 12 {
                                let data_addr = u16::from_le_bytes([pmu_bytes[8], pmu_bytes[9]]);
                                let data_size = u16::from_le_bytes([pmu_bytes[10], pmu_bytes[11]]);
                                eprintln!(
                                    "        data: addr={data_addr:#06x} size={data_size:#06x}"
                                );
                            }
                        }
                    }

                    j += 6;
                }
            }
        }
    }

    // Also scan for Falcon IMEM/DMEM patterns (PMU boot code)
    let mut falcon_boot_count = 0;
    for i in 0..data.len().saturating_sub(8) {
        // Falcon boot vector: small code with specific patterns
        // PMU boot code typically starts with a specific instruction sequence
        let w0 = u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
        // Falcon NOP = 0x00000000, but boot code typically has a jump
        // Look for Falcon-style branch instructions at plausible offsets
        if w0 == 0xF0000000 || w0 == 0xF2000000 {
            falcon_boot_count += 1;
        }
    }
    if falcon_boot_count > 0 {
        eprintln!("  Found {falcon_boot_count} potential Falcon branch instructions");
    }

    println!();
    println!("═══ VBIOS PMU Summary ═══");
    println!("  VBIOS:     {} ({} bytes)", path.display(), data.len());
    println!("  BIT headers: {}", bit_offsets.len());
    if !bit_offsets.is_empty() {
        println!("  PMU firmware is embedded in VBIOS via BIT tables.");
        println!("  Nouveau's gk110_pmu_new parses these to load PMU on init.");
        println!();
        println!("  Key insight: The K80 VBIOS already contains PMU init data.");
        println!("  Path 1 (fast): Patch nouveau to add case 0x0f2 → nvf1_chipset");
        println!("                 and let nouveau load PMU from this VBIOS natively.");
        println!("  Path 2 (hard): Parse BIT PMU tables ourselves, upload via BAR0.");
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
