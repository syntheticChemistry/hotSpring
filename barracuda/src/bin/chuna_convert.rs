// SPDX-License-Identifier: AGPL-3.0-only

//! Chuna Engine: Convert — import/export between hotSpring and ILDG/LIME.
//!
//! Converts gauge configurations between formats:
//!   - Import: ILDG/LIME → internal (for loading MILC configs into hotSpring)
//!   - Export: internal lattice → ILDG/LIME (for sharing with Bazavov/MILC)
//!   - Verify: round-trip test (write then read, check plaquette match)
//!   - Info: inspect an ILDG file's metadata without loading the gauge field
//!
//! # Usage
//!
//! ```bash
//! # Inspect an ILDG file:
//! cargo run --release --bin chuna_convert -- --info data/conf_000100.lime
//!
//! # Verify round-trip integrity:
//! cargo run --release --bin chuna_convert -- --verify data/conf_000100.lime
//!
//! # Re-export at 32-bit precision (for smaller files):
//! cargo run --release --bin chuna_convert -- \
//!   --input=data/conf_000100.lime --output=data/conf_000100_f32.lime --precision=32
//!
//! # Generate a cold-start test config:
//! cargo run --release --bin chuna_convert -- \
//!   --generate-cold --lattice=8 --beta=6.0 --output=test_cold.lime
//! ```

use hotspring_barracuda::lattice::ildg::{
    IldgMetadata, ildg_crc, read_gauge_config_file, write_gauge_config_file,
};
use hotspring_barracuda::lattice::lime::LimeReader;
use hotspring_barracuda::lattice::measurement::{
    ConfigEntry, EnsembleManifest, format_dims, parse_dims_from_args,
};
use hotspring_barracuda::lattice::qcdml::{
    QcdmlConfigInfo, QcdmlEnsembleInfo, generate_config_xml, generate_ensemble_xml,
};
use hotspring_barracuda::lattice::wilson::Lattice;

use std::io::Cursor;
use std::time::Instant;

struct CliArgs {
    mode: Mode,
}

enum Mode {
    Info { path: String },
    Verify { path: String },
    Convert {
        input: String,
        output: String,
        precision: u32,
    },
    GenerateCold {
        dims: [usize; 4],
        beta: f64,
        output: String,
    },
    EmitQcdml { path: String },
}

fn parse_args() -> CliArgs {
    let argv: Vec<String> = std::env::args().skip(1).collect();

    if argv.is_empty() {
        print_usage();
        std::process::exit(1);
    }

    // --info FILE
    if let Some(idx) = argv.iter().position(|a| a == "--info") {
        let path = argv.get(idx + 1).cloned().unwrap_or_else(|| {
            eprintln!("--info requires a file path");
            std::process::exit(1);
        });
        return CliArgs {
            mode: Mode::Info { path },
        };
    }

    // --verify FILE
    if let Some(idx) = argv.iter().position(|a| a == "--verify") {
        let path = argv.get(idx + 1).cloned().unwrap_or_else(|| {
            eprintln!("--verify requires a file path");
            std::process::exit(1);
        });
        return CliArgs {
            mode: Mode::Verify { path },
        };
    }

    // --emit-qcdml FILE
    if let Some(idx) = argv.iter().position(|a| a == "--emit-qcdml") {
        let path = argv.get(idx + 1).cloned().unwrap_or_else(|| {
            eprintln!("--emit-qcdml requires a file path");
            std::process::exit(1);
        });
        return CliArgs {
            mode: Mode::EmitQcdml { path },
        };
    }

    // --generate-cold
    if argv.iter().any(|a| a == "--generate-cold") {
        let dims = parse_dims_from_args(&argv).unwrap_or([8, 8, 8, 8]);
        let mut beta = 6.0;
        let mut output = "cold.lime".to_string();
        for arg in &argv {
            if let Some(v) = arg.strip_prefix("--beta=") {
                beta = v.parse().expect("--beta=F");
            } else if let Some(v) = arg.strip_prefix("--output=") {
                output = v.to_string();
            }
        }
        return CliArgs {
            mode: Mode::GenerateCold {
                dims,
                beta,
                output,
            },
        };
    }

    // --input + --output (convert)
    let mut input = None;
    let mut output = None;
    let mut precision = 64u32;
    for arg in &argv {
        if let Some(v) = arg.strip_prefix("--input=") {
            input = Some(v.to_string());
        } else if let Some(v) = arg.strip_prefix("--output=") {
            output = Some(v.to_string());
        } else if let Some(v) = arg.strip_prefix("--precision=") {
            precision = v.parse().expect("--precision=32|64");
        }
    }

    if let (Some(input), Some(output)) = (input, output) {
        return CliArgs {
            mode: Mode::Convert {
                input,
                output,
                precision,
            },
        };
    }

    print_usage();
    std::process::exit(1);
}

fn print_usage() {
    eprintln!("chuna_convert — ILDG/LIME gauge configuration conversion tool");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  chuna_convert --info FILE.lime");
    eprintln!("  chuna_convert --verify FILE.lime");
    eprintln!("  chuna_convert --input=IN.lime --output=OUT.lime [--precision=32|64]");
    eprintln!("  chuna_convert --generate-cold --lattice=8 --beta=6.0 --output=cold.lime");
    eprintln!("  chuna_convert --generate-cold --dims=8,8,8,16 --beta=6.0 --output=cold.lime");
    eprintln!("  chuna_convert --generate-cold --ns=8 --nt=16 --beta=6.0 --output=cold.lime");
    eprintln!("  chuna_convert --emit-qcdml FILE.lime        (generate QCDml 2.0 XML)");
}

fn main() {
    let args = parse_args();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Chuna Engine: ILDG/LIME Convert & Inspect                 ║");
    println!("║  hotSpring-barracuda — bidirectional MILC interop           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    match args.mode {
        Mode::Info { path } => cmd_info(&path),
        Mode::Verify { path } => cmd_verify(&path),
        Mode::Convert {
            input,
            output,
            precision,
        } => cmd_convert(&input, &output, precision),
        Mode::GenerateCold {
            dims,
            beta,
            output,
        } => cmd_generate_cold(dims, beta, &output),
        Mode::EmitQcdml { path } => cmd_emit_qcdml(&path),
    }
}

fn cmd_info(path: &str) {
    println!("\n  Inspecting: {path}");

    let data = std::fs::read(path).expect("read file");
    let reader = LimeReader::new(Cursor::new(&data));
    let records = reader.read_all().expect("parse LIME");

    println!("  LIME records: {}", records.len());
    for (i, rec) in records.iter().enumerate() {
        println!(
            "    [{i}] type={:<25} len={:>10} MB={} ME={}",
            rec.header.record_type,
            rec.header.data_length,
            if rec.header.message_begin { "1" } else { "0" },
            if rec.header.message_end { "1" } else { "0" },
        );
    }

    // Try to parse format XML
    for rec in &records {
        if rec.header.record_type == "ildg-format" {
            let xml = String::from_utf8_lossy(&rec.data);
            println!("\n  ildg-format XML:");
            for line in xml.lines() {
                println!("    {line}");
            }
        }
        if rec.header.record_type == "ildg-data-lfn" {
            let lfn = String::from_utf8_lossy(&rec.data);
            println!("\n  LFN: {lfn}");
        }
    }

    // Load and check
    let (lattice, meta) = read_gauge_config_file(path).expect("load config");
    let [nx, ny, nz, nt] = lattice.dims;
    println!("\n  Dimensions:  {}×{}×{}×{}", nx, ny, nz, nt);
    println!("  β:           {:.4}", lattice.beta);
    println!("  Precision:   {} bit", meta.precision_bits);
    println!("  Trajectory:  {}", meta.trajectory);
    println!("  Ensemble:    {}", meta.ensemble_id);
    println!("  Creator:     {}", meta.creator);
    println!("  Plaquette:   {:.8}", lattice.average_plaquette());
    println!("  |L|:         {:.6}", lattice.average_polyakov_loop());
    println!("  File size:   {} bytes", data.len());
}

fn cmd_verify(path: &str) {
    println!("\n  Verifying round-trip: {path}");
    let start = Instant::now();

    let (orig, meta) = read_gauge_config_file(path).expect("load original");
    let plaq_orig = orig.average_plaquette();
    println!("  Original:  ⟨P⟩={plaq_orig:.10}");

    // Write to memory
    let mut buf = Vec::new();
    hotspring_barracuda::lattice::ildg::write_gauge_config(&mut buf, &orig, &meta)
        .expect("write to buffer");

    // Read back
    let (loaded, _) =
        hotspring_barracuda::lattice::ildg::read_gauge_config(Cursor::new(&buf))
            .expect("read from buffer");
    let plaq_loaded = loaded.average_plaquette();
    println!("  Reloaded:  ⟨P⟩={plaq_loaded:.10}");

    let diff = (plaq_orig - plaq_loaded).abs();
    let tol = if meta.precision_bits == 32 { 1e-5 } else { 1e-12 };
    let pass = diff < tol;

    println!("  Δ⟨P⟩:     {diff:.2e} (tol={tol:.0e})");
    if pass {
        println!("  ✓ PASS — round-trip verified ({:.2}s)", start.elapsed().as_secs_f64());
    } else {
        println!("  ✗ FAIL — plaquette mismatch exceeds tolerance");
        std::process::exit(1);
    }
}

fn cmd_convert(input: &str, output: &str, precision: u32) {
    println!("\n  Converting: {input} → {output} (precision={precision})");
    let start = Instant::now();

    let (lattice, mut meta) = read_gauge_config_file(input).expect("load input");
    meta.precision_bits = precision;

    write_gauge_config_file(output, &lattice, &meta).expect("write output");

    let out_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
    println!(
        "  Done: {} bytes → {} bytes ({:.2}s)",
        std::fs::metadata(input).map(|m| m.len()).unwrap_or(0),
        out_size,
        start.elapsed().as_secs_f64()
    );
}

fn cmd_generate_cold(dims: [usize; 4], beta: f64, output: &str) {
    println!("\n  Generating cold-start ILDG config:");
    println!("    Lattice: {}, β={beta}", format_dims(dims));

    let lattice = Lattice::cold_start(dims, beta);
    let meta = IldgMetadata::for_lattice(&lattice, 0);
    write_gauge_config_file(output, &lattice, &meta).expect("write config");

    println!(
        "    ⟨P⟩={:.8} (should be 1.0 for cold start)",
        lattice.average_plaquette()
    );
    println!("    → {output}");
}

fn cmd_emit_qcdml(path: &str) {
    println!("\n  Generating QCDml 2.0 XML from: {path}");

    let (_lattice, meta) = read_gauge_config_file(path).expect("load config");
    let file_bytes = std::fs::read(path).expect("read file");
    let crc = ildg_crc(&file_bytes);

    let [nx, ny, nz, nt] = meta.dims;
    let manifest = EnsembleManifest::new(&meta.ensemble_id, meta.dims, meta.beta);

    let entry = ConfigEntry {
        trajectory: meta.trajectory,
        filename: path.rsplit('/').next().unwrap_or(path).to_string(),
        ildg_lfn: meta.lfn.clone(),
        checksum_crc32: None,
        checksum_ildg_crc: Some(crc),
        plaquette: meta.plaquette,
    };

    let ens_info = QcdmlEnsembleInfo::for_manifest(&manifest);
    let cfg_info = QcdmlConfigInfo::from_provenance(&manifest.provenance);

    // Ensemble XML
    let ens_xml = generate_ensemble_xml(&manifest, &ens_info);
    let ens_xml_path = path.replace(".lime", "_ensemble.xml");
    std::fs::write(&ens_xml_path, &ens_xml).expect("write ensemble XML");
    println!("  → Ensemble XML: {ens_xml_path}");

    // Config XML
    let cfg_xml = generate_config_xml(&entry, &meta, &ens_info, &cfg_info);
    let cfg_xml_path = path.replace(".lime", ".xml");
    std::fs::write(&cfg_xml_path, &cfg_xml).expect("write config XML");
    println!("  → Config XML:   {cfg_xml_path}");

    println!(
        "  ILDG CRC:       {crc}",
    );
    println!(
        "  Lattice:        {}×{}×{}×{}, β={:.4}, traj={}",
        nx, ny, nz, nt, meta.beta, meta.trajectory
    );
}
