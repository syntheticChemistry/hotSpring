use hotspring_barracuda::lattice::{
    hmc::{hmc_trajectory, HmcConfig},
    milc,
    wilson::Lattice,
};
use std::path::{Path, PathBuf};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("export") => cmd_export(&args[2..]),
        Some("import") => cmd_import(&args[2..]),
        Some("roundtrip") => cmd_roundtrip(&args[2..]),
        Some("export-cached") => cmd_export_cached(&args[2..]),
        _ => {
            eprintln!("Usage: milc_validation_loop <command> [args...]");
            eprintln!();
            eprintln!("Commands:");
            eprintln!("  export-cached <config.lat> <output.milc>  Export cached config to MILC format");
            eprintln!("  export --beta B --volume L --trajectories N --output FILE");
            eprintln!("  import --input FILE");
            eprintln!("  roundtrip --beta B --volume L --trajectories N");
            std::process::exit(1);
        }
    }
}

fn cmd_export_cached(args: &[String]) {
    if args.len() < 2 {
        eprintln!("Usage: milc_validation_loop export-cached <config.lat> <output.milc>");
        std::process::exit(1);
    }

    let input_path = Path::new(&args[0]);
    let output_path = Path::new(&args[1]);

    println!("Loading cached config: {}", input_path.display());
    let lattice = Lattice::load(input_path).unwrap_or_else(|e| {
        eprintln!("Failed to load config: {}", e);
        std::process::exit(1);
    });

    let [nx, ny, nz, nt] = lattice.dims;
    let plaq = lattice.average_plaquette();
    println!("  Dims: {}×{}×{}×{}", nx, ny, nz, nt);
    println!("  β = {:.4}", lattice.beta);
    println!("  ⟨P⟩ = {:.10}", plaq);
    println!();

    println!("Writing MILC v5 format: {}", output_path.display());
    let header = milc::write_milc_config(output_path, &lattice).unwrap_or_else(|e| {
        eprintln!("Failed to write MILC config: {:?}", e);
        std::process::exit(1);
    });

    println!("  Header plaquette: {:.10}", header.plaquette);
    println!("  Checksum: [0x{:08x}, 0x{:08x}]", header.checksum[0], header.checksum[1]);
    println!();
    println!("✓ MILC export complete");
}

fn cmd_export(args: &[String]) {
    let mut beta = 6.0;
    let mut volume = 16usize;
    let mut trajectories = 200usize;
    let mut output = PathBuf::from("/tmp/hotspring_export.milc");

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--beta" => { beta = args[i + 1].parse().unwrap(); i += 2; }
            "--volume" => { volume = args[i + 1].parse().unwrap(); i += 2; }
            "--trajectories" => { trajectories = args[i + 1].parse().unwrap(); i += 2; }
            "--output" => { output = PathBuf::from(&args[i + 1]); i += 2; }
            _ => { i += 1; }
        }
    }

    println!("═══ MILC Export: Generate + Write ═══");
    println!("  β={}, V={}⁴, {} trajectories", beta, volume, trajectories);
    println!();

    let dims = [volume, volume, volume, volume];
    println!("Thermalizing SU(3) at β={}, {}⁴...", beta, volume);
    let mut lattice = Lattice::cold_start(dims, beta);
    let mut config = HmcConfig {
        n_md_steps: 15,
        dt: 0.05,
        seed: 42,
        ..Default::default()
    };

    for t in 0..(100 + trajectories) {
        let result = hmc_trajectory(&mut lattice, &mut config);
        if t % 50 == 0 {
            let phase = if t < 100 { "therm" } else { "prod " };
            println!("  [{phase}] traj {t}: plaq={:.6}, {}",
                     result.plaquette,
                     if result.accepted { "ACC" } else { "REJ" });
        }
    }

    let plaq = lattice.average_plaquette();
    println!();
    println!("Final plaquette: {:.10}", plaq);
    println!("Writing MILC v5: {}", output.display());

    let header = milc::write_milc_config(&output, &lattice).unwrap_or_else(|e| {
        eprintln!("Failed: {:?}", e);
        std::process::exit(1);
    });

    println!("  Header plaquette: {:.10}", header.plaquette);
    println!("  Checksum: [0x{:08x}, 0x{:08x}]", header.checksum[0], header.checksum[1]);
    println!();
    println!("✓ MILC export complete: {}", output.display());
}

fn cmd_import(args: &[String]) {
    let mut input = PathBuf::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => { input = PathBuf::from(&args[i + 1]); i += 2; }
            _ => { i += 1; }
        }
    }

    if input.as_os_str().is_empty() {
        eprintln!("Usage: milc_validation_loop import --input FILE");
        std::process::exit(1);
    }

    println!("═══ MILC Import: Read + Measure ═══");
    println!("  Input: {}", input.display());
    println!();

    let (lattice, header) = milc::read_milc_config(&input).unwrap_or_else(|e| {
        eprintln!("Failed to read MILC config: {:?}", e);
        std::process::exit(1);
    });

    let [nx, ny, nz, nt] = lattice.dims;
    println!("  Dims: {}×{}×{}×{}", nx, ny, nz, nt);
    println!("  β = {:.4}", lattice.beta);
    println!("  Header plaquette: {:.10}", header.plaquette);
    println!();

    let plaq = lattice.average_plaquette();
    println!("  Measured plaquette: {:.10}", plaq);

    let header_plaq = header.plaquette as f64;
    let diff = (plaq - header_plaq).abs();
    let rel = diff / header_plaq.abs().max(1e-15);
    println!("  |Δ⟨P⟩|: {:.6e} (relative: {:.6e})", diff, rel);
    println!();

    if rel < 1e-5 {
        println!("✓ MILC import validated — plaquette matches header");
    } else {
        println!("⚠ Plaquette mismatch — investigate site ordering or precision");
    }
}

fn cmd_roundtrip(args: &[String]) {
    let mut beta = 6.0;
    let mut volume = 8usize;
    let mut trajectories = 50usize;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--beta" => { beta = args[i + 1].parse().unwrap(); i += 2; }
            "--volume" => { volume = args[i + 1].parse().unwrap(); i += 2; }
            "--trajectories" => { trajectories = args[i + 1].parse().unwrap(); i += 2; }
            _ => { i += 1; }
        }
    }

    println!("═══ MILC Round-Trip Validation ═══");
    println!("  β={}, V={}⁴, {} trajectories", beta, volume, trajectories);
    println!();

    let dims = [volume, volume, volume, volume];
    let mut lattice = Lattice::cold_start(dims, beta);
    let mut config = HmcConfig {
        n_md_steps: 15,
        dt: 0.05,
        seed: 123,
        ..Default::default()
    };

    for _ in 0..(50 + trajectories) {
        hmc_trajectory(&mut lattice, &mut config);
    }

    let plaq_before = lattice.average_plaquette();
    println!("  Original plaquette: {:.10}", plaq_before);

    let tmp_path = PathBuf::from("/tmp/milc_roundtrip_test.milc");
    let header = milc::write_milc_config(&tmp_path, &lattice).unwrap();
    println!("  Written to: {}", tmp_path.display());
    println!("  Header plaquette: {:.10}", header.plaquette);

    let (lattice_back, _header_back) = milc::read_milc_config(&tmp_path).unwrap();
    let plaq_after = lattice_back.average_plaquette();
    println!("  Read-back plaquette: {:.10}", plaq_after);

    let diff = (plaq_before - plaq_after).abs();
    println!();
    println!("  Round-trip Δ⟨P⟩: {:.6e}", diff);

    if diff < 1e-5 {
        println!();
        println!("✓ MILC round-trip validated (Δ < 10⁻⁵, limited by f32 storage)");
    } else {
        println!();
        println!("⚠ Round-trip error too large — investigate");
        std::process::exit(1);
    }
}
