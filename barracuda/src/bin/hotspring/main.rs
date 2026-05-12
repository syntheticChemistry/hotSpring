// SPDX-License-Identifier: AGPL-3.0-or-later

//! hotSpring UniBin — the eukaryotic cell.
//!
//! Single binary consolidating certification (guideStone organelle),
//! validation scenarios (experiment ribosomes), and status reporting.
//!
//! Evolved from the prokaryotic era of separate binaries during the
//! interstadial transition (May 2026).

#![forbid(unsafe_code)]

mod cli;

use clap::Parser;

fn main() {
    let parsed = cli::Cli::parse();

    match parsed.command {
        cli::Commands::Certify { layer, bare } => cmd_certify(layer, bare),
        cli::Commands::Validate {
            ref track,
            ref scenario,
            ref tier,
            list,
            ref format,
        } => cmd_validate(
            track.as_deref(),
            scenario.as_deref(),
            tier.as_deref(),
            list,
            format == "json",
        ),
        cli::Commands::Status => cmd_status(),
        cli::Commands::Version => cmd_version(),
    }
}

fn cmd_certify(layer: Option<u8>, bare: bool) {
    let max_layer = if bare {
        0
    } else {
        layer.unwrap_or(hotspring_barracuda::certification::MAX_LAYER)
    };

    let result = hotspring_barracuda::certification::certify(max_layer);
    std::process::exit(result.exit_code());
}

fn cmd_validate(
    track: Option<&str>,
    scenario_id: Option<&str>,
    tier: Option<&str>,
    list: bool,
    json_output: bool,
) {
    use hotspring_barracuda::validation::scenarios::{Tier, Track, build_registry};

    let registry = build_registry();

    if list {
        println!(
            "hotSpring Validation Scenarios ({} registered)\n",
            registry.len()
        );
        let hdr_scenario = "SCENARIO";
        let hdr_track = "TRACK";
        let hdr_tier = "TIER";
        let hdr_provenance = "PROVENANCE";
        println!("{hdr_scenario:<30} {hdr_track:<25} {hdr_tier:<6} {hdr_provenance}");
        println!("{}", "-".repeat(90));
        for s in registry.all() {
            println!(
                "{:<30} {:<25} {:<6} {}",
                s.meta.id, s.meta.track, s.meta.tier, s.meta.provenance_crate
            );
        }
        return;
    }

    let tier_filter: Option<Tier> = tier.map(|t| match t {
        "rust" => Tier::Rust,
        "live" => Tier::Live,
        "both" | "all" => Tier::Both,
        _ => {
            eprintln!("unknown tier: {t} (expected: rust, live, both)");
            std::process::exit(1);
        }
    });

    let track_filter: Option<Track> = track.and_then(|t| {
        Track::from_str_loose(t).or_else(|| {
            eprintln!("unknown track: {t}");
            std::process::exit(1);
        })
    });

    let mut harness = hotspring_barracuda::validation::ValidationHarness::new(
        "hotSpring Validation — Scenario Runner",
    );

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  hotSpring Validation — Scenario Runner                 ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    let mut ran = 0usize;
    for s in registry.all() {
        if let Some(id) = scenario_id {
            if s.meta.id != id {
                continue;
            }
        }
        if let Some(track_f) = track_filter {
            if s.meta.track != track_f {
                continue;
            }
        }
        if let Some(tier_f) = tier_filter {
            if tier_f != Tier::Both && s.meta.tier != tier_f && s.meta.tier != Tier::Both {
                continue;
            }
        }

        println!(
            "── Scenario: {} [{}] ({}) ──",
            s.meta.id, s.meta.track, s.meta.tier
        );
        (s.run)(&mut harness);
        ran += 1;
    }

    if ran == 0 {
        eprintln!("no scenarios matched the filter criteria");
        std::process::exit(1);
    }

    if json_output {
        harness.finish_json();
    } else {
        harness.finish();
    }
}

fn cmd_status() {
    use hotspring_barracuda::composition::{AtomicType, composition_health};
    use hotspring_barracuda::primal_bridge::NucleusContext;

    let ctx = NucleusContext::detect();

    println!("hotspring v{}", env!("CARGO_PKG_VERSION"));
    println!("domain: {}", hotspring_barracuda::niche::PRIMAL_DOMAIN);
    println!(
        "local methods: {} | routed: {}",
        hotspring_barracuda::niche::LOCAL_CAPABILITIES.len(),
        hotspring_barracuda::niche::ROUTED_CAPABILITIES.len(),
    );

    let names = ctx.alive_names();
    if names.is_empty() {
        println!("NUCLEUS: standalone (no primals detected)");
    } else {
        println!("NUCLEUS: {} primal(s) — {}", names.len(), names.join(", "));
    }

    let health = composition_health(&ctx);
    println!(
        "composition: tower={} node={} nest={} nucleus={}",
        health["tower_health"],
        health["node_health"],
        health["nest_health"],
        health["nucleus_health"]
    );

    for domain in AtomicType::FullNucleus.required_domains() {
        let status = if ctx.get_by_capability(domain).is_some_and(|e| e.alive) {
            "UP"
        } else {
            "DOWN"
        };
        println!("  [{status}] {domain}");
    }
}

fn cmd_version() {
    println!("hotspring {}", env!("CARGO_PKG_VERSION"));
}
