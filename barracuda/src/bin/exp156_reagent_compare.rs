// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 156: Reagent Trace Comparison
//!
//! Parses mmiotrace / BAR0 dump artifacts from agentReagents captures and
//! diffs them against sovereign pipeline register snapshots (from exp 154/155).
//!
//! Highlights divergences in SEC2, PMU, FECS, PRAMIN, and PGRAPH domains
//! to answer: "what does nouveau/nvidia do that our sovereign pipeline doesn't?"
//!
//! ## Input formats
//!
//! - **hotSpring register dumps** (exp070 JSON): `{ registers: [{ offset, value, group }] }`
//! - **agentReagents mmiotrace** (raw kernel mmiotrace text): `W 4 0xNNNNNNNN 0xVVVVVVVV`
//! - **agentReagents BAR0 snapshots** (JSON): same exp070 format from in-VM capture
//!
//! ## Usage
//!
//! ```text
//! cargo run --release --bin exp156_reagent_compare -- \
//!   --sovereign data/070/titan_v_warm.json \
//!   --reagent data/reagent_captures/nouveau_titanv.json \
//!   [--output data/156/diff_report.json]
//! ```

use std::collections::BTreeMap;
use std::fs;

use hotspring_barracuda::validation::ValidationHarness;

/// One register observation: offset → value.
type RegMap = BTreeMap<u32, RegObservation>;

#[derive(Debug, Clone)]
struct RegObservation {
    value: u32,
    name: Option<String>,
    group: Option<String>,
    #[allow(dead_code)]
    source: String,
}

#[derive(Debug, Clone, serde::Serialize)]
struct DiffEntry {
    offset: String,
    name: String,
    group: String,
    sovereign_value: String,
    reagent_value: String,
    match_status: String,
}

#[derive(Debug, Clone, serde::Serialize)]
struct DiffReport {
    sovereign_source: String,
    reagent_source: String,
    total_sovereign: usize,
    total_reagent: usize,
    total_compared: usize,
    matches: usize,
    divergences: usize,
    sovereign_only: usize,
    reagent_only: usize,
    entries: Vec<DiffEntry>,
    domain_summary: BTreeMap<String, DomainStats>,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
struct DomainStats {
    total: usize,
    matches: usize,
    divergences: usize,
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Reagent Trace Comparison — Experiment 156                 ║");
    println!("║  Sovereign pipeline vs driver capture diff                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut harness = ValidationHarness::new("exp156_reagent_compare");

    let args: Vec<String> = std::env::args().collect();
    let sovereign_path = extract_arg(&args, "--sovereign")
        .unwrap_or_else(|| "data/070/titan_v_warm.json".to_string());
    let reagent_path = extract_arg(&args, "--reagent")
        .unwrap_or_else(|| "data/reagent_captures/nouveau_titanv.json".to_string());
    let output_path = extract_arg(&args, "--output");

    println!("  Sovereign: {sovereign_path}");
    println!("  Reagent:   {reagent_path}\n");

    // ── Load sovereign dump ──
    let sovereign = load_register_dump(&sovereign_path, "sovereign");
    let reagent = load_register_dump_or_mmiotrace(&reagent_path, "reagent");

    match (&sovereign, &reagent) {
        (Ok(s), Ok(r)) => {
            harness.check_bool("sovereign dump loaded", true);
            harness.check_bool("reagent dump loaded", true);
            println!("  Sovereign registers: {}", s.len());
            println!("  Reagent registers:   {}\n", r.len());

            let report = compare_maps(s, r, &sovereign_path, &reagent_path);
            print_report(&report);

            harness.check_bool(
                "comparison produced results",
                report.total_compared > 0 || report.sovereign_only > 0 || report.reagent_only > 0,
            );

            if report.divergences > 0 {
                println!(
                    "\n  {} divergences found — these are the init steps we may be missing",
                    report.divergences
                );
            }

            if let Some(ref path) = output_path {
                let json = serde_json::to_string_pretty(&report).unwrap_or_default();
                if fs::write(path, &json).is_ok() {
                    println!("\n  Wrote report to {path}");
                }
            }
        }
        (Err(e), _) => {
            println!("  Failed to load sovereign dump: {e}");
            harness.check_bool("sovereign dump loaded", false);
        }
        (_, Err(e)) => {
            println!("  Failed to load reagent dump: {e}");
            harness.check_bool("reagent dump loaded", false);
        }
    }

    println!();
    harness.finish();
}

fn load_register_dump(path: &str, source: &str) -> Result<RegMap, String> {
    let content = fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let parsed: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| format!("parse {path}: {e}"))?;

    let registers = parsed
        .get("registers")
        .and_then(|r| r.as_array())
        .ok_or_else(|| format!("{path}: missing 'registers' array"))?;

    let mut map = RegMap::new();
    for entry in registers {
        let offset = entry
            .get("raw_offset")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .or_else(|| {
                entry
                    .get("offset")
                    .and_then(|v| v.as_str())
                    .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
            });
        let value = entry
            .get("raw_value")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .or_else(|| {
                entry
                    .get("value")
                    .and_then(|v| v.as_str())
                    .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
            });

        if let (Some(off), Some(val)) = (offset, value) {
            map.insert(
                off,
                RegObservation {
                    value: val,
                    name: entry.get("name").and_then(|v| v.as_str()).map(String::from),
                    group: entry.get("group").and_then(|v| v.as_str()).map(String::from),
                    source: source.to_string(),
                },
            );
        }
    }
    Ok(map)
}

/// Try to load as exp070 JSON first; fall back to mmiotrace text format.
fn load_register_dump_or_mmiotrace(path: &str, source: &str) -> Result<RegMap, String> {
    if let Ok(map) = load_register_dump(path, source) {
        return Ok(map);
    }
    load_mmiotrace(path, source)
}

/// Parse kernel mmiotrace format.
///
/// Kernel mmiotrace lines: `R/W <width> <timestamp> <cpu> <phys_addr> <value> <pc> <unk>`
/// We auto-detect the BAR0 base from the first entry and convert to BAR0 offsets.
fn load_mmiotrace(path: &str, source: &str) -> Result<RegMap, String> {
    let content = fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let mut map = RegMap::new();
    let mut bar0_base: Option<u64> = None;

    for line in content.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 6 || (parts[0] != "W" && parts[0] != "R") {
            continue;
        }
        let phys_addr = u64::from_str_radix(
            parts[4].trim_start_matches("0x").trim_start_matches("0X"),
            16,
        );
        let value = u64::from_str_radix(
            parts[5].trim_start_matches("0x").trim_start_matches("0X"),
            16,
        );
        let (Ok(addr), Ok(val)) = (phys_addr, value) else {
            continue;
        };

        if bar0_base.is_none() {
            bar0_base = Some(addr & !0xFF_FFFF);
        }
        let base = bar0_base.unwrap();
        if addr < base || addr >= base + 0x100_0000 {
            continue;
        }
        let off = (addr - base) as u32;
        let val = val as u32;

        map.insert(
            off,
            RegObservation {
                value: val,
                name: None,
                group: Some(classify_offset(off)),
                source: source.to_string(),
            },
        );
    }
    if map.is_empty() {
        return Err(format!("{path}: no valid mmiotrace entries found"));
    }
    Ok(map)
}

fn compare_maps(sovereign: &RegMap, reagent: &RegMap, sov_path: &str, rea_path: &str) -> DiffReport {
    let mut entries = Vec::new();
    let mut matches = 0usize;
    let mut divergences = 0usize;
    let mut sovereign_only = 0usize;
    let mut reagent_only = 0usize;
    let mut domain_stats: BTreeMap<String, DomainStats> = BTreeMap::new();

    let all_offsets: BTreeMap<u32, ()> = sovereign
        .keys()
        .chain(reagent.keys())
        .map(|&k| (k, ()))
        .collect();

    for &offset in all_offsets.keys() {
        let sov = sovereign.get(&offset);
        let rea = reagent.get(&offset);

        let group = sov
            .and_then(|r| r.group.clone())
            .or_else(|| rea.and_then(|r| r.group.clone()))
            .unwrap_or_else(|| classify_offset(offset));

        let name = sov
            .and_then(|r| r.name.clone())
            .or_else(|| rea.and_then(|r| r.name.clone()))
            .unwrap_or_else(|| format!("REG_{offset:#08x}"));

        let stats = domain_stats.entry(group.clone()).or_default();
        stats.total += 1;

        let (sov_str, rea_str, status) = match (sov, rea) {
            (Some(s), Some(r)) => {
                if s.value == r.value {
                    matches += 1;
                    stats.matches += 1;
                    (
                        format!("{:#010x}", s.value),
                        format!("{:#010x}", r.value),
                        "MATCH".to_string(),
                    )
                } else {
                    divergences += 1;
                    stats.divergences += 1;
                    (
                        format!("{:#010x}", s.value),
                        format!("{:#010x}", r.value),
                        "DIVERGE".to_string(),
                    )
                }
            }
            (Some(s), None) => {
                sovereign_only += 1;
                (format!("{:#010x}", s.value), "—".into(), "SOV_ONLY".into())
            }
            (None, Some(r)) => {
                reagent_only += 1;
                ("—".into(), format!("{:#010x}", r.value), "REA_ONLY".into())
            }
            (None, None) => unreachable!(),
        };

        entries.push(DiffEntry {
            offset: format!("{offset:#08x}"),
            name,
            group,
            sovereign_value: sov_str,
            reagent_value: rea_str,
            match_status: status,
        });
    }

    DiffReport {
        sovereign_source: sov_path.to_string(),
        reagent_source: rea_path.to_string(),
        total_sovereign: sovereign.len(),
        total_reagent: reagent.len(),
        total_compared: matches + divergences,
        matches,
        divergences,
        sovereign_only,
        reagent_only,
        entries,
        domain_summary: domain_stats,
    }
}

fn print_report(report: &DiffReport) {
    println!("━━━ Comparison Summary ━━━\n");
    println!("  Compared:     {}", report.total_compared);
    println!("  Matches:      {}", report.matches);
    println!("  Divergences:  {}", report.divergences);
    println!("  Sovereign-only: {}", report.sovereign_only);
    println!("  Reagent-only:   {}", report.reagent_only);

    println!("\n━━━ Domain Breakdown ━━━\n");
    for (domain, stats) in &report.domain_summary {
        let pct = if stats.total > 0 {
            (stats.matches as f64 / stats.total as f64) * 100.0
        } else {
            0.0
        };
        println!(
            "  {domain:<16} total={:<4} match={:<4} div={:<4} ({pct:.0}%)",
            stats.total, stats.matches, stats.divergences
        );
    }

    let diverge_entries: Vec<&DiffEntry> = report
        .entries
        .iter()
        .filter(|e| e.match_status == "DIVERGE")
        .collect();

    if !diverge_entries.is_empty() {
        println!("\n━━━ Divergences (what nouveau/nvidia does differently) ━━━\n");
        for e in diverge_entries.iter().take(50) {
            println!(
                "  [{off}] {name:<30} sovereign={sov}  reagent={rea}  ({group})",
                off = e.offset,
                name = e.name,
                sov = e.sovereign_value,
                rea = e.reagent_value,
                group = e.group,
            );
        }
        if diverge_entries.len() > 50 {
            println!("  ... and {} more", diverge_entries.len() - 50);
        }
    }

    let reagent_only: Vec<&DiffEntry> = report
        .entries
        .iter()
        .filter(|e| e.match_status == "REA_ONLY")
        .collect();

    if !reagent_only.is_empty() {
        println!("\n━━━ Reagent-Only (registers driver touches that we don't) ━━━\n");
        for e in reagent_only.iter().take(30) {
            println!(
                "  [{off}] {name:<30} = {val}  ({group})",
                off = e.offset,
                name = e.name,
                val = e.reagent_value,
                group = e.group,
            );
        }
        if reagent_only.len() > 30 {
            println!("  ... and {} more", reagent_only.len() - 30);
        }
    }
}

fn classify_offset(offset: u32) -> String {
    match offset {
        0x000000..=0x000FFF => "PMC",
        0x001000..=0x001FFF => "PBUS",
        0x002000..=0x003FFF => "PFIFO",
        0x009000..=0x009FFF => "PMASTER",
        0x00E000..=0x00EFFF => "PNVIO",
        0x020000..=0x020FFF => "THERM",
        0x088000..=0x09FFFF => "PPMU",
        0x100000..=0x109FFF => "FB",
        0x10A000..=0x10BFFF => "PMU",
        0x300000..=0x3FFFFF => "BROM",
        0x400000..=0x408FFF => "PGRAPH",
        0x409000..=0x409FFF => "FECS",
        0x41A000..=0x41AFFF => "GPCCS",
        0x700000..=0x7FFFFF => "PRAMIN",
        0x840000..=0x84FFFF => "SEC2",
        _ => "OTHER",
    }
    .to_string()
}

fn extract_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].clone())
}
