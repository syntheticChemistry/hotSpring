// SPDX-License-Identifier: AGPL-3.0-only

//! Experiment 070: Register diff tool for sovereign reverse engineering.
//!
//! Compares two JSON register dump files (from `exp070_register_dump`) and
//! produces a categorized diff: changed, unchanged, and new registers.
//!
//! Usage:
//!   cargo run --release --bin exp070_register_diff -- <baseline.json> <warm.json> [output.json]
//!
//! Examples:
//!   cargo run --release --bin exp070_register_diff -- data/070/cold_oracle.json data/070/nouveau_warm_oracle.json

use std::collections::BTreeMap;
use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: exp070_register_diff <baseline.json> <warm.json> [output.json]");
        std::process::exit(1);
    }

    let baseline_path = &args[1];
    let warm_path = &args[2];
    let output_path = args.get(3).map(String::as_str);

    let baseline: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(baseline_path).expect("read baseline"))
            .expect("parse baseline JSON");

    let warm: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(warm_path).expect("read warm"))
            .expect("parse warm JSON");

    let base_regs = extract_registers(&baseline);
    let warm_regs = extract_registers(&warm);

    let base_bdf = baseline["bdf"].as_str().unwrap_or("?");
    let warm_bdf = warm["bdf"].as_str().unwrap_or("?");

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Exp 070: Register Diff");
    println!("║  Baseline: {baseline_path} ({base_bdf})");
    println!("║  Warm:     {warm_path} ({warm_bdf})");
    println!("╠══════════════════════════════════════════════════════════════════╣");

    let mut changed = Vec::new();
    let mut unchanged = Vec::new();
    let mut changed_by_group: BTreeMap<String, Vec<serde_json::Value>> = BTreeMap::new();

    for (offset, base_entry) in &base_regs {
        if let Some(warm_entry) = warm_regs.get(offset) {
            let base_val = base_entry["raw"].as_u64().unwrap_or(0);
            let warm_val = warm_entry["raw"].as_u64().unwrap_or(0);
            let name = base_entry["name"].as_str().unwrap_or("?");
            let group = base_entry["group"].as_str().unwrap_or("?");

            if base_val == warm_val {
                unchanged.push(name.to_string());
            } else {
                println!(
                    "║ CHANGED [{offset}] {name:<36} {base_val:#010x} → {warm_val:#010x}  ({group})",
                );
                let mut entry = serde_json::Map::new();
                entry.insert("offset".into(), serde_json::Value::String(offset.clone()));
                entry.insert("name".into(), serde_json::Value::String(name.to_string()));
                entry.insert("group".into(), serde_json::Value::String(group.to_string()));
                entry.insert(
                    "baseline".into(),
                    serde_json::Value::String(format!("{base_val:#010x}")),
                );
                entry.insert(
                    "warm".into(),
                    serde_json::Value::String(format!("{warm_val:#010x}")),
                );
                let e = serde_json::Value::Object(entry.clone());
                changed.push(e.clone());
                changed_by_group
                    .entry(group.to_string())
                    .or_default()
                    .push(e);
            }
        }
    }

    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Summary: {} changed, {} unchanged out of {} registers",
        changed.len(),
        unchanged.len(),
        base_regs.len()
    );
    println!("╠══════════════════════════════════════════════════════════════════╣");

    for (group, entries) in &changed_by_group {
        println!("║  {group}: {} registers changed", entries.len());
    }
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let mut result = serde_json::Map::new();
    result.insert(
        "baseline_file".into(),
        serde_json::Value::String(baseline_path.clone()),
    );
    result.insert(
        "warm_file".into(),
        serde_json::Value::String(warm_path.clone()),
    );
    result.insert(
        "baseline_bdf".into(),
        serde_json::Value::String(base_bdf.to_string()),
    );
    result.insert(
        "warm_bdf".into(),
        serde_json::Value::String(warm_bdf.to_string()),
    );
    result.insert(
        "total_registers".into(),
        serde_json::Value::Number(base_regs.len().into()),
    );
    result.insert(
        "changed_count".into(),
        serde_json::Value::Number(changed.len().into()),
    );
    result.insert(
        "unchanged_count".into(),
        serde_json::Value::Number(unchanged.len().into()),
    );
    result.insert("changed".into(), serde_json::Value::Array(changed));
    result.insert(
        "unchanged".into(),
        serde_json::Value::Array(
            unchanged
                .into_iter()
                .map(serde_json::Value::String)
                .collect(),
        ),
    );

    let json = serde_json::to_string_pretty(&serde_json::Value::Object(result))
        .expect("JSON serialization");

    if let Some(path) = output_path {
        fs::write(path, &json).expect("write output");
        println!("\nWrote {path}");
    }
}

fn extract_registers(dump: &serde_json::Value) -> BTreeMap<String, serde_json::Value> {
    let mut map = BTreeMap::new();
    if let Some(regs) = dump["registers"].as_array() {
        for reg in regs {
            if let Some(offset) = reg["offset"].as_str() {
                map.insert(offset.to_string(), reg.clone());
            }
        }
    }
    map
}
