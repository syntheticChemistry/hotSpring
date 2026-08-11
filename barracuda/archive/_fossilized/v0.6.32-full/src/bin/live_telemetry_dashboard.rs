// SPDX-License-Identifier: AGPL-3.0-or-later

//! Live telemetry dashboard — reads JSONL sidecar and displays running stats.
//!
//! Tails the telemetry file from `validate_chuna_overnight` and prints
//! a live-updating summary of plaquette, acceptance, and timing data.
//!
//! Usage:
//!   cargo run --release --bin live_telemetry_dashboard [path/to/telemetry.jsonl]
//!
//! Future: this bridge will feed data to petalTongue via its
//! `VisualizationDataProvider` API or HTTP endpoint.

use hotspring_barracuda::telemetry_reader::TelemetryReader;
use std::collections::BTreeMap;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        hotspring_barracuda::discovery::telemetry_path("chuna_overnight_telemetry.jsonl")
            .to_string_lossy()
            .into_owned()
    });

    let mut last_count = 0;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  hotSpring Live Telemetry Dashboard                        ║");
    println!("║  Watching: {:<49}║", &path);
    println!("║  Press Ctrl+C to exit                                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    loop {
        let Ok(reader) = TelemetryReader::from_file(&path) else {
            std::thread::sleep(std::time::Duration::from_secs(2));
            continue;
        };
        let n_events = reader.events.len();

        if n_events > last_count {
            print!("\x1B[2J\x1B[H");

            println!("╔══════════════════════════════════════════════════════════════╗");
            println!("║  hotSpring Live Telemetry Dashboard                        ║");
            println!("║  {n_events} events | {:<48}║", &path);
            println!("╚══════════════════════════════════════════════════════════════╝\n");

            let sections = reader.sections();
            let mut summaries: BTreeMap<String, SectionSummary> = BTreeMap::new();

            for event in &reader.events {
                let entry = summaries.entry(event.section.clone()).or_default();

                entry.n_events += 1;
                entry.last_t = event.t;

                if let Some(v) = event.get_f64("final_plaquette") {
                    entry.final_plaquette = Some(v);
                }
                if let Some(v) = event.get_f64("plaquette") {
                    entry.last_plaquette = Some(v);
                }
                if let Some(v) = event.get_f64("final_acceptance") {
                    entry.final_acceptance = Some(v);
                }
                if let Some(v) = event.get_f64("acceptance") {
                    entry.last_acceptance = Some(v);
                }
                if let Some(v) = event.get_f64("delta_h") {
                    entry.last_delta_h = Some(v);
                }
                if let Some(v) = event.get_f64("wall_seconds") {
                    entry.wall_seconds = Some(v);
                }
            }

            if let Some(latest) = reader.events.last() {
                println!("  ▶ Active: {} (t={:.1}s)\n", latest.section, latest.t);
            }

            println!(
                "  {:<32} {:>5} {:>9} {:>8} {:>8}",
                "Section", "Evts", "⟨P⟩", "Accept", "Wall"
            );
            println!(
                "  {:<32} {:>5} {:>9} {:>8} {:>8}",
                "───────────────────────────────", "─────", "─────────", "────────", "────────"
            );

            for (name, s) in &summaries {
                let short = if name.len() > 32 {
                    format!("…{}", &name[name.len() - 31..])
                } else {
                    name.clone()
                };

                let plaq = s
                    .final_plaquette
                    .or(s.last_plaquette)
                    .map_or_else(|| "—".into(), |v| format!("{v:.6}"));
                let acc = s
                    .final_acceptance
                    .or(s.last_acceptance)
                    .map_or_else(|| "—".into(), |v| format!("{:.0}%", v * 100.0));
                let wall = s
                    .wall_seconds
                    .map_or_else(|| "—".into(), |v| format!("{v:.0}s"));

                println!(
                    "  {short:<32} {:<5} {plaq:>9} {acc:>8} {wall:>8}",
                    s.n_events
                );
            }

            // Sparklines for the most active section
            if let Some(latest) = reader.events.last() {
                let plaq_series = reader.time_series(&latest.section, "plaquette");
                if plaq_series.len() >= 2 {
                    println!("\n  ── Plaquette Trace ({}) ──", latest.section);
                    print_sparkline("  ⟨P⟩", &plaq_series);
                }

                let acc_series = reader.time_series(&latest.section, "acceptance");
                if acc_series.len() >= 2 {
                    print_sparkline("  Acc", &acc_series);
                }

                let dh_series = reader.time_series(&latest.section, "delta_h");
                if dh_series.len() >= 2 {
                    print_sparkline("  ΔH ", &dh_series);
                }
            }

            if let Some(max_t) = reader.events.iter().map(|e| e.t).reduce(f64::max) {
                println!("\n  Elapsed: {:.0}s ({:.1} min)", max_t, max_t / 60.0);
            }

            let completed = summaries
                .values()
                .filter(|s| s.wall_seconds.is_some())
                .count();
            println!("  Completed sections: {completed} / {}", sections.len());

            // Push live data to petalTongue via JSON-RPC if socket exists
            let pt_socket = "/run/user/1000/petaltongue.sock";
            let pt_status = if std::path::Path::new(pt_socket).exists() {
                if let Some(latest) = reader.events.last() {
                    let plaq_series = reader.time_series(&latest.section, "plaquette");
                    push_to_petaltongue(pt_socket, &latest.section, &plaq_series);
                }
                "✓ LIVE (UDS connected)"
            } else {
                "○ offline (start: petaltongue server)"
            };

            println!("\n  ┌─ petalTongue ─────────────────────────────────────────────┐");
            println!("  │ JSONL stream: ✓ ({n_events} events){:>25}│", "");
            println!("  │ TelemetryReader: ✓ wired                                 │");
            println!("  │ Dashboard: {pt_status:<47}│");
            println!("  └───────────────────────────────────────────────────────────┘");

            last_count = n_events;
        }

        std::thread::sleep(std::time::Duration::from_secs(5));
    }
}

#[derive(Default)]
struct SectionSummary {
    n_events: usize,
    last_t: f64,
    last_plaquette: Option<f64>,
    last_acceptance: Option<f64>,
    last_delta_h: Option<f64>,
    wall_seconds: Option<f64>,
    final_plaquette: Option<f64>,
    final_acceptance: Option<f64>,
}

/// Push live plaquette time-series to petalTongue via JSON-RPC over UDS.
/// Uses `visualization.render.grammar` with DataBinding::TimeSeries.
fn push_to_petaltongue(socket_path: &str, section: &str, plaq_series: &[(f64, f64)]) {
    use std::io::{Read, Write};
    use std::os::unix::net::UnixStream;

    if plaq_series.len() < 2 {
        return;
    }

    let x_values: Vec<f64> = plaq_series.iter().map(|(t, _)| *t).collect();
    let y_values: Vec<f64> = plaq_series.iter().map(|(_, v)| *v).collect();

    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "visualization.render.grammar",
        "params": {
            "session_id": format!("hotspring-live-{}", section.replace(' ', "_")),
            "grammar": {
                "id": "plaquette_live",
                "geometry": "Line",
                "variables": [
                    {"name": "x", "field": "x", "role": "X"},
                    {"name": "y", "field": "y", "role": "Y"}
                ],
                "scales": [
                    {"variable": "x", "scale_type": "Linear"},
                    {"variable": "y", "scale_type": "Linear"}
                ],
                "title": format!("Live ⟨P⟩ — {section}"),
                "coordinate": "Cartesian"
            },
            "data": x_values.iter().zip(y_values.iter())
                .map(|(x, y)| serde_json::json!({"x": x, "y": y}))
                .collect::<Vec<_>>(),
            "modality": "svg",
            "domain": "physics"
        }
    });

    let payload = request.to_string();
    let frame = format!("Content-Length: {}\r\n\r\n{}", payload.len(), payload);

    if let Ok(mut stream) = UnixStream::connect(socket_path) {
        stream.set_write_timeout(Some(std::time::Duration::from_millis(500))).ok();
        stream.set_read_timeout(Some(std::time::Duration::from_millis(500))).ok();
        let _ = stream.write_all(frame.as_bytes());
        let mut buf = [0u8; 1024];
        let _ = stream.read(&mut buf);
    }
}

fn print_sparkline(label: &str, series: &[(f64, f64)]) {
    let values: Vec<f64> = series.iter().map(|(_, v)| *v).collect();
    let min = values.iter().copied().reduce(f64::min).unwrap_or(0.0);
    let max = values.iter().copied().reduce(f64::max).unwrap_or(1.0);
    let range = (max - min).max(1e-10);

    let blocks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let spark: String = values
        .iter()
        .map(|v| {
            let idx = ((v - min) / range * 7.0).round() as usize;
            blocks[idx.min(7)]
        })
        .collect();

    let last = values.last().copied().unwrap_or(0.0);
    println!("{label} [{min:.4}–{max:.4}]: {spark} → {last:.6}");
}
