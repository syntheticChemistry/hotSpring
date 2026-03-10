// SPDX-License-Identifier: AGPL-3.0-only

//! JSONL telemetry reader for petalTongue visualization integration.
//!
//! Reads the structured JSONL telemetry files produced by validation binaries
//! and provides typed access to physics observables. This is the data source
//! for petalTongue's live rendering.
//!
//! # File format
//!
//! Each line is a JSON object with at minimum:
//!   - `t`: elapsed time in seconds since run start
//!   - `section`: physics domain identifier
//!   - Additional fields vary by observable type
//!
//! # Usage
//!
//! ```no_run
//! use hotspring_barracuda::telemetry_reader::TelemetryReader;
//!
//! let reader = TelemetryReader::from_file("chuna_overnight_telemetry.jsonl").unwrap();
//! for event in reader.events_for_section("p43_prod_8⁴ β=6.0") {
//!     println!("t={:.1}s plaquette={}", event.t, event.get_f64("plaquette").unwrap_or(0.0));
//! }
//! ```

use std::collections::HashMap;
use std::io::BufRead;

/// A single telemetry event parsed from JSONL.
#[derive(Clone, Debug)]
pub struct TelemetryEvent {
    /// Elapsed time since run start (seconds)
    pub t: f64,
    /// Physics section identifier
    pub section: String,
    /// All numeric fields
    pub fields: HashMap<String, f64>,
}

impl TelemetryEvent {
    /// Get a named field as f64.
    #[must_use]
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.fields.get(key).copied()
    }
}

/// Reader for JSONL telemetry files.
#[derive(Clone, Debug)]
pub struct TelemetryReader {
    /// All parsed telemetry events.
    pub events: Vec<TelemetryEvent>,
}

impl TelemetryReader {
    /// Load a telemetry file via streaming line-by-line parsing.
    ///
    /// Uses `BufReader` to avoid buffering the entire file in memory,
    /// which matters for large overnight telemetry runs.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the file cannot be opened or a line cannot be read.
    pub fn from_file(path: &str) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;
        let reader = std::io::BufReader::new(file);
        let mut events = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(|e| format!("read line in {path}: {e}"))?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Some(event) = parse_event(trimmed) {
                events.push(event);
            }
        }
        Ok(Self { events })
    }

    /// Parse JSONL from a string.
    #[must_use]
    pub fn parse_content(content: &str) -> Self {
        let events: Vec<TelemetryEvent> = content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .filter_map(parse_event)
            .collect();
        Self { events }
    }

    /// Get all events for a given section.
    #[must_use]
    pub fn events_for_section(&self, section: &str) -> Vec<&TelemetryEvent> {
        self.events
            .iter()
            .filter(|e| e.section == section)
            .collect()
    }

    /// Get all unique section names.
    #[must_use]
    pub fn sections(&self) -> Vec<String> {
        let mut sections: Vec<String> = self
            .events
            .iter()
            .map(|e| e.section.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        sections.sort();
        sections
    }

    /// Extract a time series for a given section and field.
    #[must_use]
    pub fn time_series(&self, section: &str, field: &str) -> Vec<(f64, f64)> {
        self.events_for_section(section)
            .iter()
            .filter_map(|e| e.get_f64(field).map(|v| (e.t, v)))
            .collect()
    }

    /// Total number of events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the reader has no events.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

fn parse_event(line: &str) -> Option<TelemetryEvent> {
    // Lightweight JSON parsing without pulling in serde for this hot path
    let line = line.trim();
    if !line.starts_with('{') || !line.ends_with('}') {
        return None;
    }
    let inner = &line[1..line.len() - 1];

    let mut t = 0.0;
    let mut section = String::new();
    let mut fields = HashMap::new();

    for pair in split_json_pairs(inner) {
        let (key, value) = split_kv(pair)?;
        match key {
            "t" => t = value.parse().ok()?,
            "section" => section = value.trim_matches('"').to_string(),
            "obs" => {
                // Single-value event: {"t":..., "section":..., "obs":"name", "val":...}
                // Will be handled when we see "val"
            }
            "val" => {
                // Find the corresponding "obs" key
                for pair2 in split_json_pairs(inner) {
                    if let Some((k2, v2)) = split_kv(pair2) {
                        if k2 == "obs" {
                            let obs_name = v2.trim_matches('"').to_string();
                            if let Ok(val) = value.parse::<f64>() {
                                fields.insert(obs_name, val);
                            }
                            break;
                        }
                    }
                }
            }
            _ => {
                if let Ok(val) = value.parse::<f64>() {
                    fields.insert(key.to_string(), val);
                }
            }
        }
    }

    if section.is_empty() {
        return None;
    }

    Some(TelemetryEvent { t, section, fields })
}

fn split_json_pairs(s: &str) -> Vec<&str> {
    let mut pairs = Vec::new();
    let mut depth = 0;
    let mut start = 0;
    for (i, ch) in s.char_indices() {
        match ch {
            '{' | '[' => depth += 1,
            '}' | ']' => depth -= 1,
            ',' if depth == 0 => {
                pairs.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    if start < s.len() {
        pairs.push(s[start..].trim());
    }
    pairs
}

fn split_kv(pair: &str) -> Option<(&str, &str)> {
    let colon = pair.find(':')?;
    let key = pair[..colon].trim().trim_matches('"');
    let value = pair[colon + 1..].trim();
    Some((key, value))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_value_event() {
        let line = r#"{"t":1.234,"section":"p43_prod","obs":"plaquette","val":0.593}"#;
        let reader = TelemetryReader::parse_content(line);
        assert_eq!(reader.len(), 1);
        let e = &reader.events[0];
        assert!((e.t - 1.234).abs() < 0.001);
        assert_eq!(e.section, "p43_prod");
        assert!((e.get_f64("plaquette").unwrap() - 0.593).abs() < 0.001);
    }

    #[test]
    fn parse_map_event() {
        let line = r#"{"t":2.0,"section":"bgk","mass_err":1.0e-4,"energy_err":0.05}"#;
        let reader = TelemetryReader::parse_content(line);
        assert_eq!(reader.len(), 1);
        let e = &reader.events[0];
        assert!(e.get_f64("mass_err").unwrap() < 0.001);
    }

    #[test]
    fn time_series_extraction() {
        let content = r#"{"t":1.0,"section":"flow","obs":"E","val":0.1}
{"t":2.0,"section":"flow","obs":"E","val":0.05}
{"t":3.0,"section":"other","obs":"x","val":9.0}
"#;
        let reader = TelemetryReader::parse_content(content);
        let ts = reader.time_series("flow", "E");
        assert_eq!(ts.len(), 2);
        assert!((ts[0].1 - 0.1).abs() < 0.01);
        assert!((ts[1].1 - 0.05).abs() < 0.01);
    }

    #[test]
    fn sections_list() {
        let content = r#"{"t":1.0,"section":"a","obs":"x","val":1.0}
{"t":2.0,"section":"b","obs":"y","val":2.0}
{"t":3.0,"section":"a","obs":"z","val":3.0}
"#;
        let reader = TelemetryReader::parse_content(content);
        let sections = reader.sections();
        assert_eq!(sections.len(), 2);
        assert!(sections.contains(&"a".to_string()));
        assert!(sections.contains(&"b".to_string()));
    }

    #[test]
    fn empty_input() {
        let reader = TelemetryReader::parse_content("");
        assert!(reader.is_empty());
    }
}
