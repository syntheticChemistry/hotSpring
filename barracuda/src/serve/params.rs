// SPDX-License-Identifier: AGPL-3.0-or-later

use super::{DEFAULT_LATTICE_DIMS, LATTICE_DIM_CAP};
use serde_json::Value;

pub(super) fn params_map(params: &Value) -> Option<&serde_json::Map<String, Value>> {
    match params {
        Value::Object(m) => Some(m),
        Value::Array(a) => a.first().and_then(Value::as_object),
        _ => None,
    }
}

pub(super) fn parse_usize(m: &serde_json::Map<String, Value>, key: &str, default: usize) -> usize {
    m.get(key)
        .and_then(|v| v.as_u64().or_else(|| v.as_f64().map(|f| f as u64)))
        .map_or(default, |u| u as usize)
}

pub(super) fn parse_u64(m: &serde_json::Map<String, Value>, key: &str, default: u64) -> u64 {
    m.get(key)
        .and_then(|v| v.as_u64().or_else(|| v.as_f64().map(|f| f as u64)))
        .unwrap_or(default)
}

pub(super) fn parse_f64(m: &serde_json::Map<String, Value>, key: &str, default: f64) -> f64 {
    m.get(key)
        .and_then(|v| v.as_f64().or_else(|| v.as_u64().map(|u| u as f64)))
        .unwrap_or(default)
}

pub(super) fn parse_dims(m: &serde_json::Map<String, Value>) -> [usize; 4] {
    let mut out = DEFAULT_LATTICE_DIMS;
    if let Some(Value::Array(arr)) = m.get("dims")
        && arr.len() == 4
    {
        for (i, v) in arr.iter().enumerate() {
            let n = v
                .as_u64()
                .or_else(|| v.as_f64().map(|f| f as u64))
                .unwrap_or(out[i] as u64) as usize;
            out[i] = n.clamp(2, LATTICE_DIM_CAP);
        }
    }
    out
}

pub(super) fn parse_skyrme_params(m: &serde_json::Map<String, Value>) -> Vec<f64> {
    if let Some(Value::Array(arr)) = m.get("params") {
        arr.iter()
            .filter_map(|v| v.as_f64().or_else(|| v.as_u64().map(|u| u as f64)))
            .collect()
    } else {
        Vec::new()
    }
}
