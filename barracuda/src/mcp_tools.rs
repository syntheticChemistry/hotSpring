// SPDX-License-Identifier: AGPL-3.0-or-later

//! MCP (Model Context Protocol) tool definitions for hotSpring.
//!
//! Exposes hotSpring validation capabilities as MCP-compatible tool schemas
//! for AI/LLM integration via petalTongue or Squirrel.

use serde::{Deserialize, Serialize};

/// MCP tool definition (JSON Schema compatible).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDef {
    pub name: &'static str,
    pub description: &'static str,
    pub input_schema: serde_json::Value,
}

/// All MCP tools exposed by hotSpring.
pub fn tool_definitions() -> Vec<McpToolDef> {
    vec![
        McpToolDef {
            name: "hotspring.validate_status",
            description: "Report pass/fail status of all validation binaries and their last run results",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Filter by physics domain: lattice_qcd, molecular_dynamics, nuclear_eos, transport, all",
                        "default": "all"
                    }
                }
            }),
        },
        McpToolDef {
            name: "hotspring.tolerance_check",
            description: "Look up a named tolerance constant, its value, domain, and documented rationale",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Tolerance constant name (e.g. ENERGY_DRIFT_PCT, TRANSPORT_D_STAR_CPU_GPU_PARITY)"
                    }
                },
                "required": ["name"]
            }),
        },
        McpToolDef {
            name: "hotspring.gpu_capability_report",
            description: "Report detected GPU substrates, FP64 strategies, and available compute capabilities",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        McpToolDef {
            name: "hotspring.provenance_query",
            description: "Query Python baseline provenance for a specific physics domain (script, commit, date, command)",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Physics domain: nuclear_eos, transport, ttm, screened_coulomb, pure_gauge"
                    }
                },
                "required": ["domain"]
            }),
        },
        McpToolDef {
            name: "hotspring.experiment_list",
            description: "List available validation experiments with their pass/fail binary names and descriptions",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "description": "Optional substring filter on experiment names"
                    }
                }
            }),
        },
    ]
}

/// Format tool definitions as JSON for `mcp.tools.list` responses.
#[must_use]
pub fn tools_list_json() -> serde_json::Value {
    serde_json::json!({
        "tools": tool_definitions().iter().map(|t| {
            serde_json::json!({
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
            })
        }).collect::<Vec<_>>()
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn tool_definitions_non_empty() {
        let tools = tool_definitions();
        assert!(tools.len() >= 3);
        for tool in &tools {
            assert!(!tool.name.is_empty());
            assert!(!tool.description.is_empty());
        }
    }

    #[test]
    fn tools_list_json_valid() {
        let json = tools_list_json();
        assert!(json.get("tools").is_some());
        let arr = json["tools"].as_array().unwrap();
        assert!(!arr.is_empty());
    }
}
