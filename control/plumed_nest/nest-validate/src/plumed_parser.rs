// SPDX-License-Identifier: AGPL-3.0-or-later

//! Native PLUMED .dat file parser.
//!
//! Validates PLUMED input files without requiring the plumed binary.
//! Extracts action definitions, labels, arguments, and parameters.
//!
//! This is a structural parser only — it validates syntax, not semantics.
//! Semantic validation (atom indices, action compatibility) requires the
//! PLUMED binary or a full action registry.

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlumedInput {
    pub actions: Vec<PlumedAction>,
    pub comments: Vec<String>,
    pub warnings: Vec<String>,
    pub is_valid: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlumedAction {
    pub label: Option<String>,
    pub action_type: String,
    pub args: Vec<(String, String)>,
    pub line_number: usize,
}

/// Known PLUMED action types for validation.
const KNOWN_ACTIONS: &[&str] = &[
    "TORSION", "DISTANCE", "RMSD", "COORDINATION", "CONTACTMAP",
    "COMBINE", "MATHEVAL", "CUSTOM",
    "METAD", "OPES_METAD", "OPES_METAD_EXPLORE", "PBMETAD",
    "RESTRAINT", "UPPER_WALLS", "LOWER_WALLS", "MOVINGRESTRAINT",
    "PRINT", "FLUSH", "DUMPFORCES", "DUMPATOMS",
    "MOLINFO", "WHOLEMOLECULES", "GROUP", "CENTER", "COM",
    "PUCKERING", "GYRATION",
    "FIT_TO_TEMPLATE", "CONSTANT",
    "ENERGY", "VOLUME", "CELL",
];

/// Known modules (for informational purposes).
const MODULE_ACTIONS: &[(&str, &str)] = &[
    ("OPES_METAD", "opes"),
    ("OPES_METAD_EXPLORE", "opes"),
    ("OPES_EXPANDED", "opes"),
    ("PYTORCH_MODEL", "pytorch"),
    ("DEEP_FE", "pytorch"),
    ("PYCV", "pycv"),
    ("VES_LINEAR_EXPANSION", "ves"),
    ("TD_UNIFORM", "ves"),
];

/// Parse a PLUMED .dat input file.
pub fn parse_plumed_dat(path: &Path) -> Result<PlumedInput, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;

    parse_plumed_string(&content)
}

pub fn parse_plumed_string(content: &str) -> Result<PlumedInput, String> {
    let mut actions = Vec::new();
    let mut comments = Vec::new();
    let mut warnings = Vec::new();
    let mut is_valid = true;

    let mut in_multiline = false;
    let mut multiline_buffer = String::new();
    let mut multiline_start_line = 0;

    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();

        // Skip empty lines
        if line.is_empty() {
            continue;
        }

        // Comments
        if line.starts_with('#') {
            comments.push(line.to_string());
            continue;
        }

        // Handle multiline (... syntax)
        if in_multiline {
            // Closing patterns: "..." alone, "... ACTION", or line starting with "..."
            if line == "..." || line.starts_with("...") || line.ends_with("...") {
                let extra = line.trim_start_matches("...").trim_end_matches("...").trim();
                let full_line = if extra.is_empty() {
                    multiline_buffer.clone()
                } else {
                    format!("{} {}", multiline_buffer, extra)
                };
                in_multiline = false;
                if let Some(action) = parse_action_line(&full_line, multiline_start_line) {
                    actions.push(action);
                }
                multiline_buffer.clear();
            } else {
                multiline_buffer.push(' ');
                multiline_buffer.push_str(line);
            }
            continue;
        }

        // Start of multiline
        if line.ends_with("...") {
            in_multiline = true;
            multiline_start_line = line_num + 1;
            multiline_buffer = line.trim_end_matches("...").trim().to_string();
            continue;
        }

        // Also handle "ACTION ..." start pattern
        if line.contains(" ...") && !line.ends_with("...") {
            // This is the PLUMED "ACTION ... \n PARAMS \n ... ACTION" pattern
            let parts: Vec<&str> = line.splitn(2, " ...").collect();
            in_multiline = true;
            multiline_start_line = line_num + 1;
            multiline_buffer = parts[0].to_string();
            continue;
        }

        // Single-line action
        if let Some(action) = parse_action_line(line, line_num + 1) {
            actions.push(action);
        }
    }

    if in_multiline {
        warnings.push(format!("Unclosed multiline block starting at line {multiline_start_line}"));
        is_valid = false;
    }

    // Validate actions
    for action in &actions {
        let upper = action.action_type.to_uppercase();
        if let Some((_, module)) = MODULE_ACTIONS.iter().find(|(a, _)| *a == upper.as_str()) {
            warnings.push(format!(
                "Line {}: {} requires module '{}' (may need custom PLUMED build)",
                action.line_number, action.action_type, module
            ));
        } else if !KNOWN_ACTIONS.contains(&upper.as_str()) {
            warnings.push(format!(
                "Line {}: Unknown action '{}' (may be valid in newer PLUMED versions)",
                action.line_number, action.action_type
            ));
        }
    }

    Ok(PlumedInput { actions, comments, warnings, is_valid })
}

fn parse_action_line(line: &str, line_number: usize) -> Option<PlumedAction> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }

    // Detect label: pattern
    let (label, rest) = if let Some(colon_pos) = line.find(':') {
        let before = &line[..colon_pos].trim();
        if before.chars().all(|c| c.is_alphanumeric() || c == '_') && !before.is_empty() {
            (Some(before.to_string()), line[colon_pos + 1..].trim().to_string())
        } else {
            (None, line.to_string())
        }
    } else {
        (None, line.to_string())
    };

    // Split into tokens
    let tokens: Vec<&str> = rest.split_whitespace().collect();
    if tokens.is_empty() {
        return None;
    }

    let action_type = tokens[0].to_string();
    let mut args = Vec::new();
    let mut found_label_arg = label.clone();

    for token in &tokens[1..] {
        if let Some(eq_pos) = token.find('=') {
            let key = &token[..eq_pos];
            let val = &token[eq_pos + 1..];
            if key == "LABEL" && found_label_arg.is_none() {
                found_label_arg = Some(val.to_string());
            }
            args.push((key.to_string(), val.to_string()));
        }
    }

    Some(PlumedAction {
        label: found_label_arg,
        action_type,
        args,
        line_number,
    })
}

/// Validate a PLUMED .dat file and return a structured report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub file: String,
    pub n_actions: usize,
    pub actions_summary: Vec<String>,
    pub cvs_defined: Vec<String>,
    pub biases_defined: Vec<String>,
    pub output_files: Vec<String>,
    pub modules_required: Vec<String>,
    pub warnings: Vec<String>,
    pub is_valid: bool,
}

pub fn validate_plumed_file(path: &Path) -> Result<ValidationReport, String> {
    let input = parse_plumed_dat(path)?;

    let bias_types = ["METAD", "OPES_METAD", "OPES_METAD_EXPLORE", "PBMETAD",
                      "RESTRAINT", "UPPER_WALLS", "LOWER_WALLS", "MOVINGRESTRAINT"];
    let cv_types = ["TORSION", "DISTANCE", "RMSD", "COORDINATION", "CONTACTMAP",
                    "COMBINE", "PUCKERING", "GYRATION", "MATHEVAL", "CUSTOM"];

    let cvs_defined: Vec<String> = input.actions.iter()
        .filter(|a| cv_types.contains(&a.action_type.to_uppercase().as_str()))
        .filter_map(|a| a.label.clone())
        .collect();

    let biases_defined: Vec<String> = input.actions.iter()
        .filter(|a| bias_types.contains(&a.action_type.to_uppercase().as_str()))
        .filter_map(|a| a.label.clone())
        .collect();

    let output_files: Vec<String> = input.actions.iter()
        .flat_map(|a| a.args.iter())
        .filter(|(k, _)| k == "FILE")
        .map(|(_, v)| v.clone())
        .collect();

    let modules_required: Vec<String> = input.actions.iter()
        .filter_map(|a| {
            MODULE_ACTIONS.iter()
                .find(|(action, _)| *action == a.action_type.to_uppercase())
                .map(|(_, module)| module.to_string())
        })
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let actions_summary: Vec<String> = input.actions.iter()
        .map(|a| {
            let label = a.label.as_deref().unwrap_or("_");
            format!("{}: {} (line {})", label, a.action_type, a.line_number)
        })
        .collect();

    Ok(ValidationReport {
        file: path.display().to_string(),
        n_actions: input.actions.len(),
        actions_summary,
        cvs_defined,
        biases_defined,
        output_files,
        modules_required,
        warnings: input.warnings,
        is_valid: input.is_valid,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_metad() {
        let input = r#"
# Well-tempered metadynamics
phi: TORSION ATOMS=5,7,9,15
psi: TORSION ATOMS=7,9,15,17

METAD ARG=phi,psi PACE=500 HEIGHT=1.2 SIGMA=0.35,0.35 FILE=HILLS BIASFACTOR=6.0 TEMP=300.0 LABEL=metad

PRINT STRIDE=10 ARG=phi,psi,metad.bias FILE=COLVAR
"#;
        let result = parse_plumed_string(input).unwrap();
        assert!(result.is_valid);
        assert_eq!(result.actions.len(), 4);
        assert_eq!(result.actions[0].label, Some("phi".to_string()));
        assert_eq!(result.actions[0].action_type, "TORSION");
        assert_eq!(result.actions[2].action_type, "METAD");
    }

    #[test]
    fn test_opes_input() {
        let input = r#"
hlda: COMBINE ARG=d1,d2,d3 COEFFICIENTS=0.6,0.5,0.5 PERIODIC=NO
OPES_METAD_EXPLORE LABEL=opesE ARG=hlda PACE=500 BARRIER=20 TEMP=340 FILE=KE.data
OPES_METAD LABEL=opes ARG=hlda PACE=500 BARRIER=30 TEMP=340 FILE=K.data
PRINT ARG=hlda,opes.bias FILE=COLVAR
"#;
        let result = parse_plumed_string(input).unwrap();
        assert!(result.is_valid);

        // Should warn about OPES module
        assert!(!result.warnings.is_empty());
        assert!(result.warnings.iter().any(|w| w.contains("opes")));
    }

    #[test]
    fn test_multiline_syntax() {
        let input = r#"
METAD ...
  LABEL=metad
  ARG=phi,psi
  PACE=500
  HEIGHT=1.2
... METAD

PRINT ARG=phi FILE=COLVAR
"#;
        let result = parse_plumed_string(input).unwrap();
        assert!(result.is_valid);
        assert_eq!(result.actions.len(), 2);
        assert_eq!(result.actions[0].action_type, "METAD");
    }

    #[test]
    fn test_label_colon_syntax() {
        let input = "phi: TORSION ATOMS=5,7,9,15\n";
        let result = parse_plumed_string(input).unwrap();
        assert_eq!(result.actions[0].label, Some("phi".to_string()));
        assert_eq!(result.actions[0].action_type, "TORSION");
    }
}
