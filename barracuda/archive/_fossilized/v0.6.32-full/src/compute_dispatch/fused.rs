// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fused multi-op dispatch pipeline (TensorSession evolution).
//!
//! Replaces the single-op `submit_workload` -> `retrieve_result` pattern
//! with a session that batches multiple operations and submits them as
//! a single dispatch unit. This is the hotSpring-side wiring for upstream
//! barraCuda's `TensorSession` concept (GAP-HS-027).
//!
//! # Protocol
//!
//! 1. Create a `FusedPipeline` with a session name
//! 2. Add operations via `push_op`
//! 3. Submit the entire batch via `submit`
//! 4. Retrieve fused results via `retrieve`

use crate::error::HotSpringError;
use crate::primal_bridge::NucleusContext;
use serde::{Deserialize, Serialize};

use super::retrieve_result;

/// A fused multi-op dispatch pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedPipeline {
    pub session_name: String,
    pub ops: Vec<FusedOp>,
    pub submitted_job_ids: Vec<String>,
}

/// A single operation within a fused pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedOp {
    pub shader: String,
    pub domain: String,
    pub input: serde_json::Value,
    #[serde(default)]
    pub depends_on: Vec<usize>,
}

/// Per-operation outcome from [`FusedPipeline::submit`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusedOpSubmitOutcome {
    Submitted(String),
    Failed(String),
}

/// Report returned by [`FusedPipeline::submit`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedSubmitReport {
    pub session_name: String,
    pub outcomes: Vec<FusedOpSubmitOutcome>,
}

impl FusedSubmitReport {
    #[must_use]
    pub fn all_submitted(&self) -> bool {
        self.outcomes
            .iter()
            .all(|o| matches!(o, FusedOpSubmitOutcome::Submitted(_)))
    }

    #[must_use]
    pub fn submitted_count(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|o| matches!(o, FusedOpSubmitOutcome::Submitted(_)))
            .count()
    }

    #[must_use]
    pub fn job_ids(&self) -> Vec<&str> {
        self.outcomes
            .iter()
            .filter_map(|o| match o {
                FusedOpSubmitOutcome::Submitted(id) => Some(id.as_str()),
                FusedOpSubmitOutcome::Failed(_) => None,
            })
            .collect()
    }
}

/// Result of a fused pipeline retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedResult {
    pub session_name: String,
    pub op_results: Vec<FusedOpResult>,
    pub all_succeeded: bool,
}

/// Result of a single operation within a fused pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedOpResult {
    pub index: usize,
    pub shader: String,
    pub succeeded: bool,
    pub result: Option<serde_json::Value>,
    pub error: Option<String>,
}

impl FusedPipeline {
    #[must_use]
    pub fn new(session_name: impl Into<String>) -> Self {
        Self {
            session_name: session_name.into(),
            ops: Vec::new(),
            submitted_job_ids: Vec::new(),
        }
    }

    pub fn push_op(
        &mut self,
        shader: impl Into<String>,
        domain: impl Into<String>,
        input: serde_json::Value,
    ) {
        self.ops.push(FusedOp {
            shader: shader.into(),
            domain: domain.into(),
            input,
            depends_on: Vec::new(),
        });
    }

    pub fn push_op_with_deps(
        &mut self,
        shader: impl Into<String>,
        domain: impl Into<String>,
        input: serde_json::Value,
        depends_on: Vec<usize>,
    ) {
        self.ops.push(FusedOp {
            shader: shader.into(),
            domain: domain.into(),
            input,
            depends_on,
        });
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Submit the fused pipeline via NUCLEUS IPC.
    ///
    /// Attempts `compute.dispatch.submit_fused` first; if unavailable,
    /// falls back to sequential `compute.dispatch.submit` per operation.
    pub fn submit(
        &mut self,
        nucleus: &NucleusContext,
    ) -> Result<FusedSubmitReport, HotSpringError> {
        let mut outcomes = Vec::with_capacity(self.ops.len());

        let fused_params = serde_json::json!({
            "session": self.session_name,
            "ops": self.ops,
        });

        if let Ok(resp) =
            nucleus.call_by_capability("compute", "compute.dispatch.submit_fused", fused_params)
        {
            if let Some(ids) = resp.get("job_ids").and_then(|v| v.as_array()) {
                self.submitted_job_ids = ids
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                outcomes = self
                    .submitted_job_ids
                    .iter()
                    .map(|id| FusedOpSubmitOutcome::Submitted(id.clone()))
                    .collect();
                return Ok(FusedSubmitReport {
                    session_name: self.session_name.clone(),
                    outcomes,
                });
            }
        }

        for op in &self.ops {
            let params = serde_json::json!({
                "shader": op.shader,
                "input": op.input,
                "spring": "hotSpring",
                "session": self.session_name,
            });
            match nucleus.call_by_capability("compute", "compute.dispatch.submit", params) {
                Ok(resp) => {
                    let job_id = resp
                        .get("result")
                        .and_then(|r| r.get("job_id"))
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    if let Some(id) = job_id {
                        self.submitted_job_ids.push(id.clone());
                        outcomes.push(FusedOpSubmitOutcome::Submitted(id));
                    } else {
                        let msg = "submit succeeded but response missing job_id".to_string();
                        outcomes.push(FusedOpSubmitOutcome::Failed(msg));
                    }
                }
                Err(e) => {
                    outcomes.push(FusedOpSubmitOutcome::Failed(e.to_string()));
                }
            }
        }

        Ok(FusedSubmitReport {
            session_name: self.session_name.clone(),
            outcomes,
        })
    }

    /// Retrieve results for all submitted operations.
    pub fn retrieve(&self, nucleus: &NucleusContext) -> FusedResult {
        let mut op_results = Vec::with_capacity(self.ops.len());

        for (i, job_id) in self.submitted_job_ids.iter().enumerate() {
            let shader = self.ops.get(i).map_or("unknown", |o| &o.shader).to_string();
            match retrieve_result(nucleus, job_id) {
                Ok(data) => {
                    op_results.push(FusedOpResult {
                        index: i,
                        shader,
                        succeeded: true,
                        result: Some(data),
                        error: None,
                    });
                }
                Err(e) => {
                    op_results.push(FusedOpResult {
                        index: i,
                        shader,
                        succeeded: false,
                        result: None,
                        error: Some(e.to_string()),
                    });
                }
            }
        }

        let all_succeeded = op_results.iter().all(|r| r.succeeded);

        FusedResult {
            session_name: self.session_name.clone(),
            op_results,
            all_succeeded,
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "fused pipeline tests use expect on test payloads"
)]
mod tests {
    use super::*;

    #[test]
    fn fused_pipeline_builds_ops() {
        let mut fp = FusedPipeline::new("test-session");
        assert!(fp.is_empty());

        fp.push_op("vector_add_f64", "compute", serde_json::json!({"a": [1.0]}));
        fp.push_op_with_deps(
            "reduce_sum_f64",
            "compute",
            serde_json::json!({"input": "prev"}),
            vec![0],
        );

        assert_eq!(fp.len(), 2);
        assert!(!fp.is_empty());
        assert_eq!(fp.ops[0].shader, "vector_add_f64");
        assert_eq!(fp.ops[1].depends_on, vec![0]);
    }

    #[test]
    fn fused_pipeline_serializes() {
        let mut fp = FusedPipeline::new("serialize-test");
        fp.push_op("saxpy", "compute", serde_json::json!({"alpha": 2.0}));

        let json = serde_json::to_string(&fp).expect("serialize");
        assert!(json.contains("serialize-test"));
        assert!(json.contains("saxpy"));

        let round_trip: FusedPipeline = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(round_trip.session_name, "serialize-test");
        assert_eq!(round_trip.ops.len(), 1);
    }

    #[test]
    fn fused_result_all_succeeded_logic() {
        let result = FusedResult {
            session_name: "test".into(),
            op_results: vec![
                FusedOpResult {
                    index: 0,
                    shader: "a".into(),
                    succeeded: true,
                    result: Some(serde_json::json!({})),
                    error: None,
                },
                FusedOpResult {
                    index: 1,
                    shader: "b".into(),
                    succeeded: true,
                    result: Some(serde_json::json!({})),
                    error: None,
                },
            ],
            all_succeeded: true,
        };
        assert!(result.all_succeeded);

        let partial = FusedResult {
            session_name: "test".into(),
            op_results: vec![FusedOpResult {
                index: 0,
                shader: "c".into(),
                succeeded: false,
                result: None,
                error: Some("dispatch failed".into()),
            }],
            all_succeeded: false,
        };
        assert!(!partial.all_succeeded);
    }
}
