// SPDX-License-Identifier: AGPL-3.0-or-later

//! Squirrel / neuralSpring inference over NUCLEUS JSON-RPC.
//!
//! When Squirrel is composed into the biomeOS graph, it discovers **neuralSpring**
//! (or another provider) as the backend for the `inference.*` capability domain.
//! This module is a thin typed client: it routes via
//! [`crate::primal_bridge::NucleusContext::call_by_capability`] with domain `"inference"`
//! (same discovery rule as [`crate::composition::get_by_capability`]).
//!
//! **Fallback path:** until native WGSL inference ships in neuralSpring, deployments
//! typically route through **Ollama** (or similar) behind Squirrel — same wire contract,
//! different provider. Callers should treat [`SquirrelError::Unavailable`] as “no
//! inference primal in this NUCLEUS snapshot” and continue with local heuristics or
//! barraCuda-side models when appropriate.

use crate::primal_bridge::NucleusContext;

const INFERENCE_DOMAIN: &str = "inference";

/// JSON-RPC method names for the inference domain (ecoPrimal wire contract).
mod wire {
    pub const COMPLETE: &str = "inference.complete";
    pub const EMBED: &str = "inference.embed";
    pub const MODELS: &str = "inference.models";
}

/// Errors from Squirrel / inference IPC calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SquirrelError {
    /// No alive primal advertising an `inference*` capability was discovered.
    Unavailable,
    /// JSON-RPC transport or protocol failure (connect, timeout, parse).
    Ipc(String),
    /// Response parsed but missing `result` or failed schema decode.
    InvalidResponse(String),
}

impl std::fmt::Display for SquirrelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable => write!(
                f,
                "inference provider unavailable (no inference capability in NUCLEUS)"
            ),
            Self::Ipc(s) => write!(f, "inference IPC: {s}"),
            Self::InvalidResponse(s) => write!(f, "inference invalid response: {s}"),
        }
    }
}

impl std::error::Error for SquirrelError {}

fn inference_call(
    ctx: &NucleusContext,
    method: &str,
    params: serde_json::Value,
) -> Result<serde_json::Value, SquirrelError> {
    if ctx.get_by_capability(INFERENCE_DOMAIN).is_none() {
        return Err(SquirrelError::Unavailable);
    }
    ctx.call_by_capability(INFERENCE_DOMAIN, method, params)
        .map_err(SquirrelError::Ipc)
}

fn rpc_result(resp: &serde_json::Value) -> Result<&serde_json::Value, SquirrelError> {
    if let Some(err) = resp.get("error") {
        return Err(SquirrelError::InvalidResponse(format!(
            "JSON-RPC error: {err}"
        )));
    }
    resp.get("result")
        .ok_or_else(|| SquirrelError::InvalidResponse("missing result".into()))
}

/// Text completion via `inference.complete` (ecoPrimal [`CompleteRequest`] shape).
pub fn inference_complete(
    ctx: &NucleusContext,
    prompt: &str,
    model: Option<&str>,
) -> Result<String, SquirrelError> {
    let mut obj = serde_json::Map::new();
    obj.insert(
        "prompt".into(),
        serde_json::Value::String(prompt.to_string()),
    );
    if let Some(m) = model {
        obj.insert("model".into(), serde_json::Value::String(m.to_string()));
    }
    let params = serde_json::Value::Object(obj);
    let resp = inference_call(ctx, wire::COMPLETE, params)?;
    let result = rpc_result(&resp)?;
    let text = result
        .get("text")
        .and_then(|v| v.as_str())
        .map(std::string::ToString::to_string)
        .ok_or_else(|| SquirrelError::InvalidResponse("missing text in complete result".into()))?;
    Ok(text)
}

/// Embedding vector via `inference.embed` — returns the first embedding as `f64`.
pub fn inference_embed(ctx: &NucleusContext, text: &str) -> Result<Vec<f64>, SquirrelError> {
    let params = serde_json::json!({
        "input": text,
    });
    let resp = inference_call(ctx, wire::EMBED, params)?;
    let result = rpc_result(&resp)?;
    let arr = result
        .get("embeddings")
        .and_then(|v| v.as_array())
        .ok_or_else(|| SquirrelError::InvalidResponse("missing embeddings array".into()))?;
    let first = arr
        .first()
        .ok_or_else(|| SquirrelError::InvalidResponse("empty embeddings in embed result".into()))?;
    let vec = first
        .as_array()
        .ok_or_else(|| SquirrelError::InvalidResponse("embedding is not an array".into()))?;
    let mut out = Vec::with_capacity(vec.len());
    for v in vec {
        let x = v.as_f64().ok_or_else(|| {
            SquirrelError::InvalidResponse("non-numeric embedding component".into())
        })?;
        out.push(x);
    }
    Ok(out)
}

/// List model identifiers via `inference.models`.
pub fn inference_models(ctx: &NucleusContext) -> Result<Vec<String>, SquirrelError> {
    let params = serde_json::json!({});
    let resp = inference_call(ctx, wire::MODELS, params)?;
    let result = rpc_result(&resp)?;
    let models = result
        .get("models")
        .and_then(|v| v.as_array())
        .ok_or_else(|| SquirrelError::InvalidResponse("missing models array".into()))?;
    let mut ids = Vec::with_capacity(models.len());
    for m in models {
        let id = m
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| SquirrelError::InvalidResponse("model entry missing id".into()))?;
        ids.push(id.to_string());
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primal_bridge::PrimalEndpoint;
    use std::collections::HashMap;

    fn ctx_with_inference(cap_domain: &str) -> NucleusContext {
        let caps = serde_json::json!({
            "capabilities": [format!("{cap_domain}.complete")]
        });
        let ep = PrimalEndpoint {
            name: "squirrel".into(),
            socket: "/tmp/test.sock".into(),
            alive: true,
            capabilities: Some(caps),
        };
        let mut discovered = HashMap::new();
        discovered.insert("squirrel".into(), ep);
        NucleusContext {
            discovered,
            family_id: "test".into(),
        }
    }

    #[test]
    fn get_by_capability_finds_inference() {
        let ctx = ctx_with_inference("inference");
        assert!(ctx.get_by_capability(INFERENCE_DOMAIN).is_some());
    }

    #[test]
    fn unavailable_when_no_inference() {
        let ctx = NucleusContext {
            discovered: HashMap::new(),
            family_id: "t".into(),
        };
        assert!(matches!(
            inference_complete(&ctx, "hi", None),
            Err(SquirrelError::Unavailable)
        ));
    }
}
