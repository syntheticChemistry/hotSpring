// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validates Squirrel / neuralSpring inference round-trip via NUCLEUS IPC.
//!
//! When Squirrel is available (via `inference` capability domain):
//!   1. `inference.models` — lists available models
//!   2. `inference.complete` — generates a completion
//!   3. `inference.embed` — generates an embedding vector
//!
//! When Squirrel is absent, all checks are honest skips (exit 2).

use hotspring_barracuda::primal_bridge::NucleusContext;
use hotspring_barracuda::squirrel_client;
use hotspring_barracuda::validation::CompositionResult;

fn main() {
    let ctx = NucleusContext::detect();
    let mut result = CompositionResult::new("validate_squirrel_roundtrip");

    let squirrel_alive = ctx
        .get_by_capability("inference")
        .is_some_and(|ep| ep.alive);

    result.section("Squirrel Discovery");
    if squirrel_alive {
        result.check_bool(
            "squirrel_discovered",
            true,
            "Squirrel found via capability routing",
        );
    } else {
        result.check_skip(
            "squirrel_discovered",
            "Squirrel not available — standalone or neuralSpring not running",
        );
    }

    result.section("inference.models");
    result.check_or_skip(
        "models_list",
        squirrel_alive.then_some(&ctx),
        "no Squirrel",
        |ctx, cr| match squirrel_client::inference_models(ctx) {
            Ok(models) => {
                let has_models = !models.is_empty();
                cr.check_bool(
                    "models_list",
                    has_models,
                    &format!("{} model(s) available", models.len()),
                );
            }
            Err(e) => {
                cr.check_bool("models_list", false, &format!("IPC error: {e}"));
            }
        },
    );

    result.section("inference.complete");
    result.check_or_skip(
        "complete_roundtrip",
        squirrel_alive.then_some(&ctx),
        "no Squirrel",
        |ctx, cr| match squirrel_client::inference_complete(ctx, "What is lattice QCD?", None) {
            Ok(text) => {
                let ok = !text.is_empty();
                cr.check_bool(
                    "complete_roundtrip",
                    ok,
                    &format!("got {} chars", text.len()),
                );
            }
            Err(e) => {
                cr.check_bool("complete_roundtrip", false, &format!("IPC error: {e}"));
            }
        },
    );

    result.section("inference.embed");
    result.check_or_skip(
        "embed_roundtrip",
        squirrel_alive.then_some(&ctx),
        "no Squirrel",
        |ctx, cr| match squirrel_client::inference_embed(ctx, "lattice QCD Monte Carlo") {
            Ok(embedding) => {
                let ok = !embedding.is_empty() && embedding.iter().all(|v| v.is_finite());
                cr.check_bool(
                    "embed_roundtrip",
                    ok,
                    &format!("{}-dim embedding, all finite", embedding.len()),
                );
            }
            Err(e) => {
                cr.check_bool("embed_roundtrip", false, &format!("IPC error: {e}"));
            }
        },
    );

    result.finish();
    std::process::exit(result.exit_code_skip_aware());
}
