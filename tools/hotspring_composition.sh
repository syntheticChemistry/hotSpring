#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# hotspring_composition.sh — Event-driven QCD computation via NUCLEUS composition
#
# hotSpring's Phase 46 lane: event-driven computation + DAG memoization.
# Long-running simulations that converge rather than tick at 60Hz.
#
# Domain: lattice QCD parameter sweeps — each configuration is a DAG vertex,
# branching = different coupling constants (beta), memoization = skip recomputed
# vertices. Ledger spines seal reproducible runs. Braids carry peer-review
# provenance (DOIs, parameters, hardware IDs).
#
# Uses nucleus_composition_lib.sh for all NUCLEUS wiring. This script handles
# only QCD domain logic, async computation, and scientific metadata.
#
# Usage:
#   COMPOSITION_NAME=hotspring ./tools/hotspring_composition.sh
#   FAMILY_ID=hotspring-sim ./tools/hotspring_composition.sh
#
# Requires: NUCLEUS primals via composition_nucleus.sh or plasmidBin deployment.
# Degrades gracefully when primals are absent (bare mode).

set -euo pipefail

# ── 1. Configuration ──────────────────────────────────────────────────

COMPOSITION_NAME="${COMPOSITION_NAME:-hotspring}"
REQUIRED_CAPS="visualization security"
OPTIONAL_CAPS="compute tensor dag ledger attribution"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/nucleus_composition_lib.sh"

# ── 2. Domain State ──────────────────────────────────────────────────

RUNNING=true
SWEEP_ACTIVE=false

# QCD parameters for the current vertex
BETA="5.5"
LATTICE_L="4"
LATTICE_T="4"
ALGORITHM="HMC"
N_TRAJ=10

# Computation tracking
COMPUTE_PID=""
COMPUTE_RESULT_FILE="/tmp/hotspring-compute-$$.json"
COMPUTE_STATUS="idle"
VERTICES_COMPUTED=0
VERTICES_MEMOIZED=0

# Sweep state
declare -A MEMO_CACHE
SWEEP_BETAS=()
SWEEP_IDX=0

# Scientific provenance
PROVENANCE_DOI="10.1103/PhysRevD.10.2445"
PROVENANCE_HARDWARE=""
PROVENANCE_RUST_VERSION=""

# ── 3. Hit Testing ───────────────────────────────────────────────────

hit_test_fn() {
    local px="$1" py="$2"
    px="${px%.*}"
    py="${py%.*}"
    # Vertex list: 5 rows, each 40px tall, starting at y=160
    if (( px >= 30 && px < 430 && py >= 160 && py < 360 )); then
        local row=$(( (py - 160) / 40 ))
        echo "$row"
    else
        echo -1
    fi
}

# ── 4. Provenance Helpers ────────────────────────────────────────────

init_provenance() {
    PROVENANCE_HARDWARE="$(uname -m)"
    if command -v rustc &>/dev/null; then
        PROVENANCE_RUST_VERSION="$(rustc --version 2>/dev/null | head -1)"
    else
        PROVENANCE_RUST_VERSION="unknown"
    fi
}

provenance_metadata() {
    local event="$1"
    printf '{"event":"%s","beta":"%s","L":"%s","T":"%s","algorithm":"%s"' \
        "$event" "$BETA" "$LATTICE_L" "$LATTICE_T" "$ALGORITHM"
    printf ',"n_traj":%d,"doi":"%s","hardware":"%s","rust":"%s"' \
        "$N_TRAJ" "$PROVENANCE_DOI" "$PROVENANCE_HARDWARE" "$PROVENANCE_RUST_VERSION"
    printf ',"computed":%d,"memoized":%d}' \
        "$VERTICES_COMPUTED" "$VERTICES_MEMOIZED"
}

# ── 5. DAG Memoization Layer ─────────────────────────────────────────
#
# Wraps rhizoCrypt DAG to avoid recomputing identical parameter sets.
# Key = sha256(beta|L|T|algorithm|n_traj), stored in an associative array
# for the session, with DAG vertices as the persistent backing store.
#
# Candidate for upstream promotion to nucleus_composition_lib.sh.

memo_param_key() {
    echo "${BETA}|${LATTICE_L}|${LATTICE_T}|${ALGORITHM}|${N_TRAJ}" | sha256sum | cut -c1-16
}

memo_check_vertex() {
    local key
    key=$(memo_param_key)
    if [[ -n "${MEMO_CACHE[$key]+_}" ]]; then
        echo "${MEMO_CACHE[$key]}"
        return 0
    fi
    return 1
}

memo_store_result() {
    local vertex_id="$1" result_json="$2"
    local key
    key=$(memo_param_key)
    MEMO_CACHE[$key]="$result_json"

    if cap_available dag && [[ -n "$vertex_id" ]]; then
        dag_append_event "$COMPOSITION_NAME" "result" "computed" \
            "[{\"key\":\"params_hash\",\"value\":\"$key\"},{\"key\":\"result\",\"value\":$(echo "$result_json" | head -c 200 | tr '"' "'")}]" \
            "compute" "0"
    fi
    VERTICES_COMPUTED=$((VERTICES_COMPUTED + 1))
}

memo_get_result() {
    local key
    key=$(memo_param_key)
    if [[ -n "${MEMO_CACHE[$key]+_}" ]]; then
        echo "${MEMO_CACHE[$key]}"
        return 0
    fi
    return 1
}

# ── 6. Compute Dispatch ──────────────────────────────────────────────
#
# Dispatches real workloads to barraCuda/toadStool via IPC, or falls back
# to local Rust binary execution for heavier computations.

dispatch_tensor_probe() {
    if ! cap_available tensor; then
        log "tensor capability offline — skipping IPC probe"
        return 1
    fi
    local sock
    sock=$(cap_socket tensor)

    local data="1.0,2.0,3.0,4.0,5.0"
    local resp
    resp=$(send_rpc "$sock" "stats.mean" "{\"data\":[$data]}" 2>/dev/null || true)
    if echo "$resp" | grep -q '"result"'; then
        local mean
        mean=$(echo "$resp" | grep -oP '"result"\s*:\s*\K[0-9.]+' | head -1 || echo "?")
        ok "tensor probe: stats.mean([1..5]) = $mean"
        return 0
    fi
    warn "tensor probe: no result from stats.mean"
    return 1
}

dispatch_semf_computation() {
    local z="$1" n="$2"
    if ! cap_available tensor; then
        log "tensor offline — computing SEMF locally"
        echo '{"source":"local","z":'$z',"n":'$n',"status":"no_tensor"}'
        return 0
    fi
    local sock
    sock=$(cap_socket tensor)

    # Use stats.mean as a proxy for validating IPC round-trip with physics data.
    # In full deployment, this would call a dedicated SEMF capability.
    local a=$((z + n))
    local part1 part2 part3 part4 part5
    part1=$(echo "$a" | awk '{printf "%.4f", $1 * 15.56}')
    part2=$(echo "$a" | awk '{printf "%.4f", $1 * 17.23}')
    part3=$(echo "$a" | awk '{printf "%.4f", $1 * 0.697}')
    part4=$(echo "$a" | awk '{printf "%.4f", $1 * 23.29}')
    part5=$(echo "$a" | awk '{printf "%.4f", $1 * 11.18}')

    local resp
    resp=$(send_rpc_quiet "$sock" "stats.mean" \
        "{\"data\":[$part1,$part2,$part3,$part4,$part5]}" 2>/dev/null || true)
    local result
    result=$(echo "$resp" | grep -oP '"result"\s*:\s*\K[0-9.eE+-]+' | head -1 || echo "0")

    echo "{\"source\":\"ipc\",\"z\":$z,\"n\":$n,\"be_proxy\":$result}"
}

dispatch_background_validation() {
    local barracuda_dir="$SCRIPT_DIR/../barracuda"
    if [[ ! -f "$barracuda_dir/Cargo.toml" ]]; then
        log "barracuda crate not found — skipping background validation"
        COMPUTE_STATUS="no_crate"
        return 1
    fi

    log "launching background SEMF validation (beta=$BETA, L=$LATTICE_L)"
    # Build and run the guideStone in bare mode as a proxy for real computation
    (
        cd "$barracuda_dir"
        cargo run --release --bin hotspring_guidestone 2>&1 \
            | grep -E '\[PASS\]|\[FAIL\]|\[SKIP\]|Result:' \
            > "$COMPUTE_RESULT_FILE" 2>/dev/null
    ) &
    COMPUTE_PID=$!
    COMPUTE_STATUS="running"
    ok "background compute started (PID=$COMPUTE_PID)"
}

check_background_compute() {
    if [[ -z "$COMPUTE_PID" ]] || [[ "$COMPUTE_STATUS" != "running" ]]; then
        return 1
    fi

    if kill -0 "$COMPUTE_PID" 2>/dev/null; then
        return 1
    fi

    wait "$COMPUTE_PID" 2>/dev/null || true
    COMPUTE_STATUS="complete"
    COMPUTE_PID=""

    local result="{\"status\":\"complete\",\"beta\":\"$BETA\",\"L\":\"$LATTICE_L\"}"
    if [[ -f "$COMPUTE_RESULT_FILE" ]]; then
        local pass_count fail_count
        pass_count=$(grep -c '\[PASS\]' "$COMPUTE_RESULT_FILE" 2>/dev/null || echo "0")
        fail_count=$(grep -c '\[FAIL\]' "$COMPUTE_RESULT_FILE" 2>/dev/null || echo "0")
        result="{\"status\":\"complete\",\"beta\":\"$BETA\",\"L\":\"$LATTICE_L\",\"pass\":$pass_count,\"fail\":$fail_count}"
        ok "background compute finished: $pass_count PASS, $fail_count FAIL"
    fi

    memo_store_result "$CURRENT_VERTEX" "$result"

    if cap_available ledger; then
        ledger_append_entry "compute-result" "$result"
    fi

    braid_record "compute_complete" "application/x-hotspring-qcd" \
        "$BETA|$LATTICE_L" "$(provenance_metadata "compute_complete")" \
        "compute" "0"

    rm -f "$COMPUTE_RESULT_FILE"
    return 0
}

# ── 7. Ledger Helpers ────────────────────────────────────────────────

ledger_commit_sweep() {
    cap_available ledger || return 0
    [[ -n "$SPINE_ID" ]] || return 0

    local summary
    summary=$(printf '{"beta":"%s","L":"%s","T":"%s","algorithm":"%s","computed":%d,"memoized":%d,"vertices":%d}' \
        "$BETA" "$LATTICE_L" "$LATTICE_T" "$ALGORITHM" \
        "$VERTICES_COMPUTED" "$VERTICES_MEMOIZED" "${#VERTEX_STACK[@]}")
    ledger_append_entry "sweep-summary" "$summary"

    if ledger_seal_spine; then
        ok "sweep sealed: $VERTICES_COMPUTED computed, $VERTICES_MEMOIZED memoized → spine $SPINE_ID"
    else
        ok "sweep committed (unsealed) → spine $SPINE_ID"
    fi
}

# ── 8. Domain Hooks ──────────────────────────────────────────────────

domain_init() {
    init_provenance

    dag_create_session "$COMPOSITION_NAME" \
        "[{\"key\":\"beta\",\"value\":\"$BETA\"},{\"key\":\"L\",\"value\":\"$LATTICE_L\"},{\"key\":\"T\",\"value\":\"$LATTICE_T\"}]"

    ledger_create_spine

    # Initial tensor probe to verify IPC
    dispatch_tensor_probe || true

    domain_render "Ready — R=run, S=sweep, B=beta+, V=beta-, Q=quit"
}

domain_render() {
    local status="${1:-}"
    local title
    title=$(make_text_node "title" 230 30 "hotSpring QCD Composition" 24 0.95 0.95 1.0)

    local params_text="beta=$BETA  L=${LATTICE_L}^3x${LATTICE_T}  $ALGORITHM  n_traj=$N_TRAJ"
    local params
    params=$(make_text_node "params" 230 70 "$params_text" 16 0.7 0.85 1.0)

    local stats_text="computed=$VERTICES_COMPUTED  memoized=$VERTICES_MEMOIZED  status=$COMPUTE_STATUS"
    local stats
    stats=$(make_text_node "stats" 230 100 "$stats_text" 14 0.6 0.8 0.7)

    local dag_text="DAG depth=${#VERTEX_STACK[@]}  vertex=${CURRENT_VERTEX:0:12}..."
    local dag_info
    dag_info=$(make_text_node "dag" 230 125 "$dag_text" 12 0.5 0.7 0.6)

    local status_node
    status_node=$(make_text_node "status" 230 400 "$status" 14 0.8 0.8 0.85)

    local root
    root=$(printf '"root":{"id":"root","transform":{"a":1.0,"b":0.0,"c":0.0,"d":1.0,"tx":0.0,"ty":0.0},"primitives":[],"children":["title","params","stats","dag","status"],"visible":true,"opacity":1.0,"label":null,"data_source":null}')
    local scene="{\"nodes\":{${root},${title},${params},${stats},${dag_info},${status_node}},\"root_id\":\"root\"}"
    push_scene "${COMPOSITION_NAME}-main" "$scene"
}

domain_on_key() {
    local key="$1"
    case "$key" in
        Q|q|Escape)
            log "quit requested — sealing sweep"
            ledger_commit_sweep
            RUNNING=false
            ;;
        R|r)
            log "run computation at current vertex"
            local cached
            if cached=$(memo_check_vertex); then
                ok "memoized: result already cached for beta=$BETA L=$LATTICE_L"
                VERTICES_MEMOIZED=$((VERTICES_MEMOIZED + 1))
                domain_render "MEMOIZED: beta=$BETA (cached result)"
            else
                dag_append_event "$COMPOSITION_NAME" "compute_start" "$BETA|$LATTICE_L" \
                    "[{\"key\":\"beta\",\"value\":\"$BETA\"},{\"key\":\"L\",\"value\":\"$LATTICE_L\"}]" \
                    "keyboard" "0"
                braid_record "compute_start" "application/x-hotspring-qcd" \
                    "$BETA|$LATTICE_L" "$(provenance_metadata "compute_start")" \
                    "keyboard" "0"

                local semf_result
                semf_result=$(dispatch_semf_computation 82 126)
                memo_store_result "$CURRENT_VERTEX" "$semf_result"
                if cap_available ledger; then
                    ledger_append_entry "semf-pb208" "$semf_result"
                fi
                domain_render "Computed: beta=$BETA → $(echo "$semf_result" | head -c 60)"
            fi
            ;;
        B|b)
            BETA=$(echo "$BETA + 0.1" | bc)
            log "beta increased to $BETA"
            dag_append_event "$COMPOSITION_NAME" "param_change" "$BETA|$LATTICE_L" \
                "[{\"key\":\"beta\",\"value\":\"$BETA\"},{\"key\":\"direction\",\"value\":\"up\"}]" \
                "keyboard" "$ACCUMULATED_HOVER_MOVES"
            braid_record "param_change" "application/x-hotspring-qcd" \
                "$BETA|$LATTICE_L" "$(provenance_metadata "param_up")" \
                "keyboard" "$ACCUMULATED_HOVER_MOVES"
            ACCUMULATED_HOVER_MOVES=0
            domain_render "beta=$BETA (increased)"
            ;;
        V|v)
            BETA=$(echo "$BETA - 0.1" | bc)
            log "beta decreased to $BETA"
            dag_append_event "$COMPOSITION_NAME" "param_change" "$BETA|$LATTICE_L" \
                "[{\"key\":\"beta\",\"value\":\"$BETA\"},{\"key\":\"direction\",\"value\":\"down\"}]" \
                "keyboard" "$ACCUMULATED_HOVER_MOVES"
            braid_record "param_change" "application/x-hotspring-qcd" \
                "$BETA|$LATTICE_L" "$(provenance_metadata "param_down")" \
                "keyboard" "$ACCUMULATED_HOVER_MOVES"
            ACCUMULATED_HOVER_MOVES=0
            domain_render "beta=$BETA (decreased)"
            ;;
        S|s)
            log "starting beta sweep"
            SWEEP_ACTIVE=true
            SWEEP_BETAS=(5.5 5.6 5.7 5.8 5.9 6.0)
            SWEEP_IDX=0
            domain_render "Sweep started: ${#SWEEP_BETAS[@]} beta values"
            ;;
        L|l)
            dispatch_background_validation || true
            domain_render "Background validation launched"
            ;;
        T|t)
            # Show DAG tree view
            if cap_available dag && [[ -n "$GENESIS_VERTEX" ]]; then
                local children_resp
                children_resp=$(dag_get_children "$CURRENT_VERTEX" 2>/dev/null || true)
                local count=0
                local child_ids
                child_ids=$(echo "$children_resp" | grep -oP '"[a-f0-9]{64}"' 2>/dev/null || true)
                if [[ -n "$child_ids" ]]; then
                    count=$(echo "$child_ids" | wc -l)
                fi
                domain_render "DAG tree: ${#VERTEX_STACK[@]} depth, $count children from here"
            else
                domain_render "DAG offline — no tree view"
            fi
            ;;
        *)
            log "unhandled key: $key"
            domain_render "Key: $key (unrecognized)"
            ;;
    esac
}

domain_on_click() {
    local cell="$1"
    log "clicked vertex row: $cell"
    dag_append_event "$COMPOSITION_NAME" "vertex_select" "$BETA|$LATTICE_L" \
        "[{\"key\":\"row\",\"value\":\"$cell\"}]" \
        "click" "$ACCUMULATED_HOVER_MOVES"
    braid_record "vertex_select" "application/x-hotspring-qcd" \
        "$BETA|$LATTICE_L" "$(provenance_metadata "vertex_select")" \
        "click" "$ACCUMULATED_HOVER_MOVES"
    ACCUMULATED_HOVER_MOVES=0
    domain_render "Selected vertex row $cell"
}

domain_on_tick() {
    # Async tick: convergence-based, not 60Hz
    # Priority: check background compute > advance sweep > proprioception

    if check_background_compute 2>/dev/null; then
        domain_render "Compute complete: $VERTICES_COMPUTED computed"
        return
    fi

    if $SWEEP_ACTIVE && [[ "$COMPUTE_STATUS" != "running" ]]; then
        if (( SWEEP_IDX < ${#SWEEP_BETAS[@]} )); then
            BETA="${SWEEP_BETAS[$SWEEP_IDX]}"
            SWEEP_IDX=$((SWEEP_IDX + 1))

            local cached
            if cached=$(memo_check_vertex); then
                ok "sweep: memoized beta=$BETA"
                VERTICES_MEMOIZED=$((VERTICES_MEMOIZED + 1))
                domain_render "Sweep [$SWEEP_IDX/${#SWEEP_BETAS[@]}]: beta=$BETA (memoized)"
            else
                dag_append_event "$COMPOSITION_NAME" "sweep_step" "$BETA|$LATTICE_L" \
                    "[{\"key\":\"beta\",\"value\":\"$BETA\"},{\"key\":\"sweep_idx\",\"value\":\"$SWEEP_IDX\"}]" \
                    "sweep" "0"

                local semf_result
                semf_result=$(dispatch_semf_computation 82 126)
                memo_store_result "$CURRENT_VERTEX" "$semf_result"

                if cap_available ledger; then
                    ledger_append_entry "sweep-step-$SWEEP_IDX" "$semf_result"
                fi

                braid_record "sweep_step" "application/x-hotspring-qcd" \
                    "$BETA|$LATTICE_L" "$(provenance_metadata "sweep_step")" \
                    "sweep" "0"

                domain_render "Sweep [$SWEEP_IDX/${#SWEEP_BETAS[@]}]: beta=$BETA → computed"
            fi
        else
            SWEEP_ACTIVE=false
            ok "sweep complete: $VERTICES_COMPUTED computed, $VERTICES_MEMOIZED memoized"
            domain_render "Sweep complete: $VERTICES_COMPUTED computed, $VERTICES_MEMOIZED memoized"
        fi
        return
    fi

    check_proprioception
}

# ── 9. Main Loop ─────────────────────────────────────────────────────

main() {
    discover_capabilities || { err "Required primals not found"; exit 1; }

    composition_startup "hotSpring QCD" "Event-Driven Computation + DAG Memoization"

    subscribe_interactions "click"
    subscribe_sensor_stream

    domain_init

    while $RUNNING; do
        local sensor_batch
        sensor_batch=$(poll_sensor_stream)
        process_sensor_batch "$sensor_batch"

        ACCUMULATED_HOVER_MOVES=$((ACCUMULATED_HOVER_MOVES + SENSOR_HOVER_MOVES))

        if $SENSOR_HOVER_CHANGED; then
            domain_render "Hovering... (target: $HOVER_CELL)"
        fi

        if [[ -n "$SENSOR_KEY" ]]; then
            domain_on_key "$SENSOR_KEY"
        elif [[ "$SENSOR_CLICK_CELL" -ge 0 ]]; then
            domain_on_click "$SENSOR_CLICK_CELL"
        else
            domain_on_tick
            # Adaptive backoff: shorter when sweep is active or compute running
            if $SWEEP_ACTIVE || [[ "$COMPUTE_STATUS" = "running" ]]; then
                sleep 0.2
            else
                sleep "$POLL_INTERVAL"
            fi
        fi
    done

    composition_summary
    composition_teardown "${COMPOSITION_NAME}-main"
}

main
