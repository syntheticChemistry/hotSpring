#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# nucleus_composition_lib.sh — Reusable NUCLEUS composition wiring
#
# Source this library from a domain-specific composition script to get:
#   - Capability discovery (family-aware UDS resolution)
#   - JSON-RPC transport (socat over UDS)
#   - petalTongue motor/visualization/interaction/proprioception
#   - rhizoCrypt DAG session management
#   - loamSpine ledger spine + entry + seal
#   - sweetGrass braid provenance fabric
#   - Startup personality + teardown
#   - Sensor stream processing with discrete event isolation
#
# Required variables (set before sourcing):
#   COMPOSITION_NAME  — short identifier (e.g. "ttt", "hotspring-sim")
#
# Optional variables:
#   FAMILY_ID         — socket namespace (default: $COMPOSITION_NAME)
#   SOCKET_DIR        — UDS directory (default: $XDG_RUNTIME_DIR/biomeos)
#   POLL_INTERVAL     — main loop sleep (default: 0.5)
#   REQUIRED_CAPS     — space-separated required capabilities
#   OPTIONAL_CAPS     — space-separated optional capabilities

if [[ -z "${COMPOSITION_NAME:-}" ]]; then
    echo "ERROR: COMPOSITION_NAME must be set before sourcing nucleus_composition_lib.sh" >&2
    exit 1
fi

FAMILY_ID="${FAMILY_ID:-$COMPOSITION_NAME}"
SOCKET_DIR="${XDG_RUNTIME_DIR:-/tmp}/biomeos"
POLL_INTERVAL="${POLL_INTERVAL:-0.5}"
REQUIRED_CAPS="${REQUIRED_CAPS:-visualization security}"
OPTIONAL_CAPS="${OPTIONAL_CAPS:-compute tensor dag ledger attribution}"

# ── Logging ───────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[$COMPOSITION_NAME]${NC} $*"; }
ok()   { echo -e "${GREEN}[$COMPOSITION_NAME]${NC} $*"; }
warn() { echo -e "${YELLOW}[$COMPOSITION_NAME]${NC} $*"; }
err()  { echo -e "${RED}[$COMPOSITION_NAME]${NC} $*" >&2; }

# ── Capability Discovery ─────────────────────────────────────────────

resolve_capability() {
    local capability="$1"
    local fallback_primal="${2:-}"

    local cap_sock="$SOCKET_DIR/${capability}-${FAMILY_ID}.sock"
    if [[ -S "$cap_sock" ]]; then
        echo "$cap_sock"
        return 0
    fi

    if [[ -n "$fallback_primal" ]]; then
        local primal_sock="$SOCKET_DIR/${fallback_primal}-${FAMILY_ID}.sock"
        if [[ -S "$primal_sock" ]]; then
            echo "$primal_sock"
            return 0
        fi
    fi

    return 1
}

# Capability → socket path map (populated by discover_capabilities)
declare -A CAP_SOCKETS=()

# Standard primal fallback names for each capability domain
declare -A CAP_FALLBACKS=(
    [visualization]="petaltongue"
    [security]="beardog"
    [compute]="toadstool"
    [tensor]="barracuda"
    [dag]="rhizocrypt"
    [ledger]="loamspine"
    [attribution]="sweetgrass"
    [discovery]="songbird"
    [storage]="nestgate"
    [shader]="coralreef"
    [ai]="squirrel"
    [inference]="squirrel"
    [crypto]="beardog"
    [btsp]="beardog"
)

discover_capabilities() {
    log "── Capability Discovery ──"
    local found=0 total=0

    for cap in $REQUIRED_CAPS $OPTIONAL_CAPS; do
        total=$((total + 1))
    done

    local discovered=0
    for cap in $REQUIRED_CAPS; do
        local fallback="${CAP_FALLBACKS[$cap]:-}"
        local sock
        if sock=$(resolve_capability "$cap" "$fallback"); then
            CAP_SOCKETS[$cap]="$sock"
            ok "$cap: $sock"
            discovered=$((discovered + 1))
        else
            err "$cap NOT FOUND — required"
            return 1
        fi
    done

    for cap in $OPTIONAL_CAPS; do
        local fallback="${CAP_FALLBACKS[$cap]:-}"
        local sock
        if sock=$(resolve_capability "$cap" "$fallback"); then
            CAP_SOCKETS[$cap]="$sock"
            ok "$cap: $sock"
            discovered=$((discovered + 1))
        else
            warn "$cap not found — feature disabled"
        fi
    done

    log "── Discovered $discovered/$total capabilities ──"
}

cap_socket() {
    local cap="$1"
    echo "${CAP_SOCKETS[$cap]:-}"
}

cap_available() {
    local cap="$1"
    [[ -n "${CAP_SOCKETS[$cap]:-}" ]]
}

# ── JSON-RPC Transport ────────────────────────────────────────────────

send_rpc() {
    local sock="$1" method="$2" params="$3"
    local id=$((RANDOM % 9999 + 1))
    local payload
    payload=$(printf '{"jsonrpc":"2.0","method":"%s","params":%s,"id":%d}' "$method" "$params" "$id")
    echo "$payload" | timeout 5 socat - "UNIX-CONNECT:$sock" 2>/dev/null || true
}

send_rpc_quiet() {
    local sock="$1" method="$2" params="$3"
    send_rpc "$sock" "$method" "$params" > /dev/null 2>&1 || true
}

# ── petalTongue: Motor Commands ───────────────────────────────────────

motor_set_panel() {
    local panel="$1" visible="$2"
    cap_available visualization || return
    send_rpc_quiet "$(cap_socket visualization)" "motor.set_panel" \
        "{\"panel\":\"$panel\",\"visible\":$visible}"
}

motor_continuous() {
    local enabled="$1"
    cap_available visualization || return
    send_rpc_quiet "$(cap_socket visualization)" "motor.continuous" \
        "{\"enabled\":$enabled}"
}

motor_fit_to_view() {
    cap_available visualization || return
    send_rpc_quiet "$(cap_socket visualization)" "motor.fit_to_view" "{}"
}

# ── petalTongue: Proprioception ───────────────────────────────────────

poll_proprioception() {
    cap_available visualization || { echo "{}"; return; }
    send_rpc "$(cap_socket visualization)" "proprioception.get" "{}"
}

# ── petalTongue: Scene Lifecycle ──────────────────────────────────────

dismiss_scene() {
    local session_id="$1"
    cap_available visualization || return
    send_rpc_quiet "$(cap_socket visualization)" "visualization.dismiss" \
        "{\"session_id\":\"$session_id\"}"
}

push_scene() {
    local session_id="$1" scene_json="$2"
    cap_available visualization || return
    send_rpc_quiet "$(cap_socket visualization)" "visualization.render.scene" \
        "{\"session_id\":\"$session_id\",\"scene\":$scene_json}"
}

make_text_node() {
    local id="$1" x="$2" y="$3" content="$4" size="$5" r="$6" g="$7" b="$8"
    printf '"%s":{"id":"%s","transform":{"a":1.0,"b":0.0,"c":0.0,"d":1.0,"tx":0.0,"ty":0.0},"primitives":[{"Text":{"x":%s.0,"y":%s.0,"content":"%s","font_size":%s.0,"color":{"r":%s,"g":%s,"b":%s,"a":1.0},"anchor":"Center","bold":false,"italic":false,"data_id":null}}],"children":[],"visible":true,"opacity":1.0,"label":null,"data_source":null}' \
        "$id" "$id" "$x" "$y" "$content" "$size" "$r" "$g" "$b"
}

# ── Startup Personality ───────────────────────────────────────────────
#
# composition_startup title subtitle
#   title    — main splash title (e.g. "NUCLEUS Tic-Tac-Toe")
#   subtitle — splash subtitle (e.g. "Composition Startup")

composition_startup() {
    local title="${1:-NUCLEUS $COMPOSITION_NAME}"
    local subtitle="${2:-Composition Startup}"
    local splash_id="${COMPOSITION_NAME}-splash"

    log "── Startup Personality ──"

    motor_set_panel "top_menu" false
    motor_set_panel "controls" false
    motor_set_panel "dashboard" false
    motor_set_panel "audio" false
    motor_set_panel "trust" false
    motor_set_panel "proprioception" false
    motor_continuous true

    local title_node
    title_node=$(make_text_node "sp-title" 230 120 "$title" 32 0.95 0.95 1.0)
    local sub_node
    sub_node=$(make_text_node "sp-sub" 230 160 "$subtitle" 16 0.5 0.7 0.5)

    local cap_nodes="" cap_ids="" y=210 idx=0
    for cap in $REQUIRED_CAPS $OPTIONAL_CAPS; do
        local color_r=0.4 color_g=0.4 color_b=0.5 label
        if cap_available "$cap"; then
            color_r=0.3; color_g=0.85; color_b=0.4
            label="  ✓ ${cap}"
        else
            color_r=0.7; color_g=0.7; color_b=0.3
            label="  ~ ${cap} — offline"
        fi
        local nid="sp-cap-${idx}"
        local node
        node=$(make_text_node "$nid" 230 $y "$label" 13 $color_r $color_g $color_b)
        if [[ -n "$cap_nodes" ]]; then
            cap_nodes="${cap_nodes},${node}"
            cap_ids="${cap_ids},\"${nid}\""
        else
            cap_nodes="${node}"
            cap_ids="\"${nid}\""
        fi
        y=$((y + 22))
        idx=$((idx + 1))
    done

    local status_node
    status_node=$(make_text_node "sp-status" 230 420 "Starting..." 14 0.6 0.6 0.7)

    local children="\"sp-title\",\"sp-sub\""
    [[ -n "$cap_ids" ]] && children="${children},${cap_ids}"
    children="${children},\"sp-status\""

    local all_nodes="${title_node},${sub_node}"
    [[ -n "$cap_nodes" ]] && all_nodes="${all_nodes},${cap_nodes}"
    all_nodes="${all_nodes},${status_node}"

    local root
    root=$(printf '"root":{"id":"root","transform":{"a":1.0,"b":0.0,"c":0.0,"d":1.0,"tx":0.0,"ty":0.0},"primitives":[],"children":[%s],"visible":true,"opacity":1.0,"label":null,"data_source":null}' "$children")

    local scene="{\"nodes\":{${root},${all_nodes}},\"root_id\":\"root\"}"
    push_scene "$splash_id" "$scene"
    sleep 0.5

    local proprio
    proprio=$(poll_proprioception)
    if echo "$proprio" | grep -q '"frame_rate"'; then
        local fps
        fps=$(echo "$proprio" | grep -oP '"frame_rate"\s*:\s*\K[0-9.]+' | head -1 || echo "?")
        ok "proprioception: petalTongue alive, fps=${fps}"
    fi
    sleep 0.8

    dismiss_scene "$splash_id"
    sleep 0.1
    ok "startup sequence complete"
}

# ── Security: Crypto Signing ──────────────────────────────────────────

sign_payload() {
    local message="$1"
    cap_available security || return
    local resp
    resp=$(send_rpc "$(cap_socket security)" "crypto.sign" \
        "{\"message\":\"$(echo -n "$message" | base64)\"}")
    local sig
    sig=$(echo "$resp" | grep -o '"signature":"[^"]*"' | head -1 | cut -d'"' -f4 || true)
    if [[ -n "$sig" ]]; then
        ok "signed: ${sig:0:16}..."
    else
        warn "sign returned: $(echo "$resp" | head -c 120)"
    fi
}

# ── rhizoCrypt: DAG Session Management ────────────────────────────────

DAG_SESSION=""
CURRENT_VERTEX=""
GENESIS_VERTEX=""
declare -a VERTEX_STACK=()
declare -a STATE_STACK=()
declare -a INPUT_TYPE_STACK=()
declare -a HOVER_COUNT_STACK=()

dag_create_session() {
    local domain="${1:-$COMPOSITION_NAME}"
    local genesis_metadata="${2:-[]}"
    CURRENT_VERTEX=""
    GENESIS_VERTEX=""
    VERTEX_STACK=()
    STATE_STACK=()
    INPUT_TYPE_STACK=()
    HOVER_COUNT_STACK=()
    ACCUMULATED_HOVER_MOVES=0
    braid_init_session

    cap_available dag || return

    local resp
    resp=$(send_rpc "$(cap_socket dag)" "dag.session.create" \
        "{\"description\":\"${domain}-$(date +%s)\",\"session_type\":\"General\"}")
    DAG_SESSION=$(echo "$resp" | grep -oP '"result"\s*:\s*"\K[^"]+' | head -1 || true)
    if [[ -z "$DAG_SESSION" ]]; then
        warn "dag.session.create: $(echo "$resp" | head -c 120)"
        return
    fi
    ok "DAG session: $DAG_SESSION"

    local gen_resp
    gen_resp=$(send_rpc "$(cap_socket dag)" "dag.event.append" \
        "{\"session_id\":\"$DAG_SESSION\",\"event_type\":{\"Custom\":{\"domain\":\"$domain\",\"event_name\":\"genesis\"}},\"parents\":[],\"metadata\":$genesis_metadata}")
    GENESIS_VERTEX=$(echo "$gen_resp" | grep -oP '"result"\s*:\s*"\K[^"]+' | head -1 || true)
    if [[ -n "$GENESIS_VERTEX" ]]; then
        CURRENT_VERTEX="$GENESIS_VERTEX"
        VERTEX_STACK=("$GENESIS_VERTEX")
        STATE_STACK=("")
        INPUT_TYPE_STACK=("genesis")
        HOVER_COUNT_STACK=("0")
        ok "DAG genesis: ${GENESIS_VERTEX:0:16}..."
    fi
}

dag_append_event() {
    local domain="$1" event_name="$2" state_snapshot="$3" metadata_json="$4"
    local input_type="${5:-unknown}" hover_moves="${6:-0}"
    cap_available dag || return
    [[ -n "$DAG_SESSION" ]] || return

    local parents_json="[]"
    if [[ -n "$CURRENT_VERTEX" ]]; then
        parents_json="[\"$CURRENT_VERTEX\"]"
    fi

    local resp
    resp=$(send_rpc "$(cap_socket dag)" "dag.event.append" \
        "{\"session_id\":\"$DAG_SESSION\",\"event_type\":{\"Custom\":{\"domain\":\"$domain\",\"event_name\":\"$event_name\"}},\"parents\":$parents_json,\"metadata\":$metadata_json}")

    local vertex_id
    vertex_id=$(echo "$resp" | grep -oP '"result"\s*:\s*"\K[^"]+' | head -1 || true)
    if [[ -n "$vertex_id" ]]; then
        CURRENT_VERTEX="$vertex_id"
        VERTEX_STACK+=("$vertex_id")
        STATE_STACK+=("$state_snapshot")
        INPUT_TYPE_STACK+=("$input_type")
        HOVER_COUNT_STACK+=("$hover_moves")
        ok "DAG vertex: ${vertex_id:0:16}... ($event_name via $input_type)"
    else
        warn "dag.event.append: $(echo "$resp" | head -c 120)"
    fi
}

dag_get_children() {
    local vertex_id="$1"
    cap_available dag || return
    [[ -n "$DAG_SESSION" ]] || return
    send_rpc "$(cap_socket dag)" "dag.vertex.children" \
        "{\"session_id\":\"$DAG_SESSION\",\"vertex_id\":\"$vertex_id\"}"
}

dag_get_frontier() {
    cap_available dag || return
    [[ -n "$DAG_SESSION" ]] || return
    send_rpc "$(cap_socket dag)" "dag.frontier.get" \
        "{\"session_id\":\"$DAG_SESSION\"}"
}

dag_merkle_root() {
    cap_available dag || return
    [[ -n "$DAG_SESSION" ]] || return
    local resp
    resp=$(send_rpc "$(cap_socket dag)" "dag.merkle.root" "{\"session_id\":\"$DAG_SESSION\"}")
    echo "$resp" | grep -oP '"result"\s*:\s*"\K[^"]+' | head -1 || true
}

# ── loamSpine: Ledger ─────────────────────────────────────────────────

SPINE_ID=""

ledger_create_spine() {
    local name="${1:-${COMPOSITION_NAME}-$(date +%s)}"
    local owner="${2:-${COMPOSITION_NAME}-composition}"
    cap_available ledger || return

    local resp
    resp=$(send_rpc "$(cap_socket ledger)" "spine.create" \
        "{\"name\":\"$name\",\"owner\":\"$owner\"}")
    SPINE_ID=$(echo "$resp" | grep -oP '"spine_id"\s*:\s*"\K[^"]+' | head -1 || true)
    if [[ -n "$SPINE_ID" ]]; then
        ok "ledger spine: $SPINE_ID"
    else
        warn "spine.create: $(echo "$resp" | head -c 120)"
    fi
}

ledger_append_entry() {
    local bond_id="$1" data_json="$2" committer="${3:-${COMPOSITION_NAME}-composition}"
    cap_available ledger || return
    [[ -n "$SPINE_ID" ]] || return

    local params
    params=$(printf '{"spine_id":"%s","entry_type":{"BondLedgerRecord":{"bond_id":"%s","data":%s}},"committer":"%s"}' \
        "$SPINE_ID" "$bond_id" "$data_json" "$committer")
    local resp
    resp=$(send_rpc "$(cap_socket ledger)" "entry.append" "$params")
    echo "$resp" | grep -q '"entry_hash"'
}

ledger_seal_spine() {
    cap_available ledger || return
    [[ -n "$SPINE_ID" ]] || return

    local params
    params=$(printf '{"spine_id":"%s","sealer":"%s"}' "$SPINE_ID" "${COMPOSITION_NAME}-composition")
    local resp
    resp=$(send_rpc "$(cap_socket ledger)" "spine.seal" "$params")
    if echo "$resp" | grep -q '"success":true'; then
        ok "spine sealed: $SPINE_ID"
        return 0
    else
        warn "seal: $(echo "$resp" | head -c 120)"
        return 1
    fi
}

# ── sweetGrass: Braid Provenance Fabric ───────────────────────────────

BRAID_SESSION_TAG=""
LAST_BRAID_ID=""

braid_init_session() {
    BRAID_SESSION_TAG="${COMPOSITION_NAME}-$(date +%s)-$$"
    LAST_BRAID_ID=""
}

braid_record() {
    local event_name="$1" mime_type="$2" data_content="$3" custom_json="$4"
    local input_type="${5:-unknown}" hover_moves="${6:-0}"
    cap_available attribution || return

    local data_hash
    data_hash="blake3:$(echo -n "${event_name}:${data_content}:$(date +%s%N)" | sha256sum | cut -d' ' -f1)"

    local full_custom
    full_custom=$(printf '%s' "$custom_json" | sed 's/}$//')
    full_custom=$(printf '%s,"input_type":"%s","hover_moves":%d,"session":"%s"' \
        "$full_custom" "$input_type" "$hover_moves" "$BRAID_SESSION_TAG")
    [[ -n "$LAST_BRAID_ID" ]] && full_custom=$(printf '%s,"derived_from":"%s"' "$full_custom" "$LAST_BRAID_ID")
    [[ -n "$DAG_SESSION" ]] && full_custom=$(printf '%s,"dag_session":"%s"' "$full_custom" "$DAG_SESSION")
    full_custom="${full_custom}}"

    local metadata
    metadata=$(printf '{"title":"%s:%s","tags":["%s","%s","%s","%s"],"custom":%s}' \
        "$COMPOSITION_NAME" "$event_name" \
        "$COMPOSITION_NAME" "$BRAID_SESSION_TAG" "$event_name" "$input_type" \
        "$full_custom")

    local params
    params=$(printf '{"data_hash":"%s","mime_type":"%s","size":%d,"metadata":%s}' \
        "$data_hash" "$mime_type" "${#data_content}" "$metadata")

    local resp
    resp=$(send_rpc "$(cap_socket attribution)" "braid.create" "$params")

    local braid_id
    braid_id=$(echo "$resp" | grep -oP '"@id"\s*:\s*"\K[^"]+' | head -1 || true)
    if [[ -n "$braid_id" ]]; then
        LAST_BRAID_ID="$braid_id"
        ok "braid: $event_name via $input_type → ${braid_id}"
    else
        warn "braid.create: $(echo "$resp" | head -c 120)"
    fi
}

braid_query_recent() {
    local limit="${1:-10}"
    cap_available attribution || return
    send_rpc "$(cap_socket attribution)" "braid.query" \
        "{\"filter\":{\"tag\":\"$BRAID_SESSION_TAG\",\"limit\":$limit},\"order\":\"NewestFirst\"}"
}

braid_provenance_tree() {
    cap_available attribution || return
    [[ -n "$LAST_BRAID_ID" ]] || return
    send_rpc "$(cap_socket attribution)" "provenance.graph" \
        "{\"entity\":{\"braid_id\":\"$LAST_BRAID_ID\"},\"depth\":20}"
}

# ── Interaction Stack ─────────────────────────────────────────────────

SUBSCRIBER_ID=""
SENSOR_SUB_ID=""
HOVER_CELL=-1
PROPRIO_INTERVAL=5
LAST_PROPRIO=0
ACCUMULATED_HOVER_MOVES=0

SENSOR_CLICK_CELL=-1
SENSOR_KEY=""
SENSOR_HOVER_CHANGED=false
SENSOR_INPUT_TYPE=""
SENSOR_HOVER_MOVES=0

subscribe_interactions() {
    local events="${1:-click}"
    SUBSCRIBER_ID="${COMPOSITION_NAME}-$$"
    cap_available visualization || return
    local resp
    resp=$(send_rpc "$(cap_socket visualization)" "interaction.subscribe" \
        "{\"subscriber_id\":\"$SUBSCRIBER_ID\",\"events\":[\"$events\"]}")
    if echo "$resp" | grep -q '"subscribed":true'; then
        ok "L1 semantic: subscribed (id=$SUBSCRIBER_ID)"
    else
        warn "L1 semantic: $(echo "$resp" | head -c 120)"
    fi
}

poll_interaction() {
    cap_available visualization || return
    send_rpc "$(cap_socket visualization)" "interaction.poll" \
        "{\"subscriber_id\":\"$SUBSCRIBER_ID\"}"
}

subscribe_sensor_stream() {
    cap_available visualization || return
    local resp
    resp=$(send_rpc "$(cap_socket visualization)" "interaction.sensor_stream.subscribe" "{}")
    SENSOR_SUB_ID=$(echo "$resp" | grep -oP '"subscription_id"\s*:\s*"\K[^"]+' | head -1 || true)
    if [[ -n "$SENSOR_SUB_ID" ]]; then
        ok "L2 sensor stream: subscribed (id=$SENSOR_SUB_ID)"
    else
        warn "L2 sensor stream: $(echo "$resp" | head -c 120)"
    fi
}

unsubscribe_sensor_stream() {
    [[ -n "$SENSOR_SUB_ID" ]] || return
    cap_available visualization || return
    send_rpc_quiet "$(cap_socket visualization)" "interaction.sensor_stream.unsubscribe" \
        "{\"subscription_id\":\"$SENSOR_SUB_ID\"}"
}

poll_sensor_stream() {
    [[ -n "$SENSOR_SUB_ID" ]] || return
    cap_available visualization || return
    send_rpc "$(cap_socket visualization)" "interaction.sensor_stream.poll" \
        "{\"subscription_id\":\"$SENSOR_SUB_ID\"}"
}

# Pluggable hit-test function: override in domain script
# Default: returns -1 (no spatial hit testing)
# Domain scripts should override: hit_test_fn() { your_logic "$1" "$2"; }
hit_test_fn() {
    echo -1
}

process_sensor_batch() {
    local batch="$1"
    SENSOR_CLICK_CELL=-1
    SENSOR_KEY=""
    SENSOR_HOVER_CHANGED=false
    SENSOR_INPUT_TYPE=""
    SENSOR_HOVER_MOVES=0

    if [[ -z "$batch" ]] || ! echo "$batch" | grep -q '"events"'; then return; fi

    local parsed
    parsed=$(echo "$batch" | python3 -c '
import json,sys
try:
    d=json.load(sys.stdin)
    evts=d.get("result",d).get("events",[])
except: evts=[]
mx=my=cx=cy=None; key=""; hc=0
for e in evts:
    t=e.get("type","")
    if t=="pointer_move":
        mx,my=e.get("x"),e.get("y"); hc+=1
    elif t=="click":
        cx,cy=e.get("x"),e.get("y")
    elif t=="key_press":
        key=e.get("key","")
if mx is not None: print(f"MX={mx}\nMY={my}")
if cx is not None: print(f"CX={cx}\nCY={cy}")
if key: print(f"KEY={key}")
print(f"HC={hc}")
' 2>/dev/null || echo "")

    local _mx="" _my="" _cx="" _cy="" _key="" _hc="0"
    while IFS='=' read -r k v; do
        case "$k" in
            MX) _mx="$v" ;; MY) _my="$v" ;;
            CX) _cx="$v" ;; CY) _cy="$v" ;;
            KEY) _key="$v" ;; HC) _hc="$v" ;;
        esac
    done <<< "$parsed"

    SENSOR_HOVER_MOVES="$_hc"

    if [[ -n "$_mx" ]] && [[ -n "$_my" ]]; then
        local new_hover
        new_hover=$(hit_test_fn "$_mx" "$_my")
        if [[ "$new_hover" != "$HOVER_CELL" ]]; then
            HOVER_CELL=$new_hover
            SENSOR_HOVER_CHANGED=true
        fi
    fi

    if [[ -n "$_cx" ]] && [[ -n "$_cy" ]]; then
        SENSOR_CLICK_CELL=$(hit_test_fn "$_cx" "$_cy")
        SENSOR_INPUT_TYPE="click"
    fi

    if [[ -n "$_key" ]]; then
        SENSOR_KEY="$_key"
        [[ -z "$SENSOR_INPUT_TYPE" ]] && SENSOR_INPUT_TYPE="keyboard"
    fi
}

check_proprioception() {
    local now
    now=$(date +%s)
    if (( now - LAST_PROPRIO < PROPRIO_INTERVAL )); then return; fi
    LAST_PROPRIO=$now

    local proprio
    proprio=$(poll_proprioception)
    if echo "$proprio" | grep -q '"frame_rate"'; then
        local fps activity
        fps=$(echo "$proprio" | grep -oP '"frame_rate"\s*:\s*\K[0-9.]+' | head -1 || echo "?")
        activity=$(echo "$proprio" | grep -oP '"user_interactivity"\s*:\s*"\K[^"]+' | head -1 || echo "?")
        log "  [proprio] fps=$fps activity=$activity"
    fi
}

# ── Teardown ──────────────────────────────────────────────────────────

composition_teardown() {
    local scene_ids="${*:-}"
    log "── Teardown ──"

    unsubscribe_sensor_stream
    if cap_available visualization && [[ -n "$SUBSCRIBER_ID" ]]; then
        send_rpc_quiet "$(cap_socket visualization)" "interaction.unsubscribe" \
            "{\"subscriber_id\":\"$SUBSCRIBER_ID\"}"
    fi

    for sid in $scene_ids; do
        dismiss_scene "$sid"
    done

    motor_continuous false
    sleep 0.2
    ok "composition teardown complete — petalTongue returned to clean canvas"
}

# ── Game Summary Helper ───────────────────────────────────────────────

composition_summary() {
    log "── Composition Summary ──"
    log "  DAG session: ${DAG_SESSION:-none}"
    log "  DAG vertices: ${#VERTEX_STACK[@]}"
    log "  Spine: ${SPINE_ID:-none}"

    local caps_status=""
    for cap in $REQUIRED_CAPS $OPTIONAL_CAPS; do
        if cap_available "$cap"; then
            caps_status="${caps_status} ${cap}:yes"
        else
            caps_status="${caps_status} ${cap}:no"
        fi
    done
    log "  Primals:${caps_status}"

    local merkle
    merkle=$(dag_merkle_root)
    [[ -n "$merkle" ]] && log "  Merkle root: ${merkle:0:16}..."

    if cap_available attribution; then
        local braid_resp
        braid_resp=$(braid_query_recent 5)
        local braid_count
        braid_count=$(echo "$braid_resp" | grep -oP '"total_count"\s*:\s*\K[0-9]+' | head -1 || echo "0")
        log "  Braids: $braid_count in session $BRAID_SESSION_TAG"
    fi

    local proprio
    proprio=$(poll_proprioception)
    if echo "$proprio" | grep -q '"frame_rate"'; then
        local fps scenes
        fps=$(echo "$proprio" | grep -oP '"frame_rate"\s*:\s*\K[0-9.]+' | head -1 || echo "?")
        scenes=$(echo "$proprio" | grep -oP '"active_scenes"\s*:\s*\K[0-9]+' | head -1 || echo "?")
        log "  Proprioception: fps=$fps active_scenes=$scenes"
    fi
}
