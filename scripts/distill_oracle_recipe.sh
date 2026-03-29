#!/bin/bash
# distill_oracle_recipe.sh — Feed mmiotrace through hw-learn distiller
#
# Takes a raw mmiotrace file and produces a golden init recipe (JSON)
# that GlowPlug's digital PMU can replay.
#
# Usage: ./scripts/distill_oracle_recipe.sh <mmiotrace_file> [output.json]

set -euo pipefail

TRACE="${1:?Usage: $0 <mmiotrace_file> [output.json]}"
OUTPUT="${2:-$(dirname "$TRACE")/oracle_recipe_$(date +%Y%m%d_%H%M%S).json}"
TOADSTOOL_DIR="${TOADSTOOL_DIR:-${HOME}/Development/ecoPrimals/phase1/toadStool}"
HW_LEARN_BIN="$TOADSTOOL_DIR/target/release/hw_learn_distill"

echo "=== Oracle Recipe Distillation ==="
echo "Input:  $TRACE"
echo "Output: $OUTPUT"
echo ""

# Step 1: Build hw-learn distiller if needed
if [ ! -x "$HW_LEARN_BIN" ]; then
    echo ">>> Building hw-learn distiller..."
    env -i HOME=$HOME PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${HOME}/.cargo/bin \
        cargo build --release -p hw-learn --bin hw_learn_distill \
        --manifest-path "$TOADSTOOL_DIR/Cargo.toml" 2>&1 | tail -5
fi

# Step 2: Run distiller
echo ">>> Running distiller (chip=gv100)..."
"$HW_LEARN_BIN" gv100 "$TRACE" "$TRACE" "$OUTPUT" 2>&1

echo ""
echo ">>> Recipe saved to: $OUTPUT"

# Step 3: Quick summary
if command -v python3 >/dev/null 2>&1; then
    python3 -c "
import json, sys
with open('${OUTPUT}') as f:
    r = json.load(f)
steps = r.get('steps', [])
by_func = {}
for s in steps:
    if 'RegisterWrite' in str(type(s)) or (isinstance(s, dict) and 'RegisterWrite' in s):
        rw = s.get('RegisterWrite', s) if isinstance(s, dict) else s
        func = rw.get('function', 'Unknown')
        by_func[func] = by_func.get(func, 0) + 1
print(f'Recipe: {len(steps)} steps')
for func, count in sorted(by_func.items(), key=lambda x: -x[1]):
    print(f'  {func}: {count}')
" 2>/dev/null || echo "(python summary unavailable)"
fi

echo ""
echo ">>> Next: Load in GlowPlug with:"
echo "    CORALREEF_ORACLE_RECIPE=$OUTPUT cargo test ..."
