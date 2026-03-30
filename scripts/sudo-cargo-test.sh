#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Run cargo test as root with the user's Rust toolchain.
# Preserves RUSTUP_HOME, CARGO_HOME, PATH for rustc/cargo.
#
# Usage: sudo /path/to/sudo-cargo-test.sh [cargo test args...]
set -euo pipefail
export RUSTUP_HOME="/home/biomegate/.rustup"
export CARGO_HOME="/home/biomegate/.cargo"
export PATH="$CARGO_HOME/bin:$PATH"
exec cargo "$@"
