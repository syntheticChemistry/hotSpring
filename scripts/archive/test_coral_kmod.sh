#!/usr/bin/env bash
# DEPRECATED (S276, May 27 2026)
#
# coral-kmod has been fossilized. Its RM ABI types were absorbed into
# toadstool-cylinder nv/rm_abi.rs (22 repr(C) structs). C sources archived
# to fossilRecord/primals/coralReef/coral-kmod/.
#
# The userspace RM path is now via the catalyst pipeline + rm_trigger Rust
# binary (cargo build -p toadstool-cylinder --bin rm_trigger).
#
# Original: Integration test for coral-kmod privileged path (/dev/coral-rm).
# Kept as fossil record. DO NOT RUN.
echo "DEPRECATED: coral-kmod fossilized S276. See fossilRecord/primals/coralReef/coral-kmod/FOSSILIZED.md"
exit 1
