#!/usr/bin/env python3
"""
bpf_warm_catch_guard.py — BPF-based GPU warm-catch teardown blocker.

Uses bpf_override_return to NOP four nouveau teardown functions during
unbind, preserving GDDR5/HBM2 training and GR state for VFIO handoff.

Kernel 6.17+ rejects livepatch/kprobe .ko modules due to strict R_X86_64_64
relocation checks. BPF kprobes bypass this entirely — no module loading needed.

Blocked functions:
  1. gf100_gr_fini    — prevents PGRAPH reset (preserves GPC state)
  2. nvkm_pmu_fini    — keeps PMU falcon running (FECS dependency)
  3. nvkm_mc_disable  — prevents PMC_ENABLE bit clears (preserves clocks)
  4. nvkm_fifo_fini   — prevents PFIFO teardown (preserves runlists)

Usage:
  sudo python3 bpf_warm_catch_guard.py &
  # ... unbind nouveau, rebind vfio-pci ...
  kill %1  # or Ctrl-C

Requires: python3-bcc, CONFIG_BPF_KPROBE_OVERRIDE=y
"""

import signal
import sys
from bcc import BPF

bpf_text = r"""
#include <uapi/linux/ptrace.h>

BPF_ARRAY(block_count, u64, 4);

int block_gf100_gr_fini(struct pt_regs *ctx) {
    int idx = 0;
    u64 *cnt = block_count.lookup(&idx);
    if (cnt) __sync_fetch_and_add(cnt, 1);
    bpf_override_return(ctx, 0);
    return 0;
}

int block_nvkm_pmu_fini(struct pt_regs *ctx) {
    int idx = 1;
    u64 *cnt = block_count.lookup(&idx);
    if (cnt) __sync_fetch_and_add(cnt, 1);
    bpf_override_return(ctx, 0);
    return 0;
}

int block_nvkm_mc_disable(struct pt_regs *ctx) {
    int idx = 2;
    u64 *cnt = block_count.lookup(&idx);
    if (cnt) __sync_fetch_and_add(cnt, 1);
    bpf_override_return(ctx, 0);
    return 0;
}

int block_nvkm_fifo_fini(struct pt_regs *ctx) {
    int idx = 3;
    u64 *cnt = block_count.lookup(&idx);
    if (cnt) __sync_fetch_and_add(cnt, 1);
    bpf_override_return(ctx, 0);
    return 0;
}
"""

FUNC_NAMES = [
    ("gf100_gr_fini", "block_gf100_gr_fini"),
    ("nvkm_pmu_fini", "block_nvkm_pmu_fini"),
    ("nvkm_mc_disable", "block_nvkm_mc_disable"),
    ("nvkm_fifo_fini", "block_nvkm_fifo_fini"),
]


def main():
    b = BPF(text=bpf_text)

    attached = []
    for kernel_fn, bpf_fn in FUNC_NAMES:
        try:
            b.attach_kprobe(event=kernel_fn, fn_name=bpf_fn)
            attached.append(kernel_fn)
            print(f"  armed: {kernel_fn}")
        except Exception as e:
            print(f"  skip:  {kernel_fn} ({e})")

    if not attached:
        print("ERROR: no functions armed — is nouveau loaded?")
        sys.exit(1)

    print(f"\nBPF warm-catch guard active ({len(attached)}/{len(FUNC_NAMES)} functions)")
    print("Unbind nouveau now — teardown functions will be NOP'd.")
    print("Press Ctrl-C to disarm and exit.\n")

    def on_exit(sig, frame):
        print("\nDisarming...")
        counts = b["block_count"]
        names = ["gf100_gr_fini", "nvkm_pmu_fini", "nvkm_mc_disable", "nvkm_fifo_fini"]
        for i, name in enumerate(names):
            print(f"  {name}: {counts[i].value} calls blocked")
        sys.exit(0)

    signal.signal(signal.SIGINT, on_exit)
    signal.signal(signal.SIGTERM, on_exit)

    signal.pause()


if __name__ == "__main__":
    main()
