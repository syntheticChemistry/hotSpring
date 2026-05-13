# Experiment 177: Blackwell Dispatch ABI Fixes

**Date:** April 19, 2026
**Track:** Sovereign GPU
**Hardware:** RTX 5060 (GB206, SM120, Blackwell)
**Driver:** nvidia 580.119.02 (GSP-RM)

---

## Objective

Resolve remaining Blackwell dispatch blockers from Exp 175-176. Focus on
FAULT_PDE, UVM_PAGEABLE_MEM_ACCESS failure, and SM Warp Exception root cause.

## Method

1. Traced UVM ioctl struct definitions against NVIDIA `uvm_ioctl.h` headers
2. Compared coral-driver struct sizes with kernel expectations
3. Analyzed VRAM allocation flag differences between CUDA trace and coral-kmod
4. Iterated on kmod `alloc_gpu_buffer` page size attributes

## Results

### Finding 1: UVM_PAGEABLE_MEM_ACCESS was always succeeding

The `UvmPageableMemAccessParams` struct was 4 bytes but the kernel ABI requires
8 bytes (`NvBool pageableMemAccess` + padding + `NV_STATUS rmStatus`). Our code
read `rm_status` from offset 0, which was actually the `pageableMemAccess` output
field. Value `0x00000001` = `pageableMemAccess = true` (supported). The real
`rmStatus` at offset 4 was `NV_OK (0)`.

**Impact:** All prior log messages showing "UVM_PAGEABLE_MEM_ACCESS FAILED:
status=0x00000001" were false negatives. UVM pageable memory access was always
enabled.

### Finding 2: FAULT_PDE caused by huge page attributes

The kmod's `alloc_gpu_buffer` used `PAGE_SIZE_BOTH` (attr `0x11800000`) and
`PAGE_SIZE_HUGE_2MB` (attr2 `0x00100005`) for 4KB data buffers. The RM set up
page directory entries at 2MB granularity, but only 4KB was mapped — the page
directory was incomplete. Changed to `PAGE_SIZE_4KB`, eliminating the fault.

**GPU VA spacing before:** 2MB (0x200000) between 4KB buffers
**GPU VA spacing after:** 4KB-64KB (normal page-level spacing)

### Finding 3: SM Warp Exception root cause chain

With both fixes applied, the `FAULT_PDE` is eliminated. The `SM Warp Exception:
Invalid Address Space` (ESR 0x10) is the remaining blocker. This is NOT a
memory mapping issue — it's a GR context issue:

1. `UVM_REGISTER_GPU_VASPACE` → `GPU_IN_FULL` (0x5D) on desktop Blackwell
2. No UVM fault handling → demand-paged context buffers can't be resolved
3. `GPU_PROMOTE_CTX` → `INSUFFICIENT_PERMISSIONS` from userspace
4. `GR_CTXSW_SETUP_BIND(vMemPtr=0)` → GSP tries demand-paging but no handler
5. SM faults on first CBUF access → "Invalid Address Space"

CUDA's path works because it has full UVM registration. The kmod has privilege
to call GPU_PROMOTE_CTX but currently skips it for Blackwell.

## Conclusion

Two ABI bugs fixed (struct size, page size). FAULT_PDE eliminated. Root cause
of SM Warp Exception identified as missing GR context buffer promotion from
kernel context. Next: re-enable GPU_PROMOTE_CTX in coral-kmod.
