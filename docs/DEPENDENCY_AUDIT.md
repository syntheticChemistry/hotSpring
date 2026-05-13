# hotSpring Dependency Audit — ecoBin Compliance

*Last updated: May 13, 2026*

## Pure Rust (no C/system deps)

| Dependency | Status | Notes |
|---|---|---|
| **blake3** | **RESOLVED** | `default-features = false` drops `cc` C build. Pure Rust SIMD impl. |
| serde / serde_json | Clean | Proc macro only |
| clap | Clean | Pure Rust |
| toml | Clean | Pure Rust |
| log | Clean | Pure Rust facade |
| bytemuck | Clean | Pure Rust, `#[repr(C)]` only |
| rayon | Clean | Pure Rust (uses `std::thread`) |
| rand / rand_chacha | Clean | Pure Rust CSPRNG |

## Ecosystem Boundaries (unavoidable system deps)

| Dependency | System Dep | Justification |
|---|---|---|
| **wgpu / naga** | Vulkan/Metal/D3D12 via `ash`/`libloading` | GPU validation requires graphics API. No pure-Rust GPU driver alternative exists. toadStool's sovereign VFIO path is the long-term evolution for non-wgpu dispatch. |
| **tokio** | `libc`/`mio` for async IO | Standard async runtime. Pure-Rust alternative (`smol`) would require full rewrite with no functional benefit. |

## Feature-Gated (opt-in only)

| Dependency | Feature Gate | Notes |
|---|---|---|
| **cudarc** | `cuda-validation` | CUDA FFI for NVIDIA validation. Opt-in, not required for default build. |
| **rustix** | `low-level` | Linux syscall wrappers for BAR0 MMIO experiments. Opt-in. |

## Compliance Summary

- **Default build**: Zero C dependencies (blake3 pure-Rust, no `cc` invocation)
- **`#![forbid(unsafe_code)]`** on library root and uniBin
- Feature-gated deps (`cudarc`, `rustix`) are explicitly opt-in for hardware experiments
- wgpu and tokio are ecosystem boundaries documented as acceptable
