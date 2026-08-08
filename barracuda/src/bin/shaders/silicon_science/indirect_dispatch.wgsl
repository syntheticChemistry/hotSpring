// Indirect dispatch: GPU-driven adaptive workload scheduling.
//
// The GPU writes its own dispatch parameters — no CPU roundtrip.
// For adaptive algorithms (CG iteration count, thermalization acceptance,
// error-driven refinement), the GPU decides how much work to do next.
//
// QCD application: CG solver terminates when residual < eps without
// CPU polling. Adaptive step-size control in integrator.
// Multi-scale: only re-thermalize sites where local action changed.

struct IndirectArgs {
    x: atomic<u32>,
    y: u32,
    z: u32,
    pad: u32,
}

struct Params {
    threshold: f32,
    n_elements: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> residuals: array<f32>;
@group(0) @binding(2) var<storage, read_write> dispatch_args: IndirectArgs;
@group(0) @binding(3) var<storage, read_write> active_sites: array<u32>;

// Phase 1: scan residuals, compact active sites, write dispatch args
@compute @workgroup_size(256)
fn compact_active_sites(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_elements { return; }

    let r = residuals[idx];

    // Only sites above threshold need more work
    if abs(r) > params.threshold {
        let slot = atomicAdd(&dispatch_args.x, 1u);
        active_sites[slot] = idx;
    }
}

// Phase 2: process only the active sites (dispatched indirectly)
@compute @workgroup_size(64)
fn process_active(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    // The dispatch count was written by Phase 1 — GPU self-scheduled
    let site = active_sites[slot];
    // ... perform relaxation/update on this site only ...
    // (placeholder: just mark it processed)
    active_sites[slot] = site | 0x80000000u;
}
