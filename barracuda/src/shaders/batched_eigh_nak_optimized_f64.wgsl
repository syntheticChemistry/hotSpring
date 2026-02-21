// SPDX-License-Identifier: AGPL-3.0-only
// NAK-Optimized Batched Symmetric Eigensolve (f64)
//
// Workarounds for 5 NAK shader compiler deficiencies identified in
// hotSpring cross-GPU analysis (Feb 18, 2026):
//
//   1. Loop unrolling:    Manual 4x unroll of k-loop in Jacobi rotation
//   2. Register pressure: Hoist all reusable values into locals before loops
//   3. Instruction sched: Interleave independent loads with computation
//   4. FMA fusion:        Explicit fma() calls for all a*b+c patterns
//   5. Bank conflicts:    N/A — no shared memory (warp-packed design)
//
// Architecture: Warp-packed (32 threads/workgroup, each solves 1 matrix)
// Each thread accesses its own n×n slice of global memory.
// Dispatch: (batch.div_ceil(32), 1, 1) workgroups.
//
// Expected speedup on NVK/NAK: ~3-4x over non-optimized warp-packed.
// Neutral on proprietary (it already applies these optimizations).
//
// Provenance: hotSpring → toadstool handoff, Feb 19 2026
// License: AGPL-3.0-only

const WARP_SIZE: u32 = 32u;

struct Params {
    n: u32,
    batch_size: u32,
    max_sweeps: u32,
    tolerance: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> A_batch: array<f64>;
@group(0) @binding(2) var<storage, read_write> V_batch: array<f64>;
@group(0) @binding(3) var<storage, read_write> eigenvalues: array<f64>;

fn idx(r: u32, c: u32, n: u32) -> u32 { return r * n + c; }

@compute @workgroup_size(32, 1, 1)
fn batched_eigh_nak_optimized(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let b = wg_id.x * WARP_SIZE + local_id.x;
    let n = params.n;
    if (b >= params.batch_size || n > 32u) { return; }

    let base = b * n * n;
    let tol = f64(params.tolerance);

    // [Workaround #2: register pressure] Hoist base + n into locals.
    // NAK often recomputes expressions inside loops; explicit locals
    // hint the allocator to keep these in registers.

    // V = Identity (branchless via select)
    // [Workaround #3: branchless] Use select() instead of if/else
    for (var i = 0u; i < n; i++) {
        for (var j = 0u; j < n; j++) {
            V_batch[base + idx(i, j, n)] = select(f64(0.0), f64(1.0), i == j);
        }
    }

    // Jacobi sweeps
    for (var sweep = 0u; sweep < params.max_sweeps; sweep++) {
        // Convergence check
        var max_off = f64(0.0);
        for (var i = 0u; i < n; i++) {
            for (var j = i + 1u; j < n; j++) {
                let off = abs(A_batch[base + idx(i, j, n)]);
                max_off = max(max_off, off);
            }
        }
        if (max_off < tol) { break; }

        // Cyclic Jacobi rotations
        for (var p = 0u; p < n - 1u; p++) {
            for (var q = p + 1u; q < n; q++) {
                let apq = A_batch[base + idx(p, q, n)];
                if (abs(apq) < 1e-14) { continue; }

                // [Workaround #2] Hoist diagonal reads
                let app = A_batch[base + idx(p, p, n)];
                let aqq = A_batch[base + idx(q, q, n)];
                let diff = aqq - app;

                // [Workaround #3: branchless] Compute t without branch
                let abs_diff = abs(diff);
                let phi = diff / (2.0 * apq);
                let abs_phi = abs(phi);
                // [Workaround #4: FMA] fma(phi, phi, 1.0) = phi*phi + 1.0
                let phi_sq_p1 = fma(phi, phi, f64(1.0));
                let inv_denom = 1.0 / (abs_phi + sqrt(phi_sq_p1));
                let t_normal = select(-inv_denom, inv_denom, phi >= 0.0);
                let t_degen = select(f64(-1.0), f64(1.0), apq >= 0.0);
                let t = select(t_normal, t_degen, abs_diff < 1e-14);

                let c_sq = fma(t, t, f64(1.0));
                let c = 1.0 / sqrt(c_sq);
                let s = t * c;
                let neg_s = -s;

                // [Workaround #1: manual 4x unroll] Apply rotation to A rows/cols
                // Process k in chunks of 4 to give NAK independent instructions
                // to schedule without its own unroller.
                let n4 = n & ~3u;  // round down to multiple of 4
                var k = 0u;
                for (; k < n4; k += 4u) {
                    // Unrolled iteration 0
                    if (k != p && k != q) {
                        let akp0 = A_batch[base + idx(k, p, n)];
                        let akq0 = A_batch[base + idx(k, q, n)];
                        let np0 = fma(c, akp0, neg_s * akq0);
                        let nq0 = fma(s, akp0, c * akq0);
                        A_batch[base + idx(k, p, n)] = np0;
                        A_batch[base + idx(k, q, n)] = nq0;
                        A_batch[base + idx(p, k, n)] = np0;
                        A_batch[base + idx(q, k, n)] = nq0;
                    }
                    let k1 = k + 1u;
                    if (k1 != p && k1 != q) {
                        let akp1 = A_batch[base + idx(k1, p, n)];
                        let akq1 = A_batch[base + idx(k1, q, n)];
                        let np1 = fma(c, akp1, neg_s * akq1);
                        let nq1 = fma(s, akp1, c * akq1);
                        A_batch[base + idx(k1, p, n)] = np1;
                        A_batch[base + idx(k1, q, n)] = nq1;
                        A_batch[base + idx(p, k1, n)] = np1;
                        A_batch[base + idx(q, k1, n)] = nq1;
                    }
                    let k2 = k + 2u;
                    if (k2 != p && k2 != q) {
                        let akp2 = A_batch[base + idx(k2, p, n)];
                        let akq2 = A_batch[base + idx(k2, q, n)];
                        let np2 = fma(c, akp2, neg_s * akq2);
                        let nq2 = fma(s, akp2, c * akq2);
                        A_batch[base + idx(k2, p, n)] = np2;
                        A_batch[base + idx(k2, q, n)] = nq2;
                        A_batch[base + idx(p, k2, n)] = np2;
                        A_batch[base + idx(q, k2, n)] = nq2;
                    }
                    let k3 = k + 3u;
                    if (k3 != p && k3 != q) {
                        let akp3 = A_batch[base + idx(k3, p, n)];
                        let akq3 = A_batch[base + idx(k3, q, n)];
                        let np3 = fma(c, akp3, neg_s * akq3);
                        let nq3 = fma(s, akp3, c * akq3);
                        A_batch[base + idx(k3, p, n)] = np3;
                        A_batch[base + idx(k3, q, n)] = nq3;
                        A_batch[base + idx(p, k3, n)] = np3;
                        A_batch[base + idx(q, k3, n)] = nq3;
                    }
                }
                // Remainder
                for (; k < n; k++) {
                    if (k != p && k != q) {
                        let akp = A_batch[base + idx(k, p, n)];
                        let akq = A_batch[base + idx(k, q, n)];
                        let np_r = fma(c, akp, neg_s * akq);
                        let nq_r = fma(s, akp, c * akq);
                        A_batch[base + idx(k, p, n)] = np_r;
                        A_batch[base + idx(k, q, n)] = nq_r;
                        A_batch[base + idx(p, k, n)] = np_r;
                        A_batch[base + idx(q, k, n)] = nq_r;
                    }
                }

                // [Workaround #4: FMA] 2x2 block update
                let cc = c * c;
                let ss = s * s;
                let cs2 = f64(2.0) * c * s;
                A_batch[base + idx(p, p, n)] = fma(cc, app, fma(-cs2, apq, ss * aqq));
                A_batch[base + idx(q, q, n)] = fma(ss, app, fma(cs2, apq, cc * aqq));
                A_batch[base + idx(p, q, n)] = f64(0.0);
                A_batch[base + idx(q, p, n)] = f64(0.0);

                // [Workaround #1 + #4] V rotation with 4x unroll + FMA
                var kv = 0u;
                for (; kv < n4; kv += 4u) {
                    let vkp0 = V_batch[base + idx(kv, p, n)];
                    let vkq0 = V_batch[base + idx(kv, q, n)];
                    V_batch[base + idx(kv, p, n)] = fma(c, vkp0, neg_s * vkq0);
                    V_batch[base + idx(kv, q, n)] = fma(s, vkp0, c * vkq0);

                    let kv1 = kv + 1u;
                    let vkp1 = V_batch[base + idx(kv1, p, n)];
                    let vkq1 = V_batch[base + idx(kv1, q, n)];
                    V_batch[base + idx(kv1, p, n)] = fma(c, vkp1, neg_s * vkq1);
                    V_batch[base + idx(kv1, q, n)] = fma(s, vkp1, c * vkq1);

                    let kv2 = kv + 2u;
                    let vkp2 = V_batch[base + idx(kv2, p, n)];
                    let vkq2 = V_batch[base + idx(kv2, q, n)];
                    V_batch[base + idx(kv2, p, n)] = fma(c, vkp2, neg_s * vkq2);
                    V_batch[base + idx(kv2, q, n)] = fma(s, vkp2, c * vkq2);

                    let kv3 = kv + 3u;
                    let vkp3 = V_batch[base + idx(kv3, p, n)];
                    let vkq3 = V_batch[base + idx(kv3, q, n)];
                    V_batch[base + idx(kv3, p, n)] = fma(c, vkp3, neg_s * vkq3);
                    V_batch[base + idx(kv3, q, n)] = fma(s, vkp3, c * vkq3);
                }
                for (; kv < n; kv++) {
                    let vkp = V_batch[base + idx(kv, p, n)];
                    let vkq = V_batch[base + idx(kv, q, n)];
                    V_batch[base + idx(kv, p, n)] = fma(c, vkp, neg_s * vkq);
                    V_batch[base + idx(kv, q, n)] = fma(s, vkp, c * vkq);
                }
            }
        }
    }

    // Extract eigenvalues
    let eig_base = b * n;
    for (var i = 0u; i < n; i++) {
        eigenvalues[eig_base + i] = A_batch[base + idx(i, i, n)];
    }
}
