// Subgroup (warp/wave) operations: hardware-level reductions without shared memory.
//
// On NVIDIA: subgroup = warp (32 threads). On AMD RDNA2: wave64 or wave32.
// subgroupAdd performs a butterfly reduction across the subgroup in ~3 cycles,
// using dedicated shuffle hardware — no shared memory, no barriers.
//
// QCD application: Kahan-compensated trace accumulation, CG dot products,
// action summation — all reduce operations that currently use shared memory.

struct Params {
    n_elements: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn reduce_subgroup(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32,
    @builtin(subgroup_size) sg_size: u32,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let idx = gid.x;
    var val: f32 = 0.0;
    if idx < params.n_elements {
        val = input[idx];
    }

    // Tier 1: subgroup reduction (hardware shuffle — zero shared memory)
    let sg_sum = subgroupAdd(val);

    // Tier 2: first lane of each subgroup writes to shared memory
    var shared_sums: array<f32, 8>;  // max 256/32 = 8 subgroups
    let sg_id = lid / sg_size;
    let n_subgroups = 256u / sg_size;

    if lane == 0u {
        shared_sums[sg_id] = sg_sum;
    }
    workgroupBarrier();

    // Tier 3: first subgroup reduces the shared sums
    if sg_id == 0u {
        var partial: f32 = 0.0;
        if lane < n_subgroups {
            partial = shared_sums[lane];
        }
        let wg_sum = subgroupAdd(partial);
        if lane == 0u {
            output[wg_id.x] = wg_sum;
        }
    }
}

// Exclusive prefix sum via subgroup shuffle
@compute @workgroup_size(256)
fn prefix_sum_subgroup(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let idx = gid.x;
    var val: f32 = 0.0;
    if idx < params.n_elements {
        val = input[idx];
    }

    let prefix = subgroupExclusiveAdd(val);

    if idx < params.n_elements {
        output[idx] = prefix;
    }
}
