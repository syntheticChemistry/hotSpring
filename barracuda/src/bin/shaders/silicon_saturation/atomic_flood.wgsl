// SPDX-License-Identifier: AGPL-3.0-or-later
@group(0) @binding(0) var<storage, read_write> counters: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bucket = gid.x % 256u;
    for (var i = 0u; i < 64u; i = i + 1u) {
        atomicAdd(&counters[(bucket + i) % 256u], 1u);
    }
}
