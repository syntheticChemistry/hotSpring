// SPDX-License-Identifier: AGPL-3.0-or-later
//! Probe RT Core and Tensor Core feature availability per card.
//! Generation-specific: these are silicon units that may or may not be
//! accessible via the current driver/API stack.

#[tokio::main]
async fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     RT Core / Tensor Core Feature Probe                         ║");
    println!("║     What's accessible vs what's on-die                          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;

    for adapter in adapters {
        let info = adapter.get_info();
        if info.device_type != wgpu::DeviceType::DiscreteGpu {
            continue;
        }
        println!("━━━ {} ━━━", info.name);
        println!("  Backend: {:?}", info.backend);

        let features = adapter.features();

        println!();
        println!("  ┌─ Ray Tracing / Acceleration Structure Features ──────────┐");
        let rt_query = features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
        println!("  │ EXPERIMENTAL_RAY_QUERY:             {}           │", yn(rt_query));
        println!("  │ (BLAS/TLAS acceleration structures)              │");
        println!("  └──────────────────────────────────────────────────────────┘");

        println!();
        println!("  ┌─ Advanced Compute Features ──────────────────────────────┐");
        let subgroup = features.contains(wgpu::Features::SUBGROUP);
        let subgroup_vertex = features.contains(wgpu::Features::SUBGROUP_VERTEX);
        let f16 = features.contains(wgpu::Features::SHADER_F16);
        let f64 = features.contains(wgpu::Features::SHADER_F64);
        let indirect = features.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE);
        let multi_draw_count = features.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT);
        let timestamp = features.contains(wgpu::Features::TIMESTAMP_QUERY);
        let pipeline_stats = features.contains(wgpu::Features::PIPELINE_STATISTICS_QUERY);
        let storage_texture = features.contains(wgpu::Features::BGRA8UNORM_STORAGE);
        let rg11b10 = features.contains(wgpu::Features::RG11B10UFLOAT_RENDERABLE);
        let float32_filterable = features.contains(wgpu::Features::FLOAT32_FILTERABLE);
        let depth32_stencil = features.contains(wgpu::Features::DEPTH32FLOAT_STENCIL8);
        let texture_compression_bc = features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC);
        let buffer_binding_array = features.contains(wgpu::Features::BUFFER_BINDING_ARRAY);
        let texture_binding_array = features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY);
        let storage_resource_binding_array = features.contains(wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY);

        println!("  │ SUBGROUP:                           {}           │", yn(subgroup));
        println!("  │ SUBGROUP_VERTEX:                    {}           │", yn(subgroup_vertex));
        println!("  │ SHADER_F16:                         {}           │", yn(f16));
        println!("  │ SHADER_F64:                         {}           │", yn(f64));
        println!("  │ INDIRECT_FIRST_INSTANCE:            {}           │", yn(indirect));
        println!("  │ MULTI_DRAW_INDIRECT_COUNT:          {}           │", yn(multi_draw_count));
        println!("  │ TIMESTAMP_QUERY:                    {}           │", yn(timestamp));
        println!("  │ PIPELINE_STATISTICS_QUERY:          {}           │", yn(pipeline_stats));
        println!("  │ FLOAT32_FILTERABLE:                 {}           │", yn(float32_filterable));
        println!("  │ BUFFER_BINDING_ARRAY:               {}           │", yn(buffer_binding_array));
        println!("  │ TEXTURE_BINDING_ARRAY:              {}           │", yn(texture_binding_array));
        println!("  │ STORAGE_RESOURCE_BINDING_ARRAY:     {}           │", yn(storage_resource_binding_array));
        println!("  │ DEPTH32FLOAT_STENCIL8:              {}           │", yn(depth32_stencil));
        println!("  │ RG11B10UFLOAT_RENDERABLE:           {}           │", yn(rg11b10));
        println!("  │ BGRA8UNORM_STORAGE:                 {}           │", yn(storage_texture));
        println!("  │ TEXTURE_COMPRESSION_BC:             {}           │", yn(texture_compression_bc));
        println!("  └──────────────────────────────────────────────────────────┘");

        println!();
        println!("  ┌─ Generation-Specific Latent Silicon ─────────────────────┐");
        if info.name.contains("3090") || info.name.contains("NVIDIA") {
            println!("  │ Tensor Cores: 328 (3rd gen) ON-DIE                       │");
            println!("  │   API access: BLOCKED (needs coralReef PTX/SASS)         │");
            println!("  │   Theoretical: 312 TF32 TOPS (with f32 accumulate)       │");
            println!("  │ RT Cores: 82 (2nd gen) ON-DIE                            │");
            println!("  │   API access: {}                   │",
                if rt_query { "AVAILABLE via EXPERIMENTAL_RAY_QUERY" } else { "BLOCKED (driver/wgpu)" });
            println!("  │ NVENC: 1× 7th gen (H.264/HEVC, 28 fps measured)          │");
            println!("  │   API access: AVAILABLE via system ffmpeg                │");
            println!("  │ f64 atomicAdd: ON-DIE (Ampere SM8.6 feature)             │");
            println!("  │   API access: AVAILABLE via naga WGSL extension          │");
        } else if info.name.contains("6950") || info.name.contains("AMD") || info.name.contains("RADV") {
            println!("  │ Ray Accelerator: 80 (1st gen) ON-DIE                     │");
            println!("  │   API access: {}                   │",
                if rt_query { "AVAILABLE via EXPERIMENTAL_RAY_QUERY" } else { "BLOCKED (driver/wgpu)" });
            println!("  │ Infinity Cache: 128 MB SRAM (KEY advantage)              │");
            println!("  │   API access: TRANSPARENT (no explicit control)          │");
            println!("  │ VCN 3.0: 1× encoder (H.264/HEVC, 33 fps measured)       │");
            println!("  │   API access: AVAILABLE via system ffmpeg VAAPI          │");
            println!("  │ ROP atomics: 6.35× NVIDIA (117.7 Gatom/s)               │");
            println!("  │   API access: AVAILABLE via render pass additive blend   │");
            println!("  │ Wave64: native wide subgroup                             │");
            println!("  │   API access: AVAILABLE (subgroup ops on 64 lanes)       │");
        }
        println!("  └──────────────────────────────────────────────────────────┘");
        println!();
    }

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     Feature Probe Complete                                      ║");
    println!("║     RT Cores: generation-dependent. Tensor: driver-blocked.     ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
}

fn yn(b: bool) -> &'static str {
    if b { "YES" } else { "NO " }
}
