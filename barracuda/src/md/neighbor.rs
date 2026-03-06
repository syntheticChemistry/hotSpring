// SPDX-License-Identifier: AGPL-3.0-only

//! Neighbor search algorithm selection and Verlet list GPU implementation.
//!
//! Three-tier complexity ladder:
//! - **AllPairs** — O(N²), zero overhead, for small N or when cells/dim < 3
//! - **CellList** — O(N), 27-cell stencil, for medium N with adequate grid
//! - **VerletList** — O(N) with compact neighbor arrays, uses CellList to
//!   build the list, then iterates only true neighbors within `rc + skin`
//!
//! The system auto-selects at runtime based on particle count and geometry.

use barracuda::ops::md::CellListGpu;

use crate::gpu::GpuF64;
use crate::md::config::MdConfig;
use crate::tolerances::{
    CELLLIST_MIN_CELLS_PER_DIM, MD_WORKGROUP_SIZE, VERLET_MAX_NEIGHBORS, VERLET_MIN_PARTICLES,
    VERLET_SKIN_FRACTION,
};

/// Force algorithm tier selected at runtime.
#[derive(Clone, Debug)]
pub enum ForceAlgorithm {
    /// O(N²) all-pairs: each particle loops over every other particle.
    AllPairs,
    /// O(N) cell-list: 27-cell stencil via `CellListGpu` indirect indexing.
    CellList,
    /// O(N) Verlet neighbor list: compact per-particle neighbor arrays
    /// built from cell-list, rebuilt when max displacement exceeds skin/2.
    VerletList {
        /// Skin radius beyond rc for neighbor inclusion.
        skin: f64,
    },
}

/// Selects the optimal force algorithm for the given simulation parameters.
pub struct AlgorithmSelector {
    n_particles: usize,
    cells_per_dim: usize,
    rc: f64,
}

impl AlgorithmSelector {
    /// Create a selector from an MD configuration.
    #[must_use]
    pub fn from_config(config: &MdConfig) -> Self {
        let box_side = config.box_side();
        let cells_per_dim = (box_side / config.rc).floor() as usize;
        Self {
            n_particles: config.n_particles,
            cells_per_dim,
            rc: config.rc,
        }
    }

    /// Select the best algorithm for the current parameters.
    #[must_use]
    pub fn select(&self) -> ForceAlgorithm {
        if self.cells_per_dim < CELLLIST_MIN_CELLS_PER_DIM {
            return ForceAlgorithm::AllPairs;
        }
        if self.n_particles < VERLET_MIN_PARTICLES {
            return ForceAlgorithm::AllPairs;
        }
        let skin = self.rc * VERLET_SKIN_FRACTION;
        ForceAlgorithm::VerletList { skin }
    }
}

/// GPU-resident Verlet neighbor list.
///
/// Stores a flat `[N × max_neighbors]` array of neighbor indices per particle,
/// plus a `[N]` count array. Built from `CellListGpu` by iterating the 27-cell
/// stencil and storing only neighbors within `rc + skin`.
///
/// Rebuilt when the maximum particle displacement from reference positions
/// exceeds `skin / 2`.
pub struct VerletListGpu {
    /// `[N × max_neighbors]` u32 — flat neighbor indices.
    neighbor_list_buf: wgpu::Buffer,
    /// `[N]` u32 — neighbor count per particle.
    neighbor_count_buf: wgpu::Buffer,
    /// `[N × 3]` f64 — positions at last rebuild (reference).
    ref_positions_buf: wgpu::Buffer,
    /// `[1]` u32 — max displacement (fixed-point, written by check shader).
    max_disp_buf: wgpu::Buffer,
    /// `[1]` u32 — staging buffer for CPU readback of max displacement.
    max_disp_staging: wgpu::Buffer,
    /// Maximum neighbors per particle (buffer dimension).
    pub max_neighbors: u32,
    /// Skin radius.
    skin: f64,
    /// Force cutoff radius.
    pub rc: f64,
    /// Number of particles.
    n: usize,
    /// Build pipeline (populates `neighbor_list` + `neighbor_count` from cell-list).
    build_pipeline: wgpu::ComputePipeline,
    /// Displacement check pipeline.
    check_pipeline: wgpu::ComputePipeline,
    /// Copy-reference pipeline (saves current positions as ref).
    copy_ref_pipeline: wgpu::ComputePipeline,
    /// The underlying cell-list used for O(N) construction.
    cell_list: CellListGpu,
}

impl VerletListGpu {
    /// Create a new Verlet list for `n` particles.
    pub fn new(
        gpu: &GpuF64,
        n: usize,
        box_dims: [f64; 3],
        rc: f64,
        skin: f64,
    ) -> Result<Self, crate::error::HotSpringError> {
        let device = gpu.device();
        let max_neighbors = VERLET_MAX_NEIGHBORS;

        let neighbor_list_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("verlet_neighbor_list"),
            size: (n as u64) * u64::from(max_neighbors) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let neighbor_count_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("verlet_neighbor_count"),
            size: (n as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let ref_positions_buf = gpu.create_f64_output_buffer(n * 3, "verlet_ref_positions");

        let max_disp_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("verlet_max_disp"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let max_disp_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("verlet_max_disp_staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Cell grid uses rc (not rc+skin) to ensure cells_per_dim >= 3 for
        // correct PBC stencil wrapping. The build shader searches within
        // (rc+skin)² which is safe since skin < cell_size for valid configs.
        let cell_list = CellListGpu::new(gpu.to_wgpu_device(), n, box_dims, rc)?;

        let build_pipeline = gpu.create_pipeline(SHADER_VERLET_BUILD, "verlet_build");
        let check_pipeline = gpu.create_pipeline(SHADER_VERLET_CHECK_DISP, "verlet_check_disp");
        let copy_ref_pipeline = gpu.create_pipeline(SHADER_VERLET_COPY_REF, "verlet_copy_ref");

        Ok(Self {
            neighbor_list_buf,
            neighbor_count_buf,
            ref_positions_buf,
            max_disp_buf,
            max_disp_staging,
            max_neighbors,
            skin,
            rc,
            n,
            build_pipeline,
            check_pipeline,
            copy_ref_pipeline,
            cell_list,
        })
    }

    /// Build the Verlet list from current positions. Also saves reference positions.
    pub fn build(
        &self,
        gpu: &GpuF64,
        pos_buf: &wgpu::Buffer,
        params_buf: &wgpu::Buffer,
    ) -> Result<(), crate::error::HotSpringError> {
        self.cell_list.build(pos_buf)?;

        let workgroups = self.n.div_ceil(MD_WORKGROUP_SIZE) as u32;

        let build_bg = gpu.create_bind_group(
            &self.build_pipeline,
            &[
                pos_buf,
                &self.neighbor_list_buf,
                &self.neighbor_count_buf,
                params_buf,
                self.cell_list.cell_start(),
                self.cell_list.cell_count(),
                self.cell_list.sorted_indices(),
            ],
        );
        gpu.dispatch(&self.build_pipeline, &build_bg, workgroups);

        let copy_bg = gpu.create_bind_group(
            &self.copy_ref_pipeline,
            &[pos_buf, &self.ref_positions_buf, params_buf],
        );
        gpu.dispatch(&self.copy_ref_pipeline, &copy_bg, workgroups);

        Ok(())
    }

    /// Check maximum displacement since last rebuild.
    /// Returns true if a rebuild is needed (`max_disp` > skin/2).
    #[must_use]
    pub fn needs_rebuild(
        &self,
        gpu: &GpuF64,
        pos_buf: &wgpu::Buffer,
        params_buf: &wgpu::Buffer,
    ) -> bool {
        let device = gpu.device();

        gpu.queue()
            .write_buffer(&self.max_disp_buf, 0, &0u32.to_le_bytes());

        let workgroups = self.n.div_ceil(MD_WORKGROUP_SIZE) as u32;
        let check_bg = gpu.create_bind_group(
            &self.check_pipeline,
            &[
                pos_buf,
                &self.ref_positions_buf,
                &self.max_disp_buf,
                params_buf,
            ],
        );
        gpu.dispatch(&self.check_pipeline, &check_bg, workgroups);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("verlet_disp_copy"),
        });
        encoder.copy_buffer_to_buffer(&self.max_disp_buf, 0, &self.max_disp_staging, 0, 4);
        gpu.queue().submit(std::iter::once(encoder.finish()));

        let slice = self.max_disp_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        let fixed_point = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);
        self.max_disp_staging.unmap();

        // Fixed-point: value * 1e6
        let max_disp = f64::from(fixed_point) * 1e-6;
        max_disp > self.skin * 0.5
    }

    /// Access the neighbor list buffer for force shader binding.
    #[must_use]
    pub const fn neighbor_list(&self) -> &wgpu::Buffer {
        &self.neighbor_list_buf
    }

    /// Access the neighbor count buffer for force shader binding.
    #[must_use]
    pub const fn neighbor_count(&self) -> &wgpu::Buffer {
        &self.neighbor_count_buf
    }

    /// Cell grid dimensions from the underlying cell-list.
    #[must_use]
    pub fn cell_list_grid(&self) -> (u32, u32, u32) {
        self.cell_list.grid()
    }

    /// Total cell count from the underlying cell-list.
    #[must_use]
    pub fn cell_list_n_cells(&self) -> u32 {
        self.cell_list.n_cells()
    }
}

// ── Shader sources ───────────────────────────────────────────────────

const SHADER_VERLET_BUILD: &str = include_str!("shaders/verlet_build.wgsl");
const SHADER_VERLET_CHECK_DISP: &str = include_str!("shaders/verlet_check_displacement.wgsl");
const SHADER_VERLET_COPY_REF: &str = include_str!("shaders/verlet_copy_ref.wgsl");
