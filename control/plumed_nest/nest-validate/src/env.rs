// SPDX-License-Identifier: AGPL-3.0-or-later

//! Environment detection — locate GROMACS, PLUMED, and conda without wrappers.
//!
//! Makes the binary self-contained: it finds the correct executables and
//! libraries by probing standard locations, environment variables, and
//! conda environments.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Environment {
    pub gmx_path: Option<PathBuf>,
    pub gmx_version: Option<String>,
    pub plumed_path: Option<PathBuf>,
    pub plumed_version: Option<String>,
    pub plumed_kernel: Option<PathBuf>,
    pub conda_env: Option<String>,
    pub conda_prefix: Option<PathBuf>,
    pub n_cpus: usize,
    pub gpu_available: bool,
}

impl Environment {
    pub fn detect() -> Self {
        let conda_prefix = detect_conda_prefix();
        let conda_env = conda_prefix.as_ref().and_then(|p| {
            p.file_name().map(|n| n.to_string_lossy().to_string())
        });

        let gmx_path = find_executable("gmx", conda_prefix.as_deref());
        let plumed_path = find_executable("plumed", conda_prefix.as_deref());
        let plumed_kernel = find_plumed_kernel(conda_prefix.as_deref());

        let gmx_version = gmx_path.as_ref().and_then(|p| get_gmx_version(p));
        let plumed_version = plumed_path.as_ref().and_then(|p| get_plumed_version(p));

        let n_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let gpu_available = detect_gpu();

        Environment {
            gmx_path,
            gmx_version,
            plumed_path,
            plumed_version,
            plumed_kernel,
            conda_env,
            conda_prefix,
            n_cpus,
            gpu_available,
        }
    }

    /// Construct the environment variables needed to run gmx with PLUMED.
    pub fn gmx_env(&self) -> Vec<(String, String)> {
        let mut env = Vec::new();

        if let Some(ref kernel) = self.plumed_kernel {
            env.push(("PLUMED_KERNEL".to_string(), kernel.display().to_string()));
        }

        if let Some(ref prefix) = self.conda_prefix {
            let lib_path = prefix.join("lib");
            if lib_path.is_dir() {
                env.push(("LD_LIBRARY_PATH".to_string(), lib_path.display().to_string()));
            }
        }

        env
    }

    /// Recommended thread count for simulations.
    pub fn recommended_ntomp(&self) -> usize {
        // Leave 2 cores free for system/OS
        (self.n_cpus - 2).max(1).min(16)
    }

    pub fn print_summary(&self) {
        println!("  \x1b[36mEnvironment Detection\x1b[0m");
        println!("    GMX:           {}", self.gmx_path.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "NOT FOUND".to_string()));
        println!("    GMX version:   {}", self.gmx_version.as_deref().unwrap_or("unknown"));
        println!("    PLUMED:        {}", self.plumed_path.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "NOT FOUND".to_string()));
        println!("    PLUMED ver:    {}", self.plumed_version.as_deref().unwrap_or("unknown"));
        println!("    Kernel:        {}", self.plumed_kernel.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "NOT FOUND".to_string()));
        println!("    Conda env:     {}", self.conda_env.as_deref().unwrap_or("none"));
        println!("    CPUs:          {}", self.n_cpus);
        println!("    GPU:           {}", if self.gpu_available { "detected" } else { "none" });
        println!("    Rec. threads:  {}", self.recommended_ntomp());
    }
}

fn detect_conda_prefix() -> Option<PathBuf> {
    // Check CONDA_PREFIX environment variable
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        return Some(PathBuf::from(prefix));
    }

    // Check for known conda environments
    let home = std::env::var("HOME").ok()?;
    let candidates = [
        format!("{home}/miniconda3/envs/gromacs-fel"),
        format!("{home}/anaconda3/envs/gromacs-fel"),
        format!("{home}/miniforge3/envs/gromacs-fel"),
        format!("{home}/miniconda3/envs/gromacs"),
        format!("{home}/mambaforge/envs/gromacs-fel"),
    ];

    for candidate in &candidates {
        let path = PathBuf::from(candidate);
        if path.join("bin/gmx").exists() {
            return Some(path);
        }
    }

    None
}

fn find_executable(name: &str, conda_prefix: Option<&Path>) -> Option<PathBuf> {
    // Check explicit environment variable
    let env_key = name.to_uppercase();
    if let Ok(path) = std::env::var(&env_key) {
        let p = PathBuf::from(&path);
        if p.exists() {
            return Some(p);
        }
    }

    // Check conda prefix
    if let Some(prefix) = conda_prefix {
        let conda_bin = prefix.join("bin").join(name);
        if conda_bin.exists() {
            return Some(conda_bin);
        }
    }

    // Check PATH via `which`
    Command::new("which")
        .arg(name)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim()))
}

fn find_plumed_kernel(conda_prefix: Option<&Path>) -> Option<PathBuf> {
    // Check PLUMED_KERNEL env var
    if let Ok(kernel) = std::env::var("PLUMED_KERNEL") {
        let p = PathBuf::from(&kernel);
        if p.exists() {
            return Some(p);
        }
    }

    // Check conda prefix lib
    if let Some(prefix) = conda_prefix {
        let kernel = prefix.join("lib/libplumedKernel.so");
        if kernel.exists() {
            return Some(kernel);
        }
    }

    // Check standard system locations
    let system_paths = [
        "/usr/local/lib/libplumedKernel.so",
        "/usr/lib/libplumedKernel.so",
    ];
    for path in &system_paths {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    None
}

fn get_gmx_version(gmx_path: &Path) -> Option<String> {
    Command::new(gmx_path)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| {
            let stdout = String::from_utf8_lossy(&o.stdout);
            stdout.lines()
                .find(|l| l.contains("GROMACS version"))
                .map(|l| l.trim().to_string())
        })
}

fn get_plumed_version(plumed_path: &Path) -> Option<String> {
    Command::new(plumed_path)
        .args(["info", "--version"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

fn detect_gpu() -> bool {
    // Check for NVIDIA GPU
    Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Spawn a GROMACS mdrun process with proper environment.
pub fn spawn_gmx_mdrun(
    env: &Environment,
    tpr: &Path,
    plumed_dat: &Path,
    output_prefix: &Path,
    nsteps: Option<u64>,
    checkpoint: Option<&Path>,
) -> Result<std::process::Child, String> {
    let gmx = env.gmx_path.as_ref()
        .ok_or("GROMACS not found in environment")?;

    let ntomp = env.recommended_ntomp();

    let mut cmd = Command::new(gmx);
    cmd.arg("mdrun")
        .args(["-s", &tpr.display().to_string()])
        .args(["-plumed", &plumed_dat.display().to_string()])
        .args(["-deffnm", &output_prefix.display().to_string()])
        .args(["-ntmpi", "1"])
        .args(["-ntomp", &ntomp.to_string()]);

    if let Some(steps) = nsteps {
        cmd.args(["-nsteps", &steps.to_string()]);
    }

    if let Some(cpt) = checkpoint {
        cmd.args(["-cpi", &cpt.display().to_string()])
            .arg("-append");
    }

    // Set environment
    for (key, val) in env.gmx_env() {
        cmd.env(&key, &val);
    }

    cmd.spawn().map_err(|e| format!("Failed to spawn gmx mdrun: {e}"))
}
