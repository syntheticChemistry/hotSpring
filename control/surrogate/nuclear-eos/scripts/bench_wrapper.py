#!/usr/bin/env python3
"""
Benchmark wrapper for hotSpring Python validation runs.

Captures wall-clock time, CPU energy (Intel RAPL), GPU power/temperature
(nvidia-smi), and process memory.  Outputs JSON compatible with the Rust
bench harness in barracuda/src/bench.rs.

Usage as context manager:
    from bench_wrapper import BenchPhase, HardwareInventory, save_report

    hw = HardwareInventory.detect("Eastgate")
    phases = []
    with BenchPhase("L1_SEMF_python") as bp:
        ... run computation ...
        bp.set_physics(chi2=6.62, n_evals=200)
    phases.append(bp.result())
    save_report(hw, phases, "benchmarks/nuclear-eos/results")

License: AGPL-3.0
"""

import json
import os
import re
import resource
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


# ═══════════════════════════════════════════════════════════════════
#  RAPL helpers
# ═══════════════════════════════════════════════════════════════════

RAPL_ENERGY_PATH = "/sys/class/powercap/intel-rapl:0/energy_uj"
RAPL_MAX_ENERGY_PATH = "/sys/class/powercap/intel-rapl:0/max_energy_range_uj"


def read_rapl_uj() -> Optional[int]:
    """Read current RAPL energy counter in microjoules."""
    try:
        with open(RAPL_ENERGY_PATH) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def read_rapl_max_uj() -> int:
    """Read RAPL max energy range in microjoules."""
    try:
        with open(RAPL_MAX_ENERGY_PATH) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return 2**63  # fallback


# ═══════════════════════════════════════════════════════════════════
#  nvidia-smi poller (background thread)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GpuSample:
    watts: float
    temp_c: float
    vram_mib: float
    timestamp: float  # time.monotonic()


class NvidiaSmiPoller:
    """Background thread that polls nvidia-smi for GPU metrics."""

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self.samples: List[GpuSample] = []
        self._proc = None
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        try:
            self._proc = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=power.draw,temperature.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                    f"-lms", str(self.interval_ms),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()
        except FileNotFoundError:
            pass  # nvidia-smi not available

    def _reader(self):
        if self._proc is None or self._proc.stdout is None:
            return
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split(", ")
            if len(parts) >= 3:
                try:
                    self.samples.append(GpuSample(
                        watts=float(parts[0]),
                        temp_c=float(parts[1]),
                        vram_mib=float(parts[2]),
                        timestamp=time.monotonic(),
                    ))
                except ValueError:
                    pass

    def stop(self):
        self._stop.set()
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=2)


# ═══════════════════════════════════════════════════════════════════
#  Energy Report
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EnergyReport:
    cpu_joules: float = 0.0
    gpu_joules: float = 0.0
    gpu_watts_avg: float = 0.0
    gpu_watts_peak: float = 0.0
    gpu_temp_peak_c: float = 0.0
    gpu_vram_peak_mib: float = 0.0
    gpu_samples: int = 0

    @staticmethod
    def from_monitoring(
        rapl_start: Optional[int],
        rapl_end: Optional[int],
        gpu_samples: List[GpuSample],
        wall_elapsed: float,
    ) -> "EnergyReport":
        # CPU energy
        cpu_joules = 0.0
        if rapl_start is not None and rapl_end is not None:
            delta = rapl_end - rapl_start
            if delta < 0:
                delta += read_rapl_max_uj()
            cpu_joules = delta / 1_000_000.0

        # GPU energy
        n = len(gpu_samples)
        if n == 0:
            return EnergyReport(cpu_joules=cpu_joules)

        gpu_joules = 0.0
        watts_sum = 0.0
        watts_peak = 0.0
        temp_peak = 0.0
        vram_peak = 0.0

        for i, s in enumerate(gpu_samples):
            watts_sum += s.watts
            watts_peak = max(watts_peak, s.watts)
            temp_peak = max(temp_peak, s.temp_c)
            vram_peak = max(vram_peak, s.vram_mib)
            if i > 0:
                dt = s.timestamp - gpu_samples[i - 1].timestamp
                avg_w = (s.watts + gpu_samples[i - 1].watts) / 2.0
                gpu_joules += avg_w * dt

        if n == 1:
            gpu_joules = gpu_samples[0].watts * wall_elapsed

        return EnergyReport(
            cpu_joules=cpu_joules,
            gpu_joules=gpu_joules,
            gpu_watts_avg=watts_sum / n,
            gpu_watts_peak=watts_peak,
            gpu_temp_peak_c=temp_peak,
            gpu_vram_peak_mib=vram_peak,
            gpu_samples=n,
        )


# ═══════════════════════════════════════════════════════════════════
#  BenchPhase — context manager
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PhaseResult:
    phase: str = ""
    substrate: str = "Python"
    wall_time_s: float = 0.0
    per_eval_us: float = 0.0
    n_evals: int = 0
    energy: dict = field(default_factory=dict)
    peak_rss_mb: float = 0.0
    chi2: float = 1e10
    precision_mev: float = 0.0
    notes: str = ""


class BenchPhase:
    """Context manager that captures time + energy for a benchmark phase."""

    def __init__(self, phase_name: str, substrate: str = "Python"):
        self.phase_name = phase_name
        self.substrate = substrate
        self._rapl_start = None
        self._wall_start = 0.0
        self._poller = NvidiaSmiPoller()
        self._chi2 = 1e10
        self._n_evals = 0
        self._precision_mev = 0.0
        self._notes = ""
        self._energy = EnergyReport()
        self._wall_elapsed = 0.0

    def __enter__(self):
        self._rapl_start = read_rapl_uj()
        self._poller.start()
        self._wall_start = time.monotonic()
        return self

    def __exit__(self, *args):
        self._wall_elapsed = time.monotonic() - self._wall_start
        rapl_end = read_rapl_uj()
        self._poller.stop()
        self._energy = EnergyReport.from_monitoring(
            self._rapl_start, rapl_end,
            self._poller.samples,
            self._wall_elapsed,
        )

    def set_physics(self, chi2: float = 1e10, n_evals: int = 0,
                    precision_mev: float = 0.0, notes: str = ""):
        self._chi2 = chi2
        self._n_evals = n_evals
        self._precision_mev = precision_mev
        self._notes = notes

    def result(self) -> PhaseResult:
        per_eval_us = 0.0
        if self._n_evals > 0:
            per_eval_us = self._wall_elapsed * 1e6 / self._n_evals

        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_mb = rss_kb / 1024.0

        return PhaseResult(
            phase=self.phase_name,
            substrate=self.substrate,
            wall_time_s=self._wall_elapsed,
            per_eval_us=per_eval_us,
            n_evals=self._n_evals,
            energy=asdict(self._energy),
            peak_rss_mb=peak_rss_mb,
            chi2=self._chi2,
            precision_mev=self._precision_mev,
            notes=self._notes,
        )


# ═══════════════════════════════════════════════════════════════════
#  Hardware Inventory
# ═══════════════════════════════════════════════════════════════════

@dataclass
class HardwareInventory:
    gate_name: str = "unknown"
    cpu_model: str = "unknown"
    cpu_cores: int = 0
    cpu_threads: int = 0
    cpu_cache_kb: int = 0
    ram_total_mb: int = 0
    gpu_name: str = "N/A"
    gpu_vram_mb: int = 0
    gpu_driver: str = "N/A"
    gpu_compute_cap: str = "N/A"
    os_kernel: str = "unknown"
    python_version: str = ""

    @staticmethod
    def detect(gate_name: str = "unknown") -> "HardwareInventory":
        import platform
        import sys

        hw = HardwareInventory(gate_name=gate_name)
        hw.python_version = sys.version.split()[0]
        hw.os_kernel = platform.release()

        # CPU
        try:
            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read()
            for line in cpuinfo.split("\n"):
                if line.startswith("model name"):
                    hw.cpu_model = line.split(":")[1].strip()
                    break
            hw.cpu_threads = len(re.findall(r"^processor\s", cpuinfo, re.MULTILINE))
            core_ids = set(re.findall(r"core id\s*:\s*(\d+)", cpuinfo))
            hw.cpu_cores = len(core_ids) if core_ids else hw.cpu_threads
            cache_match = re.search(r"cache size\s*:\s*(\d+)\s*KB", cpuinfo)
            if cache_match:
                hw.cpu_cache_kb = int(cache_match.group(1))
        except OSError:
            pass

        # RAM
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        hw.ram_total_mb = int(line.split()[1]) // 1024
                        break
        except OSError:
            pass

        # GPU
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total,driver_version,compute_cap",
                 "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
            parts = out.split(", ")
            if len(parts) >= 4:
                hw.gpu_name = parts[0].strip()
                hw.gpu_vram_mb = int(parts[1].strip())
                hw.gpu_driver = parts[2].strip()
                hw.gpu_compute_cap = parts[3].strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        return hw


# ═══════════════════════════════════════════════════════════════════
#  Report output
# ═══════════════════════════════════════════════════════════════════

def save_report(
    hw: HardwareInventory,
    phases: List[PhaseResult],
    output_dir: str = "benchmarks/nuclear-eos/results",
) -> str:
    """Save benchmark report as JSON.  Returns path written."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    gate_slug = hw.gate_name.lower().replace(" ", "_")
    filename = f"{gate_slug}_{timestamp.replace(':', '-')}_{hw.python_version}.json"
    path = os.path.join(output_dir, filename)

    report = {
        "timestamp": timestamp,
        "runtime": "python",
        "hardware": asdict(hw),
        "phases": [asdict(p) for p in phases],
    }

    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


def print_summary(hw: HardwareInventory, phases: List[PhaseResult]):
    """Print a human-readable summary table."""
    print()
    print("=" * 80)
    print(f"  PYTHON BENCHMARK REPORT — {hw.gate_name} ({hw.cpu_model})")
    print("=" * 80)
    print()
    header = f"  {'Phase':<18} {'Substrate':<14} {'Wall Time':>10} {'per-eval':>10} {'CPU (J)':>8} {'GPU W(avg)':>10} {'chi2':>8}"
    print(header)
    print(f"  {'─' * 78}")

    for p in phases:
        wall_str = _fmt_duration(p.wall_time_s)
        eval_str = _fmt_eval(p.per_eval_us) if p.per_eval_us > 0 else "—"
        cpu_j = f"{p.energy.get('cpu_joules', 0):.1f}" if p.energy.get("cpu_joules", 0) > 0 else "—"
        gpu_w = f"{p.energy.get('gpu_watts_avg', 0):.0f} W" if p.energy.get("gpu_watts_avg", 0) > 0 else "—"
        chi2_str = f"{p.chi2:.2f}" if p.chi2 < 1e8 else "—"
        print(f"  {p.phase:<18} {p.substrate:<14} {wall_str:>10} {eval_str:>10} {cpu_j:>8} {gpu_w:>10} {chi2_str:>8}")

    print(f"  {'─' * 78}")
    print()


def _fmt_duration(secs: float) -> str:
    if secs < 0.001:
        return f"{secs * 1e6:.1f} us"
    elif secs < 1.0:
        return f"{secs * 1e3:.1f} ms"
    elif secs < 60.0:
        return f"{secs:.2f} s"
    else:
        return f"{secs / 60:.1f} min"


def _fmt_eval(us: float) -> str:
    if us < 1000:
        return f"{us:.1f} us"
    elif us < 1_000_000:
        return f"{us / 1000:.2f} ms"
    else:
        return f"{us / 1_000_000:.2f} s"


if __name__ == "__main__":
    # Quick self-test
    hw = HardwareInventory.detect("Eastgate")
    print(f"CPU: {hw.cpu_model}")
    print(f"GPU: {hw.gpu_name} ({hw.gpu_vram_mb} MB)")
    print(f"RAM: {hw.ram_total_mb} MB")
    print(f"Kernel: {hw.os_kernel}")
    print(f"Python: {hw.python_version}")

    with BenchPhase("self_test") as bp:
        total = sum(i * i for i in range(10_000_000))
        bp.set_physics(chi2=0.0, n_evals=1, notes="self-test")

    result = bp.result()
    print(f"\nSelf-test: {result.wall_time_s:.3f}s, "
          f"CPU {result.energy.get('cpu_joules', 0):.1f}J, "
          f"RSS {result.peak_rss_mb:.0f}MB")
    print_summary(hw, [result])
