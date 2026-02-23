# NVK Setup Guide — Titan V via Open-Source Nouveau/Mesa

Reproducible checklist for running the Titan V alongside an NVIDIA proprietary
GPU (RTX 4070, RTX 3090, etc.) on the same machine. Validated on Eastgate
(Pop!_OS 22.04, Feb 21, 2026). Same steps apply to biomeGate or any node with
a Titan V in a secondary PCIe slot.

---

## Prerequisites

- Linux with Vulkan support (Pop!_OS, Ubuntu, Fedora, etc.)
- NVIDIA proprietary driver already installed for the primary GPU
- Titan V in a separate PCIe slot (check with `lspci | grep -i nvidia`)
- Build tools: `meson`, `ninja`, `python3-mako`, `libvulkan-dev`, `glslang-tools`

---

## Step 1: Identify PCIe Slots

```bash
lspci | grep -i nvidia
```

Example output (Eastgate):
```
01:00.0 VGA compatible controller: NVIDIA ... [RTX 4070]
05:00.0 VGA compatible controller: NVIDIA ... [TITAN V]
```

Note the slot IDs — these differ per motherboard/platform. Threadripper
(biomeGate) will likely assign different slot numbers due to its PCIe topology.

---

## Step 2: Build Mesa NVK

NVK is the open-source Vulkan driver for NVIDIA GPUs in Mesa. Most distros
don't ship it compiled. Build from source:

```bash
# Get Mesa source
git clone https://gitlab.freedesktop.org/mesa/mesa.git
cd mesa
git checkout mesa-25.1.5  # or latest stable

# Configure with NVK only
meson setup build \
  -Dvulkan-drivers=nouveau \
  -Dgallium-drivers= \
  -Dglx=disabled \
  -Dplatforms=x11,wayland \
  -Dbuildtype=release \
  -Dprefix=$HOME/Development/mesa-nvk-build/mesa-25.1.5/install

# Build and install (local prefix, no system contamination)
ninja -C build
ninja -C build install
```

The NVK shared library lands at:
```
~/Development/mesa-nvk-build/mesa-25.1.5/build/src/nouveau/vulkan/libvulkan_nouveau.so
```

---

## Step 3: Install the Vulkan ICD

Create an ICD JSON that points Vulkan to the local NVK build:

```bash
mkdir -p ~/.config/vulkan/icd.d

cat > ~/.config/vulkan/icd.d/nouveau_icd.json << 'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "/home/$USER/Development/mesa-nvk-build/mesa-25.1.5/build/src/nouveau/vulkan/libvulkan_nouveau.so",
        "api_version": "1.3.311"
    }
}
EOF
```

Replace `$USER` with the actual home directory path.

---

## Step 4: Configure modprobe for Dual-Driver Coexistence

The NVIDIA proprietary driver blacklists nouveau by default. We need nouveau
to bind the Titan V while nvidia keeps the primary GPU.

```bash
sudo tee /etc/modprobe.d/hotspring-nouveau-titanv.conf << 'EOF'
# hotSpring: Allow nouveau to bind Titan V alongside nvidia proprietary.
# Masks nvidia's "alias nouveau off" blacklist.
# nouveau binds Titan V; nvidia binds the primary GPU (RTX 4070/3090).
#
# After editing: sudo update-initramfs -u && reboot
install nouveau /sbin/modprobe --ignore-install nouveau
EOF

sudo update-initramfs -u
```

Reboot after this step.

---

## Step 5: Verify

After reboot, both drivers should be active:

```bash
# Check that nouveau bound the Titan V
lsmod | grep nouveau

# Check Vulkan sees both GPUs
vulkaninfo --summary
```

Expected output includes two GPUs:
- GPU0: primary GPU via nvidia proprietary
- GPU1: NVIDIA TITAN V (NVK GV100), Vulkan 1.3.x

Check SHADER_F64 support:
```bash
vulkaninfo | grep -A5 "TITAN V"
```

---

## Step 6: Validate with hotSpring

```bash
# Enumerate adapters
HOTSPRING_GPU_ADAPTER=titan cargo run --release --bin bench_gpu_fp64

# CPU-GPU parity (runs on Titan V via NVK)
HOTSPRING_GPU_ADAPTER=titan cargo run --release --bin validate_cpu_gpu_parity

# Full transport validation
HOTSPRING_GPU_ADAPTER=titan cargo run --release --bin validate_stanton_murillo

# Multi-GPU cooperative benchmark
source metalForge/nodes/biomegate.env  # or eastgate.env
cargo run --release --bin bench_multi_gpu
```

All checks should pass with identical physics to the proprietary-driver GPU.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `vulkaninfo` shows only 1 GPU | nouveau not loaded | Check modprobe config, reboot |
| NVK build fails on `nak` | Missing Rust toolchain for NAK compiler | Install `rustc` (NAK is Rust-based) |
| `SHADER_F64` not reported | Old Mesa version | Use mesa-25.1.5 or newer |
| NAK crash on `exp(f64)` | Known NVK limitation for Volta | Use `ShaderTemplate::for_driver_profile()` (auto-applies workaround) |
| Wrong GPU selected | Adapter enumeration order varies | Use `HOTSPRING_GPU_ADAPTER=titan` (name match) instead of index |

---

## Platform Notes

### Eastgate (validated)
- PCIe: RTX 4070 at `01:00.0`, Titan V at `05:00.0`
- OS: Pop!_OS 22.04
- Mesa: 25.1.5 (built from source)

### biomeGate (pending first boot)
- PCIe: RTX 3090 + Titan V — slot IDs TBD (Threadripper TRX40 topology)
- OS: TBD
- Same NVK build process applies; only PCIe slot IDs differ

---

## ToadStool Integration

ToadStool's barracuda crate auto-detects NVK via adapter info strings and
applies the Volta-specific driver profile (`DriverKind::Nvk`, `CompilerKind::Nak`,
`GpuArch::Volta`). This handles:
- exp/log workarounds (NAK limitation)
- Warp size detection (32 for Volta)
- Shader optimization flags

No manual driver profile configuration is needed — `GpuF64::new()` handles it.
