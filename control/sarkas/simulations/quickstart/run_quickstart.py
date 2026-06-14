#!/usr/bin/env python
"""
hotSpring - Sarkas Quickstart Validation
Run from: hotSpring/control/sarkas/simulations/quickstart/
Env: conda activate sarkas (or micromamba activate sarkas)

Reproduces the Sarkas quickstart tutorial headlessly.
Validates: installation, simulation correctness, basic output.
"""
import time
import os
import sys

# Ensure output goes to this directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("hotSpring - Sarkas Quickstart Validation")
print("=" * 60)

# Import sarkas
from sarkas.processes import PreProcess, Simulation, PostProcess

input_file = "yocp_quickstart.yaml"

print(f"\nInput file: {input_file}")
print(f"Working dir: {os.getcwd()}")

# Phase 1: Preprocessing
print("\n--- Phase 1: Preprocessing ---")
t0 = time.time()
preproc = PreProcess(input_file)
preproc.setup(read_yaml=True)
preproc.run()
t_pre = time.time() - t0
print(f"Preprocessing: {t_pre:.2f}s")

# Phase 2: Simulation
print("\n--- Phase 2: Simulation ---")
t0 = time.time()
sim = Simulation(input_file)
sim.setup(read_yaml=True)
sim.run()
t_sim = time.time() - t0
print(f"Simulation: {t_sim:.2f}s")

# Phase 3: Post-processing
print("\n--- Phase 3: Post-processing ---")
t0 = time.time()
postproc = PostProcess(input_file)
postproc.setup(read_yaml=True)
postproc.run()
t_post = time.time() - t0
print(f"Post-processing: {t_post:.2f}s")

# Summary
print("\n" + "=" * 60)
print("TIMING SUMMARY")
print(f"  Preprocessing:  {t_pre:.2f}s")
print(f"  Simulation:     {t_sim:.2f}s")
print(f"  Post-processing: {t_post:.2f}s")
print(f"  Total:          {t_pre + t_sim + t_post:.2f}s")
print("=" * 60)

# Check output exists
output_dir = "yocp_quickstart"
if os.path.exists(output_dir):
    print(f"\nOutput directory: {output_dir}/")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        indent = '  ' * (level + 1)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = '  ' * (level + 2)
        for file in files:
            size = os.path.getsize(os.path.join(root, file))
            print(f"{subindent}{file} ({size:,} bytes)")
else:
    print(f"\nWARNING: Output directory '{output_dir}' not found!")
    sys.exit(1)

print("\nQuickstart validation COMPLETE.")
