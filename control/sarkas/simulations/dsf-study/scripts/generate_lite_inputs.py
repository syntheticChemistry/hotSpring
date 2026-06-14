#!/usr/bin/env python
"""
Generate 'lite' (N=2000) versions of the PP Yukawa DSF study cases for Eastgate.

Takes each dsf_k*_G*_mks.yaml (PP cases only, kappa >= 1) and creates a
_lite variant with:
  - num: 2000 (instead of 10000)
  - production_steps: 30000 (instead of 80000)
  - prod_dump_step: 10 (instead of 5)
  - job_dir: dsf_k{kappa}_G{gamma}_lite
"""

import os
import re
from pathlib import Path

INPUT_DIR = Path(__file__).resolve().parent.parent / "input_files"

pp_cases = sorted(INPUT_DIR.glob("dsf_k[123]_G*_mks.yaml"))

print(f"Found {len(pp_cases)} PP cases to convert to lite")
print()

created = []
for src in pp_cases:
    content = src.read_text()
    
    # Skip if already a lite file
    if '_lite' in src.name:
        continue
    
    # Extract kappa and gamma from filename
    m = re.match(r'dsf_k(\d+)_G(\d+)_mks\.yaml', src.name)
    if not m:
        continue
    kappa, gamma = m.group(1), m.group(2)
    
    lite_name = f"dsf_k{kappa}_G{gamma}_mks_lite.yaml"
    lite_path = INPUT_DIR / lite_name
    
    # Already exists?
    if lite_path.exists():
        print(f"  {lite_name} already exists, skipping")
        continue
    
    # Modify parameters for lite
    lite_content = content
    
    # Change num
    lite_content = re.sub(r'num: 10000\b', 'num: 2000', lite_content)
    
    # Change production steps
    lite_content = re.sub(r'production_steps: 80000\b', 'production_steps: 30000', lite_content)
    
    # Change prod dump step
    lite_content = re.sub(r'prod_dump_step: 5\b', 'prod_dump_step: 10', lite_content)
    
    # Change job_dir
    lite_content = re.sub(
        rf'job_dir: dsf_k{kappa}_G{gamma}\b',
        f'job_dir: dsf_k{kappa}_G{gamma}_lite',
        lite_content
    )
    
    # Add lite header
    lite_content = lite_content.replace(
        f"# Yukawa DSF Study: kappa={kappa}, Gamma={gamma}",
        f"# Yukawa DSF Study: kappa={kappa}, Gamma={gamma} — LITE VERSION (Eastgate, N=2000)",
        1
    )
    
    # Reduce max_k_harmonics for faster post-processing
    lite_content = re.sub(r'max_k_harmonics: 30', 'max_k_harmonics: 20', lite_content)
    
    lite_path.write_text(lite_content)
    print(f"  Created: {lite_name}")
    created.append(lite_path)

print(f"\n{len(created)} lite files created")

