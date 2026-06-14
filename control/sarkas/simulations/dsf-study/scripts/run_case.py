#!/usr/bin/env python
"""
Run a single Sarkas DSF study case headlessly.

Usage:
    python run_case.py <yaml_file> [--preprocess-only] [--skip-preprocess] [--pppm-estimate]

This script follows the exact Sarkas workflow from the OCP example notebook:
    1. PreProcess  - validate parameters, set up PPPM, estimate timing
    2. Simulation  - run equilibration + production MD
    3. PostProcess - compute observables (RDF, DSF, SSF, Thermodynamics, VACF)

All output goes to the Simulations/ directory under the job_dir specified in the YAML.
"""

import argparse
import os
import sys
import time
import traceback

# Change to the script's directory so relative paths work
script_dir = os.path.dirname(os.path.abspath(__file__))
study_dir = os.path.dirname(script_dir)
os.chdir(study_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="Run Sarkas DSF study case")
    parser.add_argument("yaml_file", help="Path to YAML input file")
    parser.add_argument("--preprocess-only", action="store_true",
                        help="Only run preprocessing (parameter validation)")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing, go straight to simulation")
    parser.add_argument("--pppm-estimate", action="store_true",
                        help="Run PPPM force error estimation during preprocessing")
    parser.add_argument("--skip-simulation", action="store_true",
                        help="Skip simulation, only run post-processing")
    return parser.parse_args()


def run_preprocess(input_file, pppm_estimate=False):
    """Run Sarkas PreProcess step."""
    from sarkas.processes import PreProcess
    import matplotlib
    matplotlib.use('Agg')  # headless

    print("\n" + "=" * 60)
    print("PHASE 1: PreProcessing")
    print("=" * 60)

    t0 = time.time()
    preproc = PreProcess(input_file)
    preproc.setup(read_yaml=True)
    preproc.run(pppm_estimate=pppm_estimate, timing=True, remove=True)
    elapsed = time.time() - t0

    print(f"\nPreprocessing complete: {elapsed:.2f}s")

    # Report key parameters
    params = preproc.parameters
    print(f"\n--- Simulation Parameters ---")
    print(f"  Particles:       {params.total_num_ptcls}")
    print(f"  a_ws:            {params.a_ws:.6e} m")
    print(f"  Plasma freq:     {params.total_plasma_frequency:.6e} rad/s")
    print(f"  Coupling (Gamma):{params.coupling_constant:.4f}")
    print(f"  dt:              {preproc.integrator.dt:.6e} s")
    dt_reduced = preproc.integrator.dt * params.total_plasma_frequency
    print(f"  dt (reduced):    {dt_reduced:.6f}")
    print(f"  Eq steps:        {params.equilibration_steps}")
    print(f"  Prod steps:      {params.production_steps}")
    print(f"  Box length:      {params.box_lengths[0]:.6e} m")

    if hasattr(preproc.potential, 'pppm_on') and preproc.potential.pppm_on:
        print(f"\n--- PPPM Parameters ---")
        print(f"  Mesh:            {preproc.potential.pppm_mesh}")
        print(f"  CAO:             {preproc.potential.pppm_cao}")
        print(f"  Alpha (Ewald):   {preproc.potential.pppm_alpha_ewald:.6e}")
        print(f"  rc:              {preproc.potential.rc:.6e} m")
        if hasattr(preproc.potential, 'force_error'):
            print(f"  Force error:     {preproc.potential.force_error:.6e}")

    return elapsed


def run_simulation(input_file):
    """Run Sarkas Simulation step."""
    from sarkas.processes import Simulation

    print("\n" + "=" * 60)
    print("PHASE 2: Simulation (MD)")
    print("=" * 60)

    t0 = time.time()
    sim = Simulation(input_file)
    sim.setup(read_yaml=True)
    sim.run()
    elapsed = time.time() - t0

    print(f"\nSimulation complete: {elapsed:.2f}s")
    return elapsed


def run_postprocess(input_file):
    """Run Sarkas PostProcess step."""
    from sarkas.processes import PostProcess
    import matplotlib
    matplotlib.use('Agg')  # headless

    print("\n" + "=" * 60)
    print("PHASE 3: PostProcessing")
    print("=" * 60)

    t0 = time.time()
    postproc = PostProcess(input_file)
    postproc.setup(read_yaml=True)

    # Compute all observables that were defined in the YAML
    print("\nComputing observables...")

    # RDF
    try:
        print("  - Radial Distribution Function...", end=" ", flush=True)
        postproc.rdf.setup(postproc.parameters)
        postproc.rdf.compute()
        print("done")
    except Exception as ex:
        print(f"FAILED: {ex}")

    # Thermodynamics
    try:
        print("  - Thermodynamics...", end=" ", flush=True)
        postproc.therm.setup(postproc.parameters)
        postproc.therm.compute()
        print("done")
    except Exception as ex:
        print(f"FAILED: {ex}")

    # Dynamic Structure Factor
    try:
        print("  - Dynamic Structure Factor...", end=" ", flush=True)
        from sarkas.tools.observables import DynamicStructureFactor
        dsf = DynamicStructureFactor()
        dsf.setup(postproc.parameters)
        dsf.compute()
        print("done")
    except Exception as ex:
        print(f"FAILED: {ex}")

    # Static Structure Factor
    try:
        print("  - Static Structure Factor...", end=" ", flush=True)
        from sarkas.tools.observables import StaticStructureFactor
        ssf = StaticStructureFactor()
        ssf.setup(postproc.parameters)
        ssf.compute()
        print("done")
    except Exception as ex:
        print(f"FAILED: {ex}")

    # Velocity Autocorrelation Function
    try:
        print("  - Velocity Autocorrelation Function...", end=" ", flush=True)
        from sarkas.tools.observables import VelocityAutoCorrelationFunction
        vacf = VelocityAutoCorrelationFunction()
        vacf.setup(postproc.parameters)
        vacf.compute()
        print("done")
    except Exception as ex:
        print(f"FAILED: {ex}")

    elapsed = time.time() - t0
    print(f"\nPostprocessing complete: {elapsed:.2f}s")
    return elapsed


def main():
    args = parse_args()

    # Resolve input file path
    input_file = os.path.abspath(args.yaml_file)
    if not os.path.exists(input_file):
        # Try relative to input_files/
        alt = os.path.join(study_dir, "input_files", args.yaml_file)
        if os.path.exists(alt):
            input_file = alt
        else:
            print(f"ERROR: Input file not found: {args.yaml_file}")
            sys.exit(1)

    print("=" * 60)
    print("hotSpring - Sarkas DSF Study Runner")
    print("=" * 60)
    print(f"Input: {input_file}")
    print(f"CWD:   {os.getcwd()}")

    timings = {}
    total_t0 = time.time()

    try:
        # Phase 1: PreProcess
        if not args.skip_preprocess and not args.skip_simulation:
            timings["preprocess"] = run_preprocess(
                input_file, pppm_estimate=args.pppm_estimate
            )

        if args.preprocess_only:
            print("\n[--preprocess-only] Stopping after preprocessing.")
            return

        # Phase 2: Simulation
        if not args.skip_simulation:
            timings["simulation"] = run_simulation(input_file)

        # Phase 3: PostProcess
        timings["postprocess"] = run_postprocess(input_file)

    except Exception as ex:
        print(f"\nFATAL ERROR: {ex}")
        traceback.print_exc()
        sys.exit(1)

    total_elapsed = time.time() - total_t0

    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    for phase, t in timings.items():
        print(f"  {phase:<20} {t:>10.2f}s")
    print(f"  {'TOTAL':<20} {total_elapsed:>10.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
