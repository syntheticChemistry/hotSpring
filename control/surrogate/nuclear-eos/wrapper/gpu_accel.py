#!/usr/bin/env python3
"""
GPU Acceleration for Nuclear EOS Surrogate Learning

Uses PyTorch CUDA to accelerate the HFB self-consistency loop.
Key insight: keep ALL tensors on GPU for the entire solve, avoid
CPU↔GPU transfer overhead that kills small-matrix GPU performance.

Profile (CPU, 100Sn):
  eigh:       52% of runtime (84 calls × 28ms = 2.36s) → GPU: 6.3x
  trapezoid:  21% (70,730 calls) → vectorized on GPU
  energy:     12% → stays on GPU
  gradient:    9% → pre-computed on GPU

Hardware: NVIDIA RTX 4070 (12GB VRAM, Compute 8.9)
Target: 4.5s/nucleus (CPU) → 1-2s/nucleus (GPU)

Author: ecoPrimals — consumer GPU beats institutional Fortran
License: AGPL-3.0
"""

import numpy as np

_GPU_AVAILABLE = False
_DEVICE = None
_torch = None


def init_gpu(verbose=True):
    """Initialize GPU. Returns True if CUDA available."""
    global _GPU_AVAILABLE, _DEVICE, _torch
    try:
        import torch
        _torch = torch
        if torch.cuda.is_available():
            _DEVICE = torch.device('cuda')
            _GPU_AVAILABLE = True
            if verbose:
                name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"  GPU: {name} ({mem:.1f} GB) — CUDA {torch.version.cuda}")
        else:
            _DEVICE = torch.device('cpu')
            _GPU_AVAILABLE = False
            if verbose:
                print("  GPU: CUDA not available, staying on CPU")
    except ImportError:
        _GPU_AVAILABLE = False
        if verbose:
            print("  GPU: PyTorch not installed, staying on CPU")
    return _GPU_AVAILABLE


def is_gpu_available():
    return _GPU_AVAILABLE


# ============================================================================
# GPU-Resident HFB Solver
# ============================================================================

class GPUHFBSolver:
    """GPU-accelerated HFB self-consistency loop.

    Keeps all arrays on GPU for the entire solve, transferring only
    the final binding energy back to CPU. This avoids the per-operation
    CPU↔GPU transfer overhead that makes individual GPU calls slower
    for small (91×91) matrices.

    Strategy:
    - Pre-transfer: wf, dwf, r → GPU (once per solver)
    - Each iteration: build H, diagonalize, build density — ALL on GPU
    - Result: only binding energy comes back to CPU
    """

    def __init__(self, hfb_solver):
        """Wrap a SphericalHFB solver with GPU tensors.

        Parameters
        ----------
        hfb_solver : SphericalHFB instance (from skyrme_hfb.py)
        """
        torch = _torch
        dev = _DEVICE
        self.cpu = hfb_solver  # Keep reference for metadata
        self.ns = hfb_solver.n_states
        self.nr = hfb_solver.nr

        # Transfer basis to GPU (ONCE — these never change)
        self.wf = torch.tensor(hfb_solver.wf, dtype=torch.float64, device=dev)
        self.dwf = torch.tensor(hfb_solver.dwf, dtype=torch.float64, device=dev)
        self.r = torch.tensor(hfb_solver.r, dtype=torch.float64, device=dev)
        self.dr = float(hfb_solver.dr)
        self.r2 = self.r ** 2
        self.r2dr = self.r2 * self.dr  # r² · dr for volume integration
        self.vol = 4.0 * np.pi * self.r2dr  # 4πr² dr

        # Block structure (stays on CPU — just indices)
        self.lj_blocks = hfb_solver._lj_blocks
        self.states = hfb_solver.states
        self.degs = torch.tensor(
            [s[3] for s in hfb_solver.states],
            dtype=torch.float64, device=dev)

        # Precompute l(l+1) for each state
        self.ll1 = torch.tensor(
            [s[1] * (s[1] + 1) for s in hfb_solver.states],
            dtype=torch.float64, device=dev)

        # Precompute block indices as GPU tensors
        self._block_idx = {}
        self._block_ll1 = {}
        for (l_val, j_val), indices in self.lj_blocks.items():
            self._block_idx[(l_val, j_val)] = torch.tensor(
                indices, dtype=torch.long, device=dev)
            self._block_ll1[(l_val, j_val)] = l_val * (l_val + 1)

        # Pre-compute wf products for fast matrix elements
        # wf_r2: R_i(r) * r (for potential integrals with r² dr)
        # These are used in every iteration
        self.wf_r2 = self.wf * self.r[None, :]  # (ns, nr) — for sqrt(r²)

        # Pre-compute spin-orbit ls values
        self.ls_vals = torch.zeros(self.ns, dtype=torch.float64, device=dev)
        for i, (n, l, j, deg) in enumerate(self.states):
            if l > 0:
                self.ls_vals[i] = (j*(j+1) - l*(l+1) - 0.75) / 2.0

        # Constants from the CPU solver
        self.hw = hfb_solver.hw
        self.b = hfb_solver.b
        self.Z = hfb_solver.Z
        self.N = hfb_solver.N
        self.A = hfb_solver.A
        self.delta_p = hfb_solver.delta_p
        self.delta_n = hfb_solver.delta_n

    def _build_teff(self, f_q):
        """Build T_eff matrix entirely on GPU.

        T_eff[i,j] = ∫ f_q(r) [dR_i·dR_j·r² + l(l+1)·R_i·R_j] dr

        Vectorized per (l,j) block as matrix-vector products.
        """
        torch = _torch
        T = torch.zeros((self.ns, self.ns), dtype=torch.float64, device=_DEVICE)

        for key, indices in self._block_idx.items():
            ll1 = self._block_ll1[key]
            n_b = len(indices)

            # Block wavefunctions
            wf_b = self.wf[indices]    # (n_b, nr)
            dwf_b = self.dwf[indices]  # (n_b, nr)

            # Integrand weights
            w_deriv = f_q * self.r2 * self.dr        # (nr,) for derivative term
            w_ll = f_q * ll1 * self.dr if ll1 > 0 else None  # (nr,) for l(l+1) term

            # T_deriv[i,j] = Σ_r dwf_i(r) · dwf_j(r) · f_q(r) · r² · dr
            T_block = (dwf_b * w_deriv[None, :]) @ dwf_b.T  # (n_b, n_b)

            if w_ll is not None:
                T_block += (wf_b * w_ll[None, :]) @ wf_b.T

            # Scatter into full matrix
            T[indices[:, None], indices[None, :]] = T_block

        return T

    def _build_potential_matrix(self, U_total, W0, drho_r):
        """Build potential matrix entirely on GPU.

        V[i,j] = ∫ R_i · U · R_j · r² dr
        + W0 · <l·s> · ∫ R_i² · (dρ/dr)/r · r² dr  (diagonal)
        """
        torch = _torch
        V = torch.zeros((self.ns, self.ns), dtype=torch.float64, device=_DEVICE)

        # Weight for potential integrals
        w = U_total * self.r2dr  # (nr,)

        for key, indices in self._block_idx.items():
            n_b = len(indices)
            wf_b = self.wf[indices]  # (n_b, nr)
            V_block = (wf_b * w[None, :]) @ wf_b.T  # (n_b, n_b)
            V[indices[:, None], indices[None, :]] = V_block

        # Spin-orbit (diagonal)
        if W0 != 0.0 and drho_r is not None:
            for i in range(self.ns):
                if self.ls_vals[i] != 0.0:
                    so = torch.sum(self.wf[i]**2 * drho_r * self.r2dr)
                    V[i, i] += W0 * self.ls_vals[i] * so

        return V

    def _density_from_eigvecs(self, eigvecs, v2):
        """Compute density on GPU.

        ρ(r) = Σ_i (2j+1)v²_i |Σ_k c_ki R_k(r)|² / (4π)
        """
        torch = _torch
        weights = self.degs * v2  # (ns,)

        # φ_i(r) = eigvecs.T @ wf → (ns, nr) — all wavefunctions at once
        phi = eigvecs.T @ self.wf  # (ns, nr)

        # ρ = Σ_i w_i · |φ_i|² / (4π)
        rho = torch.sum(weights[:, None] * phi**2, dim=0) / (4.0 * np.pi)
        return torch.clamp(rho, min=1e-15)

    def _coulomb_direct_gpu(self, rho_p):
        """Coulomb direct potential on GPU."""
        torch = _torch
        # Q(r) = ∫₀ʳ ρ_p 4πr'² dr'
        charge_enclosed = torch.cumsum(rho_p * self.vol, dim=0)
        # Φ_out(r) = ∫ᵣ^∞ ρ_p 4πr' dr'
        phi_outer = torch.flip(
            torch.cumsum(torch.flip(rho_p * 4.0 * np.pi * self.r * self.dr, [0]), 0),
            [0])
        r_safe = torch.clamp(self.r, min=1e-10)
        E2 = 1.4399764  # e²/(4πε₀) in MeV·fm
        return E2 * (charge_enclosed / r_safe + phi_outer)

    def _coulomb_exchange_gpu(self, rho_p):
        """Coulomb exchange on GPU."""
        E2 = 1.4399764
        return -E2 * (3.0 / np.pi)**(1.0/3.0) * _torch.clamp(rho_p, min=0.0)**(1.0/3.0)

    def _bcs_occupations_gpu(self, energies, num_particles, delta):
        """BCS occupation numbers on GPU."""
        torch = _torch
        if num_particles <= 0:
            return torch.zeros(self.ns, dtype=torch.float64, device=_DEVICE), 0.0

        if delta < 0.01:
            # Sharp filling
            idx = torch.argsort(energies)
            v2 = torch.zeros(self.ns, dtype=torch.float64, device=_DEVICE)
            remaining = float(num_particles)
            for i in idx:
                fill = min(remaining, float(self.degs[i]))
                v2[i] = fill / float(self.degs[i])
                remaining -= fill
                if remaining <= 0:
                    break
            lam = float(energies[idx[min(
                int(torch.searchsorted(
                    torch.cumsum(self.degs[idx], 0),
                    torch.tensor(num_particles, dtype=torch.float64, device=_DEVICE))),
                len(idx)-1)]])
            return v2, lam

        # BCS with brentq on CPU (small operation)
        e_np = energies.cpu().numpy()
        d_np = self.degs.cpu().numpy()

        from scipy.optimize import brentq
        def particle_number(lam):
            ek = e_np - lam
            Ek = np.sqrt(ek**2 + delta**2)
            v2 = 0.5 * (1.0 - ek / Ek)
            return np.sum(d_np * v2) - num_particles

        e_min, e_max = float(energies.min()) - 50, float(energies.max()) + 50
        try:
            lam = brentq(particle_number, e_min, e_max, xtol=1e-6)
        except ValueError:
            # Fallback
            idx = np.argsort(e_np)
            count = 0
            for i in idx:
                count += d_np[i]
                if count >= num_particles:
                    lam = e_np[i]
                    break
            else:
                lam = e_np[idx[-1]]

        ek = energies - lam
        Ek = torch.sqrt(ek**2 + delta**2)
        v2 = 0.5 * (1.0 - ek / Ek)
        return v2, lam

    def solve(self, skyrme_params, max_iter=150, tol=0.05, mixing=0.3):
        """GPU-accelerated HF+BCS self-consistency loop.

        Entire iteration runs on GPU. Only final result transfers to CPU.
        """
        torch = _torch
        dev = _DEVICE
        from skyrme_hf import _to_dict, HBAR2_2M, E2

        p = _to_dict(skyrme_params)
        W0 = p.get("W0", 0.0)

        t0, t1, t2, t3 = p['t0'], p['t1'], p['t2'], p['t3']
        x0, x1, x2, x3 = p['x0'], p['x1'], p['x2'], p['x3']
        alpha = p['alpha']

        # Initial densities on GPU
        R_nuc = 1.2 * self.A**(1.0/3.0)
        rho0 = 3.0 * self.A / (4.0 * np.pi * R_nuc**3)
        rho_p = torch.where(self.r < R_nuc,
                            torch.tensor(rho0 * self.Z / self.A, dtype=torch.float64, device=dev),
                            torch.tensor(1e-15, dtype=torch.float64, device=dev))
        rho_n = torch.where(self.r < R_nuc,
                            torch.tensor(rho0 * self.N / self.A, dtype=torch.float64, device=dev),
                            torch.tensor(1e-15, dtype=torch.float64, device=dev))
        rho_p = torch.clamp(rho_p, min=1e-15)
        rho_n = torch.clamp(rho_n, min=1e-15)

        # Effective mass coefficients
        C0t = 0.25 * (t1 * (1 + x1/2) + t2 * (1 + x2/2))
        C1n = 0.25 * (t1 * (0.5 + x1) - t2 * (0.5 + x2))

        E_prev = 1e10
        converged = False
        results_q = {}

        for iteration in range(max_iter):
            rho = rho_p + rho_n
            rho_safe = torch.clamp(rho, min=1e-20)

            # Spin-orbit: dρ/dr / r
            if W0 != 0:
                drho = torch.zeros_like(rho)
                drho[1:-1] = (rho[2:] - rho[:-2]) / (2 * self.dr)
                drho[0] = (rho[1] - rho[0]) / self.dr
                drho[-1] = (rho[-1] - rho[-2]) / self.dr
                drho_r = drho / torch.clamp(self.r, min=0.1)
            else:
                drho_r = None

            rho_p_new = torch.zeros(self.nr, dtype=torch.float64, device=dev)
            rho_n_new = torch.zeros(self.nr, dtype=torch.float64, device=dev)

            for q, num_q, delta_q in [('p', self.Z, self.delta_p),
                                       ('n', self.N, self.delta_n)]:
                rho_q = rho_p if q == 'p' else rho_n

                # --- Skyrme potential (on GPU) ---
                U = t0 * ((1.0 + x0/2.0) * rho - (0.5 + x0) * rho_q)

                rho_alpha = rho_safe ** alpha
                rho_alpha_m1 = torch.where(rho > 1e-15, rho_safe ** (alpha - 1), torch.zeros_like(rho))
                sum_rho2 = rho_p**2 + rho_n**2

                U = U + (t3 / 12.0) * (
                    (1.0 + x3/2.0) * (alpha + 2) * rho_alpha * rho
                    - (0.5 + x3) * (
                        alpha * rho_alpha_m1 * sum_rho2
                        + 2.0 * rho_alpha * rho_q))

                if q == 'p':
                    V_C = self._coulomb_direct_gpu(rho_p)
                    V_Cx = self._coulomb_exchange_gpu(rho_p)
                    U_total = U + V_C + V_Cx
                else:
                    U_total = U

                # --- Effective mass function ---
                f_q = HBAR2_2M + C0t * rho - C1n * rho_q
                f_q = torch.clamp(f_q, min=HBAR2_2M * 0.3)

                # --- Build H = T_eff + V ---
                T_eff = self._build_teff(f_q)
                V_mat = self._build_potential_matrix(U_total, W0, drho_r)
                H = T_eff + V_mat

                # --- Diagonalize (on GPU!) ---
                eigenvalues, eigvecs = torch.linalg.eigh(H)

                # --- BCS occupation ---
                v2, lam_q = self._bcs_occupations_gpu(eigenvalues, num_q, delta_q)

                # --- New density (on GPU) ---
                rho_q_new = self._density_from_eigvecs(eigvecs, v2)

                if q == 'p':
                    rho_p_new = rho_q_new
                else:
                    rho_n_new = rho_q_new

                results_q[q] = {
                    'eigenvalues': eigenvalues,
                    'eigvecs': eigvecs,
                    'v2': v2,
                    'lambda': lam_q,
                }

            # --- Mix densities ---
            rho_p = mixing * rho_p_new + (1.0 - mixing) * rho_p
            rho_n = mixing * rho_n_new + (1.0 - mixing) * rho_n

            # --- Total energy ---
            E_total = self._compute_energy_gpu(
                rho_p, rho_n, results_q, p, C0t, C1n)

            dE = abs(E_total - E_prev)
            if dE < tol and iteration > 5:
                converged = True
                break
            E_prev = E_total

        # --- Transfer result to CPU ---
        B = -E_total if E_total < 0 else 0.0

        norm_p = torch.sum(rho_p * self.vol)
        if norm_p > 0.1:
            r_ch2 = torch.sum(rho_p * self.r**2 * self.vol) / norm_p
            r_ch = float(torch.sqrt(torch.abs(r_ch2) + 0.64))
        else:
            r_ch = 1.2 * self.A**(1.0/3.0)

        return {
            "Z": self.Z, "N": self.N, "A": self.A,
            "binding_energy_MeV": float(B),
            "charge_radius_fm": r_ch,
            "converged": converged,
            "iterations": iteration + 1,
            "delta_E_MeV": float(dE),
            "lambda_p_MeV": results_q.get('p', {}).get('lambda', 0),
            "lambda_n_MeV": results_q.get('n', {}).get('lambda', 0),
        }

    def _compute_energy_gpu(self, rho_p, rho_n, results_q, p, C0t, C1n):
        """Compute total energy entirely on GPU."""
        torch = _torch
        from skyrme_hf import HBAR2_2M, E2

        rho = rho_p + rho_n
        rho_safe = torch.clamp(rho, min=1e-20)

        t0, t1, t2, t3 = p['t0'], p['t1'], p['t2'], p['t3']
        x0, x1, x2, x3 = p['x0'], p['x1'], p['x2'], p['x3']
        alpha = p['alpha']

        # --- Kinetic energy via T_eff eigenvectors ---
        E_kin = torch.tensor(0.0, dtype=torch.float64, device=_DEVICE)
        for q_label in ('p', 'n'):
            rq = results_q[q_label]
            v2_q = rq['v2']
            evecs = rq['eigvecs']
            rho_q = rho_p if q_label == 'p' else rho_n

            f_q = torch.clamp(
                HBAR2_2M + C0t * rho - C1n * rho_q,
                min=HBAR2_2M * 0.3)

            T_eff_q = self._build_teff(f_q)

            weights = self.degs * v2_q
            # E_kin += Σ_i w_i · cᵢᵀ T_eff cᵢ
            # Vectorized: E_kin = trace(diag(w) · Cᵀ T_eff C)
            #            = Σ_i w_i (C[:,i]ᵀ @ T @ C[:,i])
            # Efficient: (T @ C) element-wise * C, sum, dot with weights
            TC = T_eff_q @ evecs  # (ns, ns)
            per_state = torch.sum(evecs * TC, dim=0)  # (ns,)
            E_kin = E_kin + torch.sum(weights * per_state)

        # --- Skyrme functional ---
        sum_rho2 = rho_p**2 + rho_n**2

        integrand_t0 = (1 + x0/2.0) * rho**2 - (0.5 + x0) * sum_rho2
        E_t0 = (t0 / 2.0) * torch.sum(integrand_t0 * self.vol)

        integrand_t3 = rho_safe**alpha * (
            (1 + x3/2.0) * rho**2 - (0.5 + x3) * sum_rho2)
        E_t3 = (t3 / 12.0) * torch.sum(integrand_t3 * self.vol)

        # --- Coulomb ---
        V_C = self._coulomb_direct_gpu(rho_p)
        E_Coul_d = 0.5 * torch.sum(V_C * rho_p * self.vol)
        V_Cx = self._coulomb_exchange_gpu(rho_p)
        E_Coul_x = torch.sum(V_Cx * rho_p * self.vol)

        # --- BCS pairing ---
        E_pair = torch.tensor(0.0, dtype=torch.float64, device=_DEVICE)
        for q, delta_q in [('p', self.delta_p), ('n', self.delta_n)]:
            v2 = results_q[q]['v2']
            u2 = 1.0 - v2
            E_pair = E_pair - delta_q * torch.sum(
                self.degs * torch.sqrt(torch.clamp(v2 * u2, min=0.0)))

        # --- Center-of-mass correction ---
        E_cm = -0.75 * self.hw

        return float(E_kin + E_t0 + E_t3 + E_Coul_d + E_Coul_x + E_pair + E_cm)


# ============================================================================
# GPU-Accelerated RBF Surrogate
# ============================================================================

class GPURBFInterpolator:
    """RBF interpolator with GPU-accelerated linear algebra.

    Uses torch.linalg.solve for the kernel system. Faster than scipy
    for N > 2000 points.
    """

    def __init__(self, X, y, kernel='thin_plate_spline'):
        torch = _torch
        dev = _DEVICE

        self.X_train = torch.tensor(X, dtype=torch.float64, device=dev)
        self.y_train = torch.tensor(y, dtype=torch.float64, device=dev)
        self.N, self.D = X.shape

        K = self._kernel_matrix(self.X_train, self.X_train)
        P = torch.cat([torch.ones(self.N, 1, dtype=torch.float64, device=dev),
                        self.X_train], dim=1)
        M = self.D + 1

        A = torch.zeros(self.N + M, self.N + M, dtype=torch.float64, device=dev)
        A[:self.N, :self.N] = K
        A[:self.N, self.N:] = P
        A[self.N:, :self.N] = P.T
        A[:self.N, :self.N] += 1e-8 * torch.eye(self.N, dtype=torch.float64, device=dev)

        rhs = torch.zeros(self.N + M, dtype=torch.float64, device=dev)
        rhs[:self.N] = self.y_train

        try:
            self.coeffs = torch.linalg.solve(A, rhs)
        except Exception:
            self.coeffs, *_ = torch.linalg.lstsq(A, rhs.unsqueeze(1))
            self.coeffs = self.coeffs.squeeze()

        self.weights = self.coeffs[:self.N]
        self.poly_coeffs = self.coeffs[self.N:]

    def _kernel_matrix(self, X1, X2):
        diff = X1[:, None, :] - X2[None, :, :]
        r = torch.norm(diff, dim=2)
        r_safe = torch.clamp(r, min=1e-20)
        K = r**2 * torch.log(r_safe)
        K[r < 1e-15] = 0.0
        return K

    def __call__(self, X_test):
        torch = _torch
        dev = _DEVICE
        X_t = torch.tensor(X_test, dtype=torch.float64, device=dev)
        K = self._kernel_matrix(X_t, self.X_train)
        P = torch.cat([torch.ones(len(X_test), 1, dtype=torch.float64, device=dev),
                        X_t], dim=1)
        return (K @ self.weights + P @ self.poly_coeffs).cpu().numpy()


# ============================================================================
# Convenience: GPU-accelerated binding energy
# ============================================================================

_gpu_solver_cache = {}


def gpu_binding_energy(Z, N, skyrme_params, max_cache=50):
    """Compute binding energy using GPU-accelerated HFB.

    Drop-in replacement for hfb_binding_energy().
    """
    key = (Z, N)
    if key not in _gpu_solver_cache:
        if len(_gpu_solver_cache) > max_cache:
            _gpu_solver_cache.clear()
        from skyrme_hfb import SphericalHFB
        cpu_solver = SphericalHFB(Z, N)
        _gpu_solver_cache[key] = GPUHFBSolver(cpu_solver)

    solver = _gpu_solver_cache[key]
    result = solver.solve(skyrme_params)
    return result["binding_energy_MeV"], result["converged"]


# ============================================================================
# Self-Test
# ============================================================================

if __name__ == "__main__":
    import time, sys
    sys.path.insert(0, '.')

    print("=" * 60)
    print("GPU HFB Solver — Benchmark vs CPU")
    print("=" * 60)

    ok = init_gpu()
    if not ok:
        print("No GPU available.")
        exit(1)

    from skyrme_hfb import SphericalHFB, hfb_binding_energy

    sly4 = [-2488.91, 486.82, -546.39, 13777.0,
            0.834, -0.344, -1.0, 1.354, 0.1667, 123.0]

    test_nuclei = [
        (28, 28, "⁵⁶Ni", 483.988),
        (50, 50, "¹⁰⁰Sn", 824.793),
        (50, 82, "¹³²Sn", 1102.851),
    ]

    print(f"\n{'Nucleus':>8}  {'CPU(s)':>7}  {'GPU(s)':>7}  {'Speed':>6}  "
          f"{'B_cpu':>8}  {'B_gpu':>8}  {'Exp':>8}  {'Match':>5}")
    print("-" * 75)

    for Z, N, name, B_exp in test_nuclei:
        # CPU
        t0 = time.time()
        B_cpu, conv_cpu = hfb_binding_energy(Z, N, sly4)
        dt_cpu = time.time() - t0

        # GPU (first call includes solver construction + GPU transfer)
        t0 = time.time()
        B_gpu, conv_gpu = gpu_binding_energy(Z, N, sly4)
        dt_gpu_first = time.time() - t0

        # GPU (second call — solver cached, only solve())
        _gpu_solver_cache.clear()  # Clear to force fresh construction
        from skyrme_hfb import SphericalHFB as SH
        cpu_s = SH(Z, N)
        gpu_s = GPUHFBSolver(cpu_s)

        t0 = time.time()
        r = gpu_s.solve(sly4)
        dt_gpu = time.time() - t0
        B_gpu2 = r["binding_energy_MeV"]

        match = "✅" if abs(B_cpu - B_gpu2) < 1.0 else f"Δ={abs(B_cpu-B_gpu2):.1f}"
        speedup = dt_cpu / dt_gpu if dt_gpu > 0 else 0

        print(f"{name:>8}  {dt_cpu:7.2f}  {dt_gpu:7.2f}  {speedup:5.1f}x  "
              f"{B_cpu:8.1f}  {B_gpu2:8.1f}  {B_exp:8.1f}  {match}")

    # Full L2 objective timing
    print(f"\n{'='*60}")
    print("Full L2 Objective Evaluation (18 nuclei)")
    print(f"{'='*60}")

    from objective import L2_FOCUSED_NUCLEI, load_experimental_data
    from skyrme_hf import nuclear_matter_properties, semf_binding_energy

    exp_data = load_experimental_data()

    # CPU timing
    from skyrme_hfb import binding_energy_l2
    t0 = time.time()
    chi2_cpu = 0.0
    n = 0
    for (Zn, Nn), (B_exp, _) in exp_data.items():
        if (Zn, Nn) not in L2_FOCUSED_NUCLEI:
            continue
        B, conv = binding_energy_l2(Zn, Nn, sly4, method="auto")
        if B > 0:
            sigma = max(0.01 * B_exp, 2.0)
            chi2_cpu += ((B - B_exp) / sigma)**2
            n += 1
    dt_cpu = time.time() - t0

    # GPU timing
    _gpu_solver_cache.clear()
    t0 = time.time()
    chi2_gpu = 0.0
    n = 0
    for (Zn, Nn), (B_exp, _) in exp_data.items():
        if (Zn, Nn) not in L2_FOCUSED_NUCLEI:
            continue
        A = Zn + Nn
        if 56 <= A <= 132:
            B, conv = gpu_binding_energy(Zn, Nn, sly4)
        else:
            B = semf_binding_energy(Zn, Nn, sly4)
            conv = True
        if B > 0:
            sigma = max(0.01 * B_exp, 2.0)
            chi2_gpu += ((B - B_exp) / sigma)**2
            n += 1
    dt_gpu_first = time.time() - t0

    # Second call (all solvers cached)
    t0 = time.time()
    chi2_gpu2 = 0.0
    n = 0
    for (Zn, Nn), (B_exp, _) in exp_data.items():
        if (Zn, Nn) not in L2_FOCUSED_NUCLEI:
            continue
        A = Zn + Nn
        if 56 <= A <= 132:
            B, conv = gpu_binding_energy(Zn, Nn, sly4)
        else:
            B = semf_binding_energy(Zn, Nn, sly4)
            conv = True
        if B > 0:
            sigma = max(0.01 * B_exp, 2.0)
            chi2_gpu2 += ((B - B_exp) / sigma)**2
            n += 1
    dt_gpu_cached = time.time() - t0

    print(f"  CPU:            {dt_cpu:6.1f}s  χ²/n = {chi2_cpu/max(n,1):.2f}")
    print(f"  GPU (cold):     {dt_gpu_first:6.1f}s  χ²/n = {chi2_gpu/max(n,1):.2f}")
    print(f"  GPU (cached):   {dt_gpu_cached:6.1f}s  χ²/n = {chi2_gpu2/max(n,1):.2f}")
    print(f"  Speedup (cached): {dt_cpu/dt_gpu_cached:.1f}x")

    est_30k = 30000 * dt_gpu_cached / 3600
    print(f"\n  Estimated 30,000 evals: {est_30k:.1f} hours")
    print(f"  (Paper used ~500 hours on institutional Fortran)")
    print(f"{'='*60}")
