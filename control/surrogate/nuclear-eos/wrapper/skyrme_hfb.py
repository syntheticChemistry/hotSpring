#!/usr/bin/env python3
"""
Spherical Skyrme HF+BCS Nuclear Structure Solver (Level 2)

Level 1: SEMF + nuclear matter properties (analytic, ~1ms)
Level 2: Self-consistent HF+BCS with separate p/n channels (~seconds)  ← THIS
Level 3: Axially deformed HFB (future — BarraCUDA target)

Upgrades from skyrme_hf.py (Level 1):
  - Separate proton/neutron Hamiltonians and densities
  - BCS pairing (constant gap approximation, Δ = 12/√A)
  - Coulomb exchange (Slater approximation)
  - Proper isospin structure in Skyrme potential (t0, t3 terms)
  - Kinetic energy operator T (not H_HO) in HO basis
  - Center-of-mass correction

Reference: Bender, Heenen, Reinhard, Rev. Mod. Phys. 75, 121 (2003)
Author: ecoPrimals — because we do science, not permissions
License: AGPL-3.0
"""

import numpy as np
from math import factorial, gamma as math_gamma
from scipy.optimize import brentq

from skyrme_hf import (
    HBAR_C, M_NUCLEON, M_PROTON, M_NEUTRON, E2, HBAR2_2M,
    PARAM_NAMES, _to_dict,
    nuclear_matter_properties,
    semf_binding_energy,
)


# =============================================================================
# Spherical HF+BCS Solver
# =============================================================================

class SphericalHFB:
    """Spherical Skyrme HF+BCS solver with separate proton/neutron channels.

    Key physics over Level 1 (SEMF):
    - Shell effects from self-consistent mean field
    - BCS pairing correlations (odd-even staggering)
    - Separate proton/neutron densities and potentials
    - Coulomb direct (numerical) + exchange (Slater)
    - Spin-orbit splitting

    Expected accuracy: ~2-5 MeV per nucleus for spherical/near-spherical.
    """

    def __init__(self, Z, N, n_shells=None, r_max=None, n_grid=None):
        self.Z = Z
        self.N = N
        self.A = Z + N

        # Adaptive basis size
        if n_shells is None:
            n_shells = min(max(int(2 * self.A**(1.0/3.0)) + 4, 8), 14)
        self.n_shells = n_shells

        # Adaptive grid
        if r_max is None:
            r_max = max(1.2 * self.A**(1.0/3.0) + 8.0, 12.0)
        if n_grid is None:
            n_grid = max(int(r_max * 12), 120)

        # Radial grid (avoid r=0)
        self.r = np.linspace(r_max / n_grid, r_max, n_grid)
        self.dr = self.r[1] - self.r[0]
        self.nr = n_grid

        # HO parameters
        self.hw = 41.0 * self.A**(-1.0/3.0)   # MeV
        self.b = HBAR_C / np.sqrt(M_NUCLEON * self.hw)  # fm

        # Pairing gaps (phenomenological constant-gap BCS)
        self.delta_p = 12.0 / np.sqrt(max(self.A, 4))
        self.delta_n = 12.0 / np.sqrt(max(self.A, 4))

        # Build HO basis and wavefunctions
        self._build_basis()
        self._compute_wavefunctions()
        self._compute_kinetic_matrix()

    # -----------------------------------------------------------------
    # Basis construction
    # -----------------------------------------------------------------
    def _build_basis(self):
        """Build (n, l, j, 2j+1) basis states."""
        self.states = []
        for N_sh in range(self.n_shells):
            for l in range(N_sh + 1):
                n = (N_sh - l) // 2
                if (N_sh - l) % 2 != 0:
                    continue
                if l > 0:
                    for j2 in [2*l - 1, 2*l + 1]:
                        self.states.append((n, l, j2 / 2.0, j2 + 1))
                else:
                    self.states.append((n, l, 0.5, 2))
        self.n_states = len(self.states)

        # Group states by (l, j) quantum numbers for block structure
        self._lj_blocks = {}
        for i, (n, l, j, deg) in enumerate(self.states):
            key = (l, j)
            self._lj_blocks.setdefault(key, []).append(i)

    def _compute_wavefunctions(self):
        """Pre-compute HO radial wavefunctions and derivatives on grid."""
        self.wf = np.zeros((self.n_states, self.nr))
        self.dwf = np.zeros((self.n_states, self.nr))
        for i, (n, l, j, deg) in enumerate(self.states):
            self.wf[i] = self._ho_radial(n, l, self.r, self.b)
            self.dwf[i] = np.gradient(self.wf[i], self.r)

    @staticmethod
    def _ho_radial(n, l, r, b):
        """Radial HO wavefunction R_{nl}(r).

        R_{nl}(r) = N_{nl} (r/b)^l exp(-r²/(2b²)) L_n^{l+1/2}(r²/b²)
        Normalized: ∫ |R_{nl}|² r² dr = 1
        """
        xi = (r / b)**2
        alpha_L = l + 0.5

        # Generalized Laguerre polynomial via stable recurrence
        if n == 0:
            L = np.ones_like(r)
        elif n == 1:
            L = 1.0 + alpha_L - xi
        else:
            Lm2 = np.ones_like(r)
            Lm1 = 1.0 + alpha_L - xi
            for k in range(2, n + 1):
                L = ((2*k - 1 + alpha_L - xi) * Lm1
                     - (k - 1 + alpha_L) * Lm2) / k
                Lm2, Lm1 = Lm1, L
            L = Lm1

        # Normalization: ∫₀^∞ |R|² r² dr = 1
        norm_sq = 2.0 * factorial(n) / (b**3 * math_gamma(n + l + 1.5))
        norm = np.sqrt(abs(norm_sq))

        return norm * (r / b)**l * np.exp(-xi / 2.0) * L

    def _compute_kinetic_matrix(self):
        """Compute kinetic energy matrix T in HO basis (analytic).

        T = H_HO - V_HO, where H_HO is diagonal and V_HO = (1/2)mω²r².

        For states with same (l,j), T is tridiagonal in n:
          T_{nn}   = (ℏω/2)(2n + l + 3/2)
          T_{n,n±1} = -(ℏω/2)√((n+δ)(n+l+1/2+δ))  where δ=0,1
        """
        ns = self.n_states
        self.T_matrix = np.zeros((ns, ns))

        for (l, j), indices in self._lj_blocks.items():
            for i_idx, i in enumerate(indices):
                n_i = self.states[i][0]

                # Diagonal: T_nn = (ℏω/2)(2n + l + 3/2)
                self.T_matrix[i, i] = (self.hw / 2.0) * (2*n_i + l + 1.5)

                # Off-diagonal: T_{n,n+1}
                for j_idx in range(i_idx + 1, len(indices)):
                    j_state = indices[j_idx]
                    n_j = self.states[j_state][0]

                    if n_j == n_i + 1:
                        # T_{n,n+1} = -(ℏω/2)√((n+1)(n+l+3/2))
                        val = -(self.hw / 2.0) * np.sqrt(
                            (n_i + 1) * (n_i + l + 1.5))
                        self.T_matrix[i, j_state] = val
                        self.T_matrix[j_state, i] = val
                    elif n_j == n_i - 1:
                        val = -(self.hw / 2.0) * np.sqrt(
                            n_i * (n_i + l + 0.5))
                        self.T_matrix[i, j_state] = val
                        self.T_matrix[j_state, i] = val

    # -----------------------------------------------------------------
    # Kinetic energy density from wavefunctions
    # -----------------------------------------------------------------
    def _compute_tau_q(self, eigvecs, v2):
        """Compute kinetic energy density τ_q(r) from eigenstates.

        τ_q(r) = Σ_i (2j+1)v²_i/(4π) [(dφ_i/dr)² + l(l+1)/r² φ_i²]

        This is the EXACT τ from wavefunctions, not the TF approximation.
        """
        degs = np.array([s[3] for s in self.states])
        tau_q = np.zeros(self.nr)
        r2_inv = 1.0 / np.maximum(self.r**2, 1e-20)

        for i in range(self.n_states):
            if degs[i] * v2[i] < 1e-12:
                continue
            # φ_i(r) = Σ_k c_{ki} R_k(r),  dφ_i/dr = Σ_k c_{ki} dR_k/dr
            c_i = eigvecs[:, i]
            phi = c_i @ self.wf
            dphi = c_i @ self.dwf
            l_i = self.states[i][1]
            tau_q += degs[i] * v2[i] / (4.0 * np.pi) * (
                dphi**2 + l_i * (l_i + 1) * r2_inv * phi**2)

        return tau_q

    # -----------------------------------------------------------------
    # Potentials
    # -----------------------------------------------------------------
    def _skyrme_potential(self, rho_p, rho_n, q, p):
        """Isospin-dependent Skyrme mean-field potential U_q(r).

        Includes t0 (contact), t3 (density-dependent), Coulomb terms.
        The t1,t2 (momentum-dependent) physics is handled through the
        effective mass in the kinetic energy operator, not here.

        Parameters
        ----------
        rho_p, rho_n : ndarray — proton/neutron densities (fm⁻³)
        q : str — 'p' or 'n'
        p : dict — Skyrme parameters

        Returns
        -------
        U : ndarray — single-particle potential (MeV)
        """
        rho = rho_p + rho_n
        rho_q = rho_p if q == 'p' else rho_n
        rho_safe = np.maximum(rho, 1e-20)

        t0, t1, t2, t3 = p['t0'], p['t1'], p['t2'], p['t3']
        x0, x1, x2, x3 = p['x0'], p['x1'], p['x2'], p['x3']
        alpha = p['alpha']

        # --- t0 contact term ---
        U = t0 * ((1.0 + x0/2.0) * rho - (0.5 + x0) * rho_q)

        # --- t3 density-dependent term ---
        rho_alpha = rho_safe**alpha
        rho_alpha_m1 = np.where(rho > 1e-15,
                                rho_safe**(alpha - 1), 0.0)
        sum_rho2 = rho_p**2 + rho_n**2

        U += (t3 / 12.0) * (
            (1.0 + x3/2.0) * (alpha + 2) * rho_alpha * rho
            - (0.5 + x3) * (
                alpha * rho_alpha_m1 * sum_rho2
                + 2.0 * rho_alpha * rho_q
            )
        )

        # --- t1,t2 momentum-dependent terms ---
        # Handled through the position-dependent effective mass in T_eff
        # (computed in the solver). Not included as an explicit potential.

        return U

    def _coulomb_direct(self, rho_p):
        """Coulomb direct potential from proton density (spherical Poisson).

        V_C(r) = 4πe² [1/r ∫₀ʳ ρ_p(r')r'² dr' + ∫ᵣ^∞ ρ_p(r')r' dr']
        """
        r = self.r
        dr = self.dr

        # Inner integral: Q(r) = ∫₀ʳ ρ_p(r')4πr'² dr'
        charge_enclosed = np.cumsum(rho_p * 4.0 * np.pi * r**2) * dr

        # Outer integral: Φ_out(r) = ∫ᵣ^∞ ρ_p(r') 4πr' dr'
        phi_outer = np.cumsum((rho_p * 4.0 * np.pi * r)[::-1])[::-1] * dr

        V_C = E2 * (charge_enclosed / np.maximum(r, 1e-10) + phi_outer)
        return V_C

    def _coulomb_exchange(self, rho_p):
        """Coulomb exchange (Slater approximation).

        V_Cx = -e²(3/π)^{1/3} ρ_p^{1/3}
        """
        return -E2 * (3.0 / np.pi)**(1.0/3.0) * np.maximum(rho_p, 0.0)**(1.0/3.0)

    # -----------------------------------------------------------------
    # BCS pairing
    # -----------------------------------------------------------------
    def _bcs_occupations(self, energies, num_particles, delta):
        """Compute BCS occupation numbers v²_k.

        Given single-particle energies ε_k and pairing gap Δ,
        find chemical potential λ from N = Σ (2j+1)v²_k,
        then v²_k = (1/2)(1 - (ε_k - λ)/E_k)
        where E_k = √((ε_k - λ)² + Δ²).

        Parameters
        ----------
        energies : ndarray — single-particle energies
        num_particles : int — target particle number
        delta : float — pairing gap (MeV)

        Returns
        -------
        v2 : ndarray — occupation probabilities
        lam : float — chemical potential
        """
        if num_particles <= 0:
            return np.zeros(self.n_states), 0.0

        degs = np.array([self.states[i][3] for i in range(self.n_states)])

        # For vanishing pairing, use sharp Fermi filling
        if delta < 0.01:
            return self._sharp_filling(energies, num_particles, degs)

        def particle_number(lam):
            ek = energies - lam
            Ek = np.sqrt(ek**2 + delta**2)
            v2 = 0.5 * (1.0 - ek / Ek)
            return np.sum(degs * v2) - num_particles

        # Bracket λ
        e_min, e_max = np.min(energies) - 50, np.max(energies) + 50

        try:
            lam = brentq(particle_number, e_min, e_max, xtol=1e-6)
        except ValueError:
            # Fallback: approximate Fermi energy from sharp filling
            lam = self._approx_fermi(energies, num_particles, degs)

        ek = energies - lam
        Ek = np.sqrt(ek**2 + delta**2)
        v2 = 0.5 * (1.0 - ek / Ek)

        return v2, lam

    def _sharp_filling(self, energies, num_particles, degs):
        """Sharp Fermi filling (Δ → 0 limit)."""
        idx = np.argsort(energies)
        v2 = np.zeros(self.n_states)
        remaining = num_particles
        for i in idx:
            fill = min(remaining, degs[i])
            v2[i] = fill / degs[i]
            remaining -= fill
            if remaining <= 0:
                break
        lam = energies[idx[min(np.searchsorted(
            np.cumsum(degs[idx]), num_particles), len(idx)-1)]]
        return v2, lam

    def _approx_fermi(self, energies, num_particles, degs):
        """Estimate Fermi energy from sharp filling."""
        idx = np.argsort(energies)
        count = 0
        for i in idx:
            count += degs[i]
            if count >= num_particles:
                return energies[i]
        return energies[idx[-1]]

    # -----------------------------------------------------------------
    # Main solver
    # -----------------------------------------------------------------
    def solve(self, skyrme_params, max_iter=150, tol=0.05, mixing=0.3):
        """Self-consistent HF+BCS iteration.

        Returns
        -------
        result : dict
            binding_energy_MeV, charge_radius_fm, converged, iterations, etc.
        """
        p = _to_dict(skyrme_params)
        W0 = p.get("W0", 0.0)
        r = self.r

        # --- Initial densities: uniform sphere ---
        R_nuc = 1.2 * self.A**(1.0/3.0)
        rho0 = 3.0 * self.A / (4.0 * np.pi * R_nuc**3)

        rho_p = np.where(r < R_nuc, rho0 * self.Z / self.A, 1e-15)
        rho_n = np.where(r < R_nuc, rho0 * self.N / self.A, 1e-15)
        rho_p = np.maximum(rho_p, 1e-15)
        rho_n = np.maximum(rho_n, 1e-15)

        E_prev = 1e10
        converged = False
        results_q = {}

        for iteration in range(max_iter):
            rho_p_new = np.zeros(self.nr)
            rho_n_new = np.zeros(self.nr)

            for q, num_q, delta_q in [('p', self.Z, self.delta_p),
                                       ('n', self.N, self.delta_n)]:
                # --- Build single-particle potential ---
                U_sky = self._skyrme_potential(rho_p, rho_n, q, p)

                if q == 'p':
                    V_C = self._coulomb_direct(rho_p)
                    V_Cx = self._coulomb_exchange(rho_p)
                    U_total = U_sky + V_C + V_Cx
                else:
                    U_total = U_sky

                # --- Build effective kinetic energy matrix T_eff ---
                # T_eff[i,j] = ∫ f_q(r) [dR_i/dr·dR_j/dr·r² + l(l+1)R_i·R_j] dr
                # where f_q(r) = ℏ²/(2m*_q) captures the t1,t2 effective mass.
                rho_q = rho_p if q == 'p' else rho_n
                t1v, t2v = p['t1'], p['t2']
                x1v, x2v = p['x1'], p['x2']
                C0t = 0.25 * (t1v * (1 + x1v/2) + t2v * (1 + x2v/2))
                C1n = 0.25 * (t1v * (0.5 + x1v) - t2v * (0.5 + x2v))
                f_q = HBAR2_2M + C0t * (rho_p + rho_n) - C1n * rho_q
                f_q = np.maximum(f_q, HBAR2_2M * 0.3)

                T_eff = np.zeros((self.n_states, self.n_states))
                for (lval, jval), indices in self._lj_blocks.items():
                    ll1 = lval * (lval + 1)
                    for ii, idx_i in enumerate(indices):
                        for jj in range(ii, len(indices)):
                            idx_j = indices[jj]
                            integrand = f_q * (
                                self.dwf[idx_i] * self.dwf[idx_j] * r**2
                                + ll1 * self.wf[idx_i] * self.wf[idx_j]
                            )
                            val = np.trapezoid(integrand, r)
                            T_eff[idx_i, idx_j] = val
                            T_eff[idx_j, idx_i] = val

                H = T_eff.copy()

                # Add potential matrix elements
                for i in range(self.n_states):
                    n_i, l_i, j_i, deg_i = self.states[i]

                    # Diagonal: <i|U|i>
                    integ = self.wf[i]**2 * U_total * r**2
                    H[i, i] += np.trapezoid(integ, r)

                    # Spin-orbit (diagonal in j)
                    if W0 != 0 and l_i > 0:
                        ls = (j_i*(j_i + 1) - l_i*(l_i + 1) - 0.75) / 2.0
                        rho_total = rho_p + rho_n
                        drho = np.gradient(rho_total, r)
                        drho_r = drho / np.maximum(r, 0.1)
                        so_int = np.trapezoid(
                            self.wf[i]**2 * drho_r * r**2, r)
                        H[i, i] += W0 * ls * so_int

                # Off-diagonal: same (l,j) block
                for (l, j_val), indices in self._lj_blocks.items():
                    for ii, idx_i in enumerate(indices):
                        for jj in range(ii + 1, len(indices)):
                            idx_j = indices[jj]
                            integ = (self.wf[idx_i] * self.wf[idx_j]
                                     * U_total * r**2)
                            val = np.trapezoid(integ, r)
                            H[idx_i, idx_j] += val
                            H[idx_j, idx_i] += val

                # --- Diagonalize ---
                eigenvalues, eigvecs = np.linalg.eigh(H)

                # --- BCS occupation ---
                v2, lam_q = self._bcs_occupations(
                    eigenvalues, num_q, delta_q)

                # --- New density from BCS-weighted eigenstates ---
                degs = np.array([s[3] for s in self.states])
                rho_q_new = np.zeros(self.nr)

                for i in range(self.n_states):
                    if degs[i] * v2[i] < 1e-12:
                        continue
                    # φ_i(r) = Σ_k c_{ki} R_k(r)
                    phi = eigvecs[:, i] @ self.wf   # shape: (nr,)
                    rho_q_new += degs[i] * v2[i] * phi**2 / (4.0 * np.pi)

                rho_q_new = np.maximum(rho_q_new, 1e-15)

                if q == 'p':
                    rho_p_new = rho_q_new
                else:
                    rho_n_new = rho_q_new

                # Store for energy calculation
                results_q[q] = {
                    'eigenvalues': eigenvalues,
                    'eigvecs': eigvecs,
                    'v2': v2,
                    'lambda': lam_q,
                }

            # --- Mix densities ---
            rho_p = mixing * rho_p_new + (1.0 - mixing) * rho_p
            rho_n = mixing * rho_n_new + (1.0 - mixing) * rho_n

            # --- Total energy from single-particle sum ---
            E_total = self._compute_energy(
                rho_p, rho_n, results_q, p)

            dE = abs(E_total - E_prev)
            if dE < tol and iteration > 5:
                converged = True
                break
            E_prev = E_total

        # --- Binding energy (positive for bound nuclei) ---
        B = -E_total if E_total < 0 else 0.0

        # --- Charge radius ---
        norm_p = np.trapezoid(rho_p * 4.0 * np.pi * r**2, r)
        if norm_p > 0.1:
            r_ch2 = np.trapezoid(
                rho_p * r**2 * 4.0 * np.pi * r**2, r) / norm_p
            r_ch = np.sqrt(abs(r_ch2) + 0.64)  # + proton charge radius²
        else:
            r_ch = 1.2 * self.A**(1.0/3.0)

        return {
            "Z": self.Z, "N": self.N, "A": self.A,
            "binding_energy_MeV": B,
            "charge_radius_fm": r_ch,
            "converged": converged,
            "iterations": iteration + 1,
            "delta_E_MeV": dE,
            "lambda_p_MeV": results_q.get('p', {}).get('lambda', 0),
            "lambda_n_MeV": results_q.get('n', {}).get('lambda', 0),
        }

    def _compute_energy(self, rho_p, rho_n, results_q, p):
        """Compute total HF+BCS energy.

        Uses: E = Σ_q Σ_i (2j+1)v²_i · <ψ_i|T|ψ_i>  (kinetic)
              + E_Skyrme[ρ_p, ρ_n]                      (functional)
              + E_Coulomb[ρ_p]                           (Coulomb)
              + E_pair                                    (pairing)
              + E_cm                                      (c.o.m. correction)
        """
        r = self.r
        rho = rho_p + rho_n
        degs = np.array([s[3] for s in self.states])

        t0, t1, t2, t3 = p['t0'], p['t1'], p['t2'], p['t3']
        x0, x1, x2, x3 = p['x0'], p['x1'], p['x2'], p['x3']
        alpha = p['alpha']

        # --- Effective kinetic energy from T_eff eigenvectors ---
        # E_kin_eff = Σ_{q,i} (2j+1) v²_i <ψ_i|T_eff_q|ψ_i>
        # This includes bare kinetic + t1,t2 contribution via effective mass.
        C0t = 0.25 * (t1 * (1 + x1/2.0) + t2 * (1 + x2/2.0))
        C1n = 0.25 * (t1 * (0.5 + x1) - t2 * (0.5 + x2))

        E_kin = 0.0
        for q_label in ('p', 'n'):
            rq = results_q[q_label]
            v2_q = rq['v2']
            evecs = rq['eigvecs']
            rho_q_loc = rho_p if q_label == 'p' else rho_n
            f_q = np.maximum(
                HBAR2_2M + C0t * rho - C1n * rho_q_loc,
                HBAR2_2M * 0.3)

            # Rebuild T_eff for this species (from converged density)
            T_eff_q = np.zeros((self.n_states, self.n_states))
            for (lv, jv), indices in self._lj_blocks.items():
                ll1 = lv * (lv + 1)
                for ii, idx_i in enumerate(indices):
                    for jj in range(ii, len(indices)):
                        idx_j = indices[jj]
                        intg = f_q * (
                            self.dwf[idx_i] * self.dwf[idx_j] * r**2
                            + ll1 * self.wf[idx_i] * self.wf[idx_j])
                        val = np.trapezoid(intg, r)
                        T_eff_q[idx_i, idx_j] = val
                        T_eff_q[idx_j, idx_i] = val

            for i in range(self.n_states):
                if degs[i] * v2_q[i] < 1e-12:
                    continue
                c_i = evecs[:, i]
                E_kin += degs[i] * v2_q[i] * (c_i @ T_eff_q @ c_i)

        # --- Skyrme functional energy ---
        rho_safe = np.maximum(rho, 1e-20)
        sum_rho2 = rho_p**2 + rho_n**2

        # E_t0 = (t0/2) ∫ [(1+x0/2)ρ² - (1/2+x0)(ρ_p²+ρ_n²)] 4πr² dr
        integrand_t0 = ((1 + x0/2.0) * rho**2
                        - (0.5 + x0) * sum_rho2)
        E_t0 = (t0 / 2.0) * np.trapezoid(
            integrand_t0 * 4.0 * np.pi * r**2, r)

        # E_t3 = (t3/12) ∫ ρ^α [(1+x3/2)ρ² - (1/2+x3)(ρ_p²+ρ_n²)] 4πr² dr
        integrand_t3 = rho_safe**alpha * (
            (1 + x3/2.0) * rho**2
            - (0.5 + x3) * sum_rho2)
        E_t3 = (t3 / 12.0) * np.trapezoid(
            integrand_t3 * 4.0 * np.pi * r**2, r)

        # NOTE: E_t1t2 is INCLUDED in E_kin above (effective mass formulation)
        E_t1t2 = 0.0

        # --- Coulomb energy ---
        V_C = self._coulomb_direct(rho_p)
        E_Coul_direct = 0.5 * np.trapezoid(
            V_C * rho_p * 4.0 * np.pi * r**2, r)
        V_Cx = self._coulomb_exchange(rho_p)
        E_Coul_exchange = np.trapezoid(
            V_Cx * rho_p * 4.0 * np.pi * r**2, r)
        E_Coul = E_Coul_direct + E_Coul_exchange

        # --- BCS pairing energy ---
        E_pair = 0.0
        for q, delta_q in [('p', self.delta_p), ('n', self.delta_n)]:
            v2 = results_q[q]['v2']
            u2 = 1.0 - v2
            # E_pair = -Δ Σ (2j+1) √(v²·u²)
            E_pair -= delta_q * np.sum(
                degs * np.sqrt(np.maximum(v2 * u2, 0.0)))

        # --- Center-of-mass correction ---
        E_cm = -0.75 * self.hw

        return E_kin + E_t0 + E_t3 + E_t1t2 + E_Coul + E_pair + E_cm


# =============================================================================
# Convenience function
# =============================================================================

# Cache HFB solvers to avoid rebuilding basis for repeated calls
_hfb_cache = {}


def hfb_binding_energy(Z, N, skyrme_params, max_cache=50):
    """Compute binding energy using HF+BCS (Level 2).

    Caches the HFB solver object for repeated calls with same (Z,N).
    """
    key = (Z, N)
    if key not in _hfb_cache:
        if len(_hfb_cache) > max_cache:
            _hfb_cache.clear()
        _hfb_cache[key] = SphericalHFB(Z, N)

    solver = _hfb_cache[key]
    result = solver.solve(skyrme_params)
    return result["binding_energy_MeV"], result["converged"]


def binding_energy_l2(Z, N, skyrme_params, method="auto"):
    """Compute binding energy — Level 2 hybrid.

    method:
      "hfb"  — spherical HF+BCS for all nuclei
      "semf" — SEMF (Level 1 fallback)
      "auto" — HFB for 56 ≤ A ≤ 132 (spherical heavy), SEMF otherwise
               HFB captures shell effects for doubly-magic heavy nuclei.
               SEMF handles light nuclei (where HO basis is inadequate)
               and very heavy deformed nuclei (where spherical HFB fails).
    """
    A = Z + N
    if method == "hfb":
        return hfb_binding_energy(Z, N, skyrme_params)
    elif method == "semf":
        return semf_binding_energy(Z, N, skyrme_params), True
    elif method == "auto":
        # HFB works best for spherical heavy nuclei (A=56-132)
        # SEMF is better for light (A<56) and deformed (A>132) nuclei
        if 56 <= A <= 132:
            return hfb_binding_energy(Z, N, skyrme_params)
        else:
            return semf_binding_energy(Z, N, skyrme_params), True
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Level 2: Spherical Skyrme HF+BCS Solver")
    print("  Separate p/n channels • BCS pairing • Coulomb exchange")
    print("  No HFBTHO. No Fortran. No permissions. Just physics.")
    print("=" * 70)

    # SLy4 parametrization (well-tested standard)
    sly4 = [-2488.91, 486.82, -546.39, 13777.0,
            0.834, -0.344, -1.0, 1.354, 0.1667, 123.0]

    test_nuclei = [
        # Doubly magic (spherical) — HFB should be accurate here
        (2,   2,  "⁴He",     28.296),
        (8,   8,  "¹⁶O",    127.619),
        (20, 20,  "⁴⁰Ca",   342.052),
        (20, 28,  "⁴⁸Ca",   415.991),
        (28, 28,  "⁵⁶Ni",   483.988),
        (50, 50,  "¹⁰⁰Sn",  824.793),
        (50, 82,  "¹³²Sn", 1102.851),
        (82, 126, "²⁰⁸Pb", 1636.430),
    ]

    print("\nHybrid Level 2 (SEMF for A<56, HFB for 56≤A≤132, SEMF for A>132):")
    print("  " + "-" * 72)

    errors_hybrid = []
    errors_semf = []
    for Z, N, name, B_exp in test_nuclei:
        import time
        A = Z + N
        t0_time = time.time()
        B_l2, conv_l2 = binding_energy_l2(Z, N, sly4, method="auto")
        dt = time.time() - t0_time

        B_semf = semf_binding_energy(Z, N, sly4)
        method = "HFB" if 56 <= A <= 132 else "SEMF"

        err_l2 = 100 * (B_l2 - B_exp) / B_exp if B_exp > 0 else 0
        err_semf = 100 * (B_semf - B_exp) / B_exp if B_exp > 0 else 0
        errors_hybrid.append(abs(err_l2))
        errors_semf.append(abs(err_semf))

        mark = "✅" if abs(err_l2) < 10 else "⚠️"
        conv_str = "conv" if conv_l2 else "NOT"
        print(f"  {mark} {name:7s}  L2={B_l2:8.1f}  SEMF={B_semf:8.1f}  "
              f"exp={B_exp:8.1f}  err={err_l2:+5.1f}%  [{method}, {dt:.1f}s]")

    print(f"\n  Level 2 (hybrid)  mean |err|: {np.mean(errors_hybrid):.1f}%")
    print(f"  Level 1 (SEMF)    mean |err|: {np.mean(errors_semf):.1f}%")

    # Nuclear matter properties (same as Level 1 — consistency check)
    print("\nNuclear Matter Properties (SLy4 — should match Level 1):")
    nmp = nuclear_matter_properties(sly4)
    print(f"  ρ₀ = {nmp['rho0_fm3']:.4f} fm⁻³  (exp: 0.16)")
    print(f"  E/A = {nmp['E_A_MeV']:.2f} MeV   (exp: -16)")
    print(f"  K∞  = {nmp['K_inf_MeV']:.0f} MeV    (exp: 230)")

    print("\n" + "=" * 70)
    print("  Level 2 solver ready for wiring into objective function")
    print("=" * 70)

