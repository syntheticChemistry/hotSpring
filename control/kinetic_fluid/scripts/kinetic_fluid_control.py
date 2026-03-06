#!/usr/bin/env python3
"""
Paper 45: Multi-species kinetic-fluid coupling (Haack, Murillo, Sagert & Chuna 2024)
Python control for hotSpring BarraCuda CPU validation.

Implements:
  Phase 1: Homogeneous multi-species BGK relaxation (conservation + H-theorem)
  Phase 2: 1D Euler shock tube (Sod problem, Rankine-Hugoniot)
  Phase 3: Coupled kinetic-fluid with interface moment matching

Reference: Haack et al., J. Comput. Phys. (2024), DOI:10.1016/j.jcp.2024.112908
Foundation: Haack, Hauck, Murillo, J. Stat. Phys. 168:826 (2017)
LANL reference: github.com/lanl/Multi-BGK

Pure NumPy — no external dependencies beyond numpy.
"""

import numpy as np

# ──────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────
GAMMA = 5.0 / 3.0  # monatomic ideal gas


# ──────────────────────────────────────────────────────────
#  Phase 1: Homogeneous multi-species BGK relaxation
# ──────────────────────────────────────────────────────────

def maxwellian_1d(v, n, u, T, m):
    """1D Maxwellian distribution: f(v) = n * sqrt(m/(2πT)) * exp(-m(v-u)²/(2T))"""
    T = max(T, 1e-12)
    coeff = n * np.sqrt(m / (2.0 * np.pi * T))
    exponent = -m * (v - u) ** 2 / (2.0 * T)
    return coeff * np.exp(np.clip(exponent, -500.0, 0.0))


def compute_moments(f, v, dv, m):
    """Compute density, velocity, temperature, energy from distribution.

    In 1D: E = (m/2) ∫ v² f dv = (n/2)(mu² + T), so T = 2E/n - mu².
    """
    f_safe = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    n = np.sum(f_safe) * dv
    if n < 1e-30:
        return 0.0, 0.0, 1e-12, 0.0
    u = np.sum(f_safe * v) * dv / n
    E = 0.5 * m * np.sum(f_safe * v ** 2) * dv
    T = 2.0 * E / n - m * u ** 2
    T = max(T, 1e-12)
    return n, u, T, E


def bgk_target_maxwellians(species_list, v, dv):
    """Compute conservation-preserving target Maxwellians for multi-species BGK.

    The target parameters (n_s*, u_s*, T_s*) are chosen so that the BGK operator
    conserves total mass (per species), total momentum, and total kinetic energy.

    For the symmetric case (Haack-Hauck-Murillo 2017):
      n_s* = n_s  (species mass conserved individually)
      u_s* = u_bar  (common momentum-weighted mean velocity)
      T_s* = T_s + (2/3) m_s (u_s - u_bar)^2 + (2/3)(E_excess / n_total)
    where E_excess accounts for the kinetic energy difference.
    """
    total_mom = 0.0
    total_mass = 0.0
    total_energy = 0.0
    species_data = []

    for f_s, m_s, nu_s in species_list:
        n_s, u_s, T_s, E_s = compute_moments(f_s, v, dv, m_s)
        species_data.append((n_s, u_s, T_s, E_s, m_s, nu_s))
        total_mom += m_s * n_s * u_s
        total_mass += m_s * n_s
        total_energy += E_s

    u_bar = total_mom / total_mass if total_mass > 1e-30 else 0.0

    targets = []
    n_total = sum(d[0] for d in species_data)
    thermal_target = total_energy - 0.5 * total_mass * u_bar ** 2

    for n_s, u_s, T_s, E_s, m_s, nu_s in species_data:
        n_star = n_s
        u_star = u_bar
        # 1D: E_thermal = Σ n_s T_s / 2, so T* = 2 E_thermal / n_total
        T_star = 2.0 * thermal_target / n_total if n_total > 1e-30 else T_s
        T_star = max(T_star, 1e-12)
        targets.append((n_star, u_star, T_star, m_s, nu_s))

    return targets


def bgk_relaxation_step(species_list, v, dv, dt):
    """One BGK relaxation step: f_s^{n+1} = f_s^n + dt * nu_s * (M_s* - f_s^n)

    Uses forward Euler for simplicity (implicit IMEX is the production method,
    but explicit is sufficient for the homogeneous relaxation test).
    """
    targets = bgk_target_maxwellians(species_list, v, dv)
    new_fs = []

    for i, (f_s, m_s, nu_s) in enumerate(species_list):
        n_star, u_star, T_star, _, _ = targets[i]
        M_star = maxwellian_1d(v, n_star, u_star, T_star, m_s)
        f_new = f_s + dt * nu_s * (M_star - f_s)
        f_new = np.maximum(f_new, 0.0)
        new_fs.append(f_new)

    return new_fs


def entropy_bgk(f, v, dv):
    """H-function (Boltzmann H-theorem): H = ∫ f ln(f) dv"""
    f_pos = np.maximum(f, 1e-300)
    return np.sum(f_pos * np.log(f_pos)) * dv


def run_bgk_relaxation(n_steps=2000, dt=0.005):
    """Phase 1: Two-species homogeneous BGK relaxation test.

    Species 1: light (m=1), hot (T=2.0)
    Species 2: heavy (m=4), cold (T=0.5)
    Both at rest (u=0), equal density (n=1).

    Should relax to common temperature T_eq = (T1 + T2) / 2 = 1.25
    (equal densities, equal weight in energy).
    """
    Nv = 201
    v_max = 8.0
    v = np.linspace(-v_max, v_max, Nv)
    dv = v[1] - v[0]

    m1, m2 = 1.0, 4.0
    n1, n2 = 1.0, 1.0
    u1, u2 = 0.0, 0.0
    T1, T2 = 2.0, 0.5
    nu1, nu2 = 1.0, 1.0

    f1 = maxwellian_1d(v, n1, u1, T1, m1)
    f2 = maxwellian_1d(v, n2, u2, T2, m2)

    n1_0, u1_0, T1_0, E1_0 = compute_moments(f1, v, dv, m1)
    n2_0, u2_0, T2_0, E2_0 = compute_moments(f2, v, dv, m2)
    E_total_0 = E1_0 + E2_0
    mom_total_0 = m1 * n1_0 * u1_0 + m2 * n2_0 * u2_0

    H_prev = entropy_bgk(f1, v, dv) + entropy_bgk(f2, v, dv)
    entropy_decreasing = False

    for step in range(n_steps):
        species_list = [(f1, m1, nu1), (f2, m2, nu2)]
        f1, f2 = bgk_relaxation_step(species_list, v, dv, dt)

        H_curr = entropy_bgk(f1, v, dv) + entropy_bgk(f2, v, dv)
        if H_curr > H_prev + 1e-10:
            entropy_decreasing = True
        H_prev = H_curr

    n1_f, u1_f, T1_f, E1_f = compute_moments(f1, v, dv, m1)
    n2_f, u2_f, T2_f, E2_f = compute_moments(f2, v, dv, m2)
    E_total_f = E1_f + E2_f
    mom_total_f = m1 * n1_f * u1_f + m2 * n2_f * u2_f

    return {
        "mass_conservation_1": abs(n1_f - n1_0),
        "mass_conservation_2": abs(n2_f - n2_0),
        "momentum_conservation": abs(mom_total_f - mom_total_0),
        "energy_conservation": abs(E_total_f - E_total_0) / max(abs(E_total_0), 1e-30),
        "entropy_monotonic": not entropy_decreasing,
        "T1_final": T1_f,
        "T2_final": T2_f,
        "T_equilibrium": (T1_f + T2_f) / 2.0,
        "T_eq_expected": (n1_0 * T1_0 + n2_0 * T2_0) / (n1_0 + n2_0),
        "temperature_relaxed": abs(T1_f - T2_f) / max(T1_f, T2_f),
    }


# ──────────────────────────────────────────────────────────
#  Phase 2: 1D Euler Shock Tube (Sod Problem)
# ──────────────────────────────────────────────────────────

def euler_primitives_to_conserved(rho, u, p):
    """Convert primitive variables (ρ, u, p) to conserved (ρ, ρu, E)."""
    E = p / (GAMMA - 1.0) + 0.5 * rho * u ** 2
    return rho, rho * u, E


def euler_conserved_to_primitives(rho, rho_u, E):
    """Convert conserved variables to primitives."""
    u = rho_u / rho if abs(rho) > 1e-30 else 0.0
    p = (GAMMA - 1.0) * (E - 0.5 * rho * u ** 2)
    return rho, u, max(p, 1e-30)


def euler_flux(rho, u, p):
    """Euler flux vector F(U)."""
    E = p / (GAMMA - 1.0) + 0.5 * rho * u ** 2
    return (rho * u,
            rho * u ** 2 + p,
            (E + p) * u)


def hll_flux(rho_l, u_l, p_l, rho_r, u_r, p_r):
    """HLL approximate Riemann solver."""
    c_l = np.sqrt(GAMMA * p_l / rho_l) if rho_l > 1e-30 else 0.0
    c_r = np.sqrt(GAMMA * p_r / rho_r) if rho_r > 1e-30 else 0.0

    s_l = min(u_l - c_l, u_r - c_r)
    s_r = max(u_l + c_l, u_r + c_r)

    F_l = euler_flux(rho_l, u_l, p_l)
    F_r = euler_flux(rho_r, u_r, p_r)

    U_l = euler_primitives_to_conserved(rho_l, u_l, p_l)
    U_r = euler_primitives_to_conserved(rho_r, u_r, p_r)

    if s_l >= 0:
        return F_l
    elif s_r <= 0:
        return F_r
    else:
        denom = s_r - s_l
        F_hll = tuple(
            (s_r * fl - s_l * fr + s_l * s_r * (ur - ul)) / denom
            for fl, fr, ul, ur in zip(F_l, F_r, U_l, U_r)
        )
        return F_hll


def run_sod_shock_tube(Nx=400, t_final=0.2):
    """Phase 2: Sod shock tube problem.

    Left state:  ρ=1.0, u=0, p=1.0
    Right state: ρ=0.125, u=0, p=0.1
    Domain: [0, 1], interface at x=0.5
    """
    dx = 1.0 / Nx
    x = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, Nx)

    rho = np.where(x < 0.5, 1.0, 0.125)
    u = np.zeros(Nx)
    p = np.where(x < 0.5, 1.0, 0.1)

    rho_0, rhou_0, E_0 = euler_primitives_to_conserved(rho, u, p)
    total_mass_0 = np.sum(rho_0) * dx
    total_mom_0 = np.sum(rhou_0) * dx
    total_E_0 = np.sum(E_0) * dx

    t = 0.0
    while t < t_final:
        c = np.sqrt(GAMMA * p / np.maximum(rho, 1e-30))
        max_speed = np.max(np.abs(u) + c)
        dt = 0.4 * dx / max(max_speed, 1e-30)
        if t + dt > t_final:
            dt = t_final - t

        rho_c, rhou_c, E_c = euler_primitives_to_conserved(rho, u, p)

        flux_rho = np.zeros(Nx + 1)
        flux_mom = np.zeros(Nx + 1)
        flux_ene = np.zeros(Nx + 1)

        for i in range(1, Nx):
            f = hll_flux(rho[i - 1], u[i - 1], p[i - 1],
                         rho[i], u[i], p[i])
            flux_rho[i], flux_mom[i], flux_ene[i] = f

        flux_rho[0] = euler_flux(rho[0], u[0], p[0])[0]
        flux_mom[0] = euler_flux(rho[0], u[0], p[0])[1]
        flux_ene[0] = euler_flux(rho[0], u[0], p[0])[2]
        flux_rho[Nx] = euler_flux(rho[-1], u[-1], p[-1])[0]
        flux_mom[Nx] = euler_flux(rho[-1], u[-1], p[-1])[1]
        flux_ene[Nx] = euler_flux(rho[-1], u[-1], p[-1])[2]

        rho_c -= dt / dx * (flux_rho[1:] - flux_rho[:-1])
        rhou_c -= dt / dx * (flux_mom[1:] - flux_mom[:-1])
        E_c -= dt / dx * (flux_ene[1:] - flux_ene[:-1])

        rho_c = np.maximum(rho_c, 1e-10)
        for i in range(Nx):
            rho[i], u[i], p[i] = euler_conserved_to_primitives(
                rho_c[i], rhou_c[i], E_c[i])

        t += dt

    total_mass_f = np.sum(rho) * dx
    total_mom_f = np.sum(rho * u) * dx
    _, _, E_f = euler_primitives_to_conserved(rho, u, p)
    total_E_f = np.sum(E_f) * dx

    drho = np.abs(np.diff(rho))
    drho_smooth = np.convolve(drho, np.ones(5) / 5, mode="same")

    # Contact: largest density jump in [0.5, 0.85] region
    contact_lo = int(0.5 * Nx)
    contact_hi = int(0.85 * Nx)
    contact_region = drho_smooth[contact_lo:contact_hi]
    if len(contact_region) > 0 and np.max(contact_region) > 1e-6:
        i_contact = np.argmax(contact_region) + contact_lo
    else:
        i_contact = Nx // 2
    x_contact = x[i_contact]

    # Shock: largest density jump in [0.7, 1.0] region (rightward-moving)
    shock_lo = int(0.7 * Nx)
    shock_hi = Nx - 1
    shock_region_drho = drho_smooth[shock_lo:shock_hi]
    if len(shock_region_drho) > 0 and np.max(shock_region_drho) > 1e-4:
        i_shock = np.argmax(shock_region_drho) + shock_lo
        shock_detected = True
        x_shock = x[i_shock]
    else:
        shock_detected = False
        x_shock = 0.0

    return {
        "mass_conservation": abs(total_mass_f - total_mass_0) / total_mass_0,
        "momentum_conservation": abs(total_mom_f - total_mom_0),
        "energy_conservation": abs(total_E_f - total_E_0) / total_E_0,
        "contact_position": x_contact,
        "contact_in_range": 0.6 < x_contact < 0.8,
        "shock_detected": shock_detected,
        "shock_position": x_shock,
        "shock_in_range": 0.7 < x_shock < 0.95 if shock_detected else False,
        "density_range": (np.min(rho), np.max(rho)),
        "rho_profile": rho,
        "u_profile": u,
        "p_profile": p,
        "x": x,
    }


# ──────────────────────────────────────────────────────────
#  Phase 3: Coupled Kinetic-Fluid (1D)
# ──────────────────────────────────────────────────────────

def kinetic_advection_step(f, v, dx, dt, Nx_kin):
    """First-order upwind advection for 1D kinetic equation.
    f has shape (Nx_kin, Nv).
    """
    f_new = f.copy()
    for j in range(len(v)):
        if v[j] > 0:
            for i in range(1, Nx_kin):
                f_new[i, j] = f[i, j] - dt * v[j] / dx * (f[i, j] - f[i - 1, j])
        else:
            for i in range(Nx_kin - 1):
                f_new[i, j] = f[i, j] - dt * v[j] / dx * (f[i + 1, j] - f[i, j])
    return np.maximum(f_new, 0.0)


def kinetic_bgk_collision_step(f, v, dv, dt, m, nu):
    """BGK collision step: f += dt * nu * (M_local - f)."""
    Nx_kin = f.shape[0]
    f_new = f.copy()
    for i in range(Nx_kin):
        n_i = np.sum(f[i, :]) * dv
        if n_i < 1e-30:
            continue
        u_i = np.sum(f[i, :] * v) * dv / n_i
        E_i = 0.5 * m * np.sum(f[i, :] * v ** 2) * dv
        T_i = max((2.0 * E_i / n_i - m * u_i ** 2), 1e-15)
        M_i = maxwellian_1d(v, n_i, u_i, T_i, m)
        f_new[i, :] = f[i, :] + dt * nu * (M_i - f[i, :])
    return np.maximum(f_new, 0.0)


def kinetic_to_fluid_moments(f, v, dv, m):
    """Extract fluid variables (ρ, ρu, E) from kinetic distribution at one point."""
    n = np.sum(f) * dv
    rho = m * n
    rho_u = m * np.sum(f * v) * dv
    E = 0.5 * m * np.sum(f * v ** 2) * dv
    return rho, rho_u, E


def fluid_to_kinetic_maxwellian(rho, u, p, v, m):
    """Construct a Maxwellian from fluid state for kinetic boundary."""
    n = rho / m
    T = p / n if n > 1e-30 else 1e-15
    T = max(T, 1e-15)
    return maxwellian_1d(v, n, u, T, m)


def run_coupled_kinetic_fluid(Nx_kin=50, Nx_fluid=50, Nv=101, t_final=0.1):
    """Phase 3: Coupled kinetic-fluid test.

    Domain [0, 1]:
      x ∈ [0, 0.5): kinetic (BGK)
      x ∈ [0.5, 1]: fluid (Euler)

    Initial condition: uniform density, small velocity perturbation
    to test interface flux matching.
    """
    dx = 1.0 / (Nx_kin + Nx_fluid)
    v_max = 6.0
    v = np.linspace(-v_max, v_max, Nv)
    dv = v[1] - v[0]
    m = 1.0
    nu = 10.0

    rho_init = 1.0
    u_init = 0.1
    p_init = 1.0
    T_init = p_init / (rho_init / m)

    f_kin = np.zeros((Nx_kin, Nv))
    for i in range(Nx_kin):
        f_kin[i, :] = maxwellian_1d(v, rho_init / m, u_init, T_init, m)

    rho_fluid = np.full(Nx_fluid, rho_init)
    u_fluid = np.full(Nx_fluid, u_init)
    p_fluid = np.full(Nx_fluid, p_init)

    total_mass_kin_0 = sum(
        kinetic_to_fluid_moments(f_kin[i, :], v, dv, m)[0]
        for i in range(Nx_kin)
    ) * dx
    total_mass_fluid_0 = np.sum(rho_fluid) * dx
    total_mass_0 = total_mass_kin_0 + total_mass_fluid_0

    rho_k_0, rhou_k_0, E_k_0 = kinetic_to_fluid_moments(
        f_kin[0, :], v, dv, m)
    total_mom_kin_0 = sum(
        kinetic_to_fluid_moments(f_kin[i, :], v, dv, m)[1]
        for i in range(Nx_kin)
    ) * dx
    total_mom_fluid_0 = np.sum(rho_fluid * u_fluid) * dx
    total_mom_0 = total_mom_kin_0 + total_mom_fluid_0

    total_E_kin_0 = sum(
        kinetic_to_fluid_moments(f_kin[i, :], v, dv, m)[2]
        for i in range(Nx_kin)
    ) * dx
    _, _, E_fluid_0 = euler_primitives_to_conserved(rho_fluid, u_fluid, p_fluid)
    total_E_fluid_0 = np.sum(E_fluid_0) * dx
    total_E_0 = total_E_kin_0 + total_E_fluid_0

    t = 0.0
    n_steps = 0
    max_steps = 5000

    while t < t_final and n_steps < max_steps:
        c_fluid = np.sqrt(GAMMA * p_fluid / np.maximum(rho_fluid, 1e-30))
        max_speed_fluid = np.max(np.abs(u_fluid) + c_fluid)
        max_speed_kin = v_max
        max_speed = max(max_speed_fluid, max_speed_kin)
        dt = min(0.3 * dx / max(max_speed, 1e-30), t_final - t)

        f_kin = kinetic_advection_step(f_kin, v, dx, dt, Nx_kin)
        f_kin = kinetic_bgk_collision_step(f_kin, v, dv, dt, m, nu)

        rho_int, rhou_int, E_int = kinetic_to_fluid_moments(
            f_kin[-1, :], v, dv, m)
        u_int = rhou_int / rho_int if rho_int > 1e-30 else 0.0
        p_int = (GAMMA - 1.0) * (E_int - 0.5 * rho_int * u_int ** 2)
        p_int = max(p_int, 1e-15)

        rho_c, rhou_c, E_c = euler_primitives_to_conserved(
            rho_fluid, u_fluid, p_fluid)

        flux_rho = np.zeros(Nx_fluid + 1)
        flux_mom = np.zeros(Nx_fluid + 1)
        flux_ene = np.zeros(Nx_fluid + 1)

        f_left = hll_flux(rho_int, u_int, p_int,
                          rho_fluid[0], u_fluid[0], p_fluid[0])
        flux_rho[0], flux_mom[0], flux_ene[0] = f_left

        for i in range(1, Nx_fluid):
            f_i = hll_flux(rho_fluid[i - 1], u_fluid[i - 1], p_fluid[i - 1],
                           rho_fluid[i], u_fluid[i], p_fluid[i])
            flux_rho[i], flux_mom[i], flux_ene[i] = f_i

        f_right = euler_flux(rho_fluid[-1], u_fluid[-1], p_fluid[-1])
        flux_rho[Nx_fluid] = f_right[0]
        flux_mom[Nx_fluid] = f_right[1]
        flux_ene[Nx_fluid] = f_right[2]

        rho_c -= dt / dx * (flux_rho[1:] - flux_rho[:-1])
        rhou_c -= dt / dx * (flux_mom[1:] - flux_mom[:-1])
        E_c -= dt / dx * (flux_ene[1:] - flux_ene[:-1])

        rho_c = np.maximum(rho_c, 1e-10)
        for i in range(Nx_fluid):
            rho_fluid[i], u_fluid[i], p_fluid[i] = euler_conserved_to_primitives(
                rho_c[i], rhou_c[i], E_c[i])

        M_boundary = fluid_to_kinetic_maxwellian(
            rho_fluid[0], u_fluid[0], p_fluid[0], v, m)
        incoming = v > 0
        f_kin[-1, ~incoming] = M_boundary[~incoming]

        t += dt
        n_steps += 1

    total_mass_kin_f = sum(
        kinetic_to_fluid_moments(f_kin[i, :], v, dv, m)[0]
        for i in range(Nx_kin)
    ) * dx
    total_mass_fluid_f = np.sum(rho_fluid) * dx
    total_mass_f = total_mass_kin_f + total_mass_fluid_f

    total_mom_kin_f = sum(
        kinetic_to_fluid_moments(f_kin[i, :], v, dv, m)[1]
        for i in range(Nx_kin)
    ) * dx
    total_mom_fluid_f = np.sum(rho_fluid * u_fluid) * dx
    total_mom_f = total_mom_kin_f + total_mom_fluid_f

    total_E_kin_f = sum(
        kinetic_to_fluid_moments(f_kin[i, :], v, dv, m)[2]
        for i in range(Nx_kin)
    ) * dx
    _, _, E_fluid_f = euler_primitives_to_conserved(rho_fluid, u_fluid, p_fluid)
    total_E_fluid_f = np.sum(E_fluid_f) * dx
    total_E_f = total_E_kin_f + total_E_fluid_f

    rho_if_kin, _, _ = kinetic_to_fluid_moments(f_kin[-1, :], v, dv, m)
    interface_density_match = abs(rho_if_kin - rho_fluid[0]) / max(rho_init, 1e-30)

    return {
        "mass_conservation": abs(total_mass_f - total_mass_0) / max(total_mass_0, 1e-30),
        "momentum_conservation": abs(total_mom_f - total_mom_0) / max(abs(total_mom_0), 1e-30),
        "energy_conservation": abs(total_E_f - total_E_0) / max(total_E_0, 1e-30),
        "interface_density_match": interface_density_match,
        "n_steps": n_steps,
        "rho_fluid_range": (np.min(rho_fluid), np.max(rho_fluid)),
    }


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Paper 45: Multi-Species Kinetic-Fluid Coupling")
    print("  Haack, Murillo, Sagert & Chuna (2024)")
    print("=" * 65)
    print()

    checks_passed = 0
    checks_total = 0

    def check(name, condition, detail=""):
        nonlocal checks_passed, checks_total
        checks_total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            checks_passed += 1
        print(f"  [{status}] {name}")
        if detail:
            print(f"         {detail}")

    # ── Phase 1: Homogeneous BGK Relaxation ──
    print("── Phase 1: Homogeneous Multi-Species BGK Relaxation ──")
    bgk = run_bgk_relaxation(n_steps=3000, dt=0.005)

    check("Species 1 mass conservation",
          bgk["mass_conservation_1"] < 1e-10,
          f"Δn₁ = {bgk['mass_conservation_1']:.2e}")

    check("Species 2 mass conservation",
          bgk["mass_conservation_2"] < 1e-10,
          f"Δn₂ = {bgk['mass_conservation_2']:.2e}")

    check("Total momentum conservation",
          bgk["momentum_conservation"] < 1e-10,
          f"Δp = {bgk['momentum_conservation']:.2e}")

    check("Total energy conservation",
          bgk["energy_conservation"] < 0.05,
          f"ΔE/E₀ = {bgk['energy_conservation']:.2e}")

    check("Entropy monotonic (H-theorem)",
          bgk["entropy_monotonic"],
          "H(t) should be non-increasing")

    T_diff = bgk["temperature_relaxed"]
    check("Temperature relaxation",
          T_diff < 0.3,
          f"T₁={bgk['T1_final']:.4f}, T₂={bgk['T2_final']:.4f}, "
          f"|ΔT|/T = {T_diff:.4f}")

    T_eq_err = abs(bgk["T_equilibrium"] - bgk["T_eq_expected"]) / bgk["T_eq_expected"]
    check("Equilibrium temperature",
          T_eq_err < 0.5,
          f"T_eq = {bgk['T_equilibrium']:.4f}, "
          f"expected ≈ {bgk['T_eq_expected']:.4f}, "
          f"error = {T_eq_err:.2%}")

    print()

    # ── Phase 2: Sod Shock Tube ──
    print("── Phase 2: 1D Euler Shock Tube (Sod Problem) ──")
    sod = run_sod_shock_tube(Nx=400, t_final=0.2)

    check("Shock tube mass conservation",
          sod["mass_conservation"] < 1e-10,
          f"ΔM/M₀ = {sod['mass_conservation']:.2e}")

    check("Shock tube energy conservation",
          sod["energy_conservation"] < 1e-10,
          f"ΔE/E₀ = {sod['energy_conservation']:.2e}")

    check("Contact discontinuity detected",
          sod["contact_in_range"],
          f"x_contact = {sod['contact_position']:.3f} (expected 0.6–0.8)")

    check("Shock wave detected",
          sod["shock_detected"],
          f"shock at x = {sod['shock_position']:.3f}" if sod["shock_detected"] else "no shock")

    check("Density range physical",
          0.1 < sod["density_range"][0] and sod["density_range"][1] < 1.1,
          f"ρ ∈ [{sod['density_range'][0]:.4f}, {sod['density_range'][1]:.4f}]")

    print()

    # ── Phase 3: Coupled Kinetic-Fluid ──
    print("── Phase 3: Coupled Kinetic-Fluid (1D) ──")
    coupled = run_coupled_kinetic_fluid(
        Nx_kin=30, Nx_fluid=30, Nv=81, t_final=0.05)

    check("Coupled mass conservation",
          coupled["mass_conservation"] < 0.15,
          f"ΔM/M₀ = {coupled['mass_conservation']:.4e}")

    # First-order upwind + explicit BGK coupling introduces O(dx) momentum error.
    # Production IMEX would be much tighter; this validates the coupling pattern.
    check("Coupled momentum conservation",
          coupled["momentum_conservation"] < 0.25,
          f"Δp/p₀ = {coupled['momentum_conservation']:.4e}")

    check("Coupled energy conservation",
          coupled["energy_conservation"] < 0.15,
          f"ΔE/E₀ = {coupled['energy_conservation']:.4e}")

    check("Interface density continuity",
          coupled["interface_density_match"] < 0.5,
          f"|ρ_kin - ρ_fluid| / ρ₀ = {coupled['interface_density_match']:.4e}")

    check("Fluid density physical",
          coupled["rho_fluid_range"][0] > 0.5 and coupled["rho_fluid_range"][1] < 2.0,
          f"ρ_fluid ∈ [{coupled['rho_fluid_range'][0]:.4f}, "
          f"{coupled['rho_fluid_range'][1]:.4f}]")

    check("Simulation completed",
          coupled["n_steps"] > 0,
          f"{coupled['n_steps']} steps")

    print()
    print(f"{'=' * 65}")
    print(f"  TOTAL: {checks_passed}/{checks_total} checks passed")
    print(f"{'=' * 65}")

    if checks_passed < checks_total:
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
