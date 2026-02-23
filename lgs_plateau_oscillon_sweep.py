#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LGS plateau-oscillon search with outgoing-wave (Sommerfeld) boundary at rho_max.

Field variable:
    u(rho, tau) = ln( Psi / Psi_n ),  rho = m_n r,  tau = m_n t

Dimensionless PDE (node-independent):
    u_tt - (u_rr + 2/r u_r) + (u_t^2 - u_r^2) + k exp(-2u) sin(a u) = 0
where:
    a = 2π / ln(phi),  k = ln(phi)/(2π)

We evolve using leapfrog / Störmer-Verlet on u and p=u_t:
    u_t = p
    p_t = Lap(u) + u_r^2 - p^2 - k exp(-2u) sin(a u)

Boundary conditions:
    - regularity at r=0: u_r(0)=0
    - outgoing wave at r=rmax (Sommerfeld):
          (∂_t + ∂_r) u = 0  at r=rmax
      implemented discretely each time step.

Core energy (dimensionless):
    I_core(t) = 4π ∫_0^{rho_core} dρ ρ^2 [ 1/2 e^{2u} (p^2 + u_r^2) + k^2(1 - cos(a u)) ]

Plateau filter (strict):
    late-time window [0.6T, 0.9T]:
        - mean core energy above Emin
        - relative slope |dI/dt| / mean(I) < rel_slope_tol
        - oscillation present: std(u0) > u0_std_min
        - relative drop across window < drop_tol
"""

import argparse
import time
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

# ----------------------------
# Constants
# ----------------------------
phi = (1.0 + 5.0**0.5) / 2.0
lnphi = np.log(phi)
a = 2.0 * np.pi / lnphi          # a = 2π / lnφ
k = lnphi / (2.0 * np.pi)        # k = lnφ / (2π)

# Safety caps to prevent catastrophic floating overflow in intermediate operations
SQ_CAP = 1e10        # cap for p^2 and ur^2 terms used in RHS and energy
U_EXP_CAP = 250.0    # cap on u when computing exp(±2u) (exp(500) is huge but still finite in float)
EPS = 1e-30

def safe_exp(x):
    # avoid overflow/underflow in exp
    return np.exp(np.clip(x, -700, 700))

def clamp_u_for_exp(u):
    # prevent exp(±2u) from becoming insane if a trajectory runs away
    return np.clip(u, -U_EXP_CAP, U_EXP_CAP)

# ----------------------------
# Spatial operators
# ----------------------------
def laplacian_radial_flux(u, dr):
    """
    3D radial Laplacian in flux form:
        ∇^2 u = (1/r^2) d/dr (r^2 u_r)
    u includes r=0 at index 0, uniform dr.
    """
    N = u.size
    r = np.arange(N) * dr
    r2 = r*r

    lap = np.zeros_like(u)

    # half-step derivative u_r at i+1/2
    ur_half = (u[1:] - u[:-1]) / dr          # length N-1
    flux_half = r2[1:] * ur_half             # length N-1

    # interior i=1..N-2
    lap[1:-1] = (flux_half[1:] - flux_half[:-1]) / (dr * (r2[1:-1] + EPS))

    # origin: u = u0 + 1/2 u2 r^2 => ∇^2 u(0) = 3 u2 ≈ 6 (u1-u0)/dr^2
    lap[0] = 6.0 * (u[1] - u[0]) / (dr*dr)

    # boundary point lap[-1] not used directly (we overwrite boundary via Sommerfeld)
    lap[-1] = 0.0
    return lap

def ur_centered(u, dr):
    ur = np.zeros_like(u)
    ur[1:-1] = (u[2:] - u[:-2]) / (2.0*dr)
    ur[0] = 0.0          # regularity
    ur[-1] = (u[-1] - u[-2]) / dr
    return ur

# ----------------------------
# RHS for p_t
# ----------------------------
def rhs_p(u, p, dr):
    lap = laplacian_radial_flux(u, dr)
    ur = ur_centered(u, dr)

    # Clip squared terms to avoid overflow during runaway cases
    ur2 = np.minimum(ur*ur, SQ_CAP)
    p2  = np.minimum(p*p,  SQ_CAP)

    u_clamped = clamp_u_for_exp(u)
    forcing = k * safe_exp(-2.0*u_clamped) * np.sin(a*u_clamped)

    # p_t = Lap(u) + u_r^2 - p^2 - k e^{-2u} sin(a u)
    return lap + ur2 - p2 - forcing

# ----------------------------
# Energy: core only (safe)
# ----------------------------
def core_energy(u, p, dr, r_core):
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(p))):
        return np.nan

    r = np.arange(u.size) * dr
    mask = r <= r_core
    if mask.sum() < 3:
        return 0.0

    ur = ur_centered(u, dr)

    # clamp u for exp and cap kinetic magnitude
    u_clamped = clamp_u_for_exp(u)
    kin = p*p + ur*ur
    kin = np.minimum(kin, SQ_CAP)

    density = 0.5 * safe_exp(2.0*u_clamped) * kin + (k*k) * (1.0 - np.cos(a*u_clamped))
    if not np.all(np.isfinite(density[mask])):
        return np.nan

    integrand = (r[mask]**2) * density[mask]
    if not np.all(np.isfinite(integrand)):
        return np.nan

    return 4.0*np.pi * np.trapz(integrand, dx=dr)

# ----------------------------
# Sommerfeld outgoing boundary
# ----------------------------
def apply_sommerfeld(u_new, u_old, dr, dt):
    """
    Enforce (∂_t + ∂_r)u=0 at outer boundary r=rmax.

    Discrete:
        (uN_new - uN_old)/dt + (uN_new - uNm1_new)/dr = 0
        => uN_new = (uN_old/dt + uNm1_new/dr) / (1/dt + 1/dr)
    """
    uN_old = u_old[-1]
    uNm1_new = u_new[-2]
    uN_new = (uN_old/dt + uNm1_new/dr) / (1.0/dt + 1.0/dr)
    u_new[-1] = uN_new

# ----------------------------
# Single run (A, sigma)
# ----------------------------
def run_case(A, sigma,
             rmax=140.0, dr=0.08, dt=0.01, tmax=2500.0,
             r_core=12.0,
             sample_every=10,
             u_blow=25.0):

    N = int(rmax/dr) + 1
    r = np.arange(N) * dr

    # initial condition: localized Gaussian bump
    u = A * np.exp(-(r/sigma)**2)
    p = np.zeros_like(u)

    # leapfrog: initialize p at half-step
    p_half = p + 0.5*dt*rhs_p(u, p, dr)

    times = []
    Icore = []
    u0 = []

    steps = int(tmax/dt)
    ok = True
    reason = ""

    for n in range(steps):
        t = n * dt

        # u^{n+1}
        u_new = u + dt * p_half

        # outgoing boundary + origin regularity
        apply_sommerfeld(u_new, u, dr, dt)
        u_new[0] = u_new[1]

        # FAIL FAST: runaway before computing RHS
        if not np.all(np.isfinite(u_new)):
            ok = False
            reason = "u NaN/Inf"
            break
        if np.max(np.abs(u_new)) > u_blow:
            ok = False
            reason = f"|u|>{u_blow}"
            break

        # p^{n+3/2}
        p_half = p_half + dt * rhs_p(u_new, p_half, dr)

        if not np.all(np.isfinite(p_half)):
            ok = False
            reason = "p NaN/Inf"
            break

        # update state
        u = u_new

        # diagnostics
        if n % sample_every == 0:
            E = core_energy(u, p_half, dr, r_core)
            if not np.isfinite(E):
                ok = False
                reason = "Energy NaN/Inf"
                break
            times.append(t)
            Icore.append(E)
            u0.append(u[0])

    return {
        "ok": ok,
        "reason": reason,
        "t_end": times[-1] if times else 0.0,
        "times": np.asarray(times, dtype=float),
        "Icore": np.asarray(Icore, dtype=float),
        "u0": np.asarray(u0, dtype=float),
        "params": dict(A=A, sigma=sigma, rmax=rmax, dr=dr, dt=dt, tmax=tmax, r_core=r_core)
    }

# ----------------------------
# Plateau filter
# ----------------------------
def plateau_metrics(times, Icore, u0, frac_start=0.6, frac_end=0.9):
    if times.size < 12:
        return None

    T = times[-1] - times[0]
    if T <= 0:
        return None

    w_start = times[0] + frac_start * T
    w_end   = times[0] + frac_end * T
    m = (times >= w_start) & (times <= w_end)
    if m.sum() < 8:
        return None

    tw = times[m]
    Ew = Icore[m]
    uw = u0[m]

    meanE = float(np.trapz(Ew, tw) / (tw[-1] - tw[0] + EPS))

    # linear fit slope
    Afit = np.vstack([tw, np.ones_like(tw)]).T
    slope, intercept = np.linalg.lstsq(Afit, Ew, rcond=None)[0]
    slope = float(slope)

    rel_slope = abs(slope) / (meanE + EPS)
    u0_std = float(np.std(uw))

    # relative drop across window
    n3 = max(2, len(Ew)//3)
    drop = float((np.median(Ew[:n3]) - np.median(Ew[-n3:])) / (meanE + EPS))

    return {
        "w_start": float(w_start),
        "w_end": float(w_end),
        "meanE": meanE,
        "slope": slope,
        "rel_slope": rel_slope,
        "u0_std": u0_std,
        "drop": drop
    }

def is_plateau(metrics, Emin=0.5, rel_slope_tol=5e-4, u0_std_min=0.02, drop_tol=0.15):
    if metrics is None:
        return False
    if metrics["meanE"] < Emin:
        return False
    if metrics["rel_slope"] > rel_slope_tol:
        return False
    if metrics["u0_std"] < u0_std_min:
        return False
    if abs(metrics["drop"]) > drop_tol:
        return False
    return True

# ----------------------------
# Sweep driver
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    # Parameter ranges
    ap.add_argument("--Amin", type=float, default=0.1)
    ap.add_argument("--Amax", type=float, default=0.8)
    ap.add_argument("--sigmin", type=float, default=2.0)
    ap.add_argument("--sigmax", type=float, default=10.0)
    ap.add_argument("--nA", type=int, default=18)
    ap.add_argument("--nsig", type=int, default=18)

    # Numerics
    ap.add_argument("--rmax", type=float, default=140.0)
    ap.add_argument("--dr", type=float, default=0.08)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--tmax", type=float, default=2500.0)
    ap.add_argument("--rcore", type=float, default=12.0)
    ap.add_argument("--sample_every", type=int, default=10)
    ap.add_argument("--u_blow", type=float, default=25.0)

    # Plateau criteria
    ap.add_argument("--Emin", type=float, default=0.5)
    ap.add_argument("--rel_slope_tol", type=float, default=5e-4)
    ap.add_argument("--u0_std_min", type=float, default=0.02)
    ap.add_argument("--drop_tol", type=float, default=0.15)

    # Misc
    ap.add_argument("--plot_best", action="store_true")
    ap.add_argument("--progress_every", type=int, default=50, help="print progress every N cases")

    args = ap.parse_args()

    # Build A grid including negative and positive ranges
    Apos = np.linspace(args.Amin, args.Amax, args.nA)
    Aneg = -np.linspace(args.Amin, args.Amax, args.nA)
    As = np.concatenate([Aneg, Apos])
    sigmas = np.linspace(args.sigmin, args.sigmax, args.nsig)

    best = None
    plateau_hits = 0
    total_cases = len(As) * len(sigmas)
    case_count = 0

    t_start = time.time()

    for A in As:
        for sig in sigmas:
            case_count += 1

            out = run_case(
                A=float(A), sigma=float(sig),
                rmax=args.rmax, dr=args.dr, dt=args.dt, tmax=args.tmax,
                r_core=args.rcore,
                sample_every=args.sample_every,
                u_blow=args.u_blow
            )

            met = plateau_metrics(out["times"], out["Icore"], out["u0"])
            ok_plateau = is_plateau(
                met,
                Emin=args.Emin,
                rel_slope_tol=args.rel_slope_tol,
                u0_std_min=args.u0_std_min,
                drop_tol=args.drop_tol
            )

            if ok_plateau:
                plateau_hits += 1
                score = met["meanE"] / (met["rel_slope"] + 1e-12)
                if (best is None) or (score > best["score"]):
                    best = {"score": score, "A": float(A), "sigma": float(sig), "out": out, "metrics": met}

            if args.progress_every > 0 and (case_count % args.progress_every == 0):
                elapsed = time.time() - t_start
                print(f"[{case_count}/{total_cases}] elapsed {elapsed:.1f}s | plateau_hits={plateau_hits} | latest A={A:+.3f}, sig={sig:.3f}, ok={out['ok']}, end={out['t_end']:.1f}, reason={out['reason']}")

    elapsed = time.time() - t_start

    print("\n==== Sweep finished ====")
    print(f"Elapsed: {elapsed:.1f} s")
    print(f"Plateau hits: {plateau_hits}")

    if best is None:
        print("No plateau oscillons found under current criteria/settings.")
        print("Try one or more of:")
        print("  - increase --tmax (e.g., 4000)")
        print("  - increase --rmax (e.g., 200)")
        print("  - reduce --dt (e.g., 0.008 or 0.005)")
        print("  - loosen slightly: --rel_slope_tol 1e-3  --drop_tol 0.25")
        return

    met = best["metrics"]
    print("\n==== Best plateau candidate ====")
    print(f"A = {best['A']:.6f}")
    print(f"sigma = {best['sigma']:.6f}")
    print(f"r_core = {args.rcore}")
    print(f"Plateau window: [{met['w_start']:.3f}, {met['w_end']:.3f}]")
    print(f"<I_E^core> (window mean) = {met['meanE']:.12g}")
    print(f"rel_slope = {met['rel_slope']:.3e}  (|dE/dt|/meanE)")
    print(f"u0_std = {met['u0_std']:.6g}")
    print(f"drop = {met['drop']:.6g}")

    print("\nUse in manuscript:")
    print("  <I_E^{core}> = {:.12g}".format(met["meanE"]))
    print("  W(n) = (phi^n / (kappa0*m_n)) * <I_E^{core}>")

    if args.plot_best and HAS_PLT:
        out = best["out"]
        plt.figure(figsize=(7,4))
        plt.plot(out["times"], out["Icore"])
        plt.xlabel(r"$\tau$")
        plt.ylabel(r"$I_E^{core}$")
        plt.title(f"Best plateau: A={best['A']:.3f}, sigma={best['sigma']:.3f}")
        plt.show()

        plt.figure(figsize=(7,4))
        plt.plot(out["times"], out["u0"])
        plt.xlabel(r"$\tau$")
        plt.ylabel(r"$u(0,\tau)$")
        plt.title("Central amplitude")
        plt.show()

if __name__ == "__main__":
    main()