# szleb_mpcrl/mpc.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple
import numpy as np
import casadi as cs

from .costs import Weights, TargetBands


@dataclass
class MPCConfig:
    horizon: int = 12
    smooth_u: float = 0.05
    solver_max_iter: int = 200
    solver_print: bool = False


def mpc_solve(
    x0: np.ndarray,
    z_seq: np.ndarray,
    u_prev: np.ndarray,
    weights: Weights,
    targets: TargetBands,
    predict: Callable[[cs.MX, cs.MX, cs.MX], cs.MX],
    u_low: np.ndarray,
    u_high: np.ndarray,
    cfg: MPCConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    MPC with soft constraints using slack variables.
    State x is fixed: [Tin, RH].
    Returns:
      - u0 (action)
      - mpc_diag: slack values at first step (sT0, sRH0)
    """
    x0 = np.asarray(x0, dtype=float).reshape(2)
    nx = 2

    H = int(cfg.horizon)
    u_dim = int(np.asarray(u_low).size)

    x = cs.MX.sym("x", nx, H + 1)
    u = cs.MX.sym("u", u_dim, H)

    # Slack variables (soft constraints)
    sT = cs.MX.sym("sT", H)     # temperature slack
    sRH = cs.MX.sym("sRH", H)   # humidity slack

    J = 0
    g = []

    # initial condition
    g.append(x[:, 0] - cs.MX(x0))

    tin_lo, tin_hi = targets.tin_opt_c
    rh_lo, rh_hi = targets.rh_opt_pct

    u_last = cs.MX(u_prev)

    for k in range(H):
        zk = cs.MX(z_seq[:, k])

        # dynamics
        x_next = predict(x[:, k], u[:, k], zk)
        g.append(x[:, k + 1] - x_next)

        Tin = x[0, k + 1]
        RH = x[1, k + 1]

        # === soft constraint inequalities via constraints g_ineq <= 0 ===
        # Tin >= lo - sT   -> (lo - sT) - Tin <= 0
        # Tin <= hi + sT   -> Tin - (hi + sT) <= 0
        g.append((tin_lo - sT[k]) - Tin)
        g.append(Tin - (tin_hi + sT[k]))

        # RH >= lo - sRH
        g.append((rh_lo - sRH[k]) - RH)
        g.append(RH - (rh_hi + sRH[k]))

        # slack non-negativity as inequality: -s <= 0
        g.append(-sT[k])
        g.append(-sRH[k])

        # quadratic penalty for remaining violation (optional)
        t_v = cs.fmax(0, tin_lo - Tin) + cs.fmax(0, Tin - tin_hi)
        rh_v = cs.fmax(0, rh_lo - RH) + cs.fmax(0, RH - rh_hi)

        # energy proxy
        energy_proxy = cs.sumsqr(u[:, k])

        # objective
        J += weights.w_temp * (t_v ** 2) + weights.w_rh * (rh_v ** 2)
        J += weights.w_energy * energy_proxy
        J += weights.w_slack_temp * (sT[k] ** 2) + weights.w_slack_rh * (sRH[k] ** 2)
        J += cfg.smooth_u * cs.sumsqr(u[:, k] - u_last)

        u_last = u[:, k]

    # decision vector includes x, u, sT, sRH
    w_dec = cs.vertcat(cs.reshape(x, -1, 1), cs.reshape(u, -1, 1), sT, sRH)
    g_dec = cs.vertcat(*g)

    n_x = nx * (H + 1)
    n_u = u_dim * H
    n_s = H + H

    # bounds for decision vars
    lbw = [-cs.inf] * n_x + list(np.tile(u_low, H)) + [0.0] * n_s
    ubw = [ cs.inf] * n_x + list(np.tile(u_high, H)) + [cs.inf] * n_s

    # constraints: first nx*(H+1) equalities are 0, rest are inequalities <= 0
    # We built g as mix: first x0 constraint + dynamics equalities, then inequalities.
    # Easiest: set all as <=0 except equalities by specifying exact 0 bounds for those indices.
    # We'll compute how many equalities: 1*(nx) for initial + H*(nx) for dynamics
    n_eq = nx + H * nx
    lbg = [0.0] * n_eq + [-cs.inf] * (int(g_dec.shape[0]) - n_eq)
    ubg = [0.0] * n_eq + [0.0] * (int(g_dec.shape[0]) - n_eq)

    nlp = {"x": w_dec, "f": J, "g": g_dec}
    opts = {
        "ipopt.max_iter": cfg.solver_max_iter,
        "ipopt.print_level": 0 if not cfg.solver_print else 4,
        "print_time": bool(cfg.solver_print),
    }
    solver = cs.nlpsol("solver", "ipopt", nlp, opts)

    # init guess
    x_init = np.tile(x0.reshape(-1, 1), (1, H + 1)).reshape(-1, 1)
    u_init = np.tile(np.asarray(u_prev).reshape(-1, 1), (1, H)).reshape(-1, 1)
    s_init = np.zeros((n_s, 1), dtype=float)
    w0 = np.vstack([x_init, u_init, s_init])

    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = np.asarray(sol["x"]).reshape(-1)

    u_opt_flat = w_opt[n_x:n_x + n_u]
    sT_opt = w_opt[n_x + n_u : n_x + n_u + H]
    sRH_opt = w_opt[n_x + n_u + H : n_x + n_u + 2 * H]

    u0 = np.asarray(u_opt_flat[:u_dim], dtype=float)

    mpc_diag = {
        "sT0": float(sT_opt[0]) if len(sT_opt) else 0.0,
        "sRH0": float(sRH_opt[0]) if len(sRH_opt) else 0.0,
    }
    return u0, mpc_diag