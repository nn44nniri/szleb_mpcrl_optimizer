# szleb_mpcrl/agent.py
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, Tuple
import pickle
import numpy as np
import casadi as cs

from .costs import Weights, TargetBands, compute_stage_cost_from_info
from .mpc import MPCConfig, mpc_solve
from .rls import RLS, RLSConfig


@dataclass
class MPCRLConfig:
    mpc: MPCConfig = field(default_factory=MPCConfig)
    rls: RLSConfig = field(default_factory=RLSConfig)

    # weight bounds
    w_min: float = 1e-6
    w_max: float = 1e6

    # disturbance dim
    nz: int = 4


def default_u_bounds_from_space(action_space):
    low = np.asarray(action_space.low, dtype=float).reshape(-1)
    high = np.asarray(action_space.high, dtype=float).reshape(-1)
    return low, high


class MPCRLAgent:
    """
    Basic structure preserved:
      - RLS predicts x_{t+1} = f(x_t, u_t, z_t)
      - MPC chooses u_t
      - Lightweight TD update tunes weights (w) with minimal overhead
    Controller state x is fixed to [Tin, RH] (2D).
    """

    def __init__(self, action_space, cfg: MPCRLConfig = MPCRLConfig()):
        self.cfg = cfg

        self.targets: Optional[TargetBands] = None
        self.w = Weights()

        self.u_low, self.u_high = default_u_bounds_from_space(action_space)
        self.nu = int(self.u_low.size)

        self.nx = 2
        self.nz = int(cfg.nz)

        self.rls = RLS(nx=self.nx, nu=self.nu, nz=self.nz, cfg=cfg.rls)

        self.u_prev = np.zeros(self.nu, dtype=float)
        self._frozen = False

        # store last MPC slack info
        self._last_mpc_diag: Dict[str, float] = {"sT0": 0.0, "sRH0": 0.0}

    def set_targets(self, targets: TargetBands) -> None:
        self.targets = targets

    def _predict_casadi(self, x: cs.MX, u: cs.MX, z: cs.MX) -> cs.MX:
        Theta = cs.MX(self.rls.Theta)
        phi = cs.vertcat(x, u, z, cs.MX.ones(1, 1))
        return Theta @ phi

    def act(self, x: np.ndarray, z_seq: np.ndarray) -> np.ndarray:
        if self.targets is None:
            raise RuntimeError("MPCRLAgent.targets not set. Call agent.set_targets(get_target_bands_from_env(env)).")
        x = np.asarray(x, dtype=float).reshape(2)

        u, mpc_diag = mpc_solve(
            x0=x,
            z_seq=z_seq,
            u_prev=self.u_prev,
            weights=self.w,
            targets=self.targets,
            predict=self._predict_casadi,
            u_low=self.u_low,
            u_high=self.u_high,
            cfg=self.cfg.mpc,
        )

        self.u_prev = u.copy()
        self._last_mpc_diag = mpc_diag
        return u

    def freeze_learning(self) -> None:
        self._frozen = True

    # ---------- TD helper ----------
    def _phi(self, diag: Dict[str, float]) -> np.ndarray:
        """
        Feature vector for lightweight TD:
          [1, temp_violation^2, rh_violation^2, energy_kwh_eq, slack^2]
        """
        tv2 = float(diag.get("temp_violation", 0.0)) ** 2
        rv2 = float(diag.get("rh_violation", 0.0)) ** 2
        e = float(diag.get("energy_kwh_eq", 0.0))
        sT0 = float(self._last_mpc_diag.get("sT0", 0.0))
        sRH0 = float(self._last_mpc_diag.get("sRH0", 0.0))
        slack2 = (sT0 ** 2) + (sRH0 ** 2)
        return np.array([1.0, tv2, rv2, e, slack2], dtype=float)

    def _value(self, phi: np.ndarray) -> float:
        """
        Simple linear value approximation using current weights as parameters.
        We reuse weights as theta for minimal overhead.
        """
        theta = np.array([
            0.0,                  # bias (unused)
            self.w.w_temp,
            self.w.w_rh,
            self.w.w_energy,
            0.5 * (self.w.w_slack_temp + self.w.w_slack_rh),
        ], dtype=float)
        return float(theta @ phi)

    def _td_update(self, cost: float, phi: np.ndarray, phi_next: np.ndarray) -> float:
        """
        TD(0) update on weights (lightweight).
        We treat 'cost' as immediate cost (we minimize cost).
        delta = cost + gamma*V(next) - V(curr)
        Update weights opposite direction of delta * features.
        """
        gamma = float(self.w.gamma)
        lr = float(self.w.td_lr)

        v = self._value(phi)
        v_next = self._value(phi_next)
        delta = float(cost + gamma * v_next - v)

        # Gradient step (very small): adjust weights directly (then clip)
        # w_temp affects feature phi[1], w_rh phi[2], w_energy phi[3], slack penalty phi[4]
        if not self._frozen:
            self.w.w_temp = float(np.clip(self.w.w_temp - lr * delta * phi[1], self.cfg.w_min, self.cfg.w_max))
            self.w.w_rh = float(np.clip(self.w.w_rh - lr * delta * phi[2], self.cfg.w_min, self.cfg.w_max))
            self.w.w_energy = float(np.clip(self.w.w_energy - lr * delta * phi[3], self.cfg.w_min, self.cfg.w_max))

            # split slack update into both penalties equally
            slack_grad = lr * delta * phi[4]
            self.w.w_slack_temp = float(np.clip(self.w.w_slack_temp - 0.5 * slack_grad, self.cfg.w_min, self.cfg.w_max))
            self.w.w_slack_rh = float(np.clip(self.w.w_slack_rh - 0.5 * slack_grad, self.cfg.w_min, self.cfg.w_max))

        return delta

    # ---------- learning ----------
    def learn_from_transition(
        self,
        x: np.ndarray,
        u: np.ndarray,
        z: np.ndarray,
        x_next: np.ndarray,
        info: Dict[str, Any],
        info_next: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Preserves structure:
          - RLS update (model learning)
          - TD update (weight learning)
        """
        if self.targets is None:
            raise RuntimeError("Targets not set.")

        x = np.asarray(x, dtype=float).reshape(2)
        u = np.asarray(u, dtype=float).reshape(self.nu)
        z = np.asarray(z, dtype=float).reshape(self.nz)
        x_next = np.asarray(x_next, dtype=float).reshape(2)

        # 1) update model
        if not self._frozen:
            self.rls.update(x=x, u=u, z=z, x_next=x_next)

        # 2) cost diagnostics from current info
        cost_now, diag_now = compute_stage_cost_from_info(info, self.targets)

        # next info might not be available; if not, use next state only to compute violations
        if info_next is not None:
            cost_next, diag_next = compute_stage_cost_from_info(info_next, self.targets)
        else:
            # build a minimal "diag_next" from x_next (no energy)
            tin_next, rh_next = float(x_next[0]), float(x_next[1])
            tv = self.targets.violation_tin(tin_next)
            rv = self.targets.violation_rh(rh_next)
            diag_next = {
                "temp_violation": float(tv),
                "rh_violation": float(rv),
                "energy_kwh_eq": 0.0,
            }

        phi = self._phi(diag_now)
        phi_next = self._phi(diag_next)

        # TD update on weights (lightweight)
        delta = self._td_update(cost=cost_now, phi=phi, phi_next=phi_next)

        return {
            "cost": float(cost_now),
            "td_delta": float(delta),
            "diag": diag_now,
            "weights": self.w,
            "mpc_diag": dict(self._last_mpc_diag),
        }

    # ---------- persistence ----------
    def save(self, path: str) -> None:
        payload = {
            "cfg": asdict(self.cfg),
            "weights": asdict(self.w),
            "u_prev": self.u_prev,
            "rls": {
                "Theta": self.rls.Theta,
                "P": self.rls.P,
                "cfg": asdict(self.rls.cfg),
            },
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load(path: str, action_space):
        with open(path, "rb") as f:
            payload = pickle.load(f)

        cfg = MPCRLConfig(
            mpc=MPCConfig(**payload["cfg"]["mpc"]),
            rls=RLSConfig(**payload["cfg"]["rls"]),
            w_min=payload["cfg"]["w_min"],
            w_max=payload["cfg"]["w_max"],
            nz=payload["cfg"]["nz"],
        )

        agent = MPCRLAgent(action_space=action_space, cfg=cfg)
        agent.w = Weights(**payload["weights"])
        agent.u_prev = payload["u_prev"]
        agent.rls.Theta = payload["rls"]["Theta"]
        agent.rls.P = payload["rls"]["P"]
        return agent