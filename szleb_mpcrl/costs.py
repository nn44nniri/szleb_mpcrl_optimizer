# szleb_mpcrl/costs.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np

GAS_KWH_PER_M3 = 10.5  # energy-equivalent conversion (approx)


@dataclass(frozen=True)
class TargetBands:
    """
    Target bands come from SZLEB env config (species-aware).
    """
    tin_opt_c: Tuple[float, float]
    rh_opt_pct: Tuple[float, float]
    vpd_opt_kpa: Tuple[float, float] | None = None

    @staticmethod
    def _clamp_violation(x: float, lo: float, hi: float) -> float:
        if not np.isfinite(x):
            return 0.0
        if x < lo:
            return lo - x
        if x > hi:
            return x - hi
        return 0.0

    def violation_tin(self, tin_c: float) -> float:
        lo, hi = self.tin_opt_c
        return self._clamp_violation(tin_c, lo, hi)

    def violation_rh(self, rh_pct: float) -> float:
        lo, hi = self.rh_opt_pct
        return self._clamp_violation(rh_pct, lo, hi)

    @classmethod
    def from_env_config(cls, env_config: Any) -> "TargetBands":
        opt = getattr(env_config, "tomato_optima", None)
        if opt is None:
            return cls(tin_opt_c=(18.0, 26.0), rh_opt_pct=(60.0, 75.0))
        tin_opt = tuple(opt.tin_opt_c)
        rh_opt = tuple(opt.rh_opt_pct)
        vpd_opt = tuple(opt.vpd_opt_kpa) if hasattr(opt, "vpd_opt_kpa") else None
        return cls(tin_opt_c=tin_opt, rh_opt_pct=rh_opt, vpd_opt_kpa=vpd_opt)


@dataclass
class Weights:
    """
    MPC stage weights (learned by lightweight TD update).
    Keep them positive and bounded in agent.
    """
    w_temp: float = 1.0
    w_rh: float = 1.0
    w_energy: float = 0.15

    # Soft constraint penalties (slacks)
    w_slack_temp: float = 50.0
    w_slack_rh: float = 50.0

    # TD learning settings (small overhead)
    gamma: float = 0.95
    td_lr: float = 1e-4


def compute_stage_cost_from_info(
    info: Dict[str, Any],
    targets: TargetBands,
) -> tuple[float, Dict[str, float]]:
    """
    Diagnostic cost from env info:
    - violation^2
    - energy_kwh_eq
    Used for reporting and TD target.
    """
    row_out = info.get("current_row_output", {}) or {}
    Tin = float(row_out.get("Tin_final_C", np.nan))
    RHin = float(row_out.get("RHin_final_pct", row_out.get("rh_in_pct", np.nan)))

    elec_kwh = float(row_out.get("total_elec_kwh", 0.0))
    gas_m3 = float(row_out.get("heater_gas_total_m3", 0.0))
    gas_kwh_eq = GAS_KWH_PER_M3 * gas_m3
    energy_kwh_eq = elec_kwh + gas_kwh_eq

    t_v = targets.violation_tin(Tin)
    rh_v = targets.violation_rh(RHin)

    # This cost is not the env reward; env reward already includes plant penalty.
    cost = (t_v ** 2) + (rh_v ** 2) + 0.01 * energy_kwh_eq

    diag = {
        "Tin": Tin,
        "RHin": RHin,
        "temp_violation": float(t_v),
        "rh_violation": float(rh_v),
        "elec_kwh": float(elec_kwh),
        "gas_m3": float(gas_m3),
        "gas_kwh_eq": float(gas_kwh_eq),
        "energy_kwh_eq": float(energy_kwh_eq),
        "cost": float(cost),
    }
    return float(cost), diag