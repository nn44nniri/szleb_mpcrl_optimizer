# szleb_mpcrl/envs.py
from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd

# Gym / Gymnasium compatibility
try:
    import gymnasium as gym
    GYMNASIUM = True
except ImportError:
    import gym  # type: ignore
    GYMNASIUM = False

from szleb_gym_rl import register_szleb_env, SZLEBEnvConfig
from .costs import TargetBands


def make_szleb_env(season_df: pd.DataFrame):
    """
    Create SZLEB-v0 with action override enabled (so MPCRL controls actuators).
    """
    register_szleb_env()

    cfg = SZLEBEnvConfig(
        elapsed_s_per_row=24 * 3600.0,
        dt_s=60.0,
        use_action_override=True,  # MPCRL provides action
    )
    env = gym.make("SZLEB-v0", season_table=season_df, config=cfg)
    return env


def get_target_bands_from_env(env) -> TargetBands:
    """
    Source of truth: env.unwrapped.config (species-aware).
    """
    cfg = getattr(env.unwrapped, "config", None)
    if cfg is None:
        # fallback (should not happen in SZLEB-v0)
        return TargetBands(tin_opt_c=(18.0, 26.0), rh_opt_pct=(60.0, 75.0))
    return TargetBands.from_env_config(cfg)


def reset_env(env):
    if GYMNASIUM:
        obs, info = env.reset()
        return obs, info
    obs = env.reset()
    return obs, {}


def step_env(env, u: np.ndarray):
    """
    Step wrapper for Gym/Gymnasium.
    """
    if GYMNASIUM:
        obs2, reward, terminated, truncated, info2 = env.step(u)
        done = bool(terminated or truncated)
        return obs2, float(reward), done, info2
    obs2, reward, done, info2 = env.step(u)
    return obs2, float(reward), bool(done), info2




# ========= NEW: control-state extractor (the key fix) =========
def extract_control_state(info: Dict[str, Any], fallback: np.ndarray | None = None) -> np.ndarray:
    """
    Returns the MPCRL state x used by RLS+MPC:
        x = [Tin_final_C, RHin_final_pct]
    This avoids relying on the raw observation layout.
    """
    row_out = info.get("current_row_output", {}) or {}

    tin = row_out.get("Tin_final_C", row_out.get("t_in_c", None))
    rh = row_out.get("RHin_final_pct", row_out.get("rh_in_pct", None))

    if tin is None or rh is None:
        # fallback to obs if env didn't provide row output (rare)
        if fallback is None:
            raise RuntimeError("Cannot extract control state: missing current_row_output and no fallback.")
        fb = np.asarray(fallback, dtype=float).reshape(-1)
        # last resort: assume first two are Tin/RH
        return np.array([float(fb[0]), float(fb[1])], dtype=float)

    return np.array([float(tin), float(rh)], dtype=float)