# evaluate_szleb_mpcrl_report.py
# Full validation (rolling windows) + CSV reports + plots.
# CSV now includes:
#  - heater_status_01, fan_status_01, vents_status_01
#  - Tout_C, RHout_pct
# for per-step and per-window summaries.

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# gym / gymnasium
try:
    import gymnasium as gym
except ImportError:
    import gym  # type: ignore

from szleb_gym_rl import register_szleb_env, SZLEBEnvConfig
from szleb_mpcrl import MPCRLAgent
from szleb_mpcrl.envs import (
    reset_env,
    step_env,
    get_target_bands_from_env,
    extract_control_state,
)
from szleb_mpcrl.costs import TargetBands


CSV_PATH = "/media/p1/datasets/weather_Alvand/Alvand_36_186002_50_064982_1640995200_1704067199_67d818b83e2ae2000820e2db.csv"
MODEL_PATH = "szleb_mpcrl_trained.pkl"

USE_COLS = ["dt", "dt_iso", "temp", "humidity", "wind_speed", "clouds_all"]

ELAPSED_S_PER_ROW = 3600.0
DT_S = 60.0

GAS_KWH_PER_M3 = 10.5

DEFAULT_TIN0_C = 20.0
DEFAULT_RHIN0_PCT = 60.0

STATUS_THRESH = {"heater": 0.5, "fan": 0.5, "vents": 0.5}


@dataclass
class EvalCfg:
    window_len_h: int = 168
    stride_h: int = 168            # set 24 for rolling weekly windows with daily stride
    save_best_k: int = 3
    save_worst_k: int = 3
    rank_metric: str = "energy"    # "energy" or "band_rate"
    out_dir: str = "eval_reports_weather_full"


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dt_iso_parsed"] = pd.to_datetime(df.get("dt_iso", pd.Series([None] * len(df))), errors="coerce", utc=True)
    if df["dt_iso_parsed"].isna().all() and "dt" in df.columns:
        df["dt_iso_parsed"] = pd.to_datetime(df["dt"], unit="s", utc=True, errors="coerce")
    return df


def _temp_to_celsius(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    med = float(np.nanmedian(s.to_numpy()))
    if med > 60.0:  # Kelvin heuristic
        return s - 273.15
    return s


def load_weather_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: c in USE_COLS)
    df = _ensure_datetime(df)
    df = df.dropna(subset=["dt", "temp", "humidity", "wind_speed", "clouds_all"]).reset_index(drop=True)

    df["t_out_c"] = _temp_to_celsius(df["temp"])
    df["rh_out_pct"] = df["humidity"].astype(float).clip(0, 100)

    clouds = df["clouds_all"].astype(float).clip(0, 100)
    df["g_sun_w_m2"] = 400.0 * (1.0 - clouds / 100.0)

    df["wind_speed_m_s"] = df["wind_speed"].astype(float).clip(lower=0)

    if df["dt_iso_parsed"].notna().any():
        df = df.sort_values("dt_iso_parsed").reset_index(drop=True)
    else:
        df = df.sort_values("dt").reset_index(drop=True)

    return df


def select_validation_slice(df: pd.DataFrame) -> pd.DataFrame:
    """
    Default validation: Nov-Dec 2022 if timestamps exist; else last 20%.
    """
    if df["dt_iso_parsed"].notna().any():
        year = df["dt_iso_parsed"].dt.year
        month = df["dt_iso_parsed"].dt.month
        mask = (year == 2022) & (month >= 11)
        df_val = df.loc[mask].copy()
        if len(df_val) > 100:
            return df_val.reset_index(drop=True)

    n = len(df)
    cut = int(0.8 * n)
    return df.iloc[cut:].reset_index(drop=True)


def make_szleb_env_hourly(season_table: pd.DataFrame):
    register_szleb_env()
    cfg = SZLEBEnvConfig(
        dt_s=DT_S,
        elapsed_s_per_row=ELAPSED_S_PER_ROW,
        use_action_override=True,
    )
    return gym.make("SZLEB-v0", season_table=season_table, config=cfg)


def build_season_table(df_window: pd.DataFrame, lai_start: float = 0.8, lai_end: float = 2.5) -> pd.DataFrame:
    n = len(df_window)
    lai = np.linspace(lai_start, lai_end, n, dtype=float)
    return pd.DataFrame({
        "day": np.arange(n, dtype=int),
        "LAI": lai,
        "t_out_c": df_window["t_out_c"].to_numpy(dtype=float),
        "rh_out_pct": df_window["rh_out_pct"].to_numpy(dtype=float),
        "g_sun_w_m2": df_window["g_sun_w_m2"].to_numpy(dtype=float),
        "t_in_c": float(DEFAULT_TIN0_C),
        "rh_in_pct": float(DEFAULT_RHIN0_PCT),
        "wind_speed_m_s": df_window["wind_speed_m_s"].to_numpy(dtype=float),
    })


def z_from_row(row: pd.Series) -> np.ndarray:
    return np.array([float(row["t_out_c"]), float(row["rh_out_pct"]), float(row["g_sun_w_m2"]), float(row["LAI"])], dtype=float)


def clamp_violation(x: float, lo: float, hi: float) -> float:
    if not np.isfinite(x):
        return 0.0
    if x < lo:
        return lo - x
    if x > hi:
        return x - hi
    return 0.0


def _infer_status(level: float, kind: str) -> int:
    if not np.isfinite(level):
        return 0
    thr = STATUS_THRESH.get(kind, 0.5)
    if level > 1.5:  # percent-like
        return int(level >= max(thr, 1.0))
    return int(level >= thr)


def extract_energy(row_out: dict) -> dict:
    elec_kwh = float(row_out.get("total_elec_kwh", 0.0))
    gas_m3 = float(row_out.get("heater_gas_total_m3", 0.0))
    gas_kwh = GAS_KWH_PER_M3 * gas_m3
    return {
        "elec_total_kwh": elec_kwh,
        "gas_total_m3": gas_m3,
        "gas_kwh_eq": gas_kwh,
        "energy_total_kwh_eq": elec_kwh + gas_kwh,
    }


def extract_triggers_decisions_status(info2: dict, u: np.ndarray) -> dict:
    """
    Ensures heater_status_01 / fan_status_01 / vents_status_01 are ALWAYS produced.
    """
    row_out = info2.get("current_row_output", {}) or {}

    heater_lvl = row_out.get("heater_pct", row_out.get("heater_activity_pct", row_out.get("heater_activity", np.nan)))
    fan_lvl = row_out.get("fan_pct", row_out.get("fan_activity_pct", row_out.get("fan_activity", np.nan)))
    vents_lvl = row_out.get("vents_open", row_out.get("vent_open", row_out.get("flaps_open", np.nan)))

    heater_lvl = float(heater_lvl) if heater_lvl == heater_lvl else np.nan
    fan_lvl = float(fan_lvl) if fan_lvl == fan_lvl else np.nan
    vents_lvl = float(vents_lvl) if vents_lvl == vents_lvl else np.nan

    heater_status = row_out.get("heater_on", row_out.get("heater_status", np.nan))
    fan_status = row_out.get("fan_on", row_out.get("fan_status", np.nan))
    vents_status = row_out.get("vents_open_status", row_out.get("vents_status", np.nan))

    heater_status = float(heater_status) if heater_status == heater_status else np.nan
    fan_status = float(fan_status) if fan_status == fan_status else np.nan
    vents_status = float(vents_status) if vents_status == vents_status else np.nan

    if not np.isfinite(heater_status):
        src = heater_lvl if np.isfinite(heater_lvl) else (float(u[0]) if u.size >= 1 else np.nan)
        heater_status = _infer_status(src, "heater")
    if not np.isfinite(fan_status):
        src = fan_lvl if np.isfinite(fan_lvl) else (float(u[1]) if u.size >= 2 else np.nan)
        fan_status = _infer_status(src, "fan")
    if not np.isfinite(vents_status):
        src = vents_lvl if np.isfinite(vents_lvl) else (float(u[2]) if u.size >= 3 else np.nan)
        vents_status = _infer_status(src, "vents")

    d = {
        "heater_activation_lvl": heater_lvl,
        "fan_activation_lvl": fan_lvl,
        "vents_activation_lvl": vents_lvl,
        "heater_status_01": int(heater_status),
        "fan_status_01": int(fan_status),
        "vents_status_01": int(vents_status),
    }
    for i in range(u.size):
        d[f"decision_u{i}"] = float(u[i])
    return d


def prediction_accuracy(agent: MPCRLAgent, x: np.ndarray, u: np.ndarray, z_now: np.ndarray, x_next: np.ndarray) -> dict:
    x_pred = agent.rls.predict(x=x, u=u, z=z_now)
    err = (x_next - x_pred).astype(float)
    return {"pred_mae": float(np.mean(np.abs(err))), "pred_rmse": float(np.sqrt(np.mean(err ** 2)))}


def run_window(env, agent: MPCRLAgent, season: pd.DataFrame, targets: TargetBands, window_id: int) -> pd.DataFrame:
    obs, info = reset_env(env)
    agent.freeze_learning()

    tin_lo, tin_hi = targets.tin_opt_c
    rh_lo, rh_hi = targets.rh_opt_pct

    H = agent.cfg.mpc.horizon
    done = False
    t = 0
    rows = []

    x = extract_control_state(info, fallback=np.asarray(obs))

    while not done:
        z_seq = np.zeros((agent.nz, H), dtype=float)
        for k in range(H):
            idx = min(t + k, len(season) - 1)
            z_seq[:, k] = z_from_row(season.iloc[idx])

        u = agent.act(x, z_seq)

        obs2, reward, done, info2 = step_env(env, u)
        x_next = extract_control_state(info2, fallback=np.asarray(obs2))

        row_out = info2.get("current_row_output", {}) or {}
        Tin = float(row_out.get("Tin_final_C", x_next[0]))
        RHin = float(row_out.get("RHin_final_pct", x_next[1]))

        t_v = clamp_violation(Tin, tin_lo, tin_hi)
        rh_v = clamp_violation(RHin, rh_lo, rh_hi)
        within = int((t_v == 0.0) and (rh_v == 0.0))

        z_now = z_from_row(season.iloc[min(t, len(season) - 1)])
        acc = prediction_accuracy(agent, x=x, u=u, z_now=z_now, x_next=x_next)

        trig = extract_triggers_decisions_status(info2, u)
        energy = extract_energy(row_out)

        row_in = season.iloc[min(t, len(season) - 1)]
        Tout = float(row_in["t_out_c"])
        RHout = float(row_in["rh_out_pct"])

        # ✅ The required fields are explicitly logged here:
        rows.append({
            "window_id": window_id,
            "step": t,
            "reward_env": float(reward),

            "Tin_C": Tin,
            "RHin_pct": RHin,

            "Tout_C": Tout,          # ✅ outdoor temp
            "RHout_pct": RHout,      # ✅ outdoor humidity

            "within_optimal_band_01": within,
            "temp_violation": float(t_v),
            "rh_violation": float(rh_v),

            **energy,
            **trig,   # ✅ heater_status_01, fan_status_01, vents_status_01 included here
            **acc,
        })

        obs, info = obs2, info2
        x = x_next
        t += 1

    return pd.DataFrame(rows)


def plot_window(df: pd.DataFrame, targets: TargetBands, out_png: str) -> None:
    t = df["step"].to_numpy()
    tin_lo, tin_hi = targets.tin_opt_c
    rh_lo, rh_hi = targets.rh_opt_pct

    fig = plt.figure(figsize=(14, 16))

    # (1) Inside Tin/RHin
    ax1 = fig.add_subplot(6, 1, 1)
    tin_color = "tab:blue"
    rhin_color = "tab:orange"

    ax1.plot(t, df["Tin_C"], label="Tin (C)", color=tin_color)
    ax1.axhline(tin_lo, linestyle="--", color=tin_color, alpha=0.6, label="Tin opt lo")
    ax1.axhline(tin_hi, linestyle="--", color=tin_color, alpha=0.6, label="Tin opt hi")
    ax1.set_ylabel("Tin (C)", color=tin_color)
    ax1.tick_params(axis="y", colors=tin_color)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Inside Parameters with Optimal Bands (from SZLEB env)")

    ax1b = ax1.twinx()
    ax1b.plot(t, df["RHin_pct"], label="RHin (%)", color=rhin_color)
    ax1b.axhline(rh_lo, linestyle="--", color=rhin_color, alpha=0.6, label="RH opt lo")
    ax1b.axhline(rh_hi, linestyle="--", color=rhin_color, alpha=0.6, label="RH opt hi")
    ax1b.set_ylabel("RH (%)", color=rhin_color)
    ax1b.tick_params(axis="y", colors=rhin_color)

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax1b.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper right")

    # (2) Outside Tout/RHout
    ax_out = fig.add_subplot(6, 1, 2)
    tout_color = "tab:green"
    rhout_color = "tab:red"

    ax_out.plot(t, df["Tout_C"], label="Tout (C)", color=tout_color)
    ax_out.set_ylabel("Tout (C)", color=tout_color)
    ax_out.tick_params(axis="y", colors=tout_color)
    ax_out.grid(True, alpha=0.3)
    ax_out.set_title("Outside Drivers")

    ax_out_b = ax_out.twinx()
    ax_out_b.plot(t, df["RHout_pct"], label="RHout (%)", color=rhout_color)
    ax_out_b.set_ylabel("RHout (%)", color=rhout_color)
    ax_out_b.tick_params(axis="y", colors=rhout_color)

    l1, lab1 = ax_out.get_legend_handles_labels()
    l2, lab2 = ax_out_b.get_legend_handles_labels()
    ax_out.legend(l1 + l2, lab1 + lab2, loc="upper right")

    # (3) Activation + decisions
    ax2 = fig.add_subplot(6, 1, 3)
    if df["heater_activation_lvl"].notna().any():
        ax2.plot(t, df["heater_activation_lvl"], label="heater_activation_lvl")
    if df["fan_activation_lvl"].notna().any():
        ax2.plot(t, df["fan_activation_lvl"], label="fan_activation_lvl")
    if df["vents_activation_lvl"].notna().any():
        v = df["vents_activation_lvl"].copy()
        if v.dropna().max() <= 1.5:
            v = v * 100.0
        ax2.plot(t, v, label="vents_activation_lvl (x100 if 0..1)")
    decision_cols = [c for c in df.columns if c.startswith("decision_u")]
    for c in decision_cols[:4]:
        ax2.plot(t, df[c], label=c)
    ax2.set_ylabel("Activation / Decisions")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")
    ax2.set_title("Trigger Activation + Decisions")

    # (4) Status
    ax3 = fig.add_subplot(6, 1, 4)
    ax3.step(t, df["heater_status_01"], where="post", label="heater_status_01")
    ax3.step(t, df["fan_status_01"], where="post", label="fan_status_01")
    ax3.step(t, df["vents_status_01"], where="post", label="vents_status_01")
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_ylabel("Status (0/1)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right")
    ax3.set_title("Trigger Status (ON/OFF)")

    # (5) Energy
    ax4 = fig.add_subplot(6, 1, 5)
    ax4.plot(t, df["elec_total_kwh"], label="electricity (kWh)")
    ax4.plot(t, df["gas_kwh_eq"], label="gas (kWh eq)")
    ax4.plot(t, df["energy_total_kwh_eq"], label="total (kWh eq)")
    ax4.set_ylabel("Energy")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper right")
    ax4.set_title("Energy Split")

    # (6) Accuracy + satisfaction
    ax5 = fig.add_subplot(6, 1, 6)
    ax5.plot(t, df["pred_mae"], label="pred_mae")
    ax5.plot(t, df["pred_rmse"], label="pred_rmse")
    ax5.plot(t, df["within_optimal_band_01"], label="within_optimal_band (0/1)")
    ax5.set_xlabel("Step (hour index)")
    ax5.set_ylabel("Accuracy / Satisfaction")
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc="upper right")
    ax5.set_title("Accuracy (RLS one-step) + Constraint Satisfaction")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def window_metrics(df: pd.DataFrame) -> dict:
    """
    Per-window summary metrics INCLUDING:
      - mean actuator status
      - mean outdoor conditions
    """
    return {
        "steps": int(len(df)),
        "band_rate": float(df["within_optimal_band_01"].mean()),
        "energy_kwh_eq": float(df["energy_total_kwh_eq"].sum()),
        "elec_kwh": float(df["elec_total_kwh"].sum()),
        "gas_m3": float(df["gas_total_m3"].sum()),
        "pred_mae": float(df["pred_mae"].mean()),
        "pred_rmse": float(df["pred_rmse"].mean()),
        "reward_total": float(df["reward_env"].sum()),

        # ✅ actuator status means over the window
        "heater_status_rate": float(df["heater_status_01"].mean()),
        "fan_status_rate": float(df["fan_status_01"].mean()),
        "vents_status_rate": float(df["vents_status_01"].mean()),

        # ✅ outdoor mean values
        "Tout_C_mean": float(df["Tout_C"].mean()),
        "RHout_pct_mean": float(df["RHout_pct"].mean()),
    }


def main():
    cfg = EvalCfg()
    os.makedirs(cfg.out_dir, exist_ok=True)

    df = load_weather_csv(CSV_PATH)
    df_val = select_validation_slice(df)

    L = cfg.window_len_h
    S = cfg.stride_h
    starts = list(range(0, max(0, len(df_val) - L + 1), S))
    if not starts:
        raise ValueError(f"Validation slice too short for window_len_h={L}. len(df_val)={len(df_val)}")

    all_summaries = []
    all_steps = []

    for w_id, st in enumerate(starts):
        df_win = df_val.iloc[st:st + L].reset_index(drop=True)
        season = build_season_table(df_win)
        env = make_szleb_env_hourly(season)

        targets = get_target_bands_from_env(env)
        agent = MPCRLAgent.load(MODEL_PATH, action_space=env.action_space)
        agent.set_targets(targets)
        agent.freeze_learning()

        df_steps = run_window(env, agent, season, targets, window_id=w_id)

        if df_win["dt_iso_parsed"].notna().any():
            t0 = str(df_win["dt_iso_parsed"].iloc[0])
            t1 = str(df_win["dt_iso_parsed"].iloc[-1])
        else:
            t0 = str(int(df_win["dt"].iloc[0]))
            t1 = str(int(df_win["dt"].iloc[-1]))

        met = window_metrics(df_steps)
        met.update({
            "window_id": w_id,
            "start_index": int(st),
            "time_start": t0,
            "time_end": t1,
            "tin_opt_lo": float(targets.tin_opt_c[0]),
            "tin_opt_hi": float(targets.tin_opt_c[1]),
            "rh_opt_lo": float(targets.rh_opt_pct[0]),
            "rh_opt_hi": float(targets.rh_opt_pct[1]),
        })

        all_summaries.append(met)
        all_steps.append(df_steps)

        print(f"[WIN {w_id:03d}/{len(starts)-1:03d}] "
              f"band_rate={met['band_rate']:.3f} energy={met['energy_kwh_eq']:.2f}kWh_eq "
              f"heater_on={met['heater_status_rate']:.2f} fan_on={met['fan_status_rate']:.2f}")

    df_summary = pd.DataFrame(all_summaries)
    df_all = pd.concat(all_steps, ignore_index=True)

    # ✅ These CSVs now include Tout_C, RHout_pct, heater_status_01, fan_status_01 in the step file
    df_summary.to_csv(os.path.join(cfg.out_dir, "validation_windows_summary.csv"), index=False)
    df_all.to_csv(os.path.join(cfg.out_dir, "validation_steps_all_windows.csv"), index=False)

    # choose best/worst windows to plot
    if cfg.rank_metric == "band_rate":
        best = df_summary.sort_values("band_rate", ascending=False).head(cfg.save_best_k)
        worst = df_summary.sort_values("band_rate", ascending=True).head(cfg.save_worst_k)
    else:
        best = df_summary.sort_values("energy_kwh_eq", ascending=True).head(cfg.save_best_k)
        worst = df_summary.sort_values("energy_kwh_eq", ascending=False).head(cfg.save_worst_k)

    for tag, subset in [("best", best), ("worst", worst)]:
        for _, row in subset.iterrows():
            wid = int(row["window_id"])
            df_w = df_all[df_all["window_id"] == wid].copy()

            targets = TargetBands(
                tin_opt_c=(float(row["tin_opt_lo"]), float(row["tin_opt_hi"])),
                rh_opt_pct=(float(row["rh_opt_lo"]), float(row["rh_opt_hi"])),
                vpd_opt_kpa=None,
            )

            png = os.path.join(cfg.out_dir, f"{tag}_window_{wid:03d}.png")
            plot_window(df_w, targets, png)

    print("\nSaved:")
    print(" - validation_windows_summary.csv   (includes status rates + outdoor means)")
    print(" - validation_steps_all_windows.csv (includes heater/fan/vents status + Tout/RHout per step)")
    print(f" - {cfg.save_best_k} best plots + {cfg.save_worst_k} worst plots")
    print("All in:", cfg.out_dir)


if __name__ == "__main__":
    main()