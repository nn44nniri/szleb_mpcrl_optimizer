from __future__ import annotations

import numpy as np
import pandas as pd

from szleb_mpcrl import make_szleb_env, MPCRLAgent
from szleb_mpcrl.envs import reset_env, step_env
from szleb_mpcrl.costs import Targets

GAS_KWH_PER_M3 = 10.5  # keep consistent with your costs.py

def build_eval_season_table(seed: int, n_days: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base_day = 100
    for i in range(n_days):
        # Example: randomized outside weather + sun; keep actuator cols present but unused since override=True
        t_out = float(rng.normal(loc=0.0, scale=8.0))
        rh_out = float(np.clip(rng.normal(loc=65.0, scale=10.0), 20.0, 100.0))
        g_sun = float(np.clip(rng.normal(loc=150.0, scale=60.0), 0.0, 800.0))
        lai = float(np.clip(1.0 + 0.03 * i + rng.normal(0, 0.05), 0.2, 6.0))
        rows.append({
            "day": base_day + i,
            "LAI": lai,
            "t_out_c": t_out,
            "t_in_c": 20.0,
            "rh_out_pct": rh_out,
            "rh_in_pct": 60.0,
            "g_sun_w_m2": g_sun,
        })
    return pd.DataFrame(rows)

def z_from_row(row: dict) -> np.ndarray:
    return np.array([
        float(row.get("t_out_c", 0.0)),
        float(row.get("rh_out_pct", 0.0)),
        float(row.get("g_sun_w_m2", 0.0)),
        float(row.get("LAI", 0.0)),
    ], dtype=float)

def clamp_violation(x: float, lo: float, hi: float) -> float:
    if x < lo: return lo - x
    if x > hi: return x - hi
    return 0.0

def evaluate_once(model_path: str, season_df: pd.DataFrame, targets: Targets):
    env = make_szleb_env(season_df)  # override=True in your wrapper
    obs, info = reset_env(env)
    obs_dim = int(np.asarray(obs).reshape(-1).size)

    agent = MPCRLAgent.load(model_path, action_space=env.action_space, obs_dim=obs_dim)
    agent.freeze_learning()  # policy fixed evaluation :contentReference[oaicite:4]{index=4}

    H = agent.cfg.mpc.horizon
    done = False
    t = 0

    # logs
    logs = []
    pred_errs = []  # (mae, rmse) per step

    while not done:
        # build z horizon (cheap “forecast” from table)
        z_seq = np.zeros((agent.nz, H), dtype=float)
        for k in range(H):
            row_i = min(t + k, len(season_df) - 1)
            z_seq[:, k] = z_from_row(season_df.iloc[row_i].to_dict())

        x = obs.reshape(-1)
        u = agent.act(x, z_seq)

        # one-step prediction accuracy (model vs true next obs)
        z_now = z_from_row(season_df.iloc[min(t, len(season_df)-1)].to_dict())
        x_pred = agent.rls.predict(x=x, u=u, z=z_now)

        obs2, reward, done, info2 = step_env(env, u)
        x_next = obs2.reshape(-1)

        err = x_next - x_pred
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        pred_errs.append((mae, rmse))

        row_out = info2.get("current_row_output", {})
        Tin = float(row_out.get("Tin_final_C", np.nan))
        RHin = float(row_out.get("RHin_final_pct", row_out.get("RH_in_pct", np.nan)))

        elec_kwh = float(row_out.get("total_elec_kwh", 0.0))
        gas_m3 = float(row_out.get("heater_gas_total_m3", 0.0))
        gas_kwh = GAS_KWH_PER_M3 * gas_m3

        t_v = clamp_violation(Tin, targets.t_min_c, targets.t_max_c) if np.isfinite(Tin) else 0.0
        rh_v = clamp_violation(RHin, targets.rh_min_pct, targets.rh_max_pct) if np.isfinite(RHin) else 0.0

        logs.append({
            "t": t,
            "day": row_out.get("day", None),
            "Tin_final_C": Tin,
            "RHin_final_pct": RHin,
            "temp_violation": t_v,
            "rh_violation": rh_v,
            "total_elec_kwh": elec_kwh,
            "heater_gas_total_m3": gas_m3,
            "gas_kwh_eq": gas_kwh,
            "reward": float(reward),
            "mae_pred": mae,
            "rmse_pred": rmse,
            **{f"u_{i}": float(u[i]) for i in range(u.size)},  # decisions
        })

        obs, info = obs2, info2
        t += 1

    df = pd.DataFrame(logs)

    # summary metrics
    constraint_ok = ((df["temp_violation"] == 0.0) & (df["rh_violation"] == 0.0)).mean()
    total_energy = float(df["total_elec_kwh"].sum() + df["gas_kwh_eq"].sum())
    avg_mae = float(df["mae_pred"].mean())
    avg_rmse = float(df["rmse_pred"].mean())
    total_return = float(df["reward"].sum())

    # action summary
    action_cols = [c for c in df.columns if c.startswith("u_")]
    action_summary = df[action_cols].agg(["mean", "min", "max"]).to_dict()

    report = {
        "steps": int(len(df)),
        "constraint_ok_rate": float(constraint_ok),
        "total_energy_kwh_eq": total_energy,
        "total_return": total_return,
        "pred_mae_avg": avg_mae,
        "pred_rmse_avg": avg_rmse,
        "action_summary": action_summary,
    }
    return df, report

def main():
    model_path = "szleb_mpcrl_trained.pkl"
    targets = Targets()

    # run multiple evaluation episodes like godspeed (episodes loop + stored data) :contentReference[oaicite:5]{index=5}
    N = 20
    reports = []
    all_dfs = []

    for ep in range(N):
        season_df = build_eval_season_table(seed=1000 + ep, n_days=30)
        df, rep = evaluate_once(model_path, season_df, targets)
        rep["episode"] = ep
        reports.append(rep)
        df["episode"] = ep
        all_dfs.append(df)

        print(f"Episode {ep}: constraint_ok_rate={rep['constraint_ok_rate']:.3f}, "
              f"energy_kwh_eq={rep['total_energy_kwh_eq']:.2f}, return={rep['total_return']:.2f}, "
              f"pred_mae={rep['pred_mae_avg']:.4f}")

    details = pd.concat(all_dfs, ignore_index=True)
    summary = pd.DataFrame(reports)

    details.to_csv("eval_details_by_step.csv", index=False)
    summary.to_csv("eval_report_by_episode.csv", index=False)

    print("\n=== EVALUATION REPORT (mean over episodes) ===")
    print({
        "constraint_ok_rate_mean": float(summary["constraint_ok_rate"].mean()),
        "constraint_ok_rate_std": float(summary["constraint_ok_rate"].std(ddof=1)),
        "energy_kwh_eq_mean": float(summary["total_energy_kwh_eq"].mean()),
        "energy_kwh_eq_std": float(summary["total_energy_kwh_eq"].std(ddof=1)),
        "return_mean": float(summary["total_return"].mean()),
        "pred_mae_mean": float(summary["pred_mae_avg"].mean()),
        "pred_rmse_mean": float(summary["pred_rmse_avg"].mean()),
    })
    print("\nSaved: eval_report_by_episode.csv, eval_details_by_step.csv")

if __name__ == "__main__":
    main()