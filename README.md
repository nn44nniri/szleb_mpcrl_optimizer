# Climate optimizer based on szleb and szleb-gym

## Summary

`szleb_mpcrl_optimizer` is a lightweight **MPC + Reinforcement Learning** controller for greenhouse climate control built around the **SZLEB-v0** Gym environment. The controller’s goal is to keep the plant’s **vital climate variables** (e.g., indoor temperature and relative humidity) **within the species-specific optimal ranges** configured inside SZLEB, while minimizing energy use. The approach follows the “MPC-based RL / MPCRL” philosophy: an MPC policy is executed online, and a small set of MPC parameters is adapted from data to reduce constraint violations under model mismatch and uncertain weather disturbances. [MALLICK2025100751](https://doi.org/10.1016/j.atech.2024.100751)


---

## Introduction

Greenhouse climate control is difficult because the real system is nonlinear and uncertain, and forecasts (e.g., weather) are imperfect. Model Predictive Control (MPC) offers strong constraint-handling and interpretability, but performance depends on having a good prediction model and correct constraint/cost tuning. Reinforcement Learning (RL) can learn from data, but pure RL can be sample-inefficient and harder to constrain.

This library combines both: **MPC remains the “policy”**, while a lightweight RL update improves a small set of parameters online/offline. This design is inspired by the MPCRL greenhouse work by Mallick et al., which learns MPC parametrization (model/cost/constraints) from data to improve climate performance under uncertainty. [MALLICK2025100751](https://doi.org/10.1016/j.atech.2024.100751)

However, instead of the policy of this paper, we have used the policy of Nauta, A et al. (2024) which is available in the szleb climate prediction library on my GitHub. This integration is a move towards a Trans-domain digital twin that I designed in my thesis. However, in the new approach, due to some problems that existed in the implementation of that thesis, instead of using a single optimizer, I plan to use two optimizers. In the previous approach, we used two domain-specific simulators managed by an optimizer and an estimator, we plan to use two optimizers and an estimator that have the optimizer and simulators embedded within themselves. 

[My thesis:Mansoorali Amiri (2025) Towards intelligent digital twins in agriculture in controlled environments: joint contributions in fruit detection by vision and trans-domain simulation.](https://doi.org/10.71781/310)



---

## Objectives

1. **Plant-safe climate:** Keep indoor climate variables in plant-optimal ranges (species-aware), using the ranges configured in the SZLEB environment.
2. **Minimum energy:** Reduce electricity and fuel (gas) usage while maintaining plant constraints.
3. **Robustness to disturbances:** Handle changing outdoor conditions (temperature/humidity/cloud cover proxies), including imperfect prediction.
4. **Edge-friendly computation:** Keep the optimizer small and fast (short horizons, simple model adaptation, lightweight learning updates).

---

## What’s “species-aware” in this project

The SZLEB-v0 environment stores plant optima (e.g., tomato temperature/humidity bands) in its environment configuration. During training and validation, the optimizer reads these bands from the environment rather than hard-coding ranges. When indoor climate leaves the optimal range, the environment returns a penalty signal (reward shaping).
This ensures **the same optimizer** can be used across plant species by changing only the SZLEB configuration.

---

## Formalism

### State, actions, disturbances

We use a compact controller state and disturbances:

* **State** (used by MPC + learning):

  * $$x_k = [Tin_k, RHin_k]^T$$
  * Extracted from SZLEB `info["current_row_output"]` (e.g., `Tin_final_C`, `RHin_final_pct`) rather than assuming a raw observation layout.

* **Action**:

  * $$u_k$$ is the actuator command vector (heater / fan / vents, depending on SZLEB-v0’s action space).

* **Disturbance / exogenous features**:

  * $$z_k = [Tout_k, RHout_k, Gsun_k, LAI_k]^T$$
  * In the OpenWeather-driven runs, $Gsun_k$ is a simple proxy derived from cloud cover; LAI can be a schedule or a simple ramp.

### Learned prediction model (fast online identification)

To keep computation low, the internal predictor is a small linear model learned online via Recursive Least Squares (RLS):

$$
x_{k+1} ≈ Θ · [ x_k ; u_k ; z_k ; 1 ]
$$

This model is not meant to be a perfect greenhouse physics model; it is a compact “control-oriented” predictor that MPC can use quickly.

### MPC problem (with soft constraints)

At each step, MPC computes a control sequence by minimizing weighted violations and energy, subject to actuator bounds and dynamics:

* **Soft constraints (slacks)** are used to keep the optimization feasible even when setpoints are temporarily unreachable:

$$
Tin_k ∈ [Tin_lo - sT_k, Tin_hi + sT_k],   sT_k ≥ 0
RHin_k ∈ [RH_lo - sRH_k, RH_hi + sRH_k], sRH_k ≥ 0
$$

* **Objective (illustrative)**:


$$min_{u_{0:H-1}, sT, sRH}  Σ_{k=0}^{H-1} [ wT * vio_Tin(x_k)^2 + wRH * vio_RH(x_k)^2 + wE * ||u_k||^2 + wST * sT_k^2 + wSRH * sRH_k^2 + wΔu * ||u_k - u_{k-1}||^2 ]$$



This matches the MPCRL spirit: MPC explicitly balances constraint satisfaction and resource efficiency, while remaining interpretable and constraint-aware. [MALLICK2025100751](https://doi.org/10.1016/j.atech.2024.100751)

### Lightweight “Rational RL-MPC” update (TD-style, small parameter vector)

Instead of training a large neural policy, we adapt a **small set of MPC weights** using a lightweight TD(0)-style update. Conceptually:

* Define a compact feature vector $φ_k$ (e.g., violation², energy, slack²).
* Define a simple value approximation $V(x_k) = θ^T φ_k$.
* Use TD error:

$$
δ_k = c_k + γ V(x_{k+1}) - V(x_k)
$$

* Update a small parameter vector (mapped to MPC weights):

$$
θ ← θ - α δ_k φ_k
$$

This keeps learning overhead minimal while still being grounded in RL/TD principles—more “rational” than ad-hoc weight nudging, and closer in intent to MPCRL-style adaptation. [MALLICK2025100751](https://doi.org/10.1016/j.atech.2024.100751)

---

## Library structure

```text
szleb_mpcrl_optimizer/
├── szleb_mpcrl/
│   ├── __init__.py
│   ├── envs.py            # SZLEB-v0 wrappers, target-band extraction, state extraction
│   ├── costs.py           # target bands + diagnostic costs + weight definitions
│   ├── rls.py             # online linear model identification (RLS)
│   ├── mpc.py             # MPC solve with soft constraints (slacks)
│   └── agent.py           # MPCRL agent: MPC policy + TD update + save/load
├── train_szleb_mpcrl_v1.py         # training + validation (tqdm progress)
├── evaluate_szleb_mpcrl_report.py  # rolling-window validation + CSV + plots
├── requirements.txt
└── README.md
```

---

## Dataset support (OpenWeather hourly CSV)


The dataset is related to the coordinates of the Rose greenhouse in Iran, Alvand Industrial Zone, collected from the site [openweathermap](https://openweathermap.org/) for the year 2022.

The training/validation scripts can load an hourly OpenWeather CSV (e.g., 2022) and map it to SZLEB inputs using only the relevant columns:

* `temp` → `t_out_c` (auto Kelvin→Celsius heuristic)
* `humidity` → `rh_out_pct`
* `clouds_all` → `g_sun_w_m2` (simple proxy)
* `wind_speed` kept for optional future extensions
* Data are fed as **1-hour rows** (`elapsed_s_per_row = 3600`).

---

## Results and outputs

### Training outputs

* Episode logs saved to CSV (reward totals, internal diagnostic costs, learned weights).
* Progress bars for:

  * dataset loading (console)
  * training episode progress (`tqdm`)
  * validation episode progress (`tqdm`)

### Validation outputs

`evaluate_szleb_mpcrl_report.py` runs rolling weekly windows over the validation slice and writes:

* `validation_windows_summary.csv`
  Per-window metrics including:

  * constraint satisfaction rate (“in optimal band”)
  * energy totals (electric + gas kWh-equivalent)
  * actuator ON-rate statistics (heater/fan/vents)
  * mean outdoor conditions (Tout/RHout)

* `validation_steps_all_windows.csv`
  Per-step traces including:

  * Tin/RHin, Tout/RHout
  * actuator activation levels and ON/OFF status
  * energy split and accuracy metrics (RLS one-step MAE/RMSE)

* Plots (`best_window_*.png`, `worst_window_*.png`) showing:

  1. indoor Tin/RHin with optimal bands
  2. outdoor Tout/RHout
  3. trigger activation + decisions
  4. trigger status (ON/OFF)
  5. energy split
  6. prediction accuracy + band satisfaction

---

## References

* Samuel Mallick, Filippo Airaldi, Azita Dabiri, Congcong Sun, Bart De Schutter. *Reinforcement learning-based model predictive control for greenhouse climate control*. Smart Agricultural Technology (2025), Vol. 10, 100751. ([TU Delft Research Portal][2])
* MPCRL greenhouse reference implementation (godspeed branch). ([GitHub][3])
* Preprint: *Reinforcement Learning-based Model Predictive Control for Greenhouse Climate Control* (arXiv). [MALLICK2025100751](https://doi.org/10.1016/j.atech.2024.100751)

