"""
Human–AI Disempowerment Minimal Model Simulator
================================================

予想される動作（重要）
- policy=delegating (u高固定)：
  - 短期満足Sは高く出やすい一方、A（自律）が徐々に低下し、R（委任）が上昇しやすい
  - 結果として severe（A低×R高×D高）イベントが増える
  - Risk と Satisfaction の相関が正になりやすい（C3を満たしやすい）
- policy=empower (u低固定)：
  - Sは控えめになりやすいが、Aが維持/回復しRが抑えられる
  - severeイベントが減りやすい（C1を満たしやすい）
  - C3（高リスクほど高評価）は弱まる/消える可能性が高い
- policy=feedback (状態フィードバック)：
  - パラメータ次第で「Sをあまり落とさずにAを維持」できる領域が出る
  - C1（稀）とC3（評価逆転）を同時に満たす “境界近傍” を探すのが目的

このスクリプトの目的
- Anthropic由来の拘束条件（C1〜C3）を「再現ターゲット」として固定し、
  最小モデルのパラメータ空間から、同時に満たす領域を探索する。
  C1: severe率が 1e-3〜1e-4 オーダー
  C2: 高負荷ドメインで severe率が上がる
  C3: Risk と Satisfaction の相関が正

実行例
- 単発実行（3ポリシー比較）:
  python simulate_disempowerment.py run --T 300 --N 2000 --seed 0
- パラメータスイープ（粗探索）:
  python simulate_disempowerment.py sweep --T 200 --N 1500 --seed 0 --out results.csv
- CSVから上位候補を表示:
  python simulate_disempowerment.py top --csv results.csv --k 20

依存
- numpy, pandas, matplotlib, numba
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
import time

# ----------------------------
# Numba Setup
# ----------------------------
try:
    from numba import njit
except ImportError:
    print("Warning: numba not found. Simulation will be slow.")
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco

# ----------------------------
# Utilities
# ----------------------------

def sigmoid_scalar(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def clip01(x: np.ndarray | float) -> np.ndarray | float:
    return np.minimum(1.0, np.maximum(0.0, x))

# JIT-compatible helpers
@njit(cache=True, fastmath=True)
def _clip01(x: float) -> float:
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return x

@njit(cache=True, fastmath=True)
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------
# Model parameters
# ----------------------------

@dataclass(frozen=True)
class Params:
    # Satisfaction model
    alpha_u: float = 2.0
    beta_r: float = 1.0
    gamma_d: float = 1.2
    delta_risk: float = 1.5

    # Reliance update
    k_r: float = 0.08
    k_d: float = 0.05
    lam_a: float = 0.6

    # Autonomy update
    k_a: float = 0.02
    k_b: float = 0.06

    # Risk proxy
    w_risk_A: float = 1.0
    w_risk_R: float = 1.0
    w_risk_D: float = 1.0
    risk_power: float = 1.0

    # Thresholds for severe (C1)
    th_A: float = 0.25
    th_R: float = 0.75
    th_D: float = 0.75

    # Vulnerability
    v_base: float = 0.10
    v_sigma: float = 0.05
    v_spike_p: float = 0.01
    v_spike_amp: float = 0.70
    th_V: float = 0.80

    # Success experience (proxy)
    eta0: float = 0.0
    etaA: float = 2.0
    etaD: float = 2.0
    etaU: float = 0.5


@dataclass(frozen=True)
class Policy:
    name: str
    u_fixed: Optional[float] = None
    u0: float = 0.50
    cA: float = 0.60
    dR: float = 0.40

    def compute_u(self, A: float, R: float) -> float:
        if self.u_fixed is not None:
            return float(clip01(self.u_fixed))
        u = self.u0 + self.cA * (1.0 - A) - self.dR * R
        return float(clip01(u))


@dataclass(frozen=True)
class Domain:
    name: str
    mu: float
    sigma: float


# ----------------------------
# Optimized Simulation Core (JIT)
# ----------------------------

@njit(cache=True, fastmath=True)
def simulate_fast_jit(
    N: int,
    T: int,
    seed: int,
    
    # Domain
    D_mu: float,
    D_sigma: float,
    
    # Policy
    u_fixed_val: float, # -1.0 if None
    u0: float,
    cA: float,
    dR: float,
    
    # Params (unpacked)
    alpha_u: float, beta_r: float, gamma_d: float, delta_risk: float,
    k_r: float, k_d: float, lam_a: float,
    k_a: float, k_b: float,
    wA: float, wR: float, wD: float, risk_power: float,
    th_A: float, th_R: float, th_D: float,
    v_base: float, v_sigma: float, v_spike_p: float, v_spike_amp: float, th_V: float,
    eta0: float, etaA: float, etaD: float, etaU: float,
    
    # Init
    A0_mean: float, R0_mean: float, A0_sd: float, R0_sd: float
):
    # ★FIX: Use standard numpy random functions for Numba compatibility
    np.random.seed(seed)
    
    # Stats accumulators
    severe_count = 0
    n_tot = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0
    sum_xy = 0.0
    
    A_end_sum = 0.0
    R_end_sum = 0.0

    for i in range(N):
        # Init agent
        A = _clip01(A0_mean + A0_sd * np.random.standard_normal())
        R = _clip01(R0_mean + R0_sd * np.random.standard_normal())
        
        # Trajectory
        for t in range(T):
            # Domain
            D = _clip01(D_mu + D_sigma * np.random.standard_normal())
            
            # Vulnerability
            v = v_base + v_sigma * np.random.standard_normal()
            if np.random.random() < v_spike_p:
                v += v_spike_amp
            V = _clip01(v)
            
            # Policy
            if u_fixed_val >= 0.0:
                U = _clip01(u_fixed_val)
            else:
                U = _clip01(u0 + cA * (1.0 - A) - dR * R)
            
            # Risk
            base_risk = wA * (1.0 - A) * wR * R * wD * D
            base_risk = _clip01(base_risk)
            if risk_power != 1.0:
                base_risk = _clip01(base_risk ** risk_power)
            
            # Satisfaction
            S = _sigmoid(alpha_u * U + beta_r * R + delta_risk * base_risk - gamma_d * D)
            
            # Severe check
            if (A < th_A) and (R > th_R) and (D > th_D) and (V > th_V):
                severe_count += 1
            
            # Accumulate stats (Risk vs S)
            x = base_risk
            y = S
            sum_x += x
            sum_y += y
            sum_x2 += x*x
            sum_y2 += y*y
            sum_xy += x*y
            n_tot += 1
            
            # Update R
            R = _clip01(R + k_r * U + lam_a * (1.0 - A) * D - k_d * (1.0 - U))
            
            # Success Experience
            E = _sigmoid(eta0 + etaA * A - etaD * D - etaU * U)
            
            # Update A
            erode = k_b * U * R
            recover = k_a * E
            A = _clip01(A + recover - erode)
            
        A_end_sum += A
        R_end_sum += R
        
    # Finalize stats
    mean_x = sum_x / max(n_tot, 1)
    mean_y = sum_y / max(n_tot, 1)
    
    exy = sum_xy / max(n_tot, 1)
    ex2 = sum_x2 / max(n_tot, 1)
    ey2 = sum_y2 / max(n_tot, 1)
    
    cov = exy - mean_x * mean_y
    varx = ex2 - mean_x * mean_x
    vary = ey2 - mean_y * mean_y
    
    corr = 0.0
    if varx > 1e-12 and vary > 1e-12:
        corr = cov / np.sqrt(varx * vary)
        
    severe_rate = severe_count / max(n_tot, 1)
    A_end_mean = A_end_sum / max(N, 1)
    R_end_mean = R_end_sum / max(N, 1)
    
    return severe_rate, corr, mean_x, mean_y, A_end_mean, R_end_mean


# ----------------------------
# Legacy Simulation (for full trajectory tracking)
# ----------------------------

def simulate_trajectory(
    T: int,
    params: Params,
    policy: Policy,
    domain: Domain,
    rng: np.random.Generator,
    A0: float = 0.8,
    R0: float = 0.2,
) -> Dict[str, np.ndarray]:
    A = np.zeros(T, dtype=np.float32)
    R = np.zeros(T, dtype=np.float32)
    D = np.zeros(T, dtype=np.float32)
    U = np.zeros(T, dtype=np.float32)
    S = np.zeros(T, dtype=np.float32)
    Risk = np.zeros(T, dtype=np.float32)
    V = np.zeros(T, dtype=np.float32)
    E = np.zeros(T, dtype=np.float32)
    Severe = np.zeros(T, dtype=int)
    R_crit = np.zeros(T, dtype=np.float32)

    A[0] = A0
    R[0] = R0

    for t in range(T - 1):
        D[t] = float(clip01(domain.mu + domain.sigma * rng.standard_normal()))
        v = params.v_base + params.v_sigma * rng.standard_normal()
        if rng.random() < params.v_spike_p:
            v += params.v_spike_amp
        V[t] = float(clip01(v))

        U[t] = policy.compute_u(A[t], R[t])

        base_risk = (
            params.w_risk_A * (1.0 - A[t]) *
            params.w_risk_R * (R[t]) *
            params.w_risk_D * (D[t])
        )
        base_risk = float(clip01(base_risk))
        if params.risk_power != 1.0:
            base_risk = float(clip01(base_risk ** params.risk_power))
        Risk[t] = base_risk

        S[t] = sigmoid_scalar(
            params.alpha_u * U[t]
            + params.beta_r * R[t]
            + params.delta_risk * Risk[t]
            - params.gamma_d * D[t]
        )

        Severe[t] = int(
            (A[t] < params.th_A)
            and (R[t] > params.th_R)
            and (D[t] > params.th_D)
            and (V[t] > params.th_V)
        )

        R[t + 1] = float(clip01(
            R[t]
            + params.k_r * U[t]
            + params.lam_a * (1.0 - A[t]) * D[t]
            - params.k_d * (1.0 - U[t])
        ))

        E_t = sigmoid_scalar(
            params.eta0
            + params.etaA * A[t]
            - params.etaD * D[t] 
            - params.etaU * U[t]
        )
        E[t] = float(E_t)

        erode = params.k_b * U[t] * R[t]
        recover = params.k_a * E[t]
        A[t + 1] = float(clip01(A[t] + recover - erode))

        if params.k_b > 1e-12:
            R_crit[t] = float(clip01((params.k_a * E_t) / params.k_b))
        else:
            R_crit[t] = 0.0

    D[-1] = D[-2]; U[-1] = U[-2]; S[-1] = S[-2]
    Risk[-1] = Risk[-2]; V[-1] = V[-2]; E[-1] = E[-2]
    Severe[-1] = Severe[-2]; R_crit[-1] = R_crit[-2]

    return {
        "A": A, "R": R, "D": D, "U": U, "S": S, "Risk": Risk, "V": V, "E": E,
        "Severe": Severe, "R_crit": R_crit
    }


def simulate_population(
    N: int,
    T: int,
    params: Params,
    policy: Policy,
    domain: Domain,
    seed: int,
    A0_mean: float = 0.8,
    R0_mean: float = 0.2,
    A0_sd: float = 0.05,
    R0_sd: float = 0.05,
    return_samples: bool = False,
    sample_cap: int = 20000,
) -> Dict[str, np.ndarray]:
    
    # ---------------------------------------------------------
    # FAST PATH (JIT) - Used during sweep when samples are not needed
    # ---------------------------------------------------------
    if not return_samples:
        u_fixed_val = policy.u_fixed if policy.u_fixed is not None else -1.0
        
        # Unpack params for JIT
        p = params
        
        severe_rate, corr, mean_x, mean_y, A_end_m, R_end_m = simulate_fast_jit(
            N, T, seed,
            domain.mu, domain.sigma,
            u_fixed_val, policy.u0, policy.cA, policy.dR,
            p.alpha_u, p.beta_r, p.gamma_d, p.delta_risk,
            p.k_r, p.k_d, p.lam_a,
            p.k_a, p.k_b,
            p.w_risk_A, p.w_risk_R, p.w_risk_D, p.risk_power,
            p.th_A, p.th_R, p.th_D,
            p.v_base, p.v_sigma, p.v_spike_p, p.v_spike_amp, p.th_V,
            p.eta0, p.etaA, p.etaD, p.etaU,
            A0_mean, R0_mean, A0_sd, R0_sd
        )
        
        return {
            "severe_rate": np.array([severe_rate], dtype=float),
            "risk_sat_corr": np.array([corr], dtype=float),
            "risk_mean": np.array([mean_x], dtype=float),
            "sat_mean": np.array([mean_y], dtype=float),
            "A_end_mean": np.array([A_end_m], dtype=float),
            "R_end_mean": np.array([R_end_m], dtype=float),
        }

    # ---------------------------------------------------------
    # SLOW PATH (Python) - Used when samples are needed for plotting
    # ---------------------------------------------------------
    rng = np.random.default_rng(seed)
    n_tot = 0
    sum_x = 0.0; sum_y = 0.0
    sum_x2 = 0.0; sum_y2 = 0.0; sum_xy = 0.0
    severe_count = 0
    A_end = np.zeros(N, dtype=np.float32)
    R_end = np.zeros(N, dtype=np.float32)
    risk_buf = []; sat_buf = []

    for i in range(N):
        A0 = float(clip01(A0_mean + A0_sd * rng.standard_normal()))
        R0 = float(clip01(R0_mean + R0_sd * rng.standard_normal()))
        traj = simulate_trajectory(T=T, params=params, policy=policy, domain=domain, rng=rng, A0=A0, R0=R0)

        x = traj["Risk"]; y = traj["S"]; sev = traj["Severe"]
        severe_count += int(np.sum(sev))
        n = x.size; n_tot += n
        
        sx = float(np.sum(x)); sy = float(np.sum(y))
        sum_x += sx; sum_y += sy
        sum_x2 += float(np.sum(x * x)); sum_y2 += float(np.sum(y * y)); sum_xy += float(np.sum(x * y))
        A_end[i] = traj["A"][-1]; R_end[i] = traj["R"][-1]

        if return_samples and len(risk_buf) < sample_cap:
            remain = sample_cap - len(risk_buf)
            if remain > 0:
                take = min(remain, n)
                risk_buf.extend(x[:take].tolist())
                sat_buf.extend(y[:take].tolist())

    mean_x = sum_x / max(n_tot, 1)
    mean_y = sum_y / max(n_tot, 1)
    exy = sum_xy / max(n_tot, 1); ex2 = sum_x2 / max(n_tot, 1); ey2 = sum_y2 / max(n_tot, 1)
    cov = exy - mean_x * mean_y
    varx = ex2 - mean_x * mean_x
    vary = ey2 - mean_y * mean_y
    corr = 0.0
    if varx > 1e-12 and vary > 1e-12:
        corr = cov / math.sqrt(varx * vary)

    out = {
        "severe_rate": np.array([severe_count / max(n_tot, 1)], dtype=float),
        "risk_sat_corr": np.array([float(corr)], dtype=float),
        "risk_mean": np.array([float(mean_x)], dtype=float),
        "sat_mean": np.array([float(mean_y)], dtype=float),
        "A_end_mean": np.array([float(np.mean(A_end))], dtype=float),
        "R_end_mean": np.array([float(np.mean(R_end))], dtype=float),
    }
    if return_samples:
        out["risk_all"] = np.array(risk_buf, dtype=float)
        out["sat_all"] = np.array(sat_buf, dtype=float)
    return out


# ----------------------------
# Constraint checks
# ----------------------------

def check_constraints(
    severe_rate_low: float,
    severe_rate_high: float,
    corr_risk_sat: float,
    target_low: float = 1e-4,
    target_high: float = 1e-3,
) -> Dict[str, bool]:
    c1 = (target_low <= severe_rate_high <= target_high)
    c2 = (severe_rate_high > severe_rate_low)
    c3 = (corr_risk_sat > 0.0)
    return {"C1": c1, "C2": c2, "C3": c3, "ALL": (c1 and c2 and c3)}


# ----------------------------
# Plotting
# ----------------------------

def plot_scatter_risk_vs_sat(risk: np.ndarray, sat: np.ndarray, title: str, outpath: Optional[str] = None):
    plt.figure()
    n = risk.size
    if n > 20000:
        idx = np.random.default_rng(0).choice(n, size=20000, replace=False)
        risk = risk[idx]
        sat = sat[idx]
    plt.scatter(risk, sat, s=2, alpha=0.3)
    plt.xlabel("Risk (disempowerment potential proxy)")
    plt.ylabel("Satisfaction (short-term proxy)")
    plt.title(title)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
    else:
        plt.show()
    plt.close()


# ----------------------------
# CLI Commands
# ----------------------------

def cmd_run(args: argparse.Namespace):
    dom_low = Domain("low", mu=args.D_low_mu, sigma=args.D_low_sigma)
    dom_high = Domain("high", mu=args.D_high_mu, sigma=args.D_high_sigma)

    policies = [
        Policy("delegating", u_fixed=args.u_delegating),
        Policy("empower", u_fixed=args.u_empower),
        Policy("feedback", u_fixed=None, u0=args.u0, cA=args.cA, dR=args.dR),
    ]

    params = Params(
        alpha_u=args.alpha_u, beta_r=args.beta_r, gamma_d=args.gamma_d,
        k_r=args.k_r, k_d=args.k_d, lam_a=args.lam_a,
        k_a=args.k_a, k_b=args.k_b,
        w_risk_A=args.wA, w_risk_R=args.wR, w_risk_D=args.wD, risk_power=args.risk_power,
        th_A=args.th_A, th_R=args.th_R, th_D=args.th_D,
    )

    rows = []
    for pol in policies:
        # Check if we need plotting samples
        need_samples = bool(args.plot and (pol.name == args.plot_policy))
        
        pop_low = simulate_population(
            N=args.N, T=args.T, params=params, policy=pol, domain=dom_low, seed=args.seed
        )

        pop_high = simulate_population(
            N=args.N, T=args.T, params=params, policy=pol, domain=dom_high, seed=args.seed + 1,
            return_samples=need_samples,
            sample_cap=20000
        )

        severe_low = float(pop_low["severe_rate"][0])
        severe_high = float(pop_high["severe_rate"][0])
        corr_val = float(pop_high["risk_sat_corr"][0])

        checks = check_constraints(severe_low, severe_high, corr_val)

        row = {
            "policy": pol.name,
            "severe_rate_low": severe_low,
            "severe_rate_high": severe_high,
            "risk_sat_corr_high": corr_val,
            "sat_mean_high": float(pop_high["sat_mean"][0]),
            "risk_mean_high": float(pop_high["risk_mean"][0]),
            "A_end_mean_high": float(pop_high["A_end_mean"][0]),
            "R_end_mean_high": float(pop_high["R_end_mean"][0]),
            **checks,
        }
        rows.append(row)

        if need_samples:
            plot_scatter_risk_vs_sat(
                pop_high["risk_all"], pop_high["sat_all"],
                title=f"Risk vs Satisfaction ({pol.name}, domain=high)",
                outpath=args.plot_out
            )
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

def sweep_worker(args_tuple):
    (
        pol, dom_low, dom_high,
        params,
        N, T, seed,
        C1_low, C1_high
    ) = args_tuple

    # These calls will now use the fast JIT path internally because return_samples=False
    pop_low = simulate_population(
        N=N, T=T, params=params,
        policy=pol, domain=dom_low,
        seed=seed
    )

    pop_high = simulate_population(
        N=N, T=T, params=params,
        policy=pol, domain=dom_high,
        seed=seed + 1
    )

    severe_low = float(pop_low["severe_rate"][0])
    severe_high = float(pop_high["severe_rate"][0])
    corr_val = float(pop_high["risk_sat_corr"][0])

    checks = check_constraints(
        severe_rate_low=severe_low,
        severe_rate_high=severe_high,
        corr_risk_sat=corr_val,
        target_low=C1_low,
        target_high=C1_high,
    )

    return {
        "policy": pol.name,
        "alpha_u": params.alpha_u,
        "k_b": params.k_b,
        "k_a": params.k_a,
        "k_r": params.k_r,
        "th_A": params.th_A,
        "severe_rate_low": severe_low,
        "severe_rate_high": severe_high,
        "risk_sat_corr_high": corr_val,
        "sat_mean_high": float(pop_high["sat_mean"][0]),
        "risk_mean_high": float(pop_high["risk_mean"][0]),
        "A_end_mean_high": float(pop_high["A_end_mean"][0]),
        "R_end_mean_high": float(pop_high["R_end_mean"][0]),
        "delta_risk": params.delta_risk,
        "v_spike_p": params.v_spike_p,
        **checks,
    }


def cmd_sweep(args: argparse.Namespace):
    dom_low = Domain("low", mu=args.D_low_mu, sigma=args.D_low_sigma)
    dom_high = Domain("high", mu=args.D_high_mu, sigma=args.D_high_sigma)

    if args.nproc is not None:
        nproc = args.nproc
    else:
        nproc = max(1, mp.cpu_count() - 1)
    
    if args.policy == "delegating":
        pol = Policy("delegating", u_fixed=args.u_delegating)
    elif args.policy == "empower":
        pol = Policy("empower", u_fixed=args.u_empower)
    elif args.policy == "feedback":
        pol = Policy("feedback", u_fixed=None, u0=args.u0, cA=args.cA, dR=args.dR)
    else:
        raise ValueError("Unknown policy")

    grid_alpha_u = np.linspace(args.alpha_u_min, args.alpha_u_max, args.alpha_u_steps)
    grid_kb = np.linspace(args.k_b_min, args.k_b_max, args.k_b_steps)
    grid_ka = np.linspace(args.k_a_min, args.k_a_max, args.k_a_steps)
    grid_kr = np.linspace(args.k_r_min, args.k_r_max, args.k_r_steps)
    grid_delta = np.linspace(args.delta_risk_min, args.delta_risk_max, args.delta_risk_steps)
    grid_vsp   = np.geomspace(args.v_spike_p_min, args.v_spike_p_max, args.v_spike_p_steps)
    grid_thA = np.linspace(args.th_A_min, args.th_A_max, args.th_A_steps)

    tasks = []
    for delta_risk in grid_delta:
        for v_spike_p in grid_vsp:
            for alpha_u in grid_alpha_u:
                for k_b in grid_kb:
                    for k_a in grid_ka:
                        for k_r in grid_kr:
                            for th_A in grid_thA:
                                params = Params(
                                    alpha_u=float(alpha_u), beta_r=args.beta_r, gamma_d=args.gamma_d,
                                    k_r=float(k_r), k_d=args.k_d, lam_a=args.lam_a,
                                    k_a=float(k_a), k_b=float(k_b),
                                    w_risk_A=args.wA, w_risk_R=args.wR, w_risk_D=args.wD,
                                    risk_power=args.risk_power,
                                    th_A=float(th_A), th_R=args.th_R, th_D=args.th_D,
                                    delta_risk=float(delta_risk),
                                    v_spike_p=float(v_spike_p)
                                )

                                tasks.append((
                                    pol, dom_low, dom_high,
                                    params,
                                    args.N, args.T, args.seed,
                                    args.C1_low, args.C1_high
                                ))

    total = len(tasks)
    print(f"Total conditions: {total}")
    print(f"Using {nproc} processes")

    rows = []
    start_time = time.time()

    with mp.Pool(processes=nproc) as pool:
        it = pool.imap_unordered(sweep_worker, tasks, chunksize=args.chunksize)
        for res in tqdm(it, total=total, desc="sweep", unit="cond"):
            rows.append(res)

    elapsed = time.time() - start_time
    print(f"\nDone. elapsed={elapsed:.1f}s, rows={len(rows)}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"saved: {args.out}")

    if "ALL" in df.columns:
        passed = df[df["ALL"] == True]
        print(f"total rows: {len(df)}; passed ALL: {len(passed)}")
    else:
        print(f"total rows: {len(df)}")


def cmd_top(args: argparse.Namespace):
    df = pd.read_csv(args.csv)
    if "ALL" in df.columns:
        df2 = df[df["ALL"] == True].copy()
    else:
        df2 = df.copy()

    if len(df2) == 0:
        print("No rows passed ALL (or CSV has no ALL). Showing closest to target instead.")
        df2 = df.copy()

    mid = 0.5 * (args.C1_low + args.C1_high)
    if "severe_rate_high" in df2.columns and "risk_sat_corr_high" in df2.columns:
        df2["score"] = -np.abs(df2["severe_rate_high"] - mid) + 0.1 * df2["risk_sat_corr_high"]
        df2 = df2.sort_values("score", ascending=False)

    print(df2.head(args.k).to_string(index=False))


# ----------------------------
# Main / Args
# ----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal Human–AI disempowerment simulator (Anthropic-constrained)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(pp: argparse.ArgumentParser):
        pp.add_argument("--T", type=int, default=300, help="timesteps per trajectory")
        pp.add_argument("--N", type=int, default=2000, help="population size")
        pp.add_argument("--seed", type=int, default=0, help="random seed")

        # Domains
        pp.add_argument("--D_low_mu", type=float, default=0.25)
        pp.add_argument("--D_low_sigma", type=float, default=0.12)
        pp.add_argument("--D_high_mu", type=float, default=0.55)
        pp.add_argument("--D_high_sigma", type=float, default=0.18)

        # Policy params
        pp.add_argument("--u_delegating", type=float, default=0.9)
        pp.add_argument("--u_empower", type=float, default=0.2)
        pp.add_argument("--u0", type=float, default=0.5)
        pp.add_argument("--cA", type=float, default=0.6)
        pp.add_argument("--dR", type=float, default=0.4)

        # Model params
        pp.add_argument("--alpha_u", type=float, default=2.0)
        pp.add_argument("--beta_r", type=float, default=1.0)
        pp.add_argument("--gamma_d", type=float, default=1.2)

        pp.add_argument("--k_r", type=float, default=0.08)
        pp.add_argument("--k_d", type=float, default=0.05)
        pp.add_argument("--lam_a", type=float, default=0.6)

        pp.add_argument("--k_a", type=float, default=0.02)
        pp.add_argument("--k_b", type=float, default=0.06)

        # Risk proxy
        pp.add_argument("--wA", type=float, default=1.0)
        pp.add_argument("--wR", type=float, default=1.0)
        pp.add_argument("--wD", type=float, default=1.0)
        pp.add_argument("--risk_power", type=float, default=1.0)

        # Severe thresholds
        pp.add_argument("--th_A", type=float, default=0.25)
        pp.add_argument("--th_R", type=float, default=0.75)
        pp.add_argument("--th_D", type=float, default=0.75)

        # delta_risk
        pp.add_argument("--delta_risk", type=float, default=1.5)

        # vulnerability params
        pp.add_argument("--v_base", type=float, default=0.10)
        pp.add_argument("--v_sigma", type=float, default=0.05)
        pp.add_argument("--v_spike_p", type=float, default=0.01)
        pp.add_argument("--v_spike_amp", type=float, default=0.70)
        pp.add_argument("--th_V", type=float, default=0.80)

        pp.add_argument("--nproc", type=int, default=None,
                      help="number of processes for parallel sweep")

    # run
    pr = sub.add_parser("run", help="run fixed comparison across policies")
    add_common(pr)
    pr.add_argument("--plot", action="store_true", help="save a scatter plot Risk vs Satisfaction (high domain)")
    pr.add_argument("--plot_policy", type=str, default="delegating", choices=["delegating","empower","feedback"])
    pr.add_argument("--plot_out", type=str, default="risk_vs_satisfaction.png")
    pr.set_defaults(func=cmd_run)

    # sweep
    ps = sub.add_parser("sweep", help="coarse parameter sweep to find regions satisfying C1-C3")
    add_common(ps)
    ps.add_argument("--policy", type=str, default="feedback", choices=["delegating","empower","feedback"])
    ps.add_argument("--out", type=str, default="results.csv")
    ps.add_argument("--verbose", action="store_true")
    ps.add_argument("--chunksize", type=int, default=10, help="multiprocessing chunksize")


    # delta_risk grid
    ps.add_argument("--delta_risk_min", type=float, default=0.0)
    ps.add_argument("--delta_risk_max", type=float, default=2.5)
    ps.add_argument("--delta_risk_steps", type=int, default=11)

    # v_spike_p grid
    ps.add_argument("--v_spike_p_min", type=float, default=1e-4)
    ps.add_argument("--v_spike_p_max", type=float, default=2e-2)
    ps.add_argument("--v_spike_p_steps", type=int, default=11)

    # C1 targets (Anthropic order)
    ps.add_argument("--C1_low", type=float, default=1e-4)
    ps.add_argument("--C1_high", type=float, default=1e-3)

    # grids
    ps.add_argument("--alpha_u_min", type=float, default=1.0)
    ps.add_argument("--alpha_u_max", type=float, default=3.0)
    ps.add_argument("--alpha_u_steps", type=int, default=5)

    ps.add_argument("--k_b_min", type=float, default=0.03)
    ps.add_argument("--k_b_max", type=float, default=0.10)
    ps.add_argument("--k_b_steps", type=int, default=5)

    ps.add_argument("--k_a_min", type=float, default=0.005)
    ps.add_argument("--k_a_max", type=float, default=0.05)
    ps.add_argument("--k_a_steps", type=int, default=5)

    ps.add_argument("--k_r_min", type=float, default=0.03)
    ps.add_argument("--k_r_max", type=float, default=0.12)
    ps.add_argument("--k_r_steps", type=int, default=5)

    ps.add_argument("--th_A_min", type=float, default=0.15)
    ps.add_argument("--th_A_max", type=float, default=0.35)
    ps.add_argument("--th_A_steps", type=int, default=5)

    ps.set_defaults(func=cmd_sweep)

    # top
    pt = sub.add_parser("top", help="show top-k candidates from sweep CSV")
    pt.add_argument("--csv", type=str, required=True)
    pt.add_argument("--k", type=int, default=20)
    pt.add_argument("--C1_low", type=float, default=1e-4)
    pt.add_argument("--C1_high", type=float, default=1e-3)
    pt.set_defaults(func=cmd_top)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()