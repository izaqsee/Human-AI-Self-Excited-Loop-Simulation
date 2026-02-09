#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase diagram utilities:
1) Extract the risk_sat_corr_high == 0 contour ("critical line") from phase_*.csv
2) Overlay policy-wise critical lines in a single figure
3) Compare "critical region" areas (by default: area where risk_sat_corr_high > 0)
   - also reports area where risk_sat_corr_high < 0
   - and a band area |risk_sat_corr_high| <= eps (near-critical)

Assumptions
- CSV contains at least: delta_risk, v_spike_p, risk_sat_corr_high
- If CSV has additional swept parameters, we aggregate them by mean over each (delta_risk, v_spike_p).

Usage examples (Windows PowerShell):
  python tools/phase_contours.py ^
    --empower phase_empower.csv ^
    --delegating phase_delegating.csv ^
    --feedback phase_feedback.csv ^
    --outdir out_phase ^
    --eps 0.05

Outputs
- out_phase/contour_overlay_risk_sat_corr0.png
- out_phase/contour_<policy>_risk_sat_corr0_points.csv  (polyline vertices)
- out_phase/area_summary.csv
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METRIC_DEFAULT = "risk_sat_corr_high"


@dataclass
class GridData:
    x: np.ndarray              # delta_risk (sorted unique)
    y: np.ndarray              # log10(v_spike_p) (sorted unique)
    Z: np.ndarray              # metric grid shape (len(y), len(x)) with NaN allowed


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _log10_safe(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if np.any(v <= 0):
        raise ValueError("v_spike_p must be > 0 to take log10.")
    return np.log10(v)


def _aggregate_to_grid(df: pd.DataFrame, metric: str) -> GridData:
    req = {"delta_risk", "v_spike_p", metric}
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Columns={list(df.columns)}")

    # Aggregate across other swept parameters by mean.
    g = (
        df.groupby(["delta_risk", "v_spike_p"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: "metric"})
    )

    x = np.sort(g["delta_risk"].unique().astype(float))
    y = np.sort(_log10_safe(g["v_spike_p"].unique().astype(float)))

    # Build grid with pivot
    g = g.assign(logv=_log10_safe(g["v_spike_p"].astype(float)))
    pivot = g.pivot(index="logv", columns="delta_risk", values="metric").reindex(index=y, columns=x)
    Z = pivot.to_numpy(dtype=float)

    return GridData(x=x, y=y, Z=Z)


def _bin_edges(coords: np.ndarray) -> np.ndarray:
    """
    Convert sorted center coordinates to bin edges (length n+1).
    Works for non-uniform spacing.
    """
    c = np.asarray(coords, dtype=float)
    if c.ndim != 1 or len(c) < 2:
        raise ValueError("Need at least 2 coordinates to compute bin edges.")
    mid = 0.5 * (c[1:] + c[:-1])
    edges = np.empty(len(c) + 1, dtype=float)
    edges[1:-1] = mid
    # Extrapolate ends
    edges[0] = c[0] - (mid[0] - c[0])
    edges[-1] = c[-1] + (c[-1] - mid[-1])
    return edges


def _area_by_sign(grid: GridData, eps: float = 0.05) -> Dict[str, float]:
    """
    Compute approximate area in (delta_risk, log10(v_spike_p)) space.
    - pos: Z > 0
    - neg: Z < 0
    - band: |Z| <= eps
    NaNs are ignored.
    """
    x_edges = _bin_edges(grid.x)
    y_edges = _bin_edges(grid.y)

    dx = np.diff(x_edges)                      # len(x)
    dy = np.diff(y_edges)                      # len(y)
    cell_area = dy[:, None] * dx[None, :]      # (len(y), len(x))

    Z = grid.Z
    valid = np.isfinite(Z)

    pos = valid & (Z > 0)
    neg = valid & (Z < 0)
    band = valid & (np.abs(Z) <= eps)

    return {
        "area_pos": float(np.sum(cell_area[pos])),
        "area_neg": float(np.sum(cell_area[neg])),
        "area_band": float(np.sum(cell_area[band])),
        "area_total_valid": float(np.sum(cell_area[valid])),
    }


def _extract_contour_points(ax, grid, level=0.0):
    """
    Extract contour polylines at given level.
    Accepts:
      - (X, Y, Z) tuple
      - GridData-like object:
          * X,Y,Z attributes (meshgrid)
          * or x,y,Z attributes where x/y are 1D axes -> meshgrid is constructed
      - dataclass GridData: uses __dataclass_fields__ keys to find candidates
    Compatible with old/new matplotlib (collections vs allsegs).
    """
    import numpy as np

    # -------- 1) tuple/list (X,Y,Z) --------
    if isinstance(grid, (tuple, list)) and len(grid) == 3:
        X, Y, Z = grid
    else:
        # -------- 2) gather field/attr names (dataclass-safe) --------
        attr_names = set(dir(grid))
        if hasattr(grid, "__dataclass_fields__"):
            attr_names |= set(grid.__dataclass_fields__.keys())

        def has(name: str) -> bool:
            return hasattr(grid, name)

        def get(name: str):
            return getattr(grid, name)

        # -------- 3) direct meshgrid fields --------
        candidates_xyz = [
            ("X", "Y", "Z"),
            ("x", "y", "z"),
            ("XX", "YY", "ZZ"),
            ("Xg", "Yg", "Zg"),
            ("X_grid", "Y_grid", "Z"),
        ]
        found = False
        for a, b, c in candidates_xyz:
            if has(a) and has(b) and has(c):
                X, Y, Z = get(a), get(b), get(c)
                found = True
                break

        # -------- 4) axis + Z (build meshgrid) --------
        if not found:
            # try likely axis names (case-insensitive)
            lower_map = {n.lower(): n for n in attr_names}

            def pick(*keys_lower):
                for k in keys_lower:
                    if k in lower_map:
                        return lower_map[k]
                return None

            z_name = pick("z", "zz", "zmat", "z_grid", "values", "val", "v")
            x_name = pick("x", "xs", "xvals", "x_values", "x_axis", "xaxis")
            y_name = pick("y", "ys", "yvals", "y_values", "y_axis", "yaxis")

            # You already have "Z" attribute visible, so prioritize that
            if has("Z"):
                z_name = "Z"

            if z_name and x_name and y_name and has(z_name) and has(x_name) and has(y_name):
                z = np.asarray(get(z_name))
                x = np.asarray(get(x_name))
                y = np.asarray(get(y_name))

                # If x,y are 1D axes -> meshgrid
                if x.ndim == 1 and y.ndim == 1:
                    X, Y = np.meshgrid(x, y, indexing="xy")
                    Z = z
                    found = True
                # If they are already meshgrids
                elif x.ndim == 2 and y.ndim == 2:
                    X, Y, Z = x, y, z
                    found = True

        if not found:
            # helpful debug: show dataclass field keys (full)
            fields = list(getattr(grid, "__dataclass_fields__", {}).keys())
            raise TypeError(
                "grid must be (X,Y,Z) or GridData with (X,Y,Z) or (x,y,Z). "
                f"Got type={type(grid)}. dataclass_fields={fields}"
            )

    cs = ax.contour(X, Y, Z, levels=[level])

    polylines = []

    # new matplotlib
    if hasattr(cs, "allsegs") and cs.allsegs:
        for level_segs in cs.allsegs:
            for seg in level_segs:
                if seg is not None and len(seg) >= 2:
                    polylines.append(seg)
    # old matplotlib
    elif hasattr(cs, "collections"):
        for col in cs.collections:
            for path in col.get_paths():
                v = path.vertices
                if v is not None and len(v) >= 2:
                    polylines.append(v)

    return polylines





def _save_polylines_csv(polylines: List[np.ndarray], out_csv: str) -> None:
    """
    Save contour polylines as a long table:
      poly_id, point_id, delta_risk, log10_v_spike_p
    """
    rows = []
    for i, poly in enumerate(polylines):
        for j, (x, y) in enumerate(poly):
            rows.append((i, j, float(x), float(y)))
    out = pd.DataFrame(rows, columns=["poly_id", "point_id", "delta_risk", "log10_v_spike_p"])
    out.to_csv(out_csv, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--empower", type=str, default=None, help="phase_empower.csv")
    ap.add_argument("--delegating", type=str, default=None, help="phase_delegating.csv")
    ap.add_argument("--feedback", type=str, default=None, help="phase_feedback.csv")
    ap.add_argument("--metric", type=str, default=METRIC_DEFAULT, help=f"metric column (default: {METRIC_DEFAULT})")
    ap.add_argument("--level", type=float, default=0.0, help="contour level (default: 0)")
    ap.add_argument("--eps", type=float, default=0.05, help="near-critical band: |metric| <= eps")
    ap.add_argument("--outdir", type=str, default="out_phase", help="output directory")
    ap.add_argument("--title", type=str, default="risk_sat_corr = 0 contour overlay", help="figure title")
    args = ap.parse_args()

    files = {k: getattr(args, k) for k in ["empower", "delegating", "feedback"] if getattr(args, k)}
    if not files:
        raise SystemExit("No input CSVs provided. Pass at least one of --empower/--delegating/--feedback")

    _ensure_dir(args.outdir)

    # Load + gridify
    grids: Dict[str, GridData] = {}
    for policy, path in files.items():
        df = pd.read_csv(path)
        grids[policy] = _aggregate_to_grid(df, args.metric)

    # Overlay plot
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(args.title)
    ax.set_xlabel("delta_risk")
    ax.set_ylabel("log10(v_spike_p)")

    # Optional: faint background of one policy (first) to orient
    first_policy = next(iter(grids.keys()))
    bg = grids[first_policy]
    Xb, Yb = np.meshgrid(bg.x, bg.y)
    im = ax.pcolormesh(Xb, Yb, bg.Z, shading="auto", alpha=0.25)
    fig.colorbar(im, ax=ax, label=args.metric)

    # Contours
    area_rows = []
    for policy, grid in grids.items():
        polylines = _extract_contour_points(ax, grid, level=args.level)

        # Style: different linestyle per policy (deterministic)
        style = {"empower": "-", "delegating": "--", "feedback": "-."}.get(policy, "-")
        # Redraw with style by plotting vertices
        for poly in polylines:
            ax.plot(poly[:, 0], poly[:, 1], linestyle=style, linewidth=2, label=f"{policy} (level={args.level})")

        out_csv = os.path.join(args.outdir, f"contour_{policy}_risk_sat_corr0_points.csv")
        _save_polylines_csv(polylines, out_csv)

        areas = _area_by_sign(grid, eps=args.eps)
        areas["policy"] = policy
        area_rows.append(areas)

    # De-duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        uniq.setdefault(l, h)
    ax.legend(list(uniq.values()), list(uniq.keys()), loc="best", framealpha=0.9)

    out_fig = os.path.join(args.outdir, "contour_overlay_risk_sat_corr0.png")
    fig.tight_layout()
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)

    # Area summary
    area_df = pd.DataFrame(area_rows).set_index("policy").sort_index()
    # Add normalized by total_valid for interpretability
    area_df["frac_pos"] = area_df["area_pos"] / area_df["area_total_valid"].replace(0, np.nan)
    area_df["frac_neg"] = area_df["area_neg"] / area_df["area_total_valid"].replace(0, np.nan)
    area_df["frac_band"] = area_df["area_band"] / area_df["area_total_valid"].replace(0, np.nan)
    out_area = os.path.join(args.outdir, "area_summary.csv")
    area_df.to_csv(out_area)

    print("Wrote:")
    print(" ", out_fig)
    print(" ", out_area)
    for policy in grids.keys():
        print(" ", os.path.join(args.outdir, f"contour_{policy}_risk_sat_corr0_points.csv"))
    print()
    print(area_df)


if __name__ == "__main__":
    main()
