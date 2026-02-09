import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors


def to_bool_series(s: pd.Series) -> pd.Series:
    """Robust bool parsing: True/False, 1/0, 'true'/'false' etc."""
    if s.dtype == bool:
        return s
    if np.issubdtype(s.dtype, np.number):
        return s.astype(int) != 0
    ss = s.astype(str).str.strip().str.lower()
    return ss.isin(["true", "1", "t", "yes", "y"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="phase.csv")
    ap.add_argument("--out", default="phase_highlight.png")
    ap.add_argument(
        "--value",
        default="severe_rate_high",
        choices=["severe_rate_high", "risk_sat_corr_high", "sat_mean_high", "risk_mean_high"],
    )

    # highlight style
    ap.add_argument("--alpha_fill", type=float, default=0.18, help="alpha for highlighting fill")
    ap.add_argument("--contour_color", default="black", help="highlight boundary color (e.g., black/white)")
    ap.add_argument("--contour_lw", type=float, default=2.5, help="highlight boundary linewidth")
    ap.add_argument("--fill_cmap", default="Greys", help="overlay colormap (recommend: Greys)")

    # rendering
    ap.add_argument("--dpi", type=int, default=200)

    # optional: manual ranges (leave None to auto)
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)

    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # ---- build ALL mask ----
    if "ALL" in df.columns:
        all_ok = to_bool_series(df["ALL"])
    elif all(c in df.columns for c in ["C1", "C2", "C3"]):
        all_ok = to_bool_series(df["C1"]) & to_bool_series(df["C2"]) & to_bool_series(df["C3"])
    else:
        raise ValueError("csv に ALL も C1/C2/C3 も見つかりません（sweep出力を確認してください）")

    df = df.copy()
    df["_ALL_"] = all_ok

    # ---- pivot: rows=v_spike_p, cols=delta_risk ----
    Z = (
        df.pivot_table(index="v_spike_p", columns="delta_risk", values=args.value, aggfunc="mean")
        .sort_index()
    )
    M = (
        df.pivot_table(index="v_spike_p", columns="delta_risk", values="_ALL_", aggfunc="max")
        .sort_index()
    )

    # Ensure same shape/index/columns
    Z = Z.reindex(index=M.index, columns=M.columns)

    x = Z.columns.values.astype(float)
    y = Z.index.values.astype(float)
    ylog = np.log10(y)

    # ---- choose colormap & norm by value type ----
    if args.value == "risk_sat_corr_high":
        # correlation: diverging and centered at 0
        cmap = plt.get_cmap("RdBu_r")
        norm = colors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    elif args.value == "severe_rate_high":
        # rates: sequential, often log-like range (tiny values)
        cmap = plt.get_cmap("YlOrRd")
        # lognorm is great for rare-event rates; guard against zeros
        zmin = np.nanmin(Z.values)
        zmax = np.nanmax(Z.values)
        # If there are many zeros, LogNorm can be unstable; use log only when positive
        if zmax > 0 and np.nanmin(Z.values[Z.values > 0]) > 0:
            # optional manual vmin/vmax override
            vmin = args.vmin if args.vmin is not None else np.nanmin(Z.values[Z.values > 0])
            vmax = args.vmax if args.vmax is not None else zmax
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            # fallback: linear
            vmin = args.vmin if args.vmin is not None else 0.0
            vmax = args.vmax if args.vmax is not None else zmax
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        # generic metrics: sequential
        cmap = plt.get_cmap("viridis")
        norm = None
        if args.vmin is not None or args.vmax is not None:
            norm = colors.Normalize(vmin=args.vmin, vmax=args.vmax)

    # ---- Plot ----
    fig, ax = plt.subplots()

    im = ax.imshow(
        Z.values,
        origin="lower",
        aspect="auto",
        extent=[x.min(), x.max(), ylog.min(), ylog.max()],
        cmap=cmap,
        norm=norm,
    )
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    })


    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(args.value)

    ax.set_xlabel("delta_risk")
    ax.set_ylabel("log10(v_spike_p)")
    ax.set_title(f"{args.value} + highlight(ALL)")

    # mask -> 0/1 grid (threshold at 0.5 to treat mean as boolean)
    M01 = (M.values >= 0.5).astype(float)

    # 1) filled overlay: make it neutral (gray) so it never fights the background
    overlay = np.where(M01 > 0.0, 1.0, np.nan)
    ax.imshow(
        overlay,
        origin="lower",
        aspect="auto",
        extent=[x.min(), x.max(), ylog.min(), ylog.max()],
        cmap=plt.get_cmap(args.fill_cmap),
        vmin=0.0,
        vmax=1.0,
        alpha=args.alpha_fill,
    )

    # 2) boundary contour: high-contrast color
    try:
        ax.contour(
            x,
            ylog,
            M01,
            levels=[0.5],
            colors=args.contour_color,
            linewidths=args.contour_lw,
        )
    except Exception as e:
        print("[warn] contour failed:", e)

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)

    print("saved:", args.out)


if __name__ == "__main__":
    main()
