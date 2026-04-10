import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np


# ── helpers ────────────────────────────────────────────────────────────────────

def _collect_values(obj, key):
    values = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and isinstance(v, (int, float)):
                values.append(float(v))
            values.extend(_collect_values(v, key))
    elif isinstance(obj, list):
        for item in obj:
            values.extend(_collect_values(item, key))
    return values


def _load_observed_direct_times(base_dir: Path, load_from_files: bool = False):
    if not load_from_files:
        return []
    candidates = sorted(base_dir.glob("*/output.json"))
    best = []
    for p in candidates:
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            vals = _collect_values(data, "direct_potential_min")
            vals = [v for v in vals if 0.0 <= v <= 120.0]
            if len(vals) > len(best):
                best = vals
        except Exception:
            continue
    return best


# ── main plot ──────────────────────────────────────────────────────────────────

def plot_dmrt_corridor(output_pdf: Path, output_png: Path, show_histogram: bool = False):

    tau_min  = 60.0   # floor
    tau_add  = 30.0   # additive offset
    mrt_low  = 60.0   # lower fixed MRT for comparison
    mrt_high = 90.0   # upper fixed MRT for comparison
    x_max    = 120.0

    x = np.linspace(0.0, x_max, 1201)

    # DMRT formula: Max(60, T_direct + 30)
    # flat at 60 for x < 30, then slope-1 line for x >= 30
    y_dmrt   = np.maximum(tau_min, x + tau_add)
    y_direct = x   # reference diagonal

    # ── figure ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=220)

    # ══ REGION 1: Unfairness Region ════════════════════════════════════════════
    # Where Fixed MRT=90 permits ride times DMRT would reject.
    # Bounded by: x=0 (left), DMRT line (top), y=mrt_high (bottom), x=60 (right)
    unfair_x_end = mrt_high - tau_add   # = 60
    mask_unfair  = x <= unfair_x_end
    ax.fill_between(
        x[mask_unfair],
        mrt_high,                   # bottom edge: Fixed MRT = 90
        y_dmrt[mask_unfair],        # top edge:    DMRT line
        color="#e8a0a0", alpha=0.50, linewidth=0,
    )

    # ══ REGION 2: Improved Service Region ══════════════════════════════════════
    # Sub-area A: x in [60,90], y between mrt_high and y_dmrt (parallelogram)
    mask_improved = x >= unfair_x_end
    ax.fill_between(
        x[mask_improved],
        mrt_high,
        y_dmrt[mask_improved],
        color="#9dd6b0", alpha=0.45, linewidth=0,
    )

    # Sub-area B: x in [90,120], y between 0 and mrt_high (rectangle)
    # Students with T_direct > Fixed MRT, completely excluded by fixed cap
    mask_rect = x >= mrt_high
    ax.fill_between(
        x[mask_rect],
        0,
        mrt_high,
        color="#9dd6b0", alpha=0.45, linewidth=0,
    )

    # ── DMRT line ──────────────────────────────────────────────────────────────
    ax.plot(x, y_dmrt, color="#0f5b78", linewidth=2.4, zorder=4)

    # kink point at T_direct = 30
    kink_x = tau_min - tau_add   # = 30
    kink_y = tau_min             # = 60
    ax.plot(kink_x, kink_y, "o", color="#0f5b78", markersize=1, zorder=5)

    # ── Fixed MRT rectangle guides ─────────────────────────────────────────────
    # Each draws an L: horizontal from x=0 to x=mrt_val,
    #                  vertical from y=0 to y=mrt_val
    for mrt_val, ls in [(mrt_low, "--"), (mrt_high, ":")]:
        ax.plot([0, mrt_val], [mrt_val, mrt_val],
                color="#4a4a4a", linewidth=1.25, linestyle=ls, zorder=3)
        ax.plot([mrt_val, mrt_val], [0, mrt_val],
                color="#4a4a4a", linewidth=1.25, linestyle=ls, zorder=3)

    # ── student distribution histogram ─────────────────────────────────────────
    if show_histogram:
        obs = _load_observed_direct_times(
            Path("experiments/experiment4_crossings")
        )
        # Fallback synthetic distribution if no data found
        if not obs:
            rng = np.random.default_rng(42)
            obs = list(rng.normal(loc=25, scale=10, size=300).clip(2, 80))

        bins = np.arange(0, x_max + 4, 4)
        counts, edges = np.histogram(obs, bins=bins)
        if counts.max() > 0:
            scale_height = 12.0
            heights = (counts / counts.max()) * scale_height
            centers = (edges[:-1] + edges[1:]) / 2.0
            widths  = np.diff(edges) * 0.88
            ax.bar(
                centers, heights, width=widths, bottom=0.0,
                color="#e07b10", alpha=0.50, edgecolor="none", zorder=2,
            )

    # ── text labels ────────────────────────────────────────────────────────────

    # Fixed MRT labels
    ax.text(2.0, mrt_low + 1.5,
            "Fixed MRT = 60 min",
            color="#3b3b3b", fontsize=8.5, ha="left", va="bottom", zorder=6)
    ax.text(2.0, mrt_high + 1.5,
            "Fixed MRT = 90 min",
            color="#3b3b3b", fontsize=8.5, ha="left", va="bottom", zorder=6)

    # DMRT label
    ax.text(
        8, 54,
        "DMRT",
        color="#0c4a61", fontsize=12, fontweight="bold",
        ha="left", va="top", zorder=6,
    )

    # Unfairness Region label: centred inside the pink region
    ax.text(
        22, 76,
        "Unfairness Region",
        color="#7a0c0c", fontsize=11, fontweight="bold",
        ha="center", va="center", zorder=6,
    )

    # Improved Service Region label
    ax.text(
        97, 100,
        "Improved Service Region",
        color="#0c4a1e", fontsize=11, fontweight="bold",
        ha="center", va="center", zorder=6,
    )

    # ── axes formatting ────────────────────────────────────────────────────────
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, x_max)
    ax.set_xlabel("Direct Travel Time to School (min)", fontsize=11)
    ax.set_ylabel("Maximum Ride Time (min)", fontsize=11)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.grid(alpha=0.18)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_pdf}")
    print(f"Saved: {output_png}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-pdf", default="dmrt_corridor.pdf")
    parser.add_argument("--output-png", default="dmrt_corridor.png")
    args    = parser.parse_args()
    out_pdf = Path(args.output_pdf)
    out_png = Path(args.output_png)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plot_dmrt_corridor(out_pdf, out_png)