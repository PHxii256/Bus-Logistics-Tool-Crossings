import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

def plot_dmrt_corridor_60_only(output_pdf: Path, output_png: Path, show_histogram: bool = False):

    tau_min  = 60.0   # floor
    tau_add  = 30.0   # additive offset
    mrt_fixed = 60.0  # fixed MRT for comparison
    x_max    = 120.0

    x = np.linspace(0.0, x_max, 1201)

    # DMRT formula: Max(60, T_direct + 30)
    # flat at 60 for x < 30, then slope-1 line for x >= 30
    y_dmrt   = np.maximum(tau_min, x + tau_add)
    y_direct = x   # reference diagonal

    # ── figure ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=220)

    # ══ REGION: Improved Service Region ═════════════════════════════════════════
    # Where DMRT permits ride times that fixed MRT=60 would reject
    # This is where y_dmrt > mrt_fixed, i.e., x > 30

    # Below: y_dmrt curve, Above: infinity (but we'll use the plot area)
    # Actually: area where DMRT > fixed MRT (purple area + green area in original)

    # Split into two parts:
    # Part 1: x in [30, 60] - area between fixed MRT=60 and DMRT (parallelogram)
    mask_part1 = (x >= 30.0) & (x <= mrt_fixed)
    ax.fill_between(
        x[mask_part1],
        mrt_fixed,           # bottom edge: Fixed MRT = 60
        y_dmrt[mask_part1],  # top edge: DMRT line
        color="#9dd6b0", alpha=0.45, linewidth=0,
    )

    # Part 2: x in [60, 120] - area where fixed MRT=60 caps students
    mask_part2 = x >= mrt_fixed
    ax.fill_between(
        x[mask_part2],
        0,
        y_dmrt[mask_part2],
        color="#9dd6b0", alpha=0.45, linewidth=0,
    )

    # ── DMRT line ──────────────────────────────────────────────────────────────
    ax.plot(x, y_dmrt, color="#0f5b78", linewidth=2.4, zorder=4)

    # kink point at T_direct = 30
    kink_x = tau_min - tau_add   # = 30
    kink_y = tau_min             # = 60
    ax.plot(kink_x, kink_y, "o", color="#0f5b78", markersize=1, zorder=5)

    # ── Fixed MRT line guide ────────────────────────────────────────────────────
    # L-shape: horizontal from x=0 to x=60, vertical from y=0 to y=60
    ax.plot([0, mrt_fixed], [mrt_fixed, mrt_fixed],
            color="#4a4a4a", linewidth=1.25, linestyle="--", zorder=3)
    ax.plot([mrt_fixed, mrt_fixed], [0, mrt_fixed],
            color="#4a4a4a", linewidth=1.25, linestyle="--", zorder=3)

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

    # Fixed MRT label
    ax.text(2.0, mrt_fixed + 1.5,
            "Fixed MRT = 60 min",
            color="#3b3b3b", fontsize=8.5, ha="left", va="bottom", zorder=6)

    # DMRT label
    ax.text(
        8, 54,
        "DMRT",
        color="#0c4a61", fontsize=12, fontweight="bold",
        ha="left", va="top", zorder=6,
    )

    # Improved Service Region label
    ax.text(
        90, 75,
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
    parser.add_argument("--output-pdf", default="dmrt_corridor_60_only.pdf")
    parser.add_argument("--output-png", default="dmrt_corridor_60_only.png")
    args    = parser.parse_args()
    out_pdf = Path(args.output_pdf)
    out_png = Path(args.output_png)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plot_dmrt_corridor_60_only(out_pdf, out_png)
