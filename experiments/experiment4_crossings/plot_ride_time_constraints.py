import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def dmrt_cap(t_direct: np.ndarray, k: float, floor_minutes: float, ceiling_minutes: float) -> np.ndarray:
    """Implemented cap in codebase: max(floor, min(k*T_direct, T_direct + ceiling))."""
    raw = k * t_direct
    absolute_ceiling = t_direct + ceiling_minutes
    return np.maximum(floor_minutes, np.minimum(raw, absolute_ceiling))


def build_plot(
    output_path: Path,
    k: float,
    floor_minutes: float,
    ceiling_minutes: float,
    t_direct_max: float,
    mrt_step: float,
) -> None:
    t_direct = np.linspace(0.0, t_direct_max, 500)
    t_max = dmrt_cap(t_direct, k=k, floor_minutes=floor_minutes, ceiling_minutes=ceiling_minutes)

    ymax = max(float(np.max(t_max)), floor_minutes) + mrt_step
    mrt_lines = np.arange(mrt_step, math.ceil(ymax / mrt_step) * mrt_step + 0.1, mrt_step)

    plt.figure(figsize=(9, 6), dpi=180)
    plt.plot(t_direct, t_max, linewidth=2.5, label=f"DMRT cap (k={k:g})")

    # Reference components of the cap equation.
    plt.plot(t_direct, k * t_direct, linewidth=1.5, alpha=0.55, label=r"$k\cdot T_{direct}$")
    plt.plot(t_direct, t_direct + ceiling_minutes, linewidth=1.5, alpha=0.55, label=rf"$T_{{direct}} + {ceiling_minutes:g}$")
    plt.axhline(floor_minutes, linewidth=1.3, alpha=0.8, color="tab:red", label=rf"Floor = {floor_minutes:g} min")

    for mrt in mrt_lines:
        plt.axhline(mrt, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
        plt.text(
            t_direct_max * 0.995,
            mrt + 0.6,
            f"MRT={int(mrt)}",
            fontsize=8,
            color="gray",
            ha="right",
            va="bottom",
        )

    plt.title(f"Actual Maximum Ride Time vs Direct Travel Time (k={k:g})")
    plt.xlabel("Direct Travel Time (minutes)")
    plt.ylabel("Actual Maximum Ride Time (minutes)")
    plt.xlim(0, t_direct_max)
    plt.ylim(0, max(mrt_lines[-1], ymax))
    plt.grid(alpha=0.25)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DMRT constraint curves for k=2 and k=3.")
    parser.add_argument(
        "--input",
        default="experiments/experiment4_crossings/input.json",
        help="Path to input.json containing constraints.",
    )
    parser.add_argument(
        "--out-dir",
        default="figures",
        help="Directory for output plot images.",
    )
    parser.add_argument(
        "--t-direct-max",
        type=float,
        default=120.0,
        help="Maximum direct travel time on x-axis (minutes).",
    )
    parser.add_argument(
        "--mrt-step",
        type=float,
        default=30.0,
        help="Spacing for dashed MRT reference lines (minutes).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    constraints = cfg.get("constraints", {})
    floor_minutes = float(constraints.get("floor_minutes", 30.0))
    ceiling_minutes = float(constraints.get("ceiling_minutes", 60.0))

    build_plot(
        output_path=out_dir / "ride_time_constraint_k2.png",
        k=2.0,
        floor_minutes=floor_minutes,
        ceiling_minutes=ceiling_minutes,
        t_direct_max=float(args.t_direct_max),
        mrt_step=float(args.mrt_step),
    )
    build_plot(
        output_path=out_dir / "ride_time_constraint_k3.png",
        k=3.0,
        floor_minutes=floor_minutes,
        ceiling_minutes=ceiling_minutes,
        t_direct_max=float(args.t_direct_max),
        mrt_step=float(args.mrt_step),
    )

    print(f"Saved: {out_dir / 'ride_time_constraint_k2.png'}")
    print(f"Saved: {out_dir / 'ride_time_constraint_k3.png'}")


if __name__ == "__main__":
    main()
