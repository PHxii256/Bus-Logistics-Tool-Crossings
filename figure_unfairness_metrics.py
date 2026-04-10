from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


PREFERRED_MODE_ORDER = [
    "DMRT Lenient",
    "DMRT Strict",
    "MRT 60",
    "MRT 90",
    "No MRT",
]

MODE_COLORS = {
    "DMRT Lenient": "#1f77b4",
    "DMRT Strict": "#2ca02c",
    "MRT 60": "#d62728",
    "MRT 90": "#9467bd",
    "No MRT": "#8c564b",
}


@dataclass
class UnfairnessRecord:
    instance_size: int
    seed: str
    mode_label: str
    ride_ratio_mean: float
    ride_ratio_std: float
    ride_ratio_max: float
    students_total: int
    source_file: Path


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        x = float(value)
        if np.isfinite(x):
            return x
        return default
    except (TypeError, ValueError):
        return default


def _derive_mode_label(run: Dict) -> str:
    constraints = ((run.get("config") or {}).get("constraints") or {})

    enabled = bool(constraints.get("enabled", True))
    if not enabled:
        return "No MRT"

    mrt_enabled = bool(constraints.get("mrt_enabled", constraints.get("mrt enabled", False)))
    mrt_value = constraints.get("mrt")
    if mrt_enabled and mrt_value is not None:
        mrt_minutes = _safe_int(mrt_value, -1)
        if mrt_minutes > 0:
            return f"MRT {mrt_minutes}"

    bidirectional = bool(constraints.get("bidirectional_check", True))
    return "DMRT Lenient" if bidirectional else "DMRT Strict"


def _mode_sort_key(label: str) -> Tuple[int, str]:
    try:
        return (PREFERRED_MODE_ORDER.index(label), label)
    except ValueError:
        return (len(PREFERRED_MODE_ORDER), label)


def _discover_eval_files(input_root: Path) -> List[Path]:
    files: List[Path] = []
    for p in input_root.rglob("evaluation.json"):
        if p == input_root / "evaluation.json":
            continue
        files.append(p)
    return sorted(files)


def _pick_mode_with_data(run: Dict) -> Tuple[Dict, int]:
    best_mode: Dict = {}
    best_total = -1
    for mode_data in (run.get("modes") or {}).values():
        if not isinstance(mode_data, dict):
            continue

        counts = (mode_data.get("counts") or {})
        total = _safe_int(counts.get("students_total"), 0)
        served = _safe_int(counts.get("students_served"), 0)
        unserved = _safe_int(counts.get("students_unserved"), 0)
        if total <= 0:
            total = max(0, served + unserved)

        if total > best_total:
            best_total = total
            best_mode = mode_data

    return best_mode, best_total


def load_unfairness_records(input_root: Path) -> List[UnfairnessRecord]:
    dedup: Dict[Tuple[int, str, str], UnfairnessRecord] = {}

    for eval_path in _discover_eval_files(input_root):
        try:
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        for run in (payload.get("runs") or []):
            config = run.get("config") or {}
            instance_size = _safe_int(config.get("n_students"), 0)
            if instance_size <= 0:
                continue

            seed = str(config.get("seed", "unknown"))
            mode_label = _derive_mode_label(run)

            mode_data, students_total = _pick_mode_with_data(run)
            if students_total <= 0:
                continue

            paper_metrics = (mode_data.get("paper_metrics") or {})
            rr_mean = _safe_float(paper_metrics.get("RideRatioMean"), None)
            rr_std = _safe_float(paper_metrics.get("RideRatioStd"), None)
            rr_max = _safe_float(paper_metrics.get("RideRatioMax"), None)
            if rr_mean is None or rr_std is None or rr_max is None:
                continue

            rec = UnfairnessRecord(
                instance_size=instance_size,
                seed=seed,
                mode_label=mode_label,
                ride_ratio_mean=rr_mean,
                ride_ratio_std=rr_std,
                ride_ratio_max=rr_max,
                students_total=students_total,
                source_file=eval_path,
            )

            key = (instance_size, seed, mode_label)
            prev = dedup.get(key)
            if prev is None or rec.students_total > prev.students_total:
                dedup[key] = rec

    return list(dedup.values())


def _instance_sizes(rows: Iterable[UnfairnessRecord]) -> List[int]:
    return sorted({r.instance_size for r in rows})


def _save_figure_with_pdf(fig, out_path: Path) -> List[Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    saved.append(out_path)

    # Always emit a companion PDF for paper workflows.
    pdf_path = out_path.with_suffix(".pdf")
    if pdf_path != out_path:
        fig.savefig(pdf_path, bbox_inches="tight")
        saved.append(pdf_path)

    return saved


def _plot_metric_panel(
    ax,
    records: List[UnfairnessRecord],
    modes: List[str],
    metric_attr: str,
    metric_title: str,
) -> None:
    x = np.arange(len(modes), dtype=float)
    means: List[float] = []
    errors: List[float] = []

    for mode in modes:
        vals = [float(getattr(r, metric_attr)) for r in records if r.mode_label == mode]
        if vals:
            arr = np.array(vals, dtype=float)
            means.append(float(np.mean(arr)))
            errors.append(float(np.std(arr, ddof=0)))
        else:
            means.append(np.nan)
            errors.append(0.0)

    bars = ax.bar(
        x,
        means,
        yerr=errors,
        color=[MODE_COLORS.get(m, "#999999") for m in modes],
        alpha=0.85,
        capsize=3,
        width=0.72,
        linewidth=0.0,
        error_kw={"ecolor": "#333333", "elinewidth": 1.0, "capthick": 1.0},
    )

    ax.set_title(metric_title, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=20, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)

    valid_tops = [
        (m + e)
        for m, e in zip(means, errors)
        if np.isfinite(m) and np.isfinite(e)
    ]
    if valid_tops:
        # Include error bars in the axis range so caps never clip at the top.
        ax.set_ylim(0.0, max(valid_tops) * 1.12)

    # Keep legend out to avoid clutter; bars are mode-colored and x labels are explicit.
    for b in bars:
        if not np.isfinite(b.get_height()):
            b.set_alpha(0.2)


def plot_instance(records: List[UnfairnessRecord], instance_size: int, out_path: Path) -> List[Path]:
    subset = [r for r in records if r.instance_size == instance_size]
    if not subset:
        raise ValueError(f"No unfairness records found for instance size {instance_size}.")

    modes = sorted({r.mode_label for r in subset}, key=_mode_sort_key)

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8), sharex=False)
    _plot_metric_panel(axes[0], subset, modes, "ride_ratio_mean", "Ride Ratio Mean")
    _plot_metric_panel(axes[1], subset, modes, "ride_ratio_std", "Ride Ratio Std")
    _plot_metric_panel(axes[2], subset, modes, "ride_ratio_max", "Ride Ratio Max")

    axes[0].set_ylabel("Value")
    fig.suptitle("Unfairness Metrics by Mode", fontsize=10, y=1.03)
    fig.tight_layout()

    saved = _save_figure_with_pdf(fig, out_path)
    plt.close(fig)
    return saved


def plot_single_metric_instance(
    records: List[UnfairnessRecord],
    instance_size: int,
    out_path: Path,
    metric_attr: str,
    metric_title: str,
) -> List[Path]:
    subset = [r for r in records if r.instance_size == instance_size]
    if not subset:
        raise ValueError(f"No unfairness records found for instance size {instance_size}.")

    modes = sorted({r.mode_label for r in subset}, key=_mode_sort_key)

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.6))
    _plot_metric_panel(ax, subset, modes, metric_attr, "")
    ax.set_ylabel("Value")
    fig.tight_layout()

    saved = _save_figure_with_pdf(fig, out_path)
    plt.close(fig)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figure: unfairness using RideRatioMean/Std/Max from evaluation.json.",
    )
    parser.add_argument(
        "--input-root",
        default="experiments/experiment4_crossings",
        help="Root directory to recursively search for run-level evaluation.json files.",
    )
    parser.add_argument(
        "--instance-size",
        type=int,
        default=None,
        help="Optional instance size filter (example: 200). If omitted, one file per size is generated.",
    )
    parser.add_argument(
        "--output",
        default="figures/fig3_unfairness_metrics.png",
        help="Output path for single instance mode; otherwise used as prefix for per-instance files.",
    )
    parser.add_argument(
        "--mean-output",
        default="figures/fig3a_ride_ratio_mean.png",
        help="Output path for separate Ride Ratio Mean figure.",
    )
    parser.add_argument(
        "--max-output",
        default="figures/fig3b_ride_ratio_max.png",
        help="Output path for separate Ride Ratio Max figure.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output = Path(args.output)
    mean_output = Path(args.mean_output)
    max_output = Path(args.max_output)

    rows = load_unfairness_records(input_root)
    if not rows:
        raise SystemExit("No unfairness records found. Check --input-root.")

    for size in _instance_sizes(rows):
        size_rows = [r for r in rows if r.instance_size == size]
        seeds = sorted({r.seed for r in size_rows})
        modes = sorted({r.mode_label for r in size_rows}, key=_mode_sort_key)
        print(
            f"Coverage |S|={size}: seeds={seeds}, modes={modes}, run_points={len(size_rows)}"
        )

    if args.instance_size is not None:
        saved_all: List[Path] = []
        saved_all.extend(plot_instance(rows, args.instance_size, output))
        saved_all.extend(plot_single_metric_instance(
            rows,
            args.instance_size,
            mean_output,
            "ride_ratio_mean",
            "Ride Ratio Mean",
        ))
        saved_all.extend(plot_single_metric_instance(
            rows,
            args.instance_size,
            max_output,
            "ride_ratio_max",
            "Ride Ratio Max",
        ))
        for p in saved_all:
            print(f"Saved: {p}")
        return

    for size in _instance_sizes(rows):
        out = output.with_name(f"{output.stem}_n{size}{output.suffix}")
        mean_out = mean_output.with_name(f"{mean_output.stem}_n{size}{mean_output.suffix}")
        max_out = max_output.with_name(f"{max_output.stem}_n{size}{max_output.suffix}")
        saved_all: List[Path] = []
        saved_all.extend(plot_instance(rows, size, out))
        saved_all.extend(plot_single_metric_instance(rows, size, mean_out, "ride_ratio_mean", "Ride Ratio Mean"))
        saved_all.extend(plot_single_metric_instance(rows, size, max_out, "ride_ratio_max", "Ride Ratio Max"))
        for p in saved_all:
            print(f"Saved: {p}")


if __name__ == "__main__":
    main()
