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


@dataclass
class RunRecord:
    instance_size: int
    seed: str
    mode_label: str
    served: int
    unserved: int
    total: int
    source_file: Path


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
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


def _extract_primary_counts(run: Dict) -> Tuple[int, int, int]:
    best = (0, 0, 0)
    modes = run.get("modes") or {}
    for mode_data in modes.values():
        counts = (mode_data or {}).get("counts") or {}
        total = _safe_int(counts.get("students_total"), 0)
        served = _safe_int(counts.get("students_served"), 0)
        unserved = _safe_int(counts.get("students_unserved"), 0)

        if total <= 0:
            total = max(0, served + unserved)

        if total > best[2]:
            best = (served, unserved, total)

    return best


def _mode_sort_key(label: str) -> Tuple[int, str]:
    try:
        return (PREFERRED_MODE_ORDER.index(label), label)
    except ValueError:
        return (len(PREFERRED_MODE_ORDER), label)


def discover_evaluation_files(input_root: Path) -> List[Path]:
    files: List[Path] = []
    for p in input_root.rglob("evaluation.json"):
        if p == input_root / "evaluation.json":
            continue
        if (p.parent / "output.json").exists():
            files.append(p)
    return sorted(files)


def load_records(input_root: Path) -> List[RunRecord]:
    eval_files = discover_evaluation_files(input_root)
    dedup: Dict[Tuple[int, str, str], RunRecord] = {}

    for eval_path in eval_files:
        try:
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        runs = payload.get("runs") or []
        for run in runs:
            config = run.get("config") or {}
            instance_size = _safe_int(config.get("n_students"), 0)
            if instance_size <= 0:
                continue

            seed = str(config.get("seed", "unknown"))
            mode_label = _derive_mode_label(run)
            served, unserved, total = _extract_primary_counts(run)
            if total <= 0:
                continue

            rec = RunRecord(
                instance_size=instance_size,
                seed=seed,
                mode_label=mode_label,
                served=served,
                unserved=unserved,
                total=total,
                source_file=eval_path,
            )

            key = (instance_size, seed, mode_label)
            prev = dedup.get(key)
            if prev is None or rec.total > prev.total:
                dedup[key] = rec

    return list(dedup.values())


def _sorted_seeds(seed_values: Iterable[str]) -> List[str]:
    def sort_key(s: str):
        try:
            return (0, int(s))
        except ValueError:
            return (1, s)

    return sorted(set(seed_values), key=sort_key)


def plot_instance(records: List[RunRecord], instance_size: int, out_path: Path) -> None:
    subset = [r for r in records if r.instance_size == instance_size]
    if not subset:
        raise ValueError(f"No records found for instance size {instance_size}.")

    seeds = _sorted_seeds(r.seed for r in subset)
    modes = sorted({r.mode_label for r in subset}, key=_mode_sort_key)
    lookup: Dict[Tuple[str, str], RunRecord] = {(r.seed, r.mode_label): r for r in subset}

    n_seeds = len(seeds)
    fig_w = 7.16
    fig_h = max(2.5, 2.0 + 0.3 * n_seeds)
    fig, axes = plt.subplots(1, n_seeds, figsize=(fig_w, fig_h), sharey=True)
    if n_seeds == 1:
        axes = [axes]

    served_color = "#1b9e77"
    unserved_color = "#d95f02"
    y_max = 0

    for ax, seed in zip(axes, seeds):
        served_vals: List[int] = []
        unserved_vals: List[int] = []

        for mode in modes:
            rec = lookup.get((seed, mode))
            served = rec.served if rec else 0
            unserved = rec.unserved if rec else 0
            served_vals.append(served)
            unserved_vals.append(unserved)
            y_max = max(y_max, served + unserved)

        x = np.arange(len(modes))
        ax.bar(x, served_vals, color=served_color, label="Served", width=0.72)
        ax.bar(x, unserved_vals, bottom=served_vals, color=unserved_color, label="Unserved", width=0.72)

        ax.set_xticks(x)
        ax.set_xticklabels(modes, rotation=20, ha="right", fontsize=8)
        ax.set_title(f"Seed {seed}", fontsize=10)
        ax.grid(axis="y", alpha=0.3, linewidth=0.6)
        ax.set_axisbelow(True)

    if y_max <= 0:
        y_max = 1

    for ax in axes:
        ax.set_ylim(0, y_max * 1.12)

    axes[0].set_ylabel("Students", fontsize=10)
    fig.suptitle(
        f"Served vs Unserved by Seed and Mode (|S|={instance_size})",
        fontsize=11,
        y=1.02,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figure 1: double-column served vs unserved by seed and mode, aggregated over all seeds.",
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
        default="figures/fig1_served_unserved_double_col.png",
        help="Output path for single instance mode; otherwise used as prefix for per-instance files.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output = Path(args.output)
    records = load_records(input_root)

    if not records:
        raise SystemExit("No valid run records found. Check --input-root.")

    for size in sorted({r.instance_size for r in records}):
        recs = [r for r in records if r.instance_size == size]
        seeds = _sorted_seeds(r.seed for r in recs)
        modes = sorted({r.mode_label for r in recs}, key=_mode_sort_key)
        print(
            f"Coverage |S|={size}: seeds={seeds}, modes={modes}, run_points={len(recs)}"
        )

    if args.instance_size is not None:
        plot_instance(records, args.instance_size, output)
        print(f"Saved: {output}")
        return

    instance_sizes = sorted({r.instance_size for r in records})
    for size in instance_sizes:
        out = output.with_name(f"{output.stem}_n{size}{output.suffix}")
        plot_instance(records, size, out)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
