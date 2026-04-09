from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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
class StudentDirectRecord:
    instance_size: int
    seed: str
    mode_label: str
    direct_minutes: float
    served: int


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
        if (p.parent / "output.json").exists():
            files.append(p)
    return sorted(files)


def _pick_best_mode_block(output_payload: Dict) -> Dict:
    best_mode: Dict = {}
    best_total = -1
    for mode_data in (output_payload.get("modes") or {}).values():
        if not isinstance(mode_data, dict):
            continue
        served = _safe_int(mode_data.get("students_served"), 0)
        unserved = _safe_int(mode_data.get("students_unserved"), 0)
        total = served + unserved
        if total > best_total:
            best_total = total
            best_mode = mode_data
    return best_mode


def load_student_direct_records(input_root: Path) -> List[StudentDirectRecord]:
    dedup: Dict[Tuple[int, str, str], Path] = {}

    for eval_path in _discover_eval_files(input_root):
        try:
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        for run in (payload.get("runs") or []):
            config = run.get("config") or {}
            n_students = _safe_int(config.get("n_students"), 0)
            if n_students <= 0:
                continue

            seed = str(config.get("seed", "unknown"))
            mode_label = _derive_mode_label(run)
            key = (n_students, seed, mode_label)

            # One run directory should represent one seed-policy pair.
            if key not in dedup:
                dedup[key] = eval_path.parent

    rows: List[StudentDirectRecord] = []
    for (instance_size, seed, mode_label), run_dir in dedup.items():
        out_json = run_dir / "output.json"
        if not out_json.exists():
            continue

        try:
            out_payload = json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            continue

        mode_block = _pick_best_mode_block(out_payload)
        if not mode_block:
            continue

        for s in (mode_block.get("students") or []):
            direct = _safe_float((s or {}).get("direct_potential_min"), None)
            if direct is None or direct <= 0:
                continue
            rows.append(
                StudentDirectRecord(
                    instance_size=instance_size,
                    seed=seed,
                    mode_label=mode_label,
                    direct_minutes=direct,
                    served=1,
                )
            )

        for s in (mode_block.get("unserved_students") or []):
            direct = _safe_float((s or {}).get("direct_potential_min"), None)
            if direct is None or direct <= 0:
                continue
            rows.append(
                StudentDirectRecord(
                    instance_size=instance_size,
                    seed=seed,
                    mode_label=mode_label,
                    direct_minutes=direct,
                    served=0,
                )
            )

    return rows


def _compute_binned_probabilities(
    rows: Sequence[StudentDirectRecord],
    bins: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    grouped: Dict[str, Dict[str, np.ndarray]] = {}
    for mode in sorted({r.mode_label for r in rows}, key=_mode_sort_key):
        mode_rows = [r for r in rows if r.mode_label == mode]
        direct = np.array([r.direct_minutes for r in mode_rows], dtype=float)
        served = np.array([r.served for r in mode_rows], dtype=float)

        total_per_bin, _ = np.histogram(direct, bins=bins)
        served_per_bin, _ = np.histogram(direct[served > 0.5], bins=bins)

        with np.errstate(divide="ignore", invalid="ignore"):
            probs = np.where(total_per_bin > 0, served_per_bin / total_per_bin, np.nan)

        grouped[mode] = {
            "total": total_per_bin,
            "served": served_per_bin,
            "probs": probs,
        }
    return grouped


def _parse_bins(bin_spec: str) -> np.ndarray:
    values = [float(x.strip()) for x in bin_spec.split(",") if x.strip()]
    if len(values) < 2:
        raise ValueError("Bin specification must include at least two edges.")
    arr = np.array(values, dtype=float)
    if not np.all(np.diff(arr) > 0):
        raise ValueError("Bin edges must be strictly increasing.")
    return arr


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


def plot_instance(
    all_rows: Sequence[StudentDirectRecord],
    instance_size: int,
    out_path: Path,
    bins: np.ndarray,
    x_max: float | None,
) -> List[Path]:
    rows = [r for r in all_rows if r.instance_size == instance_size]
    if not rows:
        raise ValueError(f"No direct-time rows for instance size {instance_size}.")

    grouped = _compute_binned_probabilities(rows, bins)
    centers = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for mode in sorted(grouped.keys(), key=_mode_sort_key):
        probs = grouped[mode]["probs"]
        total = grouped[mode]["total"]
        mask = total > 0
        if not np.any(mask):
            continue
        ax.plot(
            centers[mask],
            probs[mask],
            marker="o",
            linewidth=1.6,
            markersize=3.8,
            label=mode,
            color=MODE_COLORS.get(mode, None),
        )

    ax.set_xlabel("Direct travel time (min)")
    ax.set_ylabel("Service probability")
    ax.set_ylim(0.0, 1.05)
    x_upper = min(float(bins[-1]), float(x_max)) if x_max is not None else float(bins[-1])
    ax.set_xlim(float(bins[0]), x_upper)
    ax.grid(alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7)

    fig.tight_layout()
    saved = _save_figure_with_pdf(fig, out_path)
    plt.close(fig)
    return saved


def _instance_sizes(rows: Iterable[StudentDirectRecord]) -> List[int]:
    return sorted({r.instance_size for r in rows})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figure 2: service probability vs direct time, aggregated over all available seeds.",
    )
    parser.add_argument(
        "--input-root",
        default="experiments/experiment4_crossings",
        help="Root directory to recursively search for run-level evaluation/output pairs.",
    )
    parser.add_argument(
        "--instance-size",
        type=int,
        default=None,
        help="Optional instance size filter (example: 200). If omitted, one file per size is generated.",
    )
    parser.add_argument(
        "--bins",
        default="0,10,20,30,40,50,60,70,80,90,100,110,120",
        help="Comma-separated bin edges in minutes.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=60.0,
        help="Maximum x-axis value in minutes (default: 60).",
    )
    parser.add_argument(
        "--output",
        default="figures/fig2_service_prob_vs_direct_time.png",
        help="Output path for single instance mode; otherwise used as prefix for per-instance files.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output = Path(args.output)
    bins = _parse_bins(args.bins)

    rows = load_student_direct_records(input_root)
    if not rows:
        raise SystemExit("No student-level direct-time records found. Check --input-root.")

    for size in _instance_sizes(rows):
        size_rows = [r for r in rows if r.instance_size == size]
        seeds = sorted({r.seed for r in size_rows})
        modes = sorted({r.mode_label for r in size_rows}, key=_mode_sort_key)
        print(
            f"Coverage |S|={size}: seeds={seeds}, modes={modes}, student_rows={len(size_rows)}"
        )

    if args.instance_size is not None:
        saved = plot_instance(rows, args.instance_size, output, bins, args.x_max)
        for p in saved:
            print(f"Saved: {p}")
        return

    for size in _instance_sizes(rows):
        out = output.with_name(f"{output.stem}_n{size}{output.suffix}")
        saved = plot_instance(rows, size, out, bins, args.x_max)
        for p in saved:
            print(f"Saved: {p}")


if __name__ == "__main__":
    main()
