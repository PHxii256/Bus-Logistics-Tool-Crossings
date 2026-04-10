import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
FINAL_RESULTS = ROOT / "FINAL_RESULTS"
FIGURES_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"

MODE_KEY_TO_LABEL = {
    "strictly_constrained": "Mode A",
    "weakly_constrained": "Mode B",
    "door_to_door": "Mode C",
}
MODE_ORDER = ["Mode A", "Mode B", "Mode C"]
COMPLETE_COHORTS = {100, 200, 400}

CORE_METRICS = [
    "served_rate_pct",
    "students_served",
    "students_unserved",
    "routes_created",
    "total_route_time_min",
    "total_route_dist_km",
    "avg_walk_dist_m",
    "max_walk_dist_m",
    "alns_runtime_seconds",
    "executed_iterations",
]

INFERENTIAL_METRICS = [
    "served_rate_pct",
    "routes_created",
    "total_route_time_min",
    "total_route_dist_km",
    "avg_walk_dist_m",
    "alns_runtime_seconds",
]


def _to_float(v):
    if v is None:
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _to_int(v):
    if v is None:
        return 0
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0



def _ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)



def _cohort_from_folder(name: str) -> int:
    if not name.endswith("Students"):
        return -1
    try:
        return int(name.replace("Students", ""))
    except ValueError:
        return -1



def _load_runs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict] = []
    conv_rows: List[Dict] = []

    if not FINAL_RESULTS.exists():
        raise FileNotFoundError(f"FINAL_RESULTS folder not found: {FINAL_RESULTS}")

    for cohort_dir in sorted(FINAL_RESULTS.iterdir()):
        if not cohort_dir.is_dir():
            continue
        cohort = _cohort_from_folder(cohort_dir.name)
        if cohort <= 0:
            continue

        for seed_dir in sorted(cohort_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            output_path = seed_dir / "output.json"
            if not output_path.exists():
                continue

            with output_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            modes = payload.get("modes", {})
            for mode_key, mode_data in modes.items():
                mode = MODE_KEY_TO_LABEL.get(mode_key)
                if mode is None:
                    continue

                students_served = _to_int(mode_data.get("students_served", 0))
                students_unserved = _to_int(mode_data.get("students_unserved", 0))
                students_total = _to_int(mode_data.get("students_total", 0))
                if students_total <= 0:
                    students_total = students_served + students_unserved
                served_rate = 100.0 * students_served / students_total if students_total > 0 else np.nan

                walk_stats = mode_data.get("walk_stats", {}) or {}
                op_perf = mode_data.get("operator_performance", {}) or {}
                executed_iterations = _to_float(mode_data.get("executed_iterations", np.nan))
                if np.isnan(executed_iterations):
                    executed_iterations = _to_float(op_perf.get("executed_iterations", np.nan))

                row = {
                    "cohort": cohort,
                    "seed": seed_dir.name,
                    "mode": mode,
                    "mode_key": mode_key,
                    "students_total": students_total,
                    "students_served": students_served,
                    "students_unserved": students_unserved,
                    "served_rate_pct": served_rate,
                    "routes_created": _to_float(mode_data.get("routes_created", np.nan)),
                    "total_route_time_min": _to_float(mode_data.get("total_route_time_min", np.nan)),
                    "total_route_dist_km": _to_float(mode_data.get("total_route_dist_km", np.nan)),
                    "avg_walk_dist_m": _to_float(walk_stats.get("avg_walk_dist_m", mode_data.get("avg_walk_dist_m", np.nan))),
                    "max_walk_dist_m": _to_float(walk_stats.get("max_walk_dist_m", mode_data.get("max_walk_dist_m", np.nan))),
                    "alns_runtime_seconds": _to_float(mode_data.get("alns_runtime_seconds", np.nan)),
                    "executed_iterations": executed_iterations,
                    "matrix_precompute_total_s": _to_float(mode_data.get("matrix_precompute_total_s", np.nan)),
                    "mode_wall_time_seconds": _to_float(mode_data.get("mode_wall_time_seconds", np.nan)),
                }
                rows.append(row)

            # Convergence logs
            for suffix, mode in [("a", "Mode A"), ("b", "Mode B"), ("c", "Mode C")]:
                lp = seed_dir / f"alns_log_mode_{suffix}.json"
                if not lp.exists():
                    continue
                with lp.open("r", encoding="utf-8") as f:
                    log = json.load(f)
                for e in log.get("entries", []):
                    conv_rows.append(
                        {
                            "cohort": cohort,
                            "seed": seed_dir.name,
                            "mode": mode,
                            "iteration": _to_float(e.get("iteration", np.nan)),
                            "objective_value": _to_float(e.get("objective_value", np.nan)),
                            "best_objective": _to_float(e.get("best_objective", np.nan)),
                            "students_served": _to_float(e.get("students_served", np.nan)),
                            "temperature": _to_float(e.get("temperature", np.nan)),
                        }
                    )

    if not rows:
        raise RuntimeError("No output.json runs found in FINAL_RESULTS.")

    metrics_df = pd.DataFrame(rows)
    metrics_df["mode"] = pd.Categorical(metrics_df["mode"], categories=MODE_ORDER, ordered=True)
    metrics_df = metrics_df.sort_values(["cohort", "seed", "mode"]).reset_index(drop=True)

    conv_df = pd.DataFrame(conv_rows)
    if not conv_df.empty:
        conv_df["mode"] = pd.Categorical(conv_df["mode"], categories=MODE_ORDER, ordered=True)
        conv_df = conv_df.sort_values(["cohort", "seed", "mode", "iteration"]).reset_index(drop=True)

    return metrics_df, conv_df



def _mean_sd(x: pd.Series) -> str:
    x = x.dropna().astype(float)
    if x.empty:
        return "-"
    sd = x.std(ddof=1)
    if np.isnan(sd):
        sd = 0.0
    return f"${x.mean():.2f} \\pm {sd:.2f}$"



def _rank_biserial_from_deltas(deltas: np.ndarray) -> float:
    d = np.asarray(deltas, dtype=float)
    d = d[~np.isnan(d)]
    d = d[d != 0]
    if d.size == 0:
        return np.nan

    abs_d = np.abs(d)
    order = np.argsort(abs_d)
    ranks = np.empty_like(abs_d, dtype=float)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs_d[order[j + 1]] == abs_d[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    w_plus = ranks[d > 0].sum()
    w_minus = ranks[d < 0].sum()
    denom = w_plus + w_minus
    if denom == 0:
        return np.nan
    return (w_plus - w_minus) / denom



def _holm_adjust(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    max_so_far = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        adj = min(adj, 1.0)
        max_so_far = max(max_so_far, adj)
        adjusted[idx] = max_so_far
    return adjusted.tolist()



def _build_table1(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df[metrics_df["cohort"].isin(COMPLETE_COHORTS)].copy()
    grouped = df.groupby(["cohort", "mode"], observed=True)
    records = []
    for (cohort, mode), g in grouped:
        rec = {
            "cohort": cohort,
            "mode": mode,
            "n_seeds": g["seed"].nunique(),
        }
        for metric in CORE_METRICS:
            rec[metric] = _mean_sd(g[metric])
        records.append(rec)

    out = pd.DataFrame(records).sort_values(["cohort", "mode"])
    out.to_csv(TABLES_DIR / "table1_aggregate_metrics.csv", index=False)
    return out



def _build_table2(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df[metrics_df["cohort"].isin(COMPLETE_COHORTS)].copy()
    rows = []
    for cohort in sorted(COMPLETE_COHORTS):
        cdf = df[df["cohort"] == cohort]
        for m1, m2 in [("Mode A", "Mode B"), ("Mode B", "Mode C")]:
            s1 = set(cdf[cdf["mode"] == m1]["seed"])
            s2 = set(cdf[cdf["mode"] == m2]["seed"])
            seeds = sorted(s1 & s2)
            if not seeds:
                continue

            a = cdf[(cdf["mode"] == m1) & (cdf["seed"].isin(seeds))].set_index("seed")
            b = cdf[(cdf["mode"] == m2) & (cdf["seed"].isin(seeds))].set_index("seed")

            for metric in INFERENTIAL_METRICS:
                deltas = (a[metric] - b[metric]).astype(float)
                valid = deltas.dropna()
                if len(valid) < 3:
                    continue
                if valid.empty:
                    continue
                try:
                    stat = wilcoxon(valid.values, zero_method="wilcox", correction=False, alternative="two-sided")
                    p = float(stat.pvalue)
                    w = float(stat.statistic)
                except ValueError:
                    p = np.nan
                    w = np.nan
                rb = _rank_biserial_from_deltas(valid.values)
                rows.append(
                    {
                        "cohort": cohort,
                        "comparison": f"{m1} - {m2}",
                        "metric": metric,
                        "n_pairs": len(valid),
                        "median_delta": float(np.median(valid.values)),
                        "mean_delta": float(np.mean(valid.values)),
                        "wilcoxon_W": w,
                        "p_raw": p,
                        "effect_rank_biserial": rb,
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        out.to_csv(TABLES_DIR / "table2_wilcoxon_effects.csv", index=False)
        return out

    pvals = out["p_raw"].fillna(1.0).tolist()
    out["p_holm"] = _holm_adjust(pvals)
    out["significant_alpha_0_05"] = out["p_holm"] < 0.05
    out = out.sort_values(["cohort", "comparison", "metric"])
    out.to_csv(TABLES_DIR / "table2_wilcoxon_effects.csv", index=False)
    return out



def _to_latex_table(df: pd.DataFrame, caption: str, label: str, file_name: str) -> None:
    if df.empty:
        return
    safe = df.copy()
    safe.columns = [str(c).replace("_", " ") for c in safe.columns]

    for col in safe.columns:
        safe[col] = safe[col].map(lambda v: v.replace("_", "\\_") if isinstance(v, str) else v)

    latex = safe.to_latex(index=False, escape=False)
    wide_files = {"table1_aggregate_metrics.tex", "table2_wilcoxon_effects.tex"}
    is_wide = file_name in wide_files
    env = "table*" if is_wide else "table"

    if is_wide:
        latex = (
            "\\scriptsize\n"
            "\\setlength{\\tabcolsep}{3pt}\n"
            "\\resizebox{\\textwidth}{!}{%\n"
            f"{latex}\n"
            "}"
        )
    wrapped = (
        f"\\begin{{{env}}}[!t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{latex}\n"
        f"\\end{{{env}}}\n"
    )
    (TABLES_DIR / file_name).write_text(wrapped, encoding="utf-8")



def _figure1_mode_comparison(metrics_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    df = metrics_df[metrics_df["cohort"].isin(COMPLETE_COHORTS)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    plots = [
        ("served_rate_pct", "Served Rate (%)"),
        ("routes_created", "Routes Created"),
        ("total_route_time_min", "Total Route Time (min)"),
        ("avg_walk_dist_m", "Avg Walk Distance (m)"),
    ]
    for ax, (metric, title) in zip(axes.flatten(), plots):
        agg = df.groupby(["cohort", "mode"], observed=True)[metric].mean().reset_index()
        sns.barplot(data=agg, x="cohort", y=metric, hue="mode", hue_order=MODE_ORDER, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Students")
        ax.set_ylabel("")
        if ax is not axes.flatten()[0]:
            ax.get_legend().remove()
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.savefig(FIGURES_DIR / "figure1_mode_comparison.png", dpi=300)
    fig.savefig(FIGURES_DIR / "figure1_mode_comparison.pdf")
    plt.close(fig)



def _figure2_convergence(conv_df: pd.DataFrame) -> None:
    if conv_df.empty:
        return
    sns.set_theme(style="whitegrid")
    target = conv_df[conv_df["cohort"] == 400].copy()
    if target.empty:
        target = conv_df.copy()

    agg = target.groupby(["mode", "iteration"], observed=True)["best_objective"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    sns.lineplot(data=agg, x="iteration", y="best_objective", hue="mode", hue_order=MODE_ORDER, marker="o", ax=ax)
    ax.set_title("Convergence Profile (Mean Best Objective)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Objective")
    fig.savefig(FIGURES_DIR / "figure2_convergence_curves.png", dpi=300)
    fig.savefig(FIGURES_DIR / "figure2_convergence_curves.pdf")
    plt.close(fig)



def _figure3_scalability(metrics_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    df = metrics_df[metrics_df["cohort"].isin(COMPLETE_COHORTS)].copy()
    agg = df.groupby(["cohort", "mode"], observed=True).agg(
        alns_runtime_seconds=("alns_runtime_seconds", "mean"),
        executed_iterations=("executed_iterations", "mean"),
        served_rate_pct=("served_rate_pct", "mean"),
    ).reset_index()
    agg["iters_per_sec"] = agg["executed_iterations"] / agg["alns_runtime_seconds"].replace(0, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    sns.lineplot(data=agg, x="cohort", y="alns_runtime_seconds", hue="mode", hue_order=MODE_ORDER, marker="o", ax=axes[0])
    axes[0].set_title("Runtime vs Problem Size")
    axes[0].set_xlabel("Students")
    axes[0].set_ylabel("ALNS Runtime (s)")

    sns.lineplot(data=agg, x="cohort", y="iters_per_sec", hue="mode", hue_order=MODE_ORDER, marker="o", ax=axes[1])
    axes[1].set_title("Iterations per Second vs Problem Size")
    axes[1].set_xlabel("Students")
    axes[1].set_ylabel("Iterations / second")
    axes[1].get_legend().remove()

    fig.savefig(FIGURES_DIR / "figure3_scalability_trends.png", dpi=300)
    fig.savefig(FIGURES_DIR / "figure3_scalability_trends.pdf")
    plt.close(fig)



def _figure4_walking_distribution(metrics_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    df = metrics_df[metrics_df["cohort"].isin(COMPLETE_COHORTS)].copy()
    df_w = df.dropna(subset=["avg_walk_dist_m"]).copy()

    if df_w.empty:
        axes[0].text(0.5, 0.5, "No walk-distance data available", ha="center", va="center")
        axes[0].set_axis_off()
        axes[1].text(0.5, 0.5, "No walk-distance data available", ha="center", va="center")
        axes[1].set_axis_off()
        fig.savefig(FIGURES_DIR / "figure4_walking_distribution.png", dpi=300)
        fig.savefig(FIGURES_DIR / "figure4_walking_distribution.pdf")
        plt.close(fig)
        return

    present_modes = [m for m in MODE_ORDER if (df_w["mode"] == m).any()]
    sns.boxplot(data=df_w, x="mode", y="avg_walk_dist_m", order=present_modes, ax=axes[0])
    axes[0].set_title("Average Walk Distance by Mode")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Meters")

    sns.boxplot(data=df_w, x="cohort", y="avg_walk_dist_m", hue="mode", hue_order=present_modes, ax=axes[1])
    axes[1].set_title("Walk Distance by Cohort and Mode")
    axes[1].set_xlabel("Students")
    axes[1].set_ylabel("Meters")

    fig.savefig(FIGURES_DIR / "figure4_walking_distribution.png", dpi=300)
    fig.savefig(FIGURES_DIR / "figure4_walking_distribution.pdf")
    plt.close(fig)



def _write_manifest(table1: pd.DataFrame, table2: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    complete = metrics_df[metrics_df["cohort"].isin(COMPLETE_COHORTS)]
    by_cohort = complete.groupby("cohort")["seed"].nunique().to_dict()

    lines = [
        "# Generated Experiment Artifacts",
        "",
        f"- Complete cohorts seed counts: {by_cohort}",
        f"- Table 1 rows: {len(table1)}",
        f"- Table 2 rows: {len(table2)}",
        "- Inference policy: paired Wilcoxon + rank-biserial effect size + Holm correction.",
    ]
    (TABLES_DIR / "artifact_manifest.md").write_text("\n".join(lines), encoding="utf-8")



def main() -> None:
    _ensure_dirs()
    metrics_df, conv_df = _load_runs()

    metrics_df.to_csv(TABLES_DIR / "raw_metrics_tidy.csv", index=False)
    if not conv_df.empty:
        conv_df.to_csv(TABLES_DIR / "raw_convergence_tidy.csv", index=False)

    table1 = _build_table1(metrics_df)
    table2 = _build_table2(metrics_df)

    _to_latex_table(
        table1,
        "Aggregate Metrics by Cohort and Mode (mean $\\pm$ SD)",
        "tab:aggregate_metrics",
        "table1_aggregate_metrics.tex",
    )
    _to_latex_table(
        table2,
        "Wilcoxon Paired Comparisons with Holm-Corrected p-values and Rank-Biserial Effect Size",
        "tab:wilcoxon_effects",
        "table2_wilcoxon_effects.tex",
    )

    _figure1_mode_comparison(metrics_df)
    _figure2_convergence(conv_df)
    _figure3_scalability(metrics_df)
    _figure4_walking_distribution(metrics_df)

    _write_manifest(table1, table2, metrics_df)

    print("Generated artifacts:")
    print(f"- Figures: {FIGURES_DIR}")
    print(f"- Tables:  {TABLES_DIR}")


if __name__ == "__main__":
    main()
