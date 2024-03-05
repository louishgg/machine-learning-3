#!/usr/bin/env python3
"""Regenerate report figures and LaTeX tables from repository sources.

The script is intentionally lightweight: it does not train models or execute
notebooks. It re-scores saved submissions, validates structured summaries, and
renders publication-ready static figures for the LaTeX report.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent
ROOT = REPORT_DIR.parent
FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"
SOURCE_DIR = REPORT_DIR / "sources"

os.environ.setdefault(
    "XDG_CACHE_HOME", str(REPORT_DIR / "build" / "cache")
)
os.environ.setdefault(
    "MPLCONFIGDIR", str(REPORT_DIR / "build" / "matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
    }
)

BLUE = "#2563EB"
BLUE_LIGHT = "#93C5FD"
GOLD = "#D6A21E"
INK = "#1F2937"
GREY = "#9CA3AF"
GRID = "#E5E7EB"
WHITE = "#FFFFFF"

MODEL_LABELS = {
    "ensemble": "Ensemble",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "decision_tree": "Decision Tree",
    "catboost": "CatBoost",
    "ridge": "Ridge",
    "linear_regression": "Linear Regression",
    "knn": "KNN",
}


def haversine_km(predicted_lat_lon: np.ndarray, truth_lat_lon: np.ndarray) -> np.ndarray:
    """Vectorized spherical Haversine distance with the assignment radius."""
    pred = np.radians(np.asarray(predicted_lat_lon, dtype=float))
    truth = np.radians(np.asarray(truth_lat_lon, dtype=float))
    dlat = truth[:, 0] - pred[:, 0]
    dlon = truth[:, 1] - pred[:, 1]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(pred[:, 0]) * np.cos(truth[:, 0]) * np.sin(dlon / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * 6371.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(WHITE)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(
        FIGURE_DIR / f"{stem}.png",
        dpi=220,
        bbox_inches="tight",
        facecolor=WHITE,
    )
    plt.close(fig)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.replace("\u2013", "--").replace("\u2014", "---")


def load_zip_json(archive_path: Path, member: str) -> dict:
    with zipfile.ZipFile(archive_path) as archive:
        payload = archive.read(member)
    return json.loads(payload)


def reproduce_local_scores() -> pd.DataFrame:
    solutions = pd.read_csv(ROOT / "data" / "solutions.csv", dtype={"TRIP_ID": str})
    expected_columns = {"TRIP_ID", "LATITUDE", "LONGITUDE", "PUBLIC"}
    if set(solutions.columns) != expected_columns:
        raise ValueError(f"Unexpected solutions schema: {solutions.columns.tolist()}")
    if solutions["TRIP_ID"].duplicated().any() or len(solutions) != 320:
        raise ValueError("Solutions must contain 320 unique TRIP_ID values.")
    public_mask = solutions["PUBLIC"].astype(bool).to_numpy()
    if int(public_mask.sum()) != 160:
        raise ValueError("Expected 160 public and 160 private solution rows.")

    saved_summary = pd.read_csv(ROOT / "submission" / "scoring" / "summary.csv")
    saved_by_file = saved_summary.set_index("submission")
    rows: list[dict[str, object]] = []
    truth = solutions[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=float)

    for slug in MODEL_LABELS:
        filename = f"submission_{slug}.csv"
        submission_path = ROOT / "submission" / filename
        submission = pd.read_csv(submission_path, dtype={"TRIP_ID": str})
        if submission["TRIP_ID"].duplicated().any() or len(submission) != len(solutions):
            raise ValueError(f"{filename} does not contain 320 unique trips.")
        aligned = solutions[["TRIP_ID"]].merge(
            submission,
            on="TRIP_ID",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if aligned[["LATITUDE", "LONGITUDE"]].isna().any().any():
            raise ValueError(f"{filename} does not cover every solution trip.")
        model_distances = haversine_km(
            aligned[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=float), truth
        )
        metrics = {
            "all_mean_km": float(model_distances.mean()),
            "all_median_km": float(np.median(model_distances)),
            "all_std_km": float(model_distances.std(ddof=0)),
            "all_p90_km": float(np.quantile(model_distances, 0.90)),
            "public_mean_km": float(model_distances[public_mask].mean()),
            "private_mean_km": float(model_distances[~public_mask].mean()),
        }
        saved = saved_by_file.loc[filename]
        for metric, value in metrics.items():
            if not np.isclose(value, float(saved[metric]), atol=1e-10, rtol=0.0):
                raise ValueError(f"Independent score drift for {filename}: {metric}")
        rows.append(
            {
                "model_slug": slug,
                "model": MODEL_LABELS[slug],
                "all_mean_km": metrics["all_mean_km"],
                "public_mean_km": metrics["public_mean_km"],
                "private_mean_km": metrics["private_mean_km"],
            }
        )

    result = pd.DataFrame(rows)
    result["public_rank"] = result["public_mean_km"].rank(method="min").astype(int)
    result = result.sort_values("public_rank").reset_index(drop=True)
    return result


def plot_public_ranking(scores: pd.DataFrame) -> None:
    plot = scores.sort_values("public_mean_km", ascending=False)
    colors = [GOLD if slug == "ensemble" else BLUE for slug in plot["model_slug"]]
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    bars = ax.barh(plot["model"], plot["public_mean_km"], color=colors, edgecolor=INK, linewidth=0.45)
    ax.set_xlim(0, max(3.65, float(plot["public_mean_km"].max()) * 1.08))
    ax.set_xlabel("Mean Haversine distance (km per trip; lower is better)")
    ax.set_title("Local public-score ranking", loc="left", fontsize=13, fontweight="bold")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, plot["public_mean_km"]):
        ax.text(value + 0.035, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", color=INK, fontsize=9)
    style_axes(ax)
    fig.text(
        0.125,
        0.005,
        "Population: 160 locally designated public trips.",
        color=INK,
        fontsize=8.5,
    )
    save_figure(fig, "public_score_ranking")


def plot_ensemble_repeats(repeats: pd.DataFrame, ensemble_artifact: dict) -> None:
    mean_gain_m = float(repeats["gain_m"].mean())
    interval = ensemble_artifact["promotion"]["trip_cluster_robust_interval"]
    fig, ax = plt.subplots(figsize=(7.7, 4.25))
    x = repeats["repeat"].to_numpy(dtype=int)
    y = repeats["gain_m"].to_numpy(dtype=float)
    ax.scatter(x, y, s=72, color=BLUE, edgecolor=INK, linewidth=0.6, zorder=3)
    ax.plot(x, y, color=BLUE_LIGHT, linewidth=1.4, zorder=2)
    ax.axhline(0.0, color=INK, linewidth=1.0)
    ax.axhline(mean_gain_m, color=GOLD, linewidth=1.8, linestyle="--", label=f"Mean: {mean_gain_m:.2f} m")
    ax.axhspan(float(interval["lower_km"]) * 1000, float(interval["upper_km"]) * 1000, color=GOLD, alpha=0.16, label="95% robust interval")
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 0.055, f"{yi:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xlabel("Grouped cross-fitting repeat")
    ax.set_ylabel("Gain over Random Forest (m; higher is better)")
    ax.set_title("Ensemble gain by grouped cross-fitting repeat", loc="left", fontsize=13, fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    style_axes(ax)
    save_figure(fig, "ensemble_repeat_gain")


def plot_prefix_quantiles() -> None:
    prefix = pd.read_csv(ROOT / "outputs" / "dataset_analysis" / "tables" / "prefix_length_summary.csv")
    wanted = ["1%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%"]
    data = prefix.set_index("statistic").loc[wanted]
    x = np.arange(len(wanted))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(x, data["test_matched_prefix"], color=BLUE, marker="s", markersize=5.8, linewidth=1.8, label="Selected test-matched train prefixes", zorder=3)
    ax.plot(x, data["competition_test_prefix"], color=GOLD, marker="o", markersize=8.0, markerfacecolor=WHITE, markeredgewidth=1.8, linewidth=1.5, linestyle=":", label="Competition test prefixes", zorder=4)
    ax.plot(x, data["uniform_prefix"], color=GREY, marker="^", linewidth=1.3, linestyle="--", label="Rejected uniform-prefix strategy")
    ax.set_yscale("log")
    ax.set_xticks(x, wanted)
    ax.set_xlabel("Prefix-length quantile")
    ax.set_ylabel("Observed GPS points (logarithmic scale)")
    ax.set_title("Prefix-length quantiles by sampling strategy", loc="left", fontsize=13, fontweight="bold")
    ax.grid(axis="y", which="both", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    style_axes(ax)
    save_figure(fig, "prefix_length_quantiles")


def write_dataset_table(default_meta: dict, cat_meta: dict) -> None:
    quality = default_meta["raw"]["quality_filter_summary"]
    lines = [
        r"\begin{tabularx}{\linewidth}{@{}Xrr@{}}",
        r"\toprule",
        r"Population or representation & Rows & Features \\",
        r"\midrule",
        f"Raw training trajectories & {quality['input_rows']:,} & 9 " + r"\\",
        r"Competition test prefixes & 320 & 9 \\",
        f"Retained training trajectories & {quality['retained_rows']:,} & 8 raw fields " + r"\\",
        f"Validation training matrix (default) & {default_meta['validation']['X_train_shape'][0]:,} & {default_meta['validation']['X_train_shape'][1]:,} " + r"\\",
        f"Validation holdout matrix (default) & {default_meta['validation']['X_val_shape'][0]:,} & {default_meta['validation']['X_val_shape'][1]:,} " + r"\\",
        f"Final training matrix (default) & {default_meta['final']['X_train_shape'][0]:,} & {default_meta['final']['X_train_shape'][1]:,} " + r"\\",
        f"Final training matrix (CatBoost) & {cat_meta['final']['X_train_shape'][0]:,} & {cat_meta['final']['X_train_shape'][1]:,} " + r"\\",
        r"\bottomrule",
        r"\end{tabularx}",
        "",
    ]
    (TABLE_DIR / "dataset_summary.tex").write_text("\n".join(lines), encoding="utf-8")


def write_dataset_schema_table() -> None:
    """Write the self-contained raw CSV data dictionary used by the report."""
    rows = [
        (
            "TRIP_ID",
            "String",
            "May repeat",
            "Trip identifier.",
        ),
        (
            "CALL_TYPE",
            "Character",
            "A, B, or C",
            "Dispatch mode: central call, taxi stand, or street hail.",
        ),
        (
            "ORIGIN_CALL",
            "Integer",
            "Type A; may be missing",
            "Anonymized caller identifier when a central-call trip provides one.",
        ),
        (
            "ORIGIN_STAND",
            "Integer",
            "Type B; otherwise missing",
            "Taxi-stand identifier for stand-based trips.",
        ),
        (
            "TAXI_ID",
            "Integer",
            "442 taxis",
            "Identifier of the taxi associated with the trip.",
        ),
        (
            "TIMESTAMP",
            "Integer",
            "Unix seconds",
            "Trip start time.",
        ),
        (
            "DAY_TYPE",
            "Character",
            "A, B, or C",
            "Normal day, holiday, or day preceding a holiday, respectively.",
        ),
        (
            "MISSING_DATA",
            "Boolean",
            "True or False",
            "Whether the recorded GPS stream contains missing samples.",
        ),
        (
            "POLYLINE",
            "List as string",
            "Coordinate pairs",
            "GPS samples stored as [LONGITUDE, LATITUDE] every 15 seconds; the final training point is the destination, whereas test rows contain only the observed prefix.",
        ),
    ]
    lines = [
        r"\begin{tabularx}{\linewidth}{@{}>{\ttfamily\raggedright\arraybackslash}p{2.5cm}>{\raggedright\arraybackslash}p{1.9cm}>{\raggedright\arraybackslash}p{2.8cm}X@{}}",
        r"\toprule",
        r"\multicolumn{1}{@{}l}{Field} & Type & Values / scope & Meaning \\",
        r"\midrule",
    ]
    for field, field_type, values, meaning in rows:
        lines.append(
            f"{latex_escape(field)} & {latex_escape(field_type)} & "
            f"{latex_escape(values)} & {latex_escape(meaning)} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", ""])
    (TABLE_DIR / "dataset_schema.tex").write_text("\n".join(lines), encoding="utf-8")


def write_model_table(
    scores: pd.DataFrame, validation: pd.DataFrame, knn_artifact: dict
) -> None:
    comparison = scores.merge(
        validation[["model_slug", "val_mean_km", "fit_seconds"]],
        on="model_slug",
        how="left",
    )
    knn_metrics = knn_artifact["validation_metrics"]
    comparison.loc[comparison["model_slug"].eq("knn"), "val_mean_km"] = float(
        knn_metrics["validation_sample/haversine_mean_km"]
    )
    comparison.loc[comparison["model_slug"].eq("knn"), "fit_seconds"] = float(
        knn_metrics["fit_seconds"]
    )
    comparison = comparison.sort_values("public_rank")
    lines = [
        r"\begin{tabular}{@{}rlrrrrr@{}}",
        r"\toprule",
        r"Rank & Model & Public & Private & All & Validation & Fit (s) \\",
        r"\midrule",
    ]
    for row in comparison.itertuples(index=False):
        model = latex_escape(row.model)
        if row.model_slug == "ensemble":
            model += r"$^{\dagger}$"
            fit = "--"
        elif row.model_slug == "knn":
            model += r"$^{\ddagger}$"
            fit = f"{row.fit_seconds:.3f}"
        else:
            fit = f"{row.fit_seconds:,.1f}"
        lines.append(
            f"{row.public_rank} & {model} & {row.public_mean_km:.3f} & "
            f"{row.private_mean_km:.3f} & {row.all_mean_km:.3f} & "
            f"{row.val_mean_km:.3f} & {fit} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (TABLE_DIR / "model_comparison.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    default_meta = load_zip_json(ROOT / "data" / "preprocessed_data.zip", "data/metadata.json")
    cat_meta = load_zip_json(
        ROOT / "data" / "preprocessed_data_catboost.zip", "data/metadata.json"
    )
    scores = reproduce_local_scores()

    validation = pd.read_csv(
        SOURCE_DIR / "selected_validation_by_model.csv"
    )
    repeats = pd.read_csv(
        SOURCE_DIR / "ensemble_repeat_results.csv"
    )
    knn_artifact = json.loads(
        (ROOT / "outputs" / "artifacts" / "knn" / "best_params_knn.json").read_text(
            encoding="utf-8"
        )
    )
    ensemble_artifact = json.loads(
        (ROOT / "outputs" / "artifacts" / "ensemble" / "best_params_ensemble.json").read_text(
            encoding="utf-8"
        )
    )

    expected_models = set(MODEL_LABELS)
    if set(scores["model_slug"]) != expected_models:
        raise ValueError("The current scored model set is incomplete.")
    if set(validation["model_slug"]) != expected_models - {"knn"}:
        raise ValueError("Unexpected accepted full-validation model set.")

    plot_public_ranking(scores)
    plot_ensemble_repeats(repeats, ensemble_artifact)
    plot_prefix_quantiles()
    write_dataset_schema_table()
    write_dataset_table(default_meta, cat_meta)
    write_model_table(scores, validation, knn_artifact)

    print("Generated three PNG figures and three LaTeX tables.")


if __name__ == "__main__":
    main()
