"""Score every submission CSV and rank entries by public mean distance.

Usage:
    python code/score_submission.py
    python code/score_submission.py --submission-dir submission
    python code/score_submission.py --submission-dir submission --solutions data/solutions.csv
"""

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SUBMISSION_DIR = Path("submission")
DEFAULT_SOLUTIONS_PATH = Path("data/solutions.csv")


def default_solutions_path():
    for path in [DEFAULT_SOLUTIONS_PATH, Path("solutions.csv")]:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find solutions.csv at data/solutions.csv or solutions.csv.")


def haversine(pred, gt):
    """
    Havarsine distance between two points on the Earth surface.

    Parameters
    -----
    pred: numpy array of shape (N, 2)
        Contains predicted (LATITUDE, LONGITUDE).
    gt: numpy array of shape (N, 2)
        Contains ground-truth (LATITUDE, LONGITUDE).

    Returns
    ------
    numpy array of shape (N,)
        Contains haversine distance between predictions
        and ground truth.
    """
    pred_lat = np.radians(pred[:, 0])
    pred_long = np.radians(pred[:, 1])
    gt_lat = np.radians(gt[:, 0])
    gt_long = np.radians(gt[:, 1])

    dlat = gt_lat - pred_lat
    dlon = gt_long - pred_long

    a = np.sin(dlat/2)**2 + np.cos(pred_lat) * \
        np.cos(gt_lat) * np.sin(dlon/2)**2

    d = 2 * 6371 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return d


def load_submission(path):
    submission = pd.read_csv(path, index_col="TRIP_ID")
    expected_columns = ["LATITUDE", "LONGITUDE"]
    missing = [column for column in expected_columns if column not in submission.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return submission[expected_columns]


def score_submission(submission_path, solutions):
    submission = load_submission(submission_path)
    scored_submission = submission.loc[solutions.index]

    public_mask = solutions["PUBLIC"] == True
    private_mask = solutions["PUBLIC"] == False

    all_distances = haversine(
        pred=scored_submission[["LATITUDE", "LONGITUDE"]].values,
        gt=solutions[["LATITUDE", "LONGITUDE"]].values,
    )
    public_distances = haversine(
        pred=scored_submission.loc[public_mask, ["LATITUDE", "LONGITUDE"]].values,
        gt=solutions.loc[public_mask, ["LATITUDE", "LONGITUDE"]].values,
    )
    private_distances = haversine(
        pred=scored_submission.loc[private_mask, ["LATITUDE", "LONGITUDE"]].values,
        gt=solutions.loc[private_mask, ["LATITUDE", "LONGITUDE"]].values,
    )

    metrics = {
        "submission": submission_path.name,
        "all_mean_km": float(all_distances.mean()),
        "all_median_km": float(np.median(all_distances)),
        "all_std_km": float(all_distances.std()),
        "all_p90_km": float(np.percentile(all_distances, 90)),
        "public_mean_km": float(public_distances.mean()),
        "private_mean_km": float(private_distances.mean()),
        "n_trips": int(len(all_distances)),
    }
    return metrics, all_distances, public_distances, private_distances


def configure_matplotlib_cache():
    cache_dir = Path(tempfile.gettempdir()) / "taxi_destination_matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")


def plot_distances(all_distances, public_distances, private_distances, output_path, title):
    configure_matplotlib_cache()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_distances, bins=40, alpha=0.75, label="All test trips")
    ax.hist(public_distances, bins=40, alpha=0.45, label="Public split")
    ax.hist(private_distances, bins=40, alpha=0.45, label="Private split")
    ax.axvline(all_distances.mean(), color="red", linestyle="--", label=f"Mean: {all_distances.mean():.3f} km")
    ax.set_xlabel("Haversine distance (km)")
    ax.set_ylabel("Test trips")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def find_submission_csvs(submission_dir):
    csv_paths = sorted(submission_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No submission CSV files found in {submission_dir}")
    return csv_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Score every CSV in the local submission directory.")
    parser.add_argument("--submission-dir", type=Path, default=DEFAULT_SUBMISSION_DIR)
    parser.add_argument("--solutions", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    solutions_path = args.solutions or default_solutions_path()
    if not solutions_path.exists():
        raise FileNotFoundError(f"solutions.csv not found at {solutions_path}")

    submission_dir = args.submission_dir
    solutions = pd.read_csv(solutions_path, index_col="TRIP_ID")
    results = []

    print(f"Submission directory: {submission_dir}")
    print(f"Solutions file: {solutions_path}")
    print()

    submission_paths = find_submission_csvs(submission_dir)

    scoring_dir = submission_dir / "scoring"
    plots_dir = scoring_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for submission_path in submission_paths:
        metrics, all_distances, public_distances, private_distances = score_submission(
            submission_path=submission_path,
            solutions=solutions,
        )
        results.append(metrics)

        plot_path = plots_dir / f"{submission_path.stem}_histogram.png"
        plot_distances(
            all_distances=all_distances,
            public_distances=public_distances,
            private_distances=private_distances,
            output_path=plot_path,
            title=f"{submission_path.stem} test error distribution",
        )
        print(f"Scored {submission_path.name}; plot written to {plot_path}")

    summary = pd.DataFrame(results).sort_values("public_mean_km", ascending=True)
    summary.insert(0, "rank_by_public_mean", range(1, len(summary) + 1))

    summary_path = scoring_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    print()
    print(summary.to_string(index=False))
    print()
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise SystemExit(f"Error: {exc}")
