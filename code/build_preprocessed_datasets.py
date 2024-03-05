#!/usr/bin/env python
"""Build reusable preprocessed Porto taxi destination datasets.

This script mirrors the shared preprocessing code used by the Kaggle model
notebooks, then saves the validation-comparison and final-submission matrices
once so each model can load identical data. The default dataset retains the
existing one-hot feature contract, while the CatBoost variant preserves native
categorical strings.
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import platform
import time
import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow
import sklearn
from sklearn.model_selection import train_test_split


DEFAULT_TRAIN_PATH = Path("/kaggle/input/datasets/louishogge/taxi-dataset/data/train.csv")
DEFAULT_TEST_PATH = Path("/kaggle/input/datasets/louishogge/taxi-dataset/data/test.csv")
DEFAULT_OUTPUT_DIR = Path("/kaggle/working/preprocessed_taxi_destination")
DEFAULT_CATBOOST_OUTPUT_DIR = Path(
    "/kaggle/working/preprocessed_taxi_destination_catboost"
)

DEFAULT_DATASET_VARIANT = "default"
CATBOOST_DATASET_VARIANT = "catboost"
DATASET_VARIANTS = (DEFAULT_DATASET_VARIANT, CATBOOST_DATASET_VARIANT)

SEED = 42
VALIDATION_SIZE = 0.20
PREFIX_SAMPLE_FACTOR = 1

BROAD_SANITY_BOUNDS = {
    "lon_min": -8.85,
    "lon_max": -7.40,
    "lat_min": 40.85,
    "lat_max": 41.65,
}
DURATION_ARTIFACT_MINUTES = 240
DURATION_ARTIFACT_MAX_DISTANCE_FROM_START_KM = 20

TARGET_COLUMNS = ["END_LONG", "END_LAT"]
EDGE_POINT_COUNT = 5
GPS_FEATURE_COLUMNS = [
    feature
    for point_index in range(EDGE_POINT_COUNT * 2)
    for feature in (f"LONG_{point_index}", f"LAT_{point_index}")
]
CALL_TYPE_COLUMNS = ["CALL_TYPE_A", "CALL_TYPE_B", "CALL_TYPE_C"]
ORIGIN_CALL_FEATURE_COLUMNS = ["ORIGIN_CALL_MISSING", "ORIGIN_CALL_LOG_COUNT"]
TEMPORAL_PREFIX_FEATURE_COLUMNS = [
    "WEEK",
    "DAY",
    "QUARTER",
    "WEEK_SIN",
    "WEEK_COS",
    "DAY_SIN",
    "DAY_COS",
    "QUARTER_SIN",
    "QUARTER_COS",
    "PREFIX_LENGTH",
]
NUMERIC_METADATA_FEATURE_COLUMNS = (
    ORIGIN_CALL_FEATURE_COLUMNS + TEMPORAL_PREFIX_FEATURE_COLUMNS
)
CATBOOST_CATEGORICAL_FEATURE_COLUMNS = [
    "CALL_TYPE",
    "ORIGIN_CALL",
    "ORIGIN_STAND",
    "TAXI_ID",
]
CATBOOST_NUMERICAL_FEATURE_COLUMNS = (
    NUMERIC_METADATA_FEATURE_COLUMNS + GPS_FEATURE_COLUMNS
)
CATBOOST_FEATURE_COLUMNS = (
    CATBOOST_NUMERICAL_FEATURE_COLUMNS + CATBOOST_CATEGORICAL_FEATURE_COLUMNS
)
ORIGIN_STAND_PREFIX = "ORIGIN_STAND"
TAXI_ID_PREFIX = "TAXI_ID"
ORIGIN_STAND_CATEGORY_COLUMN = "__ORIGIN_STAND_CATEGORY"
TAXI_ID_CATEGORY_COLUMN = "__TAXI_ID_CATEGORY"
PORTO_TIMEZONE = "Europe/Lisbon"
SECONDS_PER_GPS_STEP = 15
PREFIX_LENGTH_PERCENTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
PARQUET_COMPRESSION = "snappy"
SCHEMA_VERSION = 1
CATBOOST_SCHEMA_VERSION = 2
MISSING_CATEGORY = "0"
CATBOOST_MISSING_CATEGORY = "__MISSING__"

DEFAULT_CATEGORICAL_ENCODING_POLICY = {
    "CALL_TYPE": "one_hot",
    "ORIGIN_CALL": "missing_indicator_plus_log_count",
    "ORIGIN_STAND": "one_hot_missing_as_0",
    "TAXI_ID": "one_hot",
}
CATBOOST_CATEGORICAL_ENCODING_POLICY = {
    "CALL_TYPE": "native_string",
    "ORIGIN_CALL": "native_string_plus_missing_indicator_and_log_count",
    "ORIGIN_STAND": "native_string",
    "TAXI_ID": "native_string",
}


def log(message: str) -> None:
    """Print a timestamped progress message."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def validate_dataset_variant(dataset_variant: str) -> None:
    """Require one of the supported preprocessing dataset variants."""
    if dataset_variant not in DATASET_VARIANTS:
        raise ValueError(
            f"Unsupported dataset_variant={dataset_variant!r}; "
            f"expected one of {DATASET_VARIANTS!r}."
        )


def resolve_output_dir(dataset_variant: str, output_dir: Path | None) -> Path:
    """Resolve the variant-specific output directory unless explicitly overridden."""
    validate_dataset_variant(dataset_variant)
    if output_dir is not None:
        return Path(output_dir)
    if dataset_variant == CATBOOST_DATASET_VARIANT:
        return DEFAULT_CATBOOST_OUTPUT_DIR
    return DEFAULT_OUTPUT_DIR


def schema_version_for_variant(dataset_variant: str) -> int:
    """Return the cached dataset schema version for one variant."""
    validate_dataset_variant(dataset_variant)
    if dataset_variant == CATBOOST_DATASET_VARIANT:
        return CATBOOST_SCHEMA_VERSION
    return SCHEMA_VERSION


def categorical_encoding_policy_for_variant(dataset_variant: str) -> dict[str, str]:
    """Return a copy of the categorical encoding contract for one variant."""
    validate_dataset_variant(dataset_variant)
    if dataset_variant == CATBOOST_DATASET_VARIANT:
        return CATBOOST_CATEGORICAL_ENCODING_POLICY.copy()
    return DEFAULT_CATEGORICAL_ENCODING_POLICY.copy()


def parse_n_rows(value: str | None) -> int | None:
    """Parse an optional positive row count."""
    if value is None or value.lower() in {"none", "null", ""}:
        return None
    n_rows = int(value)
    if n_rows <= 0:
        raise argparse.ArgumentTypeError("--n-rows must be a positive integer or None.")
    return n_rows


def count_csv_rows(csv_path: Path) -> int:
    """Count data rows in a plain or single-file zipped CSV."""
    if csv_path.suffix == ".zip":
        with zipfile.ZipFile(csv_path) as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(csv_names) != 1:
                raise ValueError(f"Expected exactly one CSV inside {csv_path}, found {csv_names!r}.")
            with archive.open(csv_names[0]) as csv_file:
                return max(sum(1 for _ in csv_file) - 1, 0)

    with open(csv_path, "r", encoding="utf-8") as csv_file:
        return max(sum(1 for _ in csv_file) - 1, 0)


def read_data(csv_path: Path, n_rows: int | None = None, seed: int = SEED) -> pd.DataFrame:
    """Read a trip CSV, optionally using a deterministic random row sample."""
    read_csv_kwargs = {"index_col": "TRIP_ID"}
    if n_rows is None:
        return pd.read_csv(csv_path, **read_csv_kwargs)

    total_rows = count_csv_rows(csv_path)
    if n_rows >= total_rows:
        return pd.read_csv(csv_path, **read_csv_kwargs)

    rng = np.random.default_rng(seed)
    selected_line_numbers = set(
        rng.choice(np.arange(1, total_rows + 1), size=n_rows, replace=False).tolist()
    )
    return pd.read_csv(
        csv_path,
        skiprows=lambda line_number: line_number != 0 and line_number not in selected_line_numbers,
        **read_csv_kwargs,
    )


def parse_polyline(polyline: Any) -> list:
    """Parse and validate a POLYLINE value."""
    if isinstance(polyline, list):
        parsed = polyline
    else:
        parsed = ast.literal_eval(polyline)
    if not isinstance(parsed, list):
        raise ValueError(f"POLYLINE must parse to a list, got {type(parsed)!r}")
    for point in parsed:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"Invalid GPS point in POLYLINE: {point!r}")
    return parsed


def safe_polyline_length(polyline: Any) -> float:
    """Return the parsed polyline length, or NaN when parsing fails."""
    try:
        return len(parse_polyline(polyline))
    except Exception:
        return np.nan


def get_prefix_lengths(data: pd.DataFrame, require_target: bool) -> np.ndarray:
    """Return valid observable prefix lengths for train or test rows."""
    full_lengths = data["POLYLINE"].map(safe_polyline_length).dropna().astype(int)
    if require_target:
        return (full_lengths[full_lengths >= 2] - 1).to_numpy(dtype=int)
    return full_lengths[full_lengths >= 1].to_numpy(dtype=int)


def filter_valid_polyline_rows(data: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Keep rows with usable polylines and optional targets."""
    filtered = data.copy()
    if "MISSING_DATA" in filtered.columns:
        filtered = filtered[filtered["MISSING_DATA"] == False].copy()

    lengths = filtered["POLYLINE"].map(safe_polyline_length)
    min_length = 2 if require_target else 1
    valid_length_mask = (lengths >= min_length).to_numpy()
    filtered = filtered[valid_length_mask].copy()
    filtered["__POLYLINE_LENGTH"] = lengths.to_numpy()[valid_length_mask].astype(int)
    return filtered


def is_point_inside_bounds(point: Any, bounds: dict[str, float]) -> bool:
    """Check whether one GPS point falls inside the configured bounds."""
    try:
        longitude = float(point[0])
        latitude = float(point[1])
    except Exception:
        return False
    return (
        bounds["lon_min"] <= longitude <= bounds["lon_max"]
        and bounds["lat_min"] <= latitude <= bounds["lat_max"]
    )


def max_distance_from_start_km(points: list) -> float:
    """Return the farthest distance from the first GPS point."""
    if not isinstance(points, list) or len(points) == 0:
        return np.nan

    start_longitude, start_latitude = points[0]
    longitudes = np.radians(np.asarray([point[0] for point in points], dtype=float))
    latitudes = np.radians(np.asarray([point[1] for point in points], dtype=float))
    start_longitude = np.radians(float(start_longitude))
    start_latitude = np.radians(float(start_latitude))

    delta_longitude = longitudes - start_longitude
    delta_latitude = latitudes - start_latitude
    haversine_a = (
        np.sin(delta_latitude / 2) ** 2
        + np.cos(start_latitude) * np.cos(latitudes) * np.sin(delta_longitude / 2) ** 2
    )
    distances_km = 2 * 6371 * np.arctan2(np.sqrt(haversine_a), np.sqrt(1 - haversine_a))
    return float(np.nanmax(distances_km))


def apply_train_quality_filters(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Apply train-only missing, spatial, and duration filters."""
    valid = filter_valid_polyline_rows(data, require_target=True)
    parsed_polylines = valid["POLYLINE"].map(parse_polyline)
    lengths = valid["__POLYLINE_LENGTH"].to_numpy(dtype=int)

    starts_inside_bounds = np.array([
        is_point_inside_bounds(points[0], BROAD_SANITY_BOUNDS)
        for points in parsed_polylines
    ], dtype=bool)
    destinations_inside_bounds = np.array([
        is_point_inside_bounds(points[-1], BROAD_SANITY_BOUNDS)
        for points in parsed_polylines
    ], dtype=bool)
    spatial_outlier = ~(starts_inside_bounds & destinations_inside_bounds)

    duration_minutes = np.maximum(lengths - 1, 0) * SECONDS_PER_GPS_STEP / 60
    long_duration = duration_minutes > DURATION_ARTIFACT_MINUTES
    duration_artifact = np.zeros(len(valid), dtype=bool)
    long_positions = np.flatnonzero(long_duration)
    if len(long_positions) > 0:
        max_distances = np.array([
            max_distance_from_start_km(parsed_polylines.iloc[position])
            for position in long_positions
        ], dtype=float)
        duration_artifact[long_positions] = (
            max_distances <= DURATION_ARTIFACT_MAX_DISTANCE_FROM_START_KM
        )

    keep_rows = ~spatial_outlier & ~duration_artifact
    summary = pd.Series({
        "input_rows": int(len(data)),
        "valid_polyline_rows": int(len(valid)),
        "invalid_or_missing_rows_removed": int(len(data) - len(valid)),
        "broad_spatial_rows_removed": int(spatial_outlier.sum()),
        "duration_artifact_rows_removed": int(duration_artifact.sum()),
        "spatial_and_duration_overlap_rows": int((spatial_outlier & duration_artifact).sum()),
        "retained_rows": int(keep_rows.sum()),
    })

    return valid.loc[keep_rows].drop(columns=["__POLYLINE_LENGTH"], errors="ignore").copy(), summary


def pad_first_points(prefix_points: list, n_points: int = EDGE_POINT_COUNT) -> list:
    """Return the first GPS points, padding short prefixes with the first point."""
    selected = list(prefix_points[:n_points])
    if len(selected) < n_points:
        selected = [prefix_points[0]] * (n_points - len(selected)) + selected
    return selected


def pad_last_points(prefix_points: list, n_points: int = EDGE_POINT_COUNT) -> list:
    """Return the last GPS points, padding short prefixes with the last point."""
    selected = list(prefix_points[-n_points:])
    if len(selected) < n_points:
        selected = selected + [prefix_points[-1]] * (n_points - len(selected))
    return selected


def extract_prefix_gps_features(prefix_points: list) -> dict[str, float]:
    """Build fixed-width GPS features from one observed prefix."""
    if len(prefix_points) == 0:
        raise ValueError("Cannot extract features from an empty trajectory prefix.")

    selected_points = pad_first_points(prefix_points) + pad_last_points(prefix_points)
    features = {}
    for point_index, point in enumerate(selected_points):
        features[f"LONG_{point_index}"] = float(point[0])
        features[f"LAT_{point_index}"] = float(point[1])
    return features


def timestamp_to_porto_local(timestamp_seconds: int) -> pd.Timestamp:
    """Convert a Unix timestamp to local Porto time."""
    return pd.to_datetime(timestamp_seconds, unit="s", utc=True).tz_convert(PORTO_TIMEZONE)


def cyclic_time_features(value: int, period: int, offset: int = 0) -> tuple[float, float]:
    """Encode a cyclic value as sine and cosine features."""
    angle = 2 * np.pi * ((int(value) - offset) / period)
    return float(np.sin(angle)), float(np.cos(angle))


def normalize_category_id(value: Any, missing_category: str | None = MISSING_CATEGORY) -> str | None:
    """Normalize numeric-looking categorical IDs to stable string keys."""
    if pd.isna(value):
        return missing_category

    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.upper() in {"NA", "NAN", "NONE", "NULL"}:
            return missing_category
        value = stripped

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if np.isfinite(numeric_value) and numeric_value.is_integer():
        return str(int(numeric_value))
    return str(value)


def category_sort_key(category: str) -> tuple[int, int | str]:
    """Sort numeric category keys numerically, then non-numeric keys lexicographically."""
    try:
        return (0, int(category))
    except ValueError:
        return (1, category)


def fit_categorical_encoder(
    data: pd.DataFrame,
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
) -> dict[str, Any]:
    """Fit train-only categorical encodings for ID-like fields."""
    validate_dataset_variant(dataset_variant)

    if dataset_variant == CATBOOST_DATASET_VARIANT:
        call_type_categories = sorted(
            {
                normalize_category_id(value, missing_category=CATBOOST_MISSING_CATEGORY)
                for value in data["CALL_TYPE"]
            },
            key=category_sort_key,
        )
        origin_call_categories = sorted(
            {
                normalize_category_id(value, missing_category=CATBOOST_MISSING_CATEGORY)
                for value in data["ORIGIN_CALL"]
            },
            key=category_sort_key,
        )
        origin_stand_categories = sorted(
            {
                normalize_category_id(value, missing_category=CATBOOST_MISSING_CATEGORY)
                for value in data["ORIGIN_STAND"]
            },
            key=category_sort_key,
        )
        taxi_id_categories = sorted(
            {
                normalize_category_id(value, missing_category=CATBOOST_MISSING_CATEGORY)
                for value in data["TAXI_ID"]
            },
            key=category_sort_key,
        )
        origin_call_values = data["ORIGIN_CALL"].map(
            lambda value: normalize_category_id(value, missing_category=None)
        )
        origin_call_counts = origin_call_values.dropna().value_counts().to_dict()
        known_categories = {
            "CALL_TYPE": call_type_categories,
            "ORIGIN_CALL": origin_call_categories,
            "ORIGIN_STAND": origin_stand_categories,
            "TAXI_ID": taxi_id_categories,
        }
        return {
            "dataset_variant": CATBOOST_DATASET_VARIANT,
            "missing_category": CATBOOST_MISSING_CATEGORY,
            "categorical_feature_columns": CATBOOST_CATEGORICAL_FEATURE_COLUMNS.copy(),
            "known_categories": known_categories,
            "known_category_counts": {
                column: len(categories)
                for column, categories in known_categories.items()
            },
            "origin_call_counts": {
                str(key): int(value)
                for key, value in origin_call_counts.items()
            },
        }

    origin_stand_categories = sorted(
        {normalize_category_id(value) for value in data["ORIGIN_STAND"]},
        key=category_sort_key,
    )
    taxi_id_categories = sorted(
        {normalize_category_id(value) for value in data["TAXI_ID"]},
        key=category_sort_key,
    )
    origin_call_values = data["ORIGIN_CALL"].map(
        lambda value: normalize_category_id(value, missing_category=None)
    )
    origin_call_counts = origin_call_values.dropna().value_counts().to_dict()

    return {
        "origin_stand_categories": origin_stand_categories,
        "origin_stand_columns": [
            f"{ORIGIN_STAND_PREFIX}_{category}" for category in origin_stand_categories
        ],
        "taxi_id_categories": taxi_id_categories,
        "taxi_id_columns": [f"{TAXI_ID_PREFIX}_{category}" for category in taxi_id_categories],
        "origin_call_counts": {str(key): int(value) for key, value in origin_call_counts.items()},
    }


def add_one_hot_columns(
    data: pd.DataFrame,
    category_column: str,
    categories: list[str],
    output_columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Expand one temporary category column into fixed train-fitted one-hot columns."""
    category_values = pd.Categorical(data.pop(category_column), categories=categories)
    dummies = pd.get_dummies(
        pd.Series(category_values, index=data.index),
        prefix=prefix,
        dtype=np.uint8,
    )
    dummies = dummies.reindex(columns=output_columns, fill_value=0)
    return pd.concat([data, dummies], axis=1)


def apply_categorical_encoder(
    data: pd.DataFrame,
    categorical_encoder: dict[str, Any],
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
) -> pd.DataFrame:
    """Apply fitted categorical encodings to a processed feature table."""
    validate_dataset_variant(dataset_variant)
    encoded = data.copy()
    encoded["ORIGIN_CALL_MISSING"] = encoded["ORIGIN_CALL_MISSING"].astype(np.uint8)
    encoded["ORIGIN_CALL_LOG_COUNT"] = encoded["ORIGIN_CALL_LOG_COUNT"].astype(np.float32)

    if dataset_variant == CATBOOST_DATASET_VARIANT:
        for column in CATBOOST_CATEGORICAL_FEATURE_COLUMNS:
            if column not in encoded.columns:
                raise ValueError(f"CatBoost categorical feature is missing: {column}")
            encoded[column] = encoded[column].map(
                lambda value: normalize_category_id(
                    value,
                    missing_category=CATBOOST_MISSING_CATEGORY,
                )
            )
            if encoded[column].isna().any():
                raise ValueError(
                    f"CatBoost categorical feature contains missing values: {column}"
                )
            if not encoded[column].map(lambda value: isinstance(value, str)).all():
                raise TypeError(f"CatBoost categorical feature must contain strings: {column}")

        present_targets = [column for column in TARGET_COLUMNS if column in encoded.columns]
        if present_targets and present_targets != TARGET_COLUMNS:
            raise ValueError(
                "CatBoost processed data must contain either both target columns or neither."
            )
        expected_columns = CATBOOST_FEATURE_COLUMNS + present_targets
        missing_columns = [column for column in expected_columns if column not in encoded.columns]
        unexpected_columns = [
            column
            for column in encoded.columns
            if column not in expected_columns
        ]
        if missing_columns or unexpected_columns:
            raise ValueError(
                "CatBoost processed columns do not match the schema. "
                f"Missing: {missing_columns}; unexpected: {unexpected_columns}."
            )
        return encoded.reindex(columns=expected_columns)

    encoded = add_one_hot_columns(
        encoded,
        category_column=ORIGIN_STAND_CATEGORY_COLUMN,
        categories=categorical_encoder["origin_stand_categories"],
        output_columns=categorical_encoder["origin_stand_columns"],
        prefix=ORIGIN_STAND_PREFIX,
    )
    encoded = add_one_hot_columns(
        encoded,
        category_column=TAXI_ID_CATEGORY_COLUMN,
        categories=categorical_encoder["taxi_id_categories"],
        output_columns=categorical_encoder["taxi_id_columns"],
        prefix=TAXI_ID_PREFIX,
    )
    return encoded


def describe_categorical_encoder(
    categorical_encoder: dict[str, Any],
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
) -> dict[str, Any]:
    """Return compact metadata for fitted categorical encodings."""
    validate_dataset_variant(dataset_variant)
    origin_call_counts = categorical_encoder["origin_call_counts"]

    if dataset_variant == CATBOOST_DATASET_VARIANT:
        known_categories = categorical_encoder["known_categories"]
        return {
            "call_type": {
                "encoding": "native_string",
                "category_count": len(known_categories["CALL_TYPE"]),
                "feature_columns": ["CALL_TYPE"],
            },
            "origin_call": {
                "encoding": "native_string_plus_missing_indicator_and_log_count",
                "category_count": len(known_categories["ORIGIN_CALL"]),
                "unique_non_missing_categories": len(origin_call_counts),
                "max_count": max(origin_call_counts.values(), default=0),
                "feature_columns": ["ORIGIN_CALL", *ORIGIN_CALL_FEATURE_COLUMNS],
            },
            "origin_stand": {
                "encoding": "native_string",
                "category_count": len(known_categories["ORIGIN_STAND"]),
                "feature_columns": ["ORIGIN_STAND"],
            },
            "taxi_id": {
                "encoding": "native_string",
                "category_count": len(known_categories["TAXI_ID"]),
                "feature_columns": ["TAXI_ID"],
            },
        }

    return {
        "origin_call": {
            "encoding": "missing_indicator_plus_log_count",
            "feature_columns": ORIGIN_CALL_FEATURE_COLUMNS,
            "unique_non_missing_categories": len(origin_call_counts),
            "max_count": max(origin_call_counts.values(), default=0),
        },
        "origin_stand": {
            "encoding": "one_hot_missing_as_0",
            "category_count": len(categorical_encoder["origin_stand_categories"]),
            "feature_columns": categorical_encoder["origin_stand_columns"],
        },
        "taxi_id": {
            "encoding": "one_hot",
            "category_count": len(categorical_encoder["taxi_id_categories"]),
            "feature_columns": categorical_encoder["taxi_id_columns"],
        },
    }


def build_metadata_features(
    row: pd.Series,
    prefix_length: int,
    categorical_encoder: dict[str, Any],
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
) -> dict[str, float | int | str | None]:
    """Build categorical and temporal metadata features for one trip."""
    validate_dataset_variant(dataset_variant)
    timestamp = timestamp_to_porto_local(row["TIMESTAMP"])
    call_type = row.get("CALL_TYPE")
    origin_call_category = normalize_category_id(row.get("ORIGIN_CALL"), missing_category=None)
    origin_call_missing = origin_call_category is None
    origin_call_count = (
        0
        if origin_call_missing
        else categorical_encoder["origin_call_counts"].get(origin_call_category, 0)
    )
    week = int(timestamp.isocalendar().week)
    day = int(timestamp.weekday())
    quarter = int(timestamp.hour * 4 + timestamp.minute // 15)
    week_sin, week_cos = cyclic_time_features(week, period=53, offset=1)
    day_sin, day_cos = cyclic_time_features(day, period=7)
    quarter_sin, quarter_cos = cyclic_time_features(quarter, period=96)

    numeric_features: dict[str, float | int | str | None] = {
        "ORIGIN_CALL_MISSING": int(origin_call_missing),
        "ORIGIN_CALL_LOG_COUNT": float(np.log1p(origin_call_count)),
        "WEEK": week,
        "DAY": day,
        "QUARTER": quarter,
        "WEEK_SIN": week_sin,
        "WEEK_COS": week_cos,
        "DAY_SIN": day_sin,
        "DAY_COS": day_cos,
        "QUARTER_SIN": quarter_sin,
        "QUARTER_COS": quarter_cos,
        "PREFIX_LENGTH": int(prefix_length),
    }

    if dataset_variant == CATBOOST_DATASET_VARIANT:
        return numeric_features

    features: dict[str, float | int | str | None] = {
        "ORIGIN_CALL_MISSING": numeric_features["ORIGIN_CALL_MISSING"],
        "ORIGIN_CALL_LOG_COUNT": numeric_features["ORIGIN_CALL_LOG_COUNT"],
        ORIGIN_STAND_CATEGORY_COLUMN: normalize_category_id(row.get("ORIGIN_STAND")),
        TAXI_ID_CATEGORY_COLUMN: normalize_category_id(row.get("TAXI_ID")),
        **{
            column: numeric_features[column]
            for column in TEMPORAL_PREFIX_FEATURE_COLUMNS
        },
    }
    for column in CALL_TYPE_COLUMNS:
        features[column] = 0
    call_type_column = f"CALL_TYPE_{call_type}"
    if call_type_column in CALL_TYPE_COLUMNS:
        features[call_type_column] = 1
    return features


def build_feature_row(
    row: pd.Series,
    prefix_points: list,
    categorical_encoder: dict[str, Any],
    target_point: list | tuple | None = None,
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
) -> dict[str, float | int | str | None]:
    """Build one model row from trip metadata and prefix GPS points."""
    validate_dataset_variant(dataset_variant)
    features = {}
    features.update(
        build_metadata_features(
            row,
            prefix_length=len(prefix_points),
            categorical_encoder=categorical_encoder,
            dataset_variant=dataset_variant,
        )
    )
    features.update(extract_prefix_gps_features(prefix_points))
    if dataset_variant == CATBOOST_DATASET_VARIANT:
        features.update(
            {
                "CALL_TYPE": normalize_category_id(
                    row.get("CALL_TYPE"),
                    missing_category=CATBOOST_MISSING_CATEGORY,
                ),
                "ORIGIN_CALL": normalize_category_id(
                    row.get("ORIGIN_CALL"),
                    missing_category=CATBOOST_MISSING_CATEGORY,
                ),
                "ORIGIN_STAND": normalize_category_id(
                    row.get("ORIGIN_STAND"),
                    missing_category=CATBOOST_MISSING_CATEGORY,
                ),
                "TAXI_ID": normalize_category_id(
                    row.get("TAXI_ID"),
                    missing_category=CATBOOST_MISSING_CATEGORY,
                ),
            }
        )
    if target_point is not None:
        features["END_LONG"] = float(target_point[0])
        features["END_LAT"] = float(target_point[1])
    return features


def sample_train_row_for_prefix_length(
    full_lengths: np.ndarray,
    sorted_positions: np.ndarray,
    sorted_lengths: np.ndarray,
    desired_prefix_length: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Sample a train row that can support the requested prefix length."""
    min_full_length = int(desired_prefix_length) + 1
    first_feasible = np.searchsorted(sorted_lengths, min_full_length, side="left")
    if first_feasible < len(sorted_positions):
        sorted_choice = rng.integers(first_feasible, len(sorted_positions))
        return int(sorted_positions[sorted_choice]), int(desired_prefix_length)

    longest_position = int(np.argmax(full_lengths))
    capped_prefix_length = int(max(full_lengths[longest_position] - 1, 1))
    return longest_position, capped_prefix_length


def fit_gps_scaler(data: pd.DataFrame) -> dict[str, pd.Series]:
    """Fit mean and scale statistics for GPS feature columns."""
    means = data[GPS_FEATURE_COLUMNS].mean()
    stds = data[GPS_FEATURE_COLUMNS].std(ddof=0).replace(0, 1.0)
    return {"mean": means, "std": stds}


def apply_gps_scaler(data: pd.DataFrame, gps_scaler: dict[str, pd.Series]) -> pd.DataFrame:
    """Apply fitted GPS scaling to a feature table."""
    scaled = data.copy()
    scaled[GPS_FEATURE_COLUMNS] = (scaled[GPS_FEATURE_COLUMNS] - gps_scaler["mean"]) / gps_scaler["std"]
    return scaled


def build_synthetic_prefix_dataset(
    data: pd.DataFrame,
    test_prefix_lengths: np.ndarray,
    sample_factor: float = PREFIX_SAMPLE_FACTOR,
    seed: int = SEED,
    gps_scaler: dict[str, pd.Series] | None = None,
    categorical_encoder: dict[str, Any] | None = None,
    label: str = "dataset",
    progress_every: int = 100000,
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
) -> tuple[pd.DataFrame, dict[str, pd.Series], pd.Series, dict[str, Any]]:
    """Create train or validation prefixes matched to test prefix lengths."""
    validate_dataset_variant(dataset_variant)
    valid = filter_valid_polyline_rows(data, require_target=True)
    if len(valid) == 0:
        raise ValueError("No valid training rows remain after POLYLINE filtering.")
    fitted_categorical_encoder = categorical_encoder or fit_categorical_encoder(
        valid,
        dataset_variant=dataset_variant,
    )

    full_lengths = valid["__POLYLINE_LENGTH"].to_numpy(dtype=int)
    sorted_positions = np.argsort(full_lengths)
    sorted_lengths = full_lengths[sorted_positions]
    test_prefix_lengths = np.asarray(test_prefix_lengths, dtype=int)
    test_prefix_lengths = test_prefix_lengths[test_prefix_lengths >= 1]
    if len(test_prefix_lengths) == 0:
        raise ValueError("No valid test prefix lengths available for synthetic prefix sampling.")

    rng = np.random.default_rng(seed)
    n_examples = int(len(valid) * sample_factor)
    feature_rows = []
    sampled_prefix_lengths = []
    row_index = []

    log(f"Building {label}: {n_examples:,} synthetic prefix rows.")
    for sample_index in range(n_examples):
        if progress_every > 0 and sample_index > 0 and sample_index % progress_every == 0:
            log(f"{label}: built {sample_index:,}/{n_examples:,} rows.")

        desired_prefix_length = int(rng.choice(test_prefix_lengths))
        row_position, prefix_length = sample_train_row_for_prefix_length(
            full_lengths=full_lengths,
            sorted_positions=sorted_positions,
            sorted_lengths=sorted_lengths,
            desired_prefix_length=desired_prefix_length,
            rng=rng,
        )
        row = valid.iloc[row_position]
        points = parse_polyline(row["POLYLINE"])
        if not prefix_length < len(points):
            raise AssertionError("Synthetic prefix includes the target destination point.")

        prefix_points = points[:prefix_length]
        target_point = points[-1]
        feature_rows.append(
            build_feature_row(
                row,
                prefix_points,
                categorical_encoder=fitted_categorical_encoder,
                target_point=target_point,
                dataset_variant=dataset_variant,
            )
        )
        sampled_prefix_lengths.append(prefix_length)
        row_index.append(f"{valid.index[row_position]}__prefix_{sample_index}")

    processed = pd.DataFrame(feature_rows, index=row_index)
    processed.index.name = valid.index.name
    processed = apply_categorical_encoder(
        processed,
        fitted_categorical_encoder,
        dataset_variant=dataset_variant,
    )

    fitted_scaler = gps_scaler or fit_gps_scaler(processed)
    processed = apply_gps_scaler(processed, fitted_scaler)
    sampled_prefix_lengths = pd.Series(sampled_prefix_lengths, name="PREFIX_LENGTH")
    log(f"Finished {label}: shape={processed.shape}.")
    return processed, fitted_scaler, sampled_prefix_lengths, fitted_categorical_encoder


def build_test_prefix_dataset(
    data: pd.DataFrame,
    gps_scaler: dict[str, pd.Series],
    categorical_encoder: dict[str, Any],
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
) -> pd.DataFrame:
    """Create model features from observed competition test prefixes."""
    validate_dataset_variant(dataset_variant)
    valid = filter_valid_polyline_rows(data, require_target=False)
    if len(valid) != len(data):
        raise ValueError("Competition test preprocessing would drop rows; inspect invalid test polylines before submission.")

    feature_rows = []
    for _, row in valid.iterrows():
        prefix_points = parse_polyline(row["POLYLINE"])
        feature_rows.append(
            build_feature_row(
                row,
                prefix_points,
                categorical_encoder=categorical_encoder,
                target_point=None,
                dataset_variant=dataset_variant,
            )
        )

    processed = pd.DataFrame(feature_rows, index=valid.index)
    processed.index.name = valid.index.name
    processed = apply_categorical_encoder(
        processed,
        categorical_encoder,
        dataset_variant=dataset_variant,
    )
    processed = apply_gps_scaler(processed, gps_scaler)
    return processed


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a processed table into feature and target frames."""
    X = data.drop(columns=TARGET_COLUMNS)
    y = data[TARGET_COLUMNS]
    return X, y


def align_feature_columns(
    data: pd.DataFrame,
    reference_columns: pd.Index,
    dataset_variant: str,
    data_name: str,
) -> pd.DataFrame:
    """Align a feature frame to training columns using variant-safe rules."""
    validate_dataset_variant(dataset_variant)
    if dataset_variant == DEFAULT_DATASET_VARIANT:
        return data.reindex(columns=reference_columns, fill_value=0)

    missing_columns = list(reference_columns.difference(data.columns))
    unexpected_columns = list(data.columns.difference(reference_columns))
    if missing_columns or unexpected_columns:
        raise ValueError(
            f"CatBoost feature columns differ for {data_name}. "
            f"Missing: {missing_columns}; unexpected: {unexpected_columns}."
        )
    aligned = data.reindex(columns=reference_columns)
    if list(aligned.columns) != CATBOOST_FEATURE_COLUMNS:
        raise ValueError(
            f"CatBoost feature order is invalid for {data_name}: "
            f"{list(aligned.columns)!r}."
        )
    for column in CATBOOST_CATEGORICAL_FEATURE_COLUMNS:
        if aligned[column].isna().any():
            raise ValueError(
                f"CatBoost categorical feature contains missing values in {data_name}: {column}"
            )
        if not aligned[column].map(lambda value: isinstance(value, str)).all():
            raise TypeError(
                f"CatBoost categorical feature must contain strings in {data_name}: {column}"
            )
    return aligned


def describe_prefix_lengths(**series_by_name: pd.Series | np.ndarray) -> pd.DataFrame:
    """Summarize prefix-length distributions for comparison."""
    summaries = []
    for name, values in series_by_name.items():
        summary = pd.Series(values).describe(percentiles=PREFIX_LENGTH_PERCENTILES)
        summary.name = name
        summaries.append(summary)
    return pd.concat(summaries, axis=1)


def make_json_safe(value: Any) -> Any:
    """Convert common pandas and numpy values to JSON-safe objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())
    if isinstance(value, pd.Series):
        return make_json_safe(value.to_dict())
    if isinstance(value, pd.DataFrame):
        return make_json_safe(value.to_dict())
    if isinstance(value, dict):
        return {str(key): make_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    """Write indented JSON with stable keys."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(make_json_safe(payload), file, indent=2, sort_keys=True)


def save_frame(data: pd.DataFrame, output_path: Path) -> None:
    """Save a DataFrame as compressed Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output_path, compression=PARQUET_COMPRESSION)
    log(f"Wrote {output_path} shape={data.shape}.")


def save_prefix_lengths(prefix_lengths: pd.Series, output_path: Path) -> None:
    """Save sampled prefix lengths as a one-column Parquet file."""
    save_frame(prefix_lengths.to_frame(), output_path)


def package_versions() -> dict[str, str]:
    """Return key package versions for reproducibility metadata."""
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "pyarrow": pyarrow.__version__,
    }


def build_validation_datasets(
    raw_train_data: pd.DataFrame,
    raw_test_data: pd.DataFrame,
    output_dir: Path,
    validation_size: float,
    seed: int,
    prefix_sample_factor: float,
    progress_every: int,
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
) -> dict[str, Any]:
    """Build and save shared train/validation comparison matrices."""
    validate_dataset_variant(dataset_variant)
    validation_dir = output_dir / "validation"
    log("Splitting retained train data for validation comparison.")
    raw_train_part, raw_val_part = train_test_split(
        raw_train_data,
        test_size=validation_size,
        random_state=seed,
        shuffle=True,
    )
    log(f"Raw train split shape: {raw_train_part.shape}.")
    log(f"Raw validation split shape: {raw_val_part.shape}.")

    test_prefix_lengths = get_prefix_lengths(raw_test_data, require_target=False)
    log("Building validation training matrix.")
    validation_train_result = build_synthetic_prefix_dataset(
        raw_train_part,
        test_prefix_lengths=test_prefix_lengths,
        sample_factor=prefix_sample_factor,
        seed=seed,
        label="validation/train",
        progress_every=progress_every,
        dataset_variant=dataset_variant,
    )
    train_processed, gps_scaler, train_prefix_lengths, categorical_encoder = validation_train_result
    log("Building validation holdout matrix.")
    val_processed, _, val_prefix_lengths, _ = build_synthetic_prefix_dataset(
        raw_val_part,
        test_prefix_lengths=test_prefix_lengths,
        sample_factor=prefix_sample_factor,
        seed=seed + 1,
        gps_scaler=gps_scaler,
        categorical_encoder=categorical_encoder,
        label="validation/val",
        progress_every=progress_every,
        dataset_variant=dataset_variant,
    )

    X_train, y_train = split_features_target(train_processed)
    X_val, y_val = split_features_target(val_processed)
    X_val = align_feature_columns(
        X_val,
        reference_columns=X_train.columns,
        dataset_variant=dataset_variant,
        data_name="validation/X_val",
    )

    save_frame(X_train, validation_dir / "X_train.parquet")
    save_frame(y_train, validation_dir / "y_train.parquet")
    save_frame(X_val, validation_dir / "X_val.parquet")
    save_frame(y_val, validation_dir / "y_val.parquet")
    save_prefix_lengths(train_prefix_lengths, validation_dir / "train_prefix_lengths.parquet")
    save_prefix_lengths(val_prefix_lengths, validation_dir / "val_prefix_lengths.parquet")
    joblib.dump(gps_scaler, validation_dir / "gps_scaler.joblib")
    log(f"Wrote {validation_dir / 'gps_scaler.joblib'}.")
    joblib.dump(categorical_encoder, validation_dir / "categorical_encoder.joblib")
    log(f"Wrote {validation_dir / 'categorical_encoder.joblib'}.")

    metadata = {
        "raw_train_split_shape": raw_train_part.shape,
        "raw_validation_split_shape": raw_val_part.shape,
        "processed_train_shape": train_processed.shape,
        "processed_validation_shape": val_processed.shape,
        "X_train_shape": X_train.shape,
        "y_train_shape": y_train.shape,
        "X_val_shape": X_val.shape,
        "y_val_shape": y_val.shape,
        "feature_columns": list(X_train.columns),
        "target_columns": list(y_train.columns),
        "categorical_encoding": describe_categorical_encoder(
            categorical_encoder,
            dataset_variant=dataset_variant,
        ),
        "prefix_summary": describe_prefix_lengths(
            train_synthetic=train_prefix_lengths,
            validation_synthetic=val_prefix_lengths,
            competition_test=test_prefix_lengths,
        ),
    }
    if dataset_variant == CATBOOST_DATASET_VARIANT:
        metadata.update({
            "categorical_feature_columns": CATBOOST_CATEGORICAL_FEATURE_COLUMNS.copy(),
            "numerical_feature_columns": CATBOOST_NUMERICAL_FEATURE_COLUMNS.copy(),
            "feature_dtypes": {
                column: str(dtype)
                for column, dtype in X_train.dtypes.items()
            },
        })

    del train_processed, val_processed, X_train, y_train, X_val, y_val
    del train_prefix_lengths, val_prefix_lengths, raw_train_part, raw_val_part
    del categorical_encoder
    gc.collect()
    return metadata


def build_final_datasets(
    raw_train_data: pd.DataFrame,
    raw_test_data: pd.DataFrame,
    output_dir: Path,
    seed: int,
    prefix_sample_factor: float,
    progress_every: int,
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
) -> dict[str, Any]:
    """Build and save full-train and competition-test matrices."""
    validate_dataset_variant(dataset_variant)
    final_dir = output_dir / "final"
    final_seed = seed + 100
    final_test_prefix_lengths = get_prefix_lengths(raw_test_data, require_target=False)

    log("Building final full-training matrix.")
    final_train_result = build_synthetic_prefix_dataset(
        raw_train_data,
        test_prefix_lengths=final_test_prefix_lengths,
        sample_factor=prefix_sample_factor,
        seed=final_seed,
        label="final/train",
        progress_every=progress_every,
        dataset_variant=dataset_variant,
    )
    (
        final_train_processed,
        final_gps_scaler,
        final_prefix_lengths,
        final_categorical_encoder,
    ) = final_train_result
    log("Building final competition-test matrix.")
    final_test_processed = build_test_prefix_dataset(
        raw_test_data,
        gps_scaler=final_gps_scaler,
        categorical_encoder=final_categorical_encoder,
        dataset_variant=dataset_variant,
    )
    log(f"Finished final/test: shape={final_test_processed.shape}.")

    X_final_train, y_final_train = split_features_target(final_train_processed)
    X_final_test = align_feature_columns(
        final_test_processed,
        reference_columns=X_final_train.columns,
        dataset_variant=dataset_variant,
        data_name="final/X_test",
    )

    save_frame(X_final_train, final_dir / "X_train.parquet")
    save_frame(y_final_train, final_dir / "y_train.parquet")
    save_frame(X_final_test, final_dir / "X_test.parquet")
    save_prefix_lengths(final_prefix_lengths, final_dir / "train_prefix_lengths.parquet")
    joblib.dump(final_gps_scaler, final_dir / "gps_scaler.joblib")
    log(f"Wrote {final_dir / 'gps_scaler.joblib'}.")
    joblib.dump(final_categorical_encoder, final_dir / "categorical_encoder.joblib")
    log(f"Wrote {final_dir / 'categorical_encoder.joblib'}.")

    metadata = {
        "final_seed": final_seed,
        "processed_train_shape": final_train_processed.shape,
        "processed_test_shape": final_test_processed.shape,
        "X_train_shape": X_final_train.shape,
        "y_train_shape": y_final_train.shape,
        "X_test_shape": X_final_test.shape,
        "feature_columns": list(X_final_train.columns),
        "target_columns": list(y_final_train.columns),
        "categorical_encoding": describe_categorical_encoder(
            final_categorical_encoder,
            dataset_variant=dataset_variant,
        ),
        "prefix_summary": describe_prefix_lengths(
            final_train_synthetic=final_prefix_lengths,
            competition_test=final_test_prefix_lengths,
        ),
    }
    if dataset_variant == CATBOOST_DATASET_VARIANT:
        metadata.update({
            "categorical_feature_columns": CATBOOST_CATEGORICAL_FEATURE_COLUMNS.copy(),
            "numerical_feature_columns": CATBOOST_NUMERICAL_FEATURE_COLUMNS.copy(),
            "feature_dtypes": {
                column: str(dtype)
                for column, dtype in X_final_train.dtypes.items()
            },
        })

    del final_train_processed, final_test_processed
    del X_final_train, y_final_train, X_final_test, final_prefix_lengths
    del final_categorical_encoder
    gc.collect()
    return metadata


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build reusable preprocessed Porto taxi destination datasets."
    )
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument(
        "--dataset-variant",
        choices=DATASET_VARIANTS,
        default=DEFAULT_DATASET_VARIANT,
        help=(
            "Feature contract to build. The default variant uses one-hot encoding; "
            "the catboost variant preserves native categorical strings."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. When omitted, each dataset variant uses its own "
            "directory under /kaggle/working."
        ),
    )
    parser.add_argument("--n-rows", type=parse_n_rows, default=None)
    parser.add_argument("--validation-size", type=float, default=VALIDATION_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--prefix-sample-factor", type=float, default=PREFIX_SAMPLE_FACTOR)
    parser.add_argument(
        "--skip-final",
        action="store_true",
        help="Only build validation comparison data.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100000,
        help="Print row-building progress every N rows; set 0 to disable loop progress.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments before starting the expensive build."""
    validate_dataset_variant(args.dataset_variant)
    if not args.train_path.exists():
        raise FileNotFoundError(f"Train path does not exist: {args.train_path}")
    if not args.test_path.exists():
        raise FileNotFoundError(f"Test path does not exist: {args.test_path}")
    if not 0 < args.validation_size < 1:
        raise ValueError("--validation-size must be between 0 and 1.")
    if args.prefix_sample_factor <= 0:
        raise ValueError("--prefix-sample-factor must be a positive integer.")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be 0 or a positive integer.")


def main() -> None:
    """Build and save preprocessing artifacts."""
    args = parse_args()
    validate_args(args)
    build_start_time = time.time()
    output_dir = resolve_output_dir(args.dataset_variant, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("Starting preprocessed dataset build.")
    log(f"Dataset variant: {args.dataset_variant}")
    log(f"Train path: {args.train_path}")
    log(f"Test path: {args.test_path}")
    log(f"Output directory: {output_dir.resolve()}")
    log(f"n_rows={args.n_rows}, validation_size={args.validation_size}, seed={args.seed}")
    log(f"prefix_sample_factor={args.prefix_sample_factor}, skip_final={args.skip_final}")

    log("Reading raw train data.")
    raw_train_data = read_data(args.train_path, n_rows=args.n_rows, seed=args.seed)
    log("Reading raw test data.")
    raw_test_data = read_data(args.test_path)
    raw_train_shape_before_filters = raw_train_data.shape
    raw_test_shape = raw_test_data.shape
    log(f"Raw train shape before train-only quality filters: {raw_train_shape_before_filters}.")
    log(f"Raw test shape: {raw_test_shape}.")

    log("Applying train-only quality filters.")
    raw_train_data, quality_filter_summary = apply_train_quality_filters(raw_train_data)
    log("Train quality filter summary:")
    log("\n" + quality_filter_summary.to_string())
    log(f"Raw train shape after train-only quality filters: {raw_train_data.shape}.")

    metadata: dict[str, Any] = {
        "schema_version": schema_version_for_variant(args.dataset_variant),
        "build": {
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(build_start_time)),
            "train_path": args.train_path,
            "test_path": args.test_path,
            "output_dir": output_dir.resolve(),
            "skip_final": args.skip_final,
        },
        "config": {
            "n_rows": args.n_rows,
            "validation_size": args.validation_size,
            "seed": args.seed,
            "validation_train_seed": args.seed,
            "validation_val_seed": args.seed + 1,
            "final_train_seed": args.seed + 100,
            "prefix_sample_factor": args.prefix_sample_factor,
            "broad_sanity_bounds": BROAD_SANITY_BOUNDS,
            "duration_artifact_minutes": DURATION_ARTIFACT_MINUTES,
            "duration_artifact_max_distance_from_start_km": (
                DURATION_ARTIFACT_MAX_DISTANCE_FROM_START_KM
            ),
            "target_format": "[LONG, LAT]",
        },
        "constants": {
            "target_columns": TARGET_COLUMNS,
            "edge_point_count": EDGE_POINT_COUNT,
            "gps_feature_columns": GPS_FEATURE_COLUMNS,
            "call_type_columns": (
                CALL_TYPE_COLUMNS
                if args.dataset_variant == DEFAULT_DATASET_VARIANT
                else ["CALL_TYPE"]
            ),
            "origin_call_feature_columns": ORIGIN_CALL_FEATURE_COLUMNS,
            "origin_stand_prefix": ORIGIN_STAND_PREFIX,
            "taxi_id_prefix": TAXI_ID_PREFIX,
            "categorical_encoding_policy": categorical_encoding_policy_for_variant(
                args.dataset_variant
            ),
            "porto_timezone": PORTO_TIMEZONE,
            "seconds_per_gps_step": SECONDS_PER_GPS_STEP,
            "prefix_length_percentiles": PREFIX_LENGTH_PERCENTILES,
            "parquet_compression": PARQUET_COMPRESSION,
        },
        "raw": {
            "train_shape_before_filters": raw_train_shape_before_filters,
            "test_shape": raw_test_shape,
            "train_shape_after_filters": raw_train_data.shape,
            "quality_filter_summary": quality_filter_summary,
        },
        "packages": package_versions(),
    }
    if args.dataset_variant == CATBOOST_DATASET_VARIANT:
        metadata["dataset_variant"] = CATBOOST_DATASET_VARIANT
        metadata["build"]["dataset_variant"] = CATBOOST_DATASET_VARIANT
        metadata["constants"].update(
            {
                "categorical_feature_columns": CATBOOST_CATEGORICAL_FEATURE_COLUMNS.copy(),
                "numerical_feature_columns": CATBOOST_NUMERICAL_FEATURE_COLUMNS.copy(),
                "categorical_missing_token": CATBOOST_MISSING_CATEGORY,
            }
        )

    metadata["validation"] = build_validation_datasets(
        raw_train_data=raw_train_data,
        raw_test_data=raw_test_data,
        output_dir=output_dir,
        validation_size=args.validation_size,
        seed=args.seed,
        prefix_sample_factor=args.prefix_sample_factor,
        progress_every=args.progress_every,
        dataset_variant=args.dataset_variant,
    )

    if args.skip_final:
        log("Skipping final full-data matrices because --skip-final was set.")
        metadata["final"] = None
    else:
        metadata["final"] = build_final_datasets(
            raw_train_data=raw_train_data,
            raw_test_data=raw_test_data,
            output_dir=output_dir,
            seed=args.seed,
            prefix_sample_factor=args.prefix_sample_factor,
            progress_every=args.progress_every,
            dataset_variant=args.dataset_variant,
        )

    metadata["build"]["duration_seconds"] = float(time.time() - build_start_time)
    metadata["build"]["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(metadata, output_dir / "metadata.json")
    log(f"Wrote {output_dir / 'metadata.json'}.")
    log(f"Preprocessed dataset build completed in {metadata['build']['duration_seconds']:.1f}s.")


if __name__ == "__main__":
    main()
