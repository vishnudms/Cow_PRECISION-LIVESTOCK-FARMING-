from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\cow")

FEATURE_FILE = (
    PROJECT_ROOT
    / "output"
    / "bcs_dataset"
    / "bcs_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "bcs_dataset"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "provisional_bcs_estimates.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

BCS_MIN = 1.00
BCS_MAX = 5.00
BCS_STEP = 0.25


# ============================================================
# HELPERS
# ============================================================

def robust_normalize(series):
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).astype(float)

    if values.notna().sum() == 0:
        return pd.Series(
            np.full(
                len(values),
                0.5,
            ),
            index=series.index,
        )

    low = np.nanpercentile(
        values,
        10,
    )

    high = np.nanpercentile(
        values,
        90,
    )

    if (
        not np.isfinite(low)
        or not np.isfinite(high)
        or high <= low
    ):

        return pd.Series(
            np.full(
                len(values),
                0.5,
            ),
            index=series.index,
        )

    result = (
        (values - low)
        / (high - low)
    )

    return result.clip(
        0.0,
        1.0,
    )


def round_to_quarter(value):

    return (
        round(
            value / BCS_STEP
        )
        * BCS_STEP
    )


# ============================================================
# ESTIMATOR
# ============================================================

def estimate_bcs(df):

    df = df.copy()

    required = [
        "cow_id",
        "image_name",

        "bcs_compactness",
        "bcs_area_to_bbox_ratio",

        "bcs_elongation_ratio",

        "bcs_middle_depth_norm",
        "bcs_end_min_depth_norm",
        "bcs_end_max_depth_norm",

        "bcs_mid_to_end_ratio",
        "bcs_body_depth_cv",

        "bcs_top_contour_std_norm",
        "bcs_bottom_contour_std_norm",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing required BCS features:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # Normalize model inputs
    # --------------------------------------------------------

    compactness = robust_normalize(
        df["bcs_compactness"]
    )

    fill_ratio = robust_normalize(
        df["bcs_area_to_bbox_ratio"]
    )

    mid_depth = robust_normalize(
        df["bcs_middle_depth_norm"]
    )

    end_depth = robust_normalize(
        (
            df["bcs_end_min_depth_norm"]
            + df["bcs_end_max_depth_norm"]
        ) / 2.0
    )

    mid_to_end = robust_normalize(
        df["bcs_mid_to_end_ratio"]
    )

    contour_bottom = robust_normalize(
        df["bcs_bottom_contour_std_norm"]
    )

    contour_top = robust_normalize(
        df["bcs_top_contour_std_norm"]
    )

    # --------------------------------------------------------
    # Provisional visual-condition index
    #
    # This remains a heuristic.
    # It is NOT a learned veterinary model.
    # --------------------------------------------------------

    shape_score = (
        0.22 * compactness
        + 0.16 * fill_ratio
        + 0.24 * mid_depth
        + 0.14 * end_depth
        + 0.12 * mid_to_end
        + 0.07 * contour_bottom
        + 0.05 * contour_top
    )

    shape_score = shape_score.clip(
        0.0,
        1.0,
    )

    provisional_bcs = (
        BCS_MIN
        + shape_score
        * (
            BCS_MAX
            - BCS_MIN
        )
    )

    provisional_bcs = provisional_bcs.apply(
        round_to_quarter
    )

    provisional_bcs = provisional_bcs.clip(
        BCS_MIN,
        BCS_MAX,
    )

    # --------------------------------------------------------
    # FEATURE-QUALITY CONFIDENCE
    #
    # This measures feature stability/quality only.
    # It does NOT mean "probability the BCS is correct".
    # --------------------------------------------------------

    quality_components = pd.concat(
        [
            compactness,
            fill_ratio,
            mid_depth,
            end_depth,
            mid_to_end,
        ],
        axis=1,
    )

    row_std = (
        quality_components
        .std(
            axis=1,
            ddof=0,
        )
        .fillna(0.0)
    )

    confidence = (
        1.0
        - np.clip(
            row_std / 0.35,
            0.0,
            1.0,
        )
    )

    # Penalize extreme/unstable contour measurements.
    body_depth_cv = pd.to_numeric(
        df["bcs_body_depth_cv"],
        errors="coerce",
    ).fillna(0.0)

    contour_penalty = np.clip(
        body_depth_cv / 1.5,
        0.0,
        0.5,
    )

    confidence = (
        confidence
        * (
            1.0
            - contour_penalty
        )
    )

    confidence = confidence.clip(
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    df["provisional_bcs"] = (
        provisional_bcs.round(2)
    )

    df["provisional_bcs_confidence"] = (
        confidence.round(3)
    )

    df["bcs_estimation_method"] = (
        "ORIENTATION_TOLERANT_SILHOUETTE_HEURISTIC"
    )

    df["bcs_confidence_type"] = (
        "FEATURE_QUALITY_NOT_VETERINARY_ACCURACY"
    )

    df["bcs_source"] = (
        "PROVISIONAL_AUTO_ESTIMATE"
    )

    df["label_status"] = (
        "PROVISIONAL_NOT_GROUND_TRUTH"
    )

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_output(df):

    print()
    print("=" * 80)
    print("IMPROVED PROVISIONAL BCS VALIDATION")
    print("=" * 80)

    print(
        f"Rows : {len(df)}"
    )

    if len(df) == 0:
        raise RuntimeError(
            "No BCS estimates generated."
        )

    # --------------------------------------------------------
    # Missing
    # --------------------------------------------------------

    important = [
        "cow_id",
        "image_name",
        "provisional_bcs",
        "provisional_bcs_confidence",
    ]

    missing = (
        df[important]
        .isna()
        .sum()
    )

    if missing.sum() == 0:
        print("[OK] No missing values.")
    else:
        print("[WARNING] Missing values:")
        print(
            missing[
                missing > 0
            ]
        )

    # --------------------------------------------------------
    # Identity
    # --------------------------------------------------------

    print()
    print("IDENTITY")
    print("-" * 80)

    print(
        f"Unique cows : "
        f"{df['cow_id'].nunique()}"
    )

    print(
        f"Duplicate rows : "
        f"{df.duplicated().sum()}"
    )

    # --------------------------------------------------------
    # BCS
    # --------------------------------------------------------

    print()
    print("PROVISIONAL BCS DISTRIBUTION")
    print("-" * 80)

    print(
        df["provisional_bcs"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()
    print("BCS STATISTICS")
    print("-" * 80)

    print(
        df["provisional_bcs"]
        .describe()
        .round(3)
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    print()
    print("FEATURE-QUALITY CONFIDENCE")
    print("-" * 80)

    print(
        df["provisional_bcs_confidence"]
        .describe()
        .round(3)
    )

    print()
    print(
        "[IMPORTANT] Confidence measures feature quality/stability."
    )

    print(
        "It is NOT veterinary prediction accuracy."
    )

    print(
        "Provisional BCS values are NOT ground-truth labels."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("COW PLF - IMPROVED PROVISIONAL BCS ESTIMATOR")
    print("=" * 80)

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"BCS feature dataset not found:\n"
            f"{FEATURE_FILE}"
        )

    print()
    print("[LOAD]")
    print(
        FEATURE_FILE
    )

    df = pd.read_csv(
        FEATURE_FILE
    )

    print(
        f"[OK] Rows : {len(df)}"
    )

    print()
    print(
        "[BCS] Generating improved provisional estimates..."
    )

    result = estimate_bcs(
        df
    )

    validate_output(
        result
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 80)
    print("IMPROVED PROVISIONAL BCS DATASET COMPLETE")
    print("=" * 80)

    print(
        f"Rows   : {len(result)}"
    )

    print(
        f"Output : {OUTPUT_FILE}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()