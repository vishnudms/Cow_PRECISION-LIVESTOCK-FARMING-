from pathlib import Path

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(r"D:\cow")

FEATURE_FILE = (
    PROJECT_ROOT
    / "output"
    / "bcs_dataset"
    / "bcs_features.csv"
)


def main():

    print("=" * 80)
    print("COW PLF - BCS FEATURE QUALITY VALIDATION")
    print("=" * 80)

    df = pd.read_csv(FEATURE_FILE)

    print()
    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    # -------------------------------------------------------------
    # Missing values
    # -------------------------------------------------------------

    print()
    print("-" * 80)
    print("MISSING VALUES")
    print("-" * 80)

    missing = df.isna().sum()

    if missing.sum() == 0:
        print("[OK] No missing values.")
    else:
        print(missing[missing > 0])

    # -------------------------------------------------------------
    # Duplicate cows
    # -------------------------------------------------------------

    print()
    print("-" * 80)
    print("IDENTITY CHECK")
    print("-" * 80)

    duplicate_cows = df["cow_id"].duplicated().sum()

    print(f"Unique cows     : {df['cow_id'].nunique()}")
    print(f"Duplicate rows   : {duplicate_cows}")

    # -------------------------------------------------------------
    # Numeric feature ranges
    # -------------------------------------------------------------

    feature_columns = [
        "body_length_px",
        "body_depth_px",
        "mask_area_px",
        "elongation_ratio",
        "rectangle_ratio",
        "bcs_mask_area_px",
        "bcs_bbox_width_px",
        "bcs_bbox_height_px",
        "bcs_body_length_px",
        "bcs_body_depth_px",
        "bcs_elongation_ratio",
        "bcs_compactness",
        "bcs_top_contour_std_px",
        "bcs_bottom_contour_std_px",
        "bcs_mid_body_depth_px",
    ]

    print()
    print("-" * 80)
    print("FEATURE RANGES")
    print("-" * 80)

    summary = df[feature_columns].describe().T

    print(
        summary[
            [
                "min",
                "mean",
                "std",
                "max",
            ]
        ].round(3).to_string()
    )

    # -------------------------------------------------------------
    # Infinite values
    # -------------------------------------------------------------

    print()
    print("-" * 80)
    print("NUMERICAL VALIDITY")
    print("-" * 80)

    numeric = df[feature_columns]

    infinite_count = np.isinf(
        numeric.to_numpy()
    ).sum()

    print(
        f"Infinite values : {infinite_count}"
    )

    # -------------------------------------------------------------
    # Constant features
    # -------------------------------------------------------------

    print()
    print("-" * 80)
    print("LOW-VARIANCE / CONSTANT FEATURES")
    print("-" * 80)

    constant_features = []

    for column in feature_columns:

        if df[column].nunique() <= 1:
            constant_features.append(column)

    if constant_features:
        for column in constant_features:
            print(f"[WARNING] {column}")
    else:
        print("[OK] No constant features.")

    # -------------------------------------------------------------
    # Weight distribution
    # -------------------------------------------------------------

    print()
    print("-" * 80)
    print("WEIGHT DISTRIBUTION")
    print("-" * 80)

    print(
        df["actual_weight_kg"]
        .describe()
        .round(2)
        .to_string()
    )

    # -------------------------------------------------------------
    # Correlation with weight
    #
    # IMPORTANT:
    # This is NOT BCS validation.
    # It is only a sanity check that the body features
    # contain meaningful biological/size variation.
    # -------------------------------------------------------------

    print()
    print("-" * 80)
    print("FEATURE ↔ WEIGHT CORRELATION")
    print("-" * 80)

    correlations = (
        df[feature_columns + ["actual_weight_kg"]]
        .corr()["actual_weight_kg"]
        .drop("actual_weight_kg")
        .sort_values(
            key=lambda x: x.abs(),
            ascending=False,
        )
    )

    print(
        correlations.round(3).to_string()
    )

    # -------------------------------------------------------------
    # Final status
    # -------------------------------------------------------------

    print()
    print("=" * 80)
    print("BCS FEATURE VALIDATION COMPLETE")
    print("=" * 80)

    if (
        len(df) == 71
        and missing.sum() == 0
        and duplicate_cows == 0
        and infinite_count == 0
    ):
        print("[PASS] Dataset structure is healthy.")
    else:
        print("[WARNING] Dataset requires review.")


if __name__ == "__main__":
    main()