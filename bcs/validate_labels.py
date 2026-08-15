from pathlib import Path

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(r"D:\cow")

LABEL_FILE = (
    PROJECT_ROOT
    / "bcs"
    / "labels"
    / "bcs_labeling_dataset.csv"
)


VALID_MIN = 1.0
VALID_MAX = 5.0
STEP = 0.25


def main():

    print("=" * 80)
    print("COW PLF - BCS LABEL VALIDATOR")
    print("=" * 80)

    if not LABEL_FILE.exists():
        raise FileNotFoundError(
            f"Label file not found: {LABEL_FILE}"
        )

    df = pd.read_csv(LABEL_FILE)

    print()
    print(f"Rows : {len(df)}")

    required_columns = [
        "cow_id",
        "image_name",
        "bcs_score",
        "bcs_source",
        "assessor_id",
        "assessment_notes",
        "label_status",
    ]

    print()
    print("-" * 80)
    print("COLUMN CHECK")
    print("-" * 80)

    missing_columns = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing_columns:
        print("[FAIL] Missing columns:")
        for c in missing_columns:
            print(f"  - {c}")
        return

    print("[OK] All required columns present.")

    # ------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------

    print()
    print("-" * 80)
    print("IDENTITY CHECK")
    print("-" * 80)

    print(
        f"Unique cow IDs   : {df['cow_id'].nunique()}"
    )

    duplicate_cows = df["cow_id"].duplicated().sum()
    duplicate_images = df["image_name"].duplicated().sum()

    print(
        f"Duplicate cows   : {duplicate_cows}"
    )

    print(
        f"Duplicate images : {duplicate_images}"
    )

    # ------------------------------------------------------------
    # BCS presence
    # ------------------------------------------------------------

    print()
    print("-" * 80)
    print("BCS LABEL STATUS")
    print("-" * 80)

    bcs_numeric = pd.to_numeric(
        df["bcs_score"],
        errors="coerce",
    )

    labeled = bcs_numeric.notna()

    print(
        f"Labeled   : {labeled.sum()}"
    )

    print(
        f"Unlabeled : {(~labeled).sum()}"
    )

    # ------------------------------------------------------------
    # Validate BCS range
    # ------------------------------------------------------------

    invalid_range = []

    for index, value in bcs_numeric.items():

        if pd.isna(value):
            continue

        if value < VALID_MIN or value > VALID_MAX:
            invalid_range.append(
                (index, value)
            )

    print()
    print("-" * 80)
    print("BCS RANGE CHECK")
    print("-" * 80)

    if invalid_range:
        print("[FAIL] Invalid BCS values:")

        for index, value in invalid_range:
            print(
                f"  Row {index + 2}: {value}"
            )
    else:
        print(
            "[OK] All BCS values are within 1.00-5.00."
        )

    # ------------------------------------------------------------
    # Validate increments
    # ------------------------------------------------------------

    invalid_increment = []

    for index, value in bcs_numeric.items():

        if pd.isna(value):
            continue

        scaled = value / STEP

        if not np.isclose(
            scaled,
            round(scaled),
            atol=1e-8,
        ):
            invalid_increment.append(
                (index, value)
            )

    print()
    print("-" * 80)
    print("BCS INCREMENT CHECK")
    print("-" * 80)

    if invalid_increment:

        print(
            "[FAIL] Values are not multiples of 0.25:"
        )

        for index, value in invalid_increment:
            print(
                f"  Row {index + 2}: {value}"
            )

    else:

        print(
            "[OK] All BCS values use 0.25 increments."
        )

    # ------------------------------------------------------------
    # Source / assessor
    # ------------------------------------------------------------

    print()
    print("-" * 80)
    print("ASSESSOR METADATA")
    print("-" * 80)

    labeled_df = df[labeled]

    missing_source = (
        labeled_df["bcs_source"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    missing_assessor = (
        labeled_df["assessor_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(
        f"Labeled rows without source   : {missing_source}"
    )

    print(
        f"Labeled rows without assessor : {missing_assessor}"
    )

    # ------------------------------------------------------------
    # Label status
    # ------------------------------------------------------------

    print()
    print("-" * 80)
    print("LABEL STATUS")
    print("-" * 80)

    print(
        df["label_status"]
        .fillna("EMPTY")
        .value_counts()
        .to_string()
    )

    # ------------------------------------------------------------
    # BCS distribution
    # ------------------------------------------------------------

    print()
    print("-" * 80)
    print("BCS DISTRIBUTION")
    print("-" * 80)

    if not labeled.empty:

        print(
            bcs_numeric[labeled]
            .value_counts()
            .sort_index()
            .to_string()
        )

    else:

        print(
            "[INFO] No BCS labels entered yet."
        )

    # ------------------------------------------------------------
    # Final decision
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("VALIDATION RESULT")
    print("=" * 80)

    structural_ok = (
        len(missing_columns) == 0
        and duplicate_cows == 0
        and duplicate_images == 0
    )

    label_values_ok = (
        len(invalid_range) == 0
        and len(invalid_increment) == 0
    )

    if structural_ok and label_values_ok:

        if labeled.sum() == len(df):

            print(
                "[PASS] All cows have valid BCS labels."
            )

            if (
                missing_source == 0
                and missing_assessor == 0
            ):
                print(
                    "[PASS] Label metadata is complete."
                )
            else:
                print(
                    "[WARNING] Some label metadata is missing."
                )

            print()
            print(
                "READY FOR BCS DATASET MERGING."
            )

        else:

            print(
                "[PASS] Label structure is valid."
            )

            print(
                f"[WAIT] {len(df) - labeled.sum()} "
                "cows still need reference BCS labels."
            )

    else:

        print(
            "[FAIL] Fix the validation errors before training."
        )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()