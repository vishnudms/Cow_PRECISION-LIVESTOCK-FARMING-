from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\cow")

INPUT = (
    PROJECT_ROOT
    / "output"
    / "bcs_dataset"
    / "provisional_bcs_estimates.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("COW PLF - PROVISIONAL BCS SANITY VALIDATION")
    print("=" * 80)

    if not INPUT.exists():

        raise FileNotFoundError(
            f"Provisional BCS dataset not found:\n{INPUT}"
        )

    df = pd.read_csv(INPUT)

    print()
    print(f"Rows : {len(df)}")

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required = [
        "cow_id",
        "image_name",
        "provisional_bcs",
        "provisional_bcs_confidence",
        "bcs_estimation_method",
        "bcs_confidence_type",
        "bcs_source",
        "label_status",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing columns:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # Identity
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("IDENTITY")
    print("-" * 80)

    unique_cows = df["cow_id"].nunique()
    duplicate_ids = df["cow_id"].duplicated().sum()

    print(f"Unique cows    : {unique_cows}")
    print(f"Duplicate IDs  : {duplicate_ids}")

    if duplicate_ids == 0:
        print("[OK] No duplicate cow IDs.")
    else:
        print("[WARNING] Duplicate cow IDs detected.")

    # --------------------------------------------------------
    # BCS range
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("BCS RANGE")
    print("-" * 80)

    minimum = df["provisional_bcs"].min()
    maximum = df["provisional_bcs"].max()

    print(f"Minimum : {minimum:.2f}")
    print(f"Maximum : {maximum:.2f}")

    valid_range = df["provisional_bcs"].between(
        1.0,
        5.0,
    ).all()

    if valid_range:
        print("[OK] BCS range valid.")
    else:
        print("[FAIL] BCS outside 1.00-5.00.")

    # --------------------------------------------------------
    # Quarter increment validation
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("BCS INCREMENT VALIDATION")
    print("-" * 80)

    quarter_valid = (
        (df["provisional_bcs"] * 4)
        .round()
        .sub(df["provisional_bcs"] * 4)
        .abs()
        .lt(1e-8)
        .all()
    )

    if quarter_valid:
        print("[OK] All BCS values use 0.25 increments.")
    else:
        print("[FAIL] Invalid BCS increment detected.")

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("PROVISIONAL BCS DISTRIBUTION")
    print("-" * 80)

    print(
        df["provisional_bcs"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("BCS STATISTICS")
    print("-" * 80)

    print(
        df["provisional_bcs"]
        .describe()
        .round(3)
        .to_string()
    )

    # --------------------------------------------------------
    # Feature-quality confidence
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("FEATURE-QUALITY CONFIDENCE")
    print("-" * 80)

    print(
        df["provisional_bcs_confidence"]
        .describe()
        .round(3)
        .to_string()
    )

    print()
    print(
        "[IMPORTANT] This is feature-quality/stability confidence."
    )
    print(
        "It is NOT probability that the BCS is clinically correct."
    )

    # --------------------------------------------------------
    # Low-confidence cases
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("LOW FEATURE-QUALITY CASES")
    print("-" * 80)

    low_conf = df[
        df["provisional_bcs_confidence"] < 0.20
    ][
        [
            "cow_id",
            "image_name",
            "provisional_bcs",
            "provisional_bcs_confidence",
        ]
    ].sort_values(
        "provisional_bcs_confidence"
    )

    print(
        f"Count : {len(low_conf)}"
    )

    if len(low_conf) > 0:
        print(
            low_conf.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Extreme provisional scores
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("EXTREME PROVISIONAL SCORES")
    print("-" * 80)

    extremes = df[
        (df["provisional_bcs"] <= 1.50)
        |
        (df["provisional_bcs"] >= 4.00)
    ][
        [
            "cow_id",
            "image_name",
            "provisional_bcs",
            "provisional_bcs_confidence",
        ]
    ].sort_values(
        "provisional_bcs"
    )

    print(
        f"Count : {len(extremes)}"
    )

    if len(extremes) > 0:
        print(
            extremes.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Method integrity
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("METHOD INTEGRITY")
    print("-" * 80)

    print(
        "Method:"
    )

    print(
        df["bcs_estimation_method"]
        .dropna()
        .unique()
        .tolist()
    )

    print(
        "Source:"
    )

    print(
        df["bcs_source"]
        .dropna()
        .unique()
        .tolist()
    )

    print(
        "Status:"
    )

    print(
        df["label_status"]
        .dropna()
        .unique()
        .tolist()
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    structural_pass = (
        len(df) == 71
        and unique_cows == 71
        and duplicate_ids == 0
        and valid_range
        and quarter_valid
    )

    print()
    print("=" * 80)
    print("VALIDATION RESULT")
    print("=" * 80)

    if structural_pass:

        print()
        print("[PASS] Provisional BCS dataset is structurally valid.")

    else:

        print()
        print("[FAIL] Structural validation failed.")

    print()
    print("This does NOT establish BCS prediction accuracy.")
    print("Independent reference BCS labels are still required")
    print("for a scientifically validated supervised model.")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()