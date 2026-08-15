from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\cow")

MANIFEST = (
    PROJECT_ROOT
    / "output"
    / "bcs_training"
    / "dryad_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "bcs_training"
)


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# HELPER
# ============================================================

def split_class_groups(
    class_df: pd.DataFrame,
    random_state: int,
):
    """
    Split cow groups inside one source class.

    For classes with only one cow group, the group is placed
    entirely in training because it cannot safely appear in
    validation/test without reusing the same cow.
    """

    class_df = (
        class_df
        .drop_duplicates("cow_group")
        .reset_index(drop=True)
    )

    n_groups = len(class_df)

    if n_groups < 2:

        return (
            class_df.copy(),
            pd.DataFrame(
                columns=class_df.columns
            ),
            pd.DataFrame(
                columns=class_df.columns
            ),
        )

    # --------------------------------------------------------
    # Small classes
    # --------------------------------------------------------

    if n_groups == 2:

        train = class_df.iloc[[0]].copy()
        test = class_df.iloc[[1]].copy()

        return (
            train,
            pd.DataFrame(
                columns=class_df.columns
            ),
            test,
        )

    # --------------------------------------------------------
    # Normal classes
    # --------------------------------------------------------

    train_groups, temp_groups = train_test_split(
        class_df,
        test_size=0.30,
        random_state=random_state,
    )

    val_groups, test_groups = train_test_split(
        temp_groups,
        test_size=0.50,
        random_state=random_state,
    )

    return (
        train_groups.reset_index(drop=True),
        val_groups.reset_index(drop=True),
        test_groups.reset_index(drop=True),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("DRYAD BCS - COW GROUP SPLITTER")
    print("=" * 80)

    if not MANIFEST.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST}"
        )

    df = pd.read_csv(
        MANIFEST
    )

    required = [
        "image_path",
        "source_class",
        "cow_group",
        "image_name",
        "image_type",
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
    # Verify one source class per cow group
    # --------------------------------------------------------

    ownership = (
        df.groupby(
            "cow_group"
        )["source_class"]
        .nunique()
    )

    shared = ownership[
        ownership > 1
    ]

    if not shared.empty:

        raise RuntimeError(
            "A cow group appears in multiple source classes:\n"
            + shared.to_string()
        )

    print()
    print(
        f"Images      : {len(df)}"
    )

    print(
        f"Cow groups  : "
        f"{df['cow_group'].nunique()}"
    )

    # --------------------------------------------------------
    # Unique cow-group table
    # --------------------------------------------------------

    cows = (
        df[
            [
                "cow_group",
                "source_class",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Split each source class separately
    # --------------------------------------------------------

    train_parts = []
    val_parts = []
    test_parts = []

    for source_class in sorted(
        cows["source_class"].unique()
    ):

        class_cows = cows[
            cows["source_class"]
            == source_class
        ].copy()

        train_cows, val_cows, test_cows = (
            split_class_groups(
                class_cows,
                random_state=(
                    RANDOM_STATE
                    + int(source_class)
                ),
            )
        )

        train_parts.append(
            train_cows
        )

        val_parts.append(
            val_cows
        )

        test_parts.append(
            test_cows
        )

    train_cows = pd.concat(
        train_parts,
        ignore_index=True,
    )

    val_cows = pd.concat(
        val_parts,
        ignore_index=True,
    )

    test_cows = pd.concat(
        test_parts,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Build split map
    # --------------------------------------------------------

    split_map = {}

    for cow in train_cows["cow_group"]:
        split_map[cow] = "train"

    for cow in val_cows["cow_group"]:
        split_map[cow] = "validation"

    for cow in test_cows["cow_group"]:
        split_map[cow] = "test"

    df["split"] = (
        df["cow_group"]
        .map(split_map)
    )

    if df["split"].isna().any():

        missing_groups = (
            df.loc[
                df["split"].isna(),
                "cow_group",
            ]
            .unique()
            .tolist()
        )

        raise RuntimeError(
            "Unassigned cow groups:\n"
            + str(missing_groups)
        )

    # --------------------------------------------------------
    # Leakage checks
    # --------------------------------------------------------

    train_set = set(
        train_cows["cow_group"]
    )

    val_set = set(
        val_cows["cow_group"]
    )

    test_set = set(
        test_cows["cow_group"]
    )

    train_val = (
        train_set & val_set
    )

    train_test = (
        train_set & test_set
    )

    val_test = (
        val_set & test_set
    )

    if (
        train_val
        or train_test
        or val_test
    ):

        raise RuntimeError(
            "COW-GROUP DATA LEAKAGE DETECTED."
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("SPLIT SUMMARY")
    print("=" * 80)

    summary = (
        df.groupby("split")
        .agg(
            images=("image_name", "count"),
            cow_groups=("cow_group", "nunique"),
        )
        .reindex(
            [
                "train",
                "validation",
                "test",
            ]
        )
    )

    print(
        summary.to_string()
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("CLASS DISTRIBUTION BY COW GROUP")
    print("=" * 80)

    class_summary = (
        df.groupby(
            [
                "split",
                "source_class",
            ]
        )
        .agg(
            images=("image_name", "count"),
            cow_groups=("cow_group", "nunique"),
        )
        .sort_index()
    )

    print(
        class_summary.to_string()
    )

    # --------------------------------------------------------
    # Special singleton report
    # --------------------------------------------------------

    singleton_classes = (
        cows.groupby(
            "source_class"
        )
        .size()
    )

    singleton_classes = (
        singleton_classes[
            singleton_classes == 1
        ]
        .index
        .tolist()
    )

    if singleton_classes:

        print()
        print("=" * 80)
        print("SINGLETON SOURCE CLASSES")
        print("=" * 80)

        for cls in singleton_classes:

            group = cows.loc[
                cows["source_class"] == cls,
                "cow_group",
            ].iloc[0]

            image_count = len(
                df[
                    df["cow_group"]
                    == group
                ]
            )

            print(
                f"Class {cls}: "
                f"{group} → "
                f"{image_count} images → TRAIN ONLY"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_path = (
        OUTPUT_DIR
        / "dryad_manifest_split.csv"
    )

    train_path = (
        OUTPUT_DIR
        / "train.csv"
    )

    val_path = (
        OUTPUT_DIR
        / "validation.csv"
    )

    test_path = (
        OUTPUT_DIR
        / "test.csv"
    )

    cow_split_path = (
        OUTPUT_DIR
        / "cow_level_split.csv"
    )

    df.to_csv(
        full_path,
        index=False,
    )

    df[
        df["split"] == "train"
    ].to_csv(
        train_path,
        index=False,
    )

    df[
        df["split"] == "validation"
    ].to_csv(
        val_path,
        index=False,
    )

    df[
        df["split"] == "test"
    ].to_csv(
        test_path,
        index=False,
    )

    cows["split"] = (
        cows["cow_group"]
        .map(split_map)
    )

    cows.to_csv(
        cow_split_path,
        index=False,
    )

    # --------------------------------------------------------
    # Final leakage result
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("LEAKAGE CHECK")
    print("=" * 80)

    print(
        f"Train ∩ Validation : "
        f"{len(train_val)}"
    )

    print(
        f"Train ∩ Test       : "
        f"{len(train_test)}"
    )

    print(
        f"Validation ∩ Test  : "
        f"{len(val_test)}"
    )

    if not (
        train_val
        or train_test
        or val_test
    ):

        print(
            "[PASS] No cow-group leakage."
        )

    print()
    print("=" * 80)
    print("FILES CREATED")
    print("=" * 80)

    print(
        f"Full manifest : {full_path}"
    )

    print(
        f"Train         : {train_path}"
    )

    print(
        f"Validation    : {val_path}"
    )

    print(
        f"Test          : {test_path}"
    )

    print(
        f"Cow split     : {cow_split_path}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()