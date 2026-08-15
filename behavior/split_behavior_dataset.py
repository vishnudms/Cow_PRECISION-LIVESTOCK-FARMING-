from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\cow")

INPUT_FILE = (
    PROJECT_ROOT
    / "output"
    / "behavior"
    / "behavior_annotations.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "behavior"
)


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("COW PLF - BEHAVIOR SEQUENCE-LEVEL SPLITTER")
    print("=" * 80)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "image_path",
        "image_name",
        "sequence_id",
        "x",
        "y",
        "width",
        "height",
        "standing",
        "lying",
        "foraging",
        "drinking",
        "rumination",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing columns:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    if df["sequence_id"].isna().any():
        raise RuntimeError(
            "Missing sequence IDs detected."
        )

    print()
    print(f"Annotations : {len(df)}")
    print(f"Frames      : {df['image_name'].nunique()}")
    print(f"Sequences   : {df['sequence_id'].nunique()}")

    # --------------------------------------------------------
    # Unique sequences
    # --------------------------------------------------------

    groups = (
        df[["sequence_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Train vs temporary
    # --------------------------------------------------------

    splitter_1 = GroupShuffleSplit(
        n_splits=1,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_STATE,
    )

    train_idx, temp_idx = next(
        splitter_1.split(
            df,
            groups=df["sequence_id"],
        )
    )

    train_sequences = set(
        df.iloc[train_idx]["sequence_id"]
    )

    temp_sequences = set(
        df.iloc[temp_idx]["sequence_id"]
    )

    # --------------------------------------------------------
    # Temporary -> validation/test
    #
    # Split sequences directly so no sequence can cross splits.
    # --------------------------------------------------------

    temp_df = df[
        df["sequence_id"].isin(
            temp_sequences
        )
    ].copy()

    splitter_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=RANDOM_STATE + 1,
    )

    val_idx, test_idx = next(
        splitter_2.split(
            temp_df,
            groups=temp_df["sequence_id"],
        )
    )

    validation_sequences = set(
        temp_df.iloc[val_idx]["sequence_id"]
    )

    test_sequences = set(
        temp_df.iloc[test_idx]["sequence_id"]
    )

    # --------------------------------------------------------
    # Assign split
    # --------------------------------------------------------

    def assign_split(sequence_id):

        if sequence_id in train_sequences:
            return "train"

        if sequence_id in validation_sequences:
            return "validation"

        if sequence_id in test_sequences:
            return "test"

        raise RuntimeError(
            f"Sequence was not assigned: {sequence_id}"
        )

    df["split"] = df[
        "sequence_id"
    ].map(assign_split)

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    train_set = set(
        df.loc[
            df["split"] == "train",
            "sequence_id",
        ]
    )

    val_set = set(
        df.loc[
            df["split"] == "validation",
            "sequence_id",
        ]
    )

    test_set = set(
        df.loc[
            df["split"] == "test",
            "sequence_id",
        ]
    )

    train_val = train_set & val_set
    train_test = train_set & test_set
    val_test = val_set & test_set

    if train_val or train_test or val_test:
        raise RuntimeError(
            "SEQUENCE LEAKAGE DETECTED."
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
            annotations=("annotation_id", "count"),
            frames=("image_name", "nunique"),
            sequences=("sequence_id", "nunique"),
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
    # Behavior distribution
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("BEHAVIOR DISTRIBUTION BY SPLIT")
    print("=" * 80)

    behaviors = [
        "standing",
        "lying",
        "foraging",
        "drinking",
        "rumination",
    ]

    behavior_summary = []

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        part = df[
            df["split"] == split_name
        ]

        row = {
            "split": split_name,
            "annotations": len(part),
        }

        for behavior in behaviors:
            row[behavior] = int(
                part[behavior].sum()
            )

        behavior_summary.append(
            row
        )

    print(
        pd.DataFrame(
            behavior_summary
        ).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Sequence leakage
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("SEQUENCE LEAKAGE CHECK")
    print("=" * 80)

    print(
        f"Train ∩ Validation : {len(train_val)}"
    )

    print(
        f"Train ∩ Test       : {len(train_test)}"
    )

    print(
        f"Validation ∩ Test  : {len(val_test)}"
    )

    print(
        "[PASS] No sequence leakage."
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
        / "behavior_annotations_split.csv"
    )

    train_path = (
        OUTPUT_DIR
        / "train.csv"
    )

    validation_path = (
        OUTPUT_DIR
        / "validation.csv"
    )

    test_path = (
        OUTPUT_DIR
        / "test.csv"
    )

    sequence_split_path = (
        OUTPUT_DIR
        / "sequence_split.csv"
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
        validation_path,
        index=False,
    )

    df[
        df["split"] == "test"
    ].to_csv(
        test_path,
        index=False,
    )

    sequence_rows = []

    for sequence_id in sorted(
        df["sequence_id"].unique(),
        key=str,
    ):

        sequence_rows.append(
            {
                "sequence_id": sequence_id,
                "split": assign_split(
                    sequence_id
                ),
            }
        )

    pd.DataFrame(
        sequence_rows
    ).to_csv(
        sequence_split_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("FILES CREATED")
    print("=" * 80)

    print(
        f"Full      : {full_path}"
    )

    print(
        f"Train     : {train_path}"
    )

    print(
        f"Validation: {validation_path}"
    )

    print(
        f"Test      : {test_path}"
    )

    print(
        f"Sequences : {sequence_split_path}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()