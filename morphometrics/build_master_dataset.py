"""
===============================================================================
COW PLF - MASTER IMAGE + GROUND-TRUTH DATASET
===============================================================================

Purpose
-------
Combine:

1. Ground-truth cattle measurements + actual weight
2. YOLO/view-angle/image-derived morphometric features

One row = one photographed cow.

Primary detection rule:
    For each image, select the detection with the largest segmentation
    mask area.

Why:
    Each image represents one target cow. Small secondary detections are
    treated as unwanted detections/background detections.

Important:
    This script does NOT train a model.
    It only creates the clean master dataset.

Inputs
------
D:/cow/output/morphometrics_dataset/paired_cow_dataset.csv
D:/cow/output/morphometrics_dataset/morphometrics_dataset.csv

Output
------
D:/cow/output/morphometrics_dataset/master_cow_dataset.csv
===============================================================================
"""

from pathlib import Path
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(r"D:\cow")

OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "morphometrics_dataset"
)

PAIRED_FILE = (
    OUTPUT_DIR
    / "paired_cow_dataset.csv"
)

VISION_FILE = (
    OUTPUT_DIR
    / "morphometrics_dataset.csv"
)

MASTER_FILE = (
    OUTPUT_DIR
    / "master_cow_dataset.csv"
)


# =============================================================================
# REQUIRED COLUMNS
# =============================================================================

PAIRED_REQUIRED = [
    "cow_id",
    "image_name",
    "oblique_body_length_cm",
    "withers_height_cm",
    "heart_girth_cm",
    "hip_length_cm",
    "actual_weight_kg",
]

VISION_REQUIRED = [
    "image_name",
    "cow_index",
    "detection_confidence",
    "view_class",
    "view_confidence",
    "measurement_valid",
    "orientation_deg",
    "horizontal_angle_deg",
    "elongation_ratio",
    "rectangle_ratio",
    "mask_area_px",
    "center_x",
    "center_y",
    "bbox_x",
    "bbox_y",
    "bbox_width_px",
    "bbox_height_px",
    "body_length_px",
    "body_depth_px",
]


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 80)
    print("COW PLF - MASTER DATASET BUILDER")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Check files
    # -------------------------------------------------------------------------

    if not PAIRED_FILE.exists():
        raise FileNotFoundError(
            f"Paired dataset not found:\n{PAIRED_FILE}"
        )

    if not VISION_FILE.exists():
        raise FileNotFoundError(
            f"Vision dataset not found:\n{VISION_FILE}"
        )

    # -------------------------------------------------------------------------
    # Load files
    # -------------------------------------------------------------------------

    print()
    print("[LOAD] Ground-truth paired dataset")

    paired = pd.read_csv(
        PAIRED_FILE
    )

    print(
        f"       Rows: {len(paired)}"
    )

    print()
    print("[LOAD] YOLO morphometrics dataset")

    vision = pd.read_csv(
        VISION_FILE
    )

    print(
        f"       Rows: {len(vision)}"
    )

    # -------------------------------------------------------------------------
    # Validate columns
    # -------------------------------------------------------------------------

    missing_paired = [
        c for c in PAIRED_REQUIRED
        if c not in paired.columns
    ]

    missing_vision = [
        c for c in VISION_REQUIRED
        if c not in vision.columns
    ]

    if missing_paired:
        raise ValueError(
            f"Missing paired columns: {missing_paired}"
        )

    if missing_vision:
        raise ValueError(
            f"Missing vision columns: {missing_vision}"
        )

    # -------------------------------------------------------------------------
    # Ensure one primary detection per image
    # -------------------------------------------------------------------------

    print()
    print("[VISION] Selecting primary detection per image")

    # Largest segmentation mask = primary cow candidate.
    #
    # We intentionally do NOT filter to SIDE_VIEW here.
    # A real target cow can be present but not be measurable.
    # We want to retain that information in the master dataset.

    vision_sorted = vision.sort_values(
        by=[
            "image_name",
            "mask_area_px",
            "detection_confidence",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )

    primary = (
        vision_sorted
        .drop_duplicates(
            subset=["image_name"],
            keep="first",
        )
        .copy()
    )

    print(
        f"       Total detections : {len(vision)}"
    )

    print(
        f"       Primary images   : {len(primary)}"
    )

    # -------------------------------------------------------------------------
    # Rename primary detection fields
    # -------------------------------------------------------------------------

    primary = primary.rename(
        columns={
            "cow_index": "selected_detection_index",
            "detection_confidence":
                "image_detection_confidence",
            "view_confidence":
                "image_view_confidence",
        }
    )

    # -------------------------------------------------------------------------
    # Merge using image filename
    # -------------------------------------------------------------------------

    print()
    print("[MERGE] Combining ground truth + vision features")

    master = paired.merge(
        primary,
        on="image_name",
        how="left",
        validate="one_to_one",
    )

    # -------------------------------------------------------------------------
    # Sort by cow number
    # -------------------------------------------------------------------------

    master = master.sort_values(
        by="cow_id"
    ).reset_index(
        drop=True
    )

    # -------------------------------------------------------------------------
    # Add status column
    # -------------------------------------------------------------------------

    def measurement_status(row):

        if pd.isna(row["view_class"]):
            return "NO_VISION_RESULT"

        if bool(row["measurement_valid"]):
            return "MEASUREMENT_VALID"

        return "VIEW_REJECTED"

    master["measurement_status"] = (
        master.apply(
            measurement_status,
            axis=1,
        )
    )

    # -------------------------------------------------------------------------
    # Reorder columns
    # -------------------------------------------------------------------------

    desired_order = [
        "cow_id",
        "image_name",

        # Ground truth
        "oblique_body_length_cm",
        "withers_height_cm",
        "heart_girth_cm",
        "hip_length_cm",
        "actual_weight_kg",

        # Vision quality
        "image_detection_confidence",
        "view_class",
        "image_view_confidence",
        "measurement_valid",
        "measurement_status",

        # View geometry
        "orientation_deg",
        "horizontal_angle_deg",
        "elongation_ratio",
        "rectangle_ratio",

        # Segmentation geometry
        "mask_area_px",
        "bbox_width_px",
        "bbox_height_px",
        "body_length_px",
        "body_depth_px",

        # Position
        "center_x",
        "center_y",
        "bbox_x",
        "bbox_y",

        # Image dimensions
        "image_width_px",
        "image_height_px",

        # Internal detection number
        "selected_detection_index",
    ]

    existing_columns = [
        c for c in desired_order
        if c in master.columns
    ]

    remaining_columns = [
        c for c in master.columns
        if c not in existing_columns
    ]

    master = master[
        existing_columns
        +
        remaining_columns
    ]

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    master.to_csv(
        MASTER_FILE,
        index=False,
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("MASTER DATASET COMPLETE")
    print("=" * 80)

    print(
        f"Ground-truth cows       : {len(paired)}"
    )

    print(
        f"Master rows             : {len(master)}"
    )

    print(
        f"Vision detections       : {len(vision)}"
    )

    print()
    print("View classification:")
    print(
        master["view_class"]
        .value_counts(dropna=False)
        .to_string()
    )

    print()
    print("Measurement status:")
    print(
        master["measurement_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    missing_vision = master[
        master["view_class"].isna()
    ]

    if len(missing_vision) > 0:

        print()
        print("[WARNING] Images without vision results:")
        print(
            missing_vision[
                ["cow_id", "image_name"]
            ].to_string(index=False)
        )

    # -------------------------------------------------------------------------
    # Basic weight statistics
    # -------------------------------------------------------------------------

    print()
    print("Ground-truth weight statistics:")
    print(
        master["actual_weight_kg"]
        .describe()
        .to_string()
    )

    print()
    print("Sample of master dataset:")
    print(
        master.head(10).to_string(
            index=False
        )
    )

    print()
    print(
        f"[SAVED] {MASTER_FILE}"
    )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()