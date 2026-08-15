from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "yolo26m-seg.pt"

MASTER_DATASET = (
    PROJECT_ROOT
    / "output"
    / "morphometrics_dataset"
    / "master_cow_dataset.csv"
)

IMAGE_DIR = PROJECT_ROOT / "dataset"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "bcs_dataset"
)

OUTPUT_CSV = OUTPUT_DIR / "bcs_features.csv"

# ---------------------------------------------------------------------
# IMPORT BCS FEATURE EXTRACTOR
# ---------------------------------------------------------------------

sys.path.insert(0, str(PROJECT_ROOT))

from bcs.extract_features import extract_bcs_features


# ---------------------------------------------------------------------
# IMAGE SEARCH
# ---------------------------------------------------------------------

def find_image(image_name: str) -> Path | None:
    """Find an image in the same dataset directory used by morphometrics."""

    image_path = IMAGE_DIR / image_name

    if image_path.exists() and image_path.is_file():
        return image_path

    return None

# ---------------------------------------------------------------------
# MASK CONVERSION
# ---------------------------------------------------------------------

def mask_to_binary(mask, width: int, height: int):
    """
    Convert YOLO segmentation mask into full-resolution
    binary mask.
    """

    if mask is None:
        return None

    arr = np.asarray(mask)

    if arr.ndim != 2:
        return None

    binary = (arr > 0.5).astype(np.uint8)

    if binary.shape[1] != width or binary.shape[0] != height:
        binary = cv2.resize(
            binary,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    return binary


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print()
    print("=" * 80)
    print("COW PLF - BCS FEATURE DATASET BUILDER")
    print("=" * 80)

    # -------------------------------------------------------------
    # Validate files
    # -------------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found:\n{MODEL_PATH}"
        )

    if not MASTER_DATASET.exists():
        raise FileNotFoundError(
            f"Master dataset not found:\n{MASTER_DATASET}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Load master dataset
    # -------------------------------------------------------------

    print()
    print("[LOAD] Master dataset")

    master = pd.read_csv(MASTER_DATASET)

    print(
        f"       Rows: {len(master)}"
    )

    required_columns = [
        "cow_id",
        "image_name",
        "view_class",
        "measurement_valid",
        "actual_weight_kg",
    ]

    missing = [
        column
        for column in required_columns
        if column not in master.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # -------------------------------------------------------------
    # Select valid measurements
    # -------------------------------------------------------------

    valid = master[
        (master["view_class"] == "SIDE_VIEW_GOOD")
        & (master["measurement_valid"] == True)
    ].copy()

    print(
        f"[VALID] Side-view cows: {len(valid)}"
    )

    # -------------------------------------------------------------
    # Load YOLO
    # -------------------------------------------------------------

    print()
    print("[MODEL] Loading YOLO26m segmentation model...")
    print(f"        {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    print("[OK] YOLO26m-seg loaded.")

    # -------------------------------------------------------------
    # Process images
    # -------------------------------------------------------------

    rows = []

    total = len(valid)

    for index, row in valid.iterrows():

        image_name = str(row["image_name"])

        print()
        print("-" * 80)
        print(
            f"[{len(rows) + 1}/{total}] {image_name}"
        )

        image_path = find_image(image_name)

        if image_path is None:

            print(
                f"[WARNING] Image not found: {image_name}"
            )

            continue

        image = cv2.imread(str(image_path))

        if image is None:

            print(
                f"[WARNING] Could not read: {image_path}"
            )

            continue

        height, width = image.shape[:2]

        # ---------------------------------------------------------
        # YOLO inference
        # ---------------------------------------------------------

        try:

            results = model.predict(
                source=image,
                verbose=False,
                conf=0.25,
                imgsz=1280,
            )

        except Exception as exc:

            print(
                f"[ERROR] YOLO inference failed: {exc}"
            )

            continue

        if not results:

            print("[WARNING] No YOLO result.")

            continue

        result = results[0]

        if result.masks is None:

            print("[WARNING] No segmentation masks.")

            continue

        masks = result.masks.data

        if len(masks) == 0:

            print("[WARNING] Empty segmentation.")

            continue

        # ---------------------------------------------------------
        # Find primary cow detection
        #
        # We use the detection closest to the center of the
        # ground-truth bounding box stored in the master dataset.
        # ---------------------------------------------------------

        target_cx = float(row["center_x"])
        target_cy = float(row["center_y"])

        best_mask = None
        best_distance = float("inf")

        for mask_index in range(len(masks)):

            raw_mask = masks[mask_index].cpu().numpy()

            binary_mask = mask_to_binary(
                raw_mask,
                width,
                height,
            )

            if binary_mask is None:
                continue

            ys, xs = np.where(binary_mask > 0)

            if len(xs) == 0:
                continue

            mask_cx = float(np.mean(xs))
            mask_cy = float(np.mean(ys))

            distance = (
                (mask_cx - target_cx) ** 2
                + (mask_cy - target_cy) ** 2
            )

            if distance < best_distance:

                best_distance = distance
                best_mask = binary_mask

        if best_mask is None:

            print(
                "[WARNING] Could not select primary mask."
            )

            continue

        # ---------------------------------------------------------
        # Extract BCS features
        # ---------------------------------------------------------

        features = extract_bcs_features(
            best_mask
        )

        # ---------------------------------------------------------
        # Build output row
        # ---------------------------------------------------------

        output = {
            "cow_id": row["cow_id"],
            "image_name": image_name,

            # Ground truth
            "actual_weight_kg": row["actual_weight_kg"],

            # Existing morphometrics
            "oblique_body_length_cm": row[
                "oblique_body_length_cm"
            ],
            "withers_height_cm": row[
                "withers_height_cm"
            ],
            "heart_girth_cm": row[
                "heart_girth_cm"
            ],
            "hip_length_cm": row[
                "hip_length_cm"
            ],

            # Existing vision features
            "body_length_px": row[
                "body_length_px"
            ],
            "body_depth_px": row[
                "body_depth_px"
            ],
            "mask_area_px": row[
                "mask_area_px"
            ],
            "elongation_ratio": row[
                "elongation_ratio"
            ],
            "rectangle_ratio": row[
                "rectangle_ratio"
            ],

            # BCS features
            **features,
        }

        rows.append(output)

        print(
            f"[OK] Features extracted | "
            f"Mask area={features['bcs_mask_area_px']:.0f} px"
        )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    print()
    print("=" * 80)
    print("SAVING BCS DATASET")
    print("=" * 80)

    if not rows:

        raise RuntimeError(
            "No BCS feature rows were generated."
        )

    dataset = pd.DataFrame(rows)

    dataset.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print()
    print(
        f"BCS feature rows : {len(dataset)}"
    )

    print(
        f"CSV              : {OUTPUT_CSV}"
    )

    print()
    print("=" * 80)
    print("BCS FEATURE DATASET COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()