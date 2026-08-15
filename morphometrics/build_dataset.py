"""
===============================================================================
COW PLF - MORPHOMETRICS DATASET BUILDER
===============================================================================

Purpose
-------
Process all cow images in D:/cow/dataset and build an image-derived
morphometrics dataset.

This script:
    1. Loads YOLO26m segmentation model
    2. Detects cows
    3. Extracts each cow segmentation mask
    4. Uses the existing CowViewAngleAnalyzer
    5. Keeps side-view validation consistent with the project
    6. Calculates pixel-based morphometric features
    7. Saves annotated images
    8. Saves all results to CSV

IMPORTANT
---------
These are PIXEL measurements.

They are NOT centimetres or kilograms.

Camera calibration will be implemented later.

Existing files are not modified.
===============================================================================
"""

from pathlib import Path
import sys
import math
import csv

import cv2
import numpy as np
from ultralytics import YOLO


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(r"D:\cow")

DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_PATH = PROJECT_ROOT / "models" / "yolo26m-seg.pt"

OUTPUT_DIR = PROJECT_ROOT / "output" / "morphometrics_dataset"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
CSV_PATH = OUTPUT_DIR / "morphometrics_dataset.csv"

SRC_DIR = PROJECT_ROOT / "src"

# Make src available for:
# from cow_plf.analytics.view_angle import CowViewAngleAnalyzer
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from cow_plf.analytics.view_angle import CowViewAngleAnalyzer


# =============================================================================
# CONFIGURATION
# =============================================================================

CONF_THRESHOLD = 0.25

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# =============================================================================
# HELPERS
# =============================================================================

def find_images():
    """Return all supported images in the dataset directory."""

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    images = [
        p
        for p in DATASET_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(
        images,
        key=lambda p: p.name.lower()
    )


def mask_to_binary(mask, width, height):
    """
    Convert YOLO mask data into a full-resolution binary mask.
    """

    if mask is None:
        return None

    try:
        arr = np.asarray(mask)
    except Exception:
        return None

    if arr.size == 0:
        return None

    if arr.ndim > 2:
        arr = np.squeeze(arr)

    if arr.ndim != 2:
        return None

    if arr.dtype.kind == "f":
        binary = (
            arr > 0.5
        ).astype(np.uint8) * 255
    else:
        binary = (
            arr > 0
        ).astype(np.uint8) * 255

    if (
        binary.shape[1] != width
        or binary.shape[0] != height
    ):
        binary = cv2.resize(
            binary,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

    return binary


def calculate_pixel_geometry(binary_mask):
    """
    Calculate simple pixel-based body geometry.

    Returns
    -------
    dict
        bounding-box width/height
        PCA-aligned body length/depth
        mask area
    """

    if binary_mask is None:
        return {
            "bbox_x": 0,
            "bbox_y": 0,
            "bbox_width_px": 0.0,
            "bbox_height_px": 0.0,
            "body_length_px": 0.0,
            "body_depth_px": 0.0,
            "mask_area_px": 0,
        }

    ys, xs = np.where(binary_mask > 0)

    if len(xs) < 20:
        return {
            "bbox_x": 0,
            "bbox_y": 0,
            "bbox_width_px": 0.0,
            "bbox_height_px": 0.0,
            "body_length_px": 0.0,
            "body_depth_px": 0.0,
            "mask_area_px": int(len(xs)),
        }

    # -------------------------------------------------------------------------
    # Bounding box
    # -------------------------------------------------------------------------

    x_min = int(xs.min())
    x_max = int(xs.max())

    y_min = int(ys.min())
    y_max = int(ys.max())

    bbox_width = float(x_max - x_min + 1)
    bbox_height = float(y_max - y_min + 1)

    # -------------------------------------------------------------------------
    # PCA body axis
    # -------------------------------------------------------------------------

    points = np.column_stack(
        (
            xs.astype(np.float32),
            ys.astype(np.float32),
        )
    )

    center = np.mean(
        points,
        axis=0
    )

    centered = points - center

    covariance = np.cov(
        centered,
        rowvar=False
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    order = np.argsort(
        eigenvalues
    )[::-1]

    eigenvectors = eigenvectors[
        :,
        order
    ]

    major_axis = eigenvectors[:, 0]
    minor_axis = eigenvectors[:, 1]

    # -------------------------------------------------------------------------
    # Project all mask pixels onto PCA axes
    # -------------------------------------------------------------------------

    major_projection = centered @ major_axis
    minor_projection = centered @ minor_axis

    body_length = float(
        major_projection.max()
        -
        major_projection.min()
    )

    body_depth = float(
        minor_projection.max()
        -
        minor_projection.min()
    )

    return {
        "bbox_x": x_min,
        "bbox_y": y_min,
        "bbox_width_px": bbox_width,
        "bbox_height_px": bbox_height,
        "body_length_px": body_length,
        "body_depth_px": body_depth,
        "mask_area_px": int(len(xs)),
    }


def draw_result(
    image,
    binary_mask,
    result,
    cow_index,
    detection_confidence,
):
    """Draw segmentation and measurement information."""

    output = image.copy()

    if binary_mask is not None:

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            output,
            contours,
            -1,
            (0, 255, 0),
            4,
        )

    # -------------------------------------------------------------------------
    # Text
    # -------------------------------------------------------------------------

    x = int(result["bbox_x"])
    y = int(result["bbox_y"])

    if x < 10:
        x = 10

    if y < 30:
        y = 30

    lines = [
        f"COW {cow_index}",
        f"View: {result['view_class']}",
        f"Valid: {result['measurement_valid']}",
        f"View conf: {result['view_confidence']:.3f}",
        f"Detection conf: {detection_confidence:.3f}",
        f"Length px: {result['body_length_px']:.1f}",
        f"Depth px: {result['body_depth_px']:.1f}",
        f"BBox: {result['bbox_width_px']:.1f} x {result['bbox_height_px']:.1f}",
    ]

    line_height = 34

    for i, text in enumerate(lines):

        cv2.putText(
            output,
            text,
            (
                x,
                y + i * line_height,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 80)
    print("COW PLF - MORPHOMETRICS DATASET BUILDER")
    print("=" * 80)

    print()
    print("[CONFIG]")
    print("-" * 80)
    print(f"Dataset : {DATASET_DIR}")
    print(f"Model   : {MODEL_PATH}")
    print(f"Output  : {OUTPUT_DIR}")
    print(f"CSV     : {CSV_PATH}")

    # -------------------------------------------------------------------------
    # Prepare output directories
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ANNOTATED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------------------------------
    # Find images
    # -------------------------------------------------------------------------

    images = find_images()

    print()
    print(f"[DATASET] Found {len(images)} images")

    if not images:
        print("[ERROR] No supported images found.")
        return

    # -------------------------------------------------------------------------
    # Load YOLO
    # -------------------------------------------------------------------------

    print()
    print("[MODEL] Loading YOLO segmentation model...")
    print(f"[VISION] Loading model: {MODEL_PATH}")

    model = YOLO(
        str(MODEL_PATH)
    )

    print("[VISION] YOLO26m-seg loaded.")

    # -------------------------------------------------------------------------
    # View analyzer
    # -------------------------------------------------------------------------

    analyzer = CowViewAngleAnalyzer()

    # -------------------------------------------------------------------------
    # CSV fields
    # -------------------------------------------------------------------------

    fieldnames = [
        "image_name",
        "image_width_px",
        "image_height_px",
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

    rows = []

    total_cows = 0
    total_valid = 0
    total_good = 0
    total_acceptable = 0
    total_invalid = 0
    total_failures = 0

    # -------------------------------------------------------------------------
    # Process images
    # -------------------------------------------------------------------------

    for image_number, image_path in enumerate(
        images,
        start=1
    ):

        print()
        print("=" * 80)
        print(
            f"IMAGE {image_number}/{len(images)}"
        )
        print("=" * 80)
        print(
            f"File: {image_path.name}"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                "[ERROR] Could not read image."
            )

            total_failures += 1
            continue

        image_height, image_width = (
            image.shape[:2]
        )

        print(
            f"Resolution: {image_width} x {image_height}"
        )

        # ---------------------------------------------------------------------
        # YOLO inference
        # ---------------------------------------------------------------------

        try:

            results = model.predict(
                source=image,
                conf=CONF_THRESHOLD,
                verbose=False,
            )

        except Exception as exc:

            print(
                f"[ERROR] YOLO inference failed: {exc}"
            )

            total_failures += 1
            continue

        if not results:

            print(
                "[DETECTION] 0 cow(s)"
            )
            continue

        result = results[0]

        if result.masks is None:

            print(
                "[DETECTION] 0 cow(s)"
            )
            continue

        masks = result.masks.data

        boxes = result.boxes

        cow_count = len(masks)

        print(
            f"[DETECTION] {cow_count} cow(s)"
        )

        annotated = image.copy()

        # ---------------------------------------------------------------------
        # Process every detected cow
        # ---------------------------------------------------------------------

        for cow_index in range(cow_count):

            total_cows += 1

            print()
            print(
                f"COW {cow_index + 1}"
            )
            print("-" * 80)

            try:

                # Detection confidence
                detection_confidence = float(
                    boxes.conf[cow_index].item()
                )

                # YOLO mask
                raw_mask = masks[cow_index].cpu().numpy()

                binary_mask = mask_to_binary(
                    raw_mask,
                    image_width,
                    image_height,
                )

                if binary_mask is None:

                    print(
                        "[ERROR] Invalid segmentation mask."
                    )

                    total_failures += 1
                    continue

                # -------------------------------------------------------------
                # Existing view analyzer
                # -------------------------------------------------------------

                view_result = analyzer.analyze(
                    binary_mask,
                    frame_width=image_width,
                    frame_height=image_height,
                )

                # -------------------------------------------------------------
                # Pixel geometry
                # -------------------------------------------------------------

                geometry = calculate_pixel_geometry(
                    binary_mask
                )

                # -------------------------------------------------------------
                # Combine
                # -------------------------------------------------------------

                row = {
                    "image_name": image_path.name,
                    "image_width_px": image_width,
                    "image_height_px": image_height,
                    "cow_index": cow_index + 1,

                    "detection_confidence":
                        round(
                            detection_confidence,
                            6
                        ),

                    "view_class":
                        view_result["view_class"],

                    "view_confidence":
                        round(
                            view_result["confidence"],
                            6
                        ),

                    "measurement_valid":
                        view_result["measurement_valid"],

                    "orientation_deg":
                        round(
                            view_result["orientation_deg"],
                            6
                        ),

                    "horizontal_angle_deg":
                        round(
                            view_result[
                                "horizontal_angle_deg"
                            ],
                            6
                        ),

                    "elongation_ratio":
                        round(
                            view_result[
                                "elongation_ratio"
                            ],
                            6
                        ),

                    "rectangle_ratio":
                        round(
                            view_result[
                                "rectangle_ratio"
                            ],
                            6
                        ),

                    "mask_area_px":
                        view_result[
                            "mask_area_px"
                        ],

                    "center_x":
                        round(
                            view_result["center_x"],
                            3
                        ),

                    "center_y":
                        round(
                            view_result["center_y"],
                            3
                        ),

                    "bbox_x":
                        geometry["bbox_x"],

                    "bbox_y":
                        geometry["bbox_y"],

                    "bbox_width_px":
                        round(
                            geometry[
                                "bbox_width_px"
                            ],
                            3
                        ),

                    "bbox_height_px":
                        round(
                            geometry[
                                "bbox_height_px"
                            ],
                            3
                        ),

                    "body_length_px":
                        round(
                            geometry[
                                "body_length_px"
                            ],
                            3
                        ),

                    "body_depth_px":
                        round(
                            geometry[
                                "body_depth_px"
                            ],
                            3
                        ),
                }

                rows.append(
                    row
                )

                # -------------------------------------------------------------
                # Statistics
                # -------------------------------------------------------------

                if row["measurement_valid"]:

                    total_valid += 1

                if (
                    row["view_class"]
                    ==
                    "SIDE_VIEW_GOOD"
                ):

                    total_good += 1

                elif (
                    row["view_class"]
                    ==
                    "SIDE_VIEW_ACCEPTABLE"
                ):

                    total_acceptable += 1

                else:

                    total_invalid += 1

                # -------------------------------------------------------------
                # Console output
                # -------------------------------------------------------------

                print(
                    f"view_class         : "
                    f"{row['view_class']}"
                )

                print(
                    f"confidence         : "
                    f"{row['view_confidence']:.4f}"
                )

                print(
                    f"measurement_valid  : "
                    f"{row['measurement_valid']}"
                )

                print(
                    f"orientation_deg    : "
                    f"{row['orientation_deg']:.2f}"
                )

                print(
                    f"elongation_ratio   : "
                    f"{row['elongation_ratio']:.4f}"
                )

                print(
                    f"rectangle_ratio    : "
                    f"{row['rectangle_ratio']:.4f}"
                )

                print(
                    f"mask_area_px       : "
                    f"{row['mask_area_px']}"
                )

                print(
                    f"body_length_px     : "
                    f"{row['body_length_px']:.2f}"
                )

                print(
                    f"body_depth_px      : "
                    f"{row['body_depth_px']:.2f}"
                )

                # -------------------------------------------------------------
                # Draw
                # -------------------------------------------------------------

                annotated = draw_result(
                    annotated,
                    binary_mask,
                    row,
                    cow_index + 1,
                    detection_confidence,
                )

            except Exception as exc:

                print(
                    f"[ERROR] Cow analysis failed: {exc}"
                )

                total_failures += 1

        # ---------------------------------------------------------------------
        # Save annotated image
        # ---------------------------------------------------------------------

        output_name = (
            image_path.stem
            +
            "_morphometrics.jpg"
        )

        output_path = (
            ANNOTATED_DIR
            /
            output_name
        )

        cv2.imwrite(
            str(output_path),
            annotated
        )

        print()
        print(
            f"[SAVED] {output_path}"
        )

    # =========================================================================
    # SAVE CSV
    # =========================================================================

    print()
    print("=" * 80)
    print("SAVING DATASET")
    print("=" * 80)

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print()
    print("=" * 80)
    print("MORPHOMETRICS DATASET BUILD COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Images processed        : "
        f"{len(images)}"
    )

    print(
        f"Total cows detected     : "
        f"{total_cows}"
    )

    print(
        f"Side views GOOD         : "
        f"{total_good}"
    )

    print(
        f"Side views ACCEPTABLE   : "
        f"{total_acceptable}"
    )

    print(
        f"Invalid/other views     : "
        f"{total_invalid}"
    )

    print(
        f"Measurement valid       : "
        f"{total_valid}"
    )

    print(
        f"Analysis failures       : "
        f"{total_failures}"
    )

    print()
    print(
        f"CSV dataset             : "
        f"{CSV_PATH}"
    )

    print(
        f"Annotated images        : "
        f"{ANNOTATED_DIR}"
    )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()