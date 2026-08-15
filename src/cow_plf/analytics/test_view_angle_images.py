"""
===============================================================================
COW PLF - REAL COW IMAGE VIEW ANGLE TEST
===============================================================================

Tests the existing CowViewAngleAnalyzer against real cow images.

Dataset:
    D:\cow\dataset

Output:
    D:\cow\output\view_angle_test

This test does NOT calculate weight or BCS.
It determines whether the cow is in a suitable side-view orientation
for reliable morphometric measurement.
===============================================================================
"""

from pathlib import Path
import sys
import cv2


# =============================================================================
# PROJECT
# =============================================================================

PROJECT_ROOT = Path(r"D:\cow")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# IMPORTS
# =============================================================================

from src.cow_plf.analytics.view_angle import CowViewAngleAnalyzer
from src.cow_plf.vision.detector import CowDetector


# =============================================================================
# CONFIG
# =============================================================================

DATASET_DIR = PROJECT_ROOT / "dataset"

OUTPUT_DIR = (
    PROJECT_ROOT /
    "output" /
    "view_angle_test"
)

MODEL_PATH = (
    PROJECT_ROOT /
    "models" /
    "yolo26m-seg.pt"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# SETTINGS
# =============================================================================

CONFIDENCE = 0.20
IOU = 0.50
IMAGE_SIZE = 1280
DEVICE = 0


# =============================================================================
# HEADER
# =============================================================================

print()
print("=" * 80)
print("COW PLF - REAL COW IMAGE VIEW ANGLE TEST")
print("=" * 80)

print()
print("[CONFIG]")
print("-" * 80)

print(f"Dataset : {DATASET_DIR}")
print(f"Model   : {MODEL_PATH}")
print(f"Output  : {OUTPUT_DIR}")


# =============================================================================
# CHECKS
# =============================================================================

if not DATASET_DIR.exists():

    print()
    print("[ERROR] Dataset folder not found:")
    print(DATASET_DIR)

    raise SystemExit(1)


if not MODEL_PATH.exists():

    print()
    print("[ERROR] Model not found:")
    print(MODEL_PATH)

    raise SystemExit(1)


# =============================================================================
# FIND IMAGES
# =============================================================================

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

image_files = sorted(
    [
        p
        for p in DATASET_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in extensions
    ]
)


if not image_files:

    print()
    print("[ERROR] No images found.")

    raise SystemExit(1)


print()
print(
    f"[DATASET] Found {len(image_files)} images"
)


# =============================================================================
# LOAD YOLO
# =============================================================================

print()
print("[MODEL] Loading YOLO segmentation model...")

detector = CowDetector(
    model_path=MODEL_PATH,
    confidence=CONFIDENCE,
    iou=IOU,
    image_size=IMAGE_SIZE,
    device=DEVICE,
    tracker="bytetrack.yaml"
)


# =============================================================================
# VIEW ANGLE ANALYZER
# =============================================================================

view_analyzer = CowViewAngleAnalyzer()


# =============================================================================
# STATISTICS
# =============================================================================

total_images = 0
images_with_cows = 0

total_cows = 0

side_good = 0
side_acceptable = 0
invalid_views = 0
analysis_failures = 0


# =============================================================================
# PROCESS
# =============================================================================

for index, image_path in enumerate(
    image_files,
    start=1
):

    total_images += 1

    print()
    print("=" * 80)

    print(
        f"IMAGE {index}/{len(image_files)}"
    )

    print("=" * 80)

    print(
        f"File: {image_path.name}"
    )

    # -------------------------------------------------------------------------
    # READ
    # -------------------------------------------------------------------------

    frame = cv2.imread(
        str(image_path)
    )

    if frame is None:

        print(
            "[ERROR] Could not read image."
        )

        continue

    height, width = frame.shape[:2]

    print(
        f"Resolution: {width} x {height}"
    )

    # -------------------------------------------------------------------------
    # YOLO
    # -------------------------------------------------------------------------

    try:

        detections = detector.predict(
            frame
        )

    except Exception as exc:

        print(
            f"[ERROR] Detection failed: {exc}"
        )

        continue


    if not detections:

        print(
            "[NO COW] No cow detected."
        )

        output_path = (
            OUTPUT_DIR /
            f"{image_path.stem}_no_detection.jpg"
        )

        cv2.imwrite(
            str(output_path),
            frame
        )

        continue


    images_with_cows += 1

    print(
        f"[DETECTION] {len(detections)} cow(s)"
    )


    # -------------------------------------------------------------------------
    # ANNOTATED IMAGE
    # -------------------------------------------------------------------------

    annotated = frame.copy()


    # =========================================================================
    # EACH COW
    # =========================================================================

    for cow_number, detection in enumerate(
        detections,
        start=1
    ):

        total_cows += 1

        # ---------------------------------------------------------------------
        # BBOX
        # ---------------------------------------------------------------------

        x1, y1, x2, y2 = map(
            int,
            detection.box
        )

        x1 = max(
            0,
            min(
                x1,
                width - 1
            )
        )

        y1 = max(
            0,
            min(
                y1,
                height - 1
            )
        )

        x2 = max(
            x1 + 1,
            min(
                x2,
                width
            )
        )

        y2 = max(
            y1 + 1,
            min(
                y2,
                height
            )
        )


        # ---------------------------------------------------------------------
        # MASK
        # ---------------------------------------------------------------------

        mask = detection.mask

        if mask is None:

            print(
                f"[COW {cow_number}] No segmentation mask."
            )

            analysis_failures += 1

            continue


        # ---------------------------------------------------------------------
        # IMPORTANT
        # ---------------------------------------------------------------------
        #
        # Your existing CowViewAngleAnalyzer.prepare_mask() does not accept
        # width/height.
        #
        # Therefore we first convert the YOLO mask ourselves.
        #

        try:

            mask = cv2.resize(
                mask.astype("uint8"),
                (
                    width,
                    height
                ),
                interpolation=cv2.INTER_NEAREST
            )

            binary_mask = (
                mask > 0
            ).astype(
                "uint8"
            ) * 255

        except Exception as exc:

            print(
                f"[COW {cow_number}] "
                f"Mask preparation failed: {exc}"
            )

            analysis_failures += 1

            continue


        # ---------------------------------------------------------------------
        # KEEP LARGEST COMPONENT
        # ---------------------------------------------------------------------

        try:

            binary_mask = (
                view_analyzer.keep_largest_component(
                    binary_mask
                )
            )

        except Exception as exc:

            print(
                f"[COW {cow_number}] "
                f"Mask cleaning failed: {exc}"
            )

            analysis_failures += 1

            continue


        if binary_mask is None:

            print(
                f"[COW {cow_number}] Empty mask."
            )

            analysis_failures += 1

            continue


        # ---------------------------------------------------------------------
        # VIEW ANALYSIS
        # ---------------------------------------------------------------------

        try:

            result = (
                view_analyzer.analyze(
                    binary_mask
                )
            )

        except Exception as exc:

            print()
            print(
                f"[COW {cow_number}] "
                f"View analysis failed:"
            )

            print(
                f"    {exc}"
            )

            analysis_failures += 1

            continue


        # ---------------------------------------------------------------------
        # RESULTS
        # ---------------------------------------------------------------------

        view_class = result.get(
            "view_class",
            "UNKNOWN"
        )

        confidence = float(
            result.get(
                "confidence",
                0.0
            )
        )

        measurement_valid = bool(
            result.get(
                "measurement_valid",
                False
            )
        )

        orientation = float(
            result.get(
                "orientation_deg",
                0.0
            )
        )

        elongation = float(
            result.get(
                "elongation_ratio",
                0.0
            )
        )

        rectangle_ratio = float(
            result.get(
                "rectangle_ratio",
                0.0
            )
        )

        mask_area = int(
            result.get(
                "mask_area_px",
                cv2.countNonZero(
                    binary_mask
                )
            )
        )


        # ---------------------------------------------------------------------
        # STATISTICS
        # ---------------------------------------------------------------------

        if view_class == "SIDE_VIEW_GOOD":

            side_good += 1

        elif view_class == "SIDE_VIEW_ACCEPTABLE":

            side_acceptable += 1

        else:

            invalid_views += 1


        # ---------------------------------------------------------------------
        # TERMINAL OUTPUT
        # ---------------------------------------------------------------------

        print()
        print(
            f"COW {cow_number}"
        )

        print(
            "-" * 80
        )

        print(
            f"view_class         : {view_class}"
        )

        print(
            f"confidence         : {confidence:.4f}"
        )

        print(
            f"measurement_valid  : {measurement_valid}"
        )

        print(
            f"orientation_deg    : {orientation:.2f}"
        )

        print(
            f"elongation_ratio   : {elongation:.4f}"
        )

        print(
            f"rectangle_ratio    : {rectangle_ratio:.4f}"
        )

        print(
            f"mask_area_px       : {mask_area}"
        )


        # =========================================================================
        # DRAW BOX
        # =========================================================================

        cv2.rectangle(
            annotated,
            (
                x1,
                y1
            ),
            (
                x2,
                y2
            ),
            (255, 255, 255),
            4
        )


        # =========================================================================
        # CENTER
        # =========================================================================

        center_x = int(
            (x1 + x2) / 2
        )

        center_y = int(
            (y1 + y2) / 2
        )


        # =========================================================================
        # DRAW CENTER POINT
        # =========================================================================

        cv2.circle(
            annotated,
            (
                center_x,
                center_y
            ),
            8,
            (255, 255, 255),
            -1
        )


        # =========================================================================
        # COW NUMBER
        # =========================================================================

        number = str(
            cow_number
        )

        font = cv2.FONT_HERSHEY_SIMPLEX

        font_scale = 1.0

        thickness = 3

        (
            text_w,
            text_h
        ), baseline = cv2.getTextSize(
            number,
            font,
            font_scale,
            thickness
        )

        padding = 12


        # -------------------------------------------------------------------------
        # NUMBER BACKGROUND
        # -------------------------------------------------------------------------

        cv2.rectangle(
            annotated,

            (
                center_x - text_w // 2 - padding,
                center_y - text_h - padding
            ),

            (
                center_x + text_w // 2 + padding,
                center_y + baseline + padding
            ),

            (0, 0, 0),

            -1
        )


        # -------------------------------------------------------------------------
        # NUMBER
        # -------------------------------------------------------------------------

        cv2.putText(
            annotated,

            number,

            (
                center_x - text_w // 2,
                center_y
            ),

            font,

            font_scale,

            (255, 255, 255),

            thickness,

            cv2.LINE_AA
        )


        # =========================================================================
        # VIEW LABEL
        # =========================================================================

        label = (
            f"{view_class} "
            f"| {confidence:.2f}"
        )

        (
            label_w,
            label_h
        ), label_base = cv2.getTextSize(
            label,
            font,
            0.75,
            2
        )

        label_x = x1

        label_y = max(
            40,
            y1 - 15
        )


        cv2.rectangle(
            annotated,

            (
                label_x,
                label_y - label_h - 10
            ),

            (
                label_x + label_w + 12,
                label_y + 5
            ),

            (0, 0, 0),

            -1
        )


        cv2.putText(
            annotated,

            label,

            (
                label_x + 6,
                label_y
            ),

            font,

            0.75,

            (255, 255, 255),

            2,

            cv2.LINE_AA
        )


    # =========================================================================
    # SAVE
    # =========================================================================

    output_path = (
        OUTPUT_DIR /
        f"{image_path.stem}_view_angle.jpg"
    )

    cv2.imwrite(
        str(output_path),
        annotated
    )

    print()
    print(
        f"[SAVED] {output_path}"
    )


# =============================================================================
# SUMMARY
# =============================================================================

print()
print("=" * 80)
print("VIEW ANGLE IMAGE TEST COMPLETE")
print("=" * 80)

print()

print(
    f"Images processed        : {total_images}"
)

print(
    f"Images with cows        : {images_with_cows}"
)

print(
    f"Total cows detected     : {total_cows}"
)

print(
    f"Good side views         : {side_good}"
)

print(
    f"Acceptable side views   : {side_acceptable}"
)

print(
    f"Invalid/other views     : {invalid_views}"
)

print(
    f"Analysis failures       : {analysis_failures}"
)

print()

print(
    f"Annotated output        : {OUTPUT_DIR}"
)

print()
print("=" * 80)