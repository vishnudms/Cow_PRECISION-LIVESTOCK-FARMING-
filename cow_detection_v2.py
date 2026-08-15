"""
===============================================================================
COW DETECTION V2
===============================================================================

YOLO26m-seg
+ Segmentation
+ Tracking
+ Mask cleanup
+ Duplicate filtering
+ Partial-cow filtering
+ Occlusion awareness
+ Stable visual detection

GREEN = CLEAN / USABLE COW
RED   = DETECTED BUT POOR / PARTIAL / OCCLUDED

This version focuses ONLY on improving detection quality.

NO morphometric measurement is performed here.

===============================================================================
"""

import cv2
import os
import numpy as np

from collections import deque
from ultralytics import YOLO


# =============================================================================
# CONFIGURATION
# =============================================================================

VIDEO_PATH = r"D:\cow\videos\cow_video10.mp4"

MODEL_PATH = r"D:\cow\models\yolo26m-seg.pt"

OUTPUT_DIR = r"D:\cow\output\cow_detection"

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "cow_detection_v2.mp4"
)


# =============================================================================
# YOLO CONFIGURATION
# =============================================================================

CONFIDENCE_THRESHOLD = 0.35

IOU_THRESHOLD = 0.50

IMAGE_SIZE = 960

COW_CLASS_ID = 19


# =============================================================================
# TRACKING
# =============================================================================

TRACKER_CONFIG = "bytetrack.yaml"

TRACK_PERSIST = True


# =============================================================================
# MASK QUALITY
# =============================================================================

MIN_MASK_AREA = 3000

MIN_COMPONENT_AREA = 500

MORPH_KERNEL_SIZE = 5


# =============================================================================
# PARTIAL COW DETECTION
# =============================================================================

EDGE_MARGIN = 8

MIN_VISIBLE_RATIO = 0.55


# =============================================================================
# OVERLAP
# =============================================================================

MAX_DUPLICATE_IOU = 0.80

MAX_HEAVY_OVERLAP = 0.65


# =============================================================================
# VISUALIZATION
# =============================================================================

SHOW_MASK = True

SHOW_BOX = True

SHOW_ID = True

SHOW_STATUS = True

SHOW_TRACK_TRAIL = True


# =============================================================================
# COLORS
# =============================================================================

GREEN = (0, 255, 0)

RED = (0, 0, 255)

YELLOW = (0, 255, 255)

WHITE = (255, 255, 255)

BLACK = (0, 0, 0)


# =============================================================================
# OUTPUT
# =============================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =============================================================================
# TRACK TRAILS
# =============================================================================

track_history = {}

MAX_TRAIL_LENGTH = 30


# =============================================================================
# LOAD MODEL
# =============================================================================

print()
print("=" * 80)
print("COW DETECTION V2")
print("=" * 80)
print()

print("[INFO] Loading YOLO26 segmentation model...")
print(f"[INFO] Model: {MODEL_PATH}")

model = YOLO(
    MODEL_PATH
)

print("[OK] YOLO26 loaded successfully.")
print()


# =============================================================================
# MASK CLEANING
# =============================================================================

def clean_mask(mask):
    """
    Clean segmentation mask.

    Operations:
        1. Morphological opening
        2. Morphological closing
        3. Remove tiny disconnected components
    """

    mask = (
        mask > 0
    ).astype(
        np.uint8
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            MORPH_KERNEL_SIZE,
            MORPH_KERNEL_SIZE
        )
    )

    # Remove small noise
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Fill small holes
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    cleaned = np.zeros_like(
        mask
    )

    for label in range(
        1,
        num_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= MIN_COMPONENT_AREA:

            cleaned[
                labels == label
            ] = 1

    return cleaned


# =============================================================================
# BOUNDING BOX
# =============================================================================

def mask_bbox(mask):

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:

        return None

    x1 = int(
        np.min(xs)
    )

    y1 = int(
        np.min(ys)
    )

    x2 = int(
        np.max(xs)
    )

    y2 = int(
        np.max(ys)
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


# =============================================================================
# BOX AREA
# =============================================================================

def bbox_area(box):

    x1, y1, x2, y2 = box

    return max(
        0,
        x2 - x1
    ) * max(
        0,
        y2 - y1
    )


# =============================================================================
# MASK AREA
# =============================================================================

def mask_area(mask):

    return int(
        np.sum(mask)
    )


# =============================================================================
# BOX IOU
# =============================================================================

def calculate_iou(
    box_a,
    box_b
):

    ax1, ay1, ax2, ay2 = box_a

    bx1, by1, bx2, by2 = box_b

    ix1 = max(
        ax1,
        bx1
    )

    iy1 = max(
        ay1,
        by1
    )

    ix2 = min(
        ax2,
        bx2
    )

    iy2 = min(
        ay2,
        by2
    )

    iw = max(
        0,
        ix2 - ix1
    )

    ih = max(
        0,
        iy2 - iy1
    )

    intersection = (
        iw * ih
    )

    if intersection <= 0:

        return 0.0

    area_a = bbox_area(
        box_a
    )

    area_b = bbox_area(
        box_b
    )

    union = (
        area_a +
        area_b -
        intersection
    )

    if union <= 0:

        return 0.0

    return (
        intersection /
        union
    )


# =============================================================================
# MASK OVERLAP
# =============================================================================

def calculate_mask_overlap(
    mask_a,
    mask_b
):

    intersection = np.logical_and(
        mask_a > 0,
        mask_b > 0
    ).sum()

    area_a = np.sum(
        mask_a > 0
    )

    area_b = np.sum(
        mask_b > 0
    )

    smaller_area = min(
        area_a,
        area_b
    )

    if smaller_area <= 0:

        return 0.0

    return float(
        intersection /
        smaller_area
    )


# =============================================================================
# EDGE TOUCH TEST
# =============================================================================

def touches_edge(
    mask,
    width,
    height
):

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:

        return False

    left = np.min(xs)

    right = np.max(xs)

    top = np.min(ys)

    bottom = np.max(ys)

    return (
        left <= EDGE_MARGIN
        or
        right >= width - EDGE_MARGIN
        or
        top <= EDGE_MARGIN
        or
        bottom >= height - EDGE_MARGIN
    )


# =============================================================================
# MASK FILL RATIO
# =============================================================================

def mask_fill_ratio(
    mask,
    box
):

    area = mask_area(
        mask
    )

    box_area_value = bbox_area(
        box
    )

    if box_area_value <= 0:

        return 0.0

    return float(
        area /
        box_area_value
    )


# =============================================================================
# TRACK CENTER
# =============================================================================

def get_center(
    box
):

    x1, y1, x2, y2 = box

    return (
        int(
            (x1 + x2) / 2
        ),
        int(
            (y1 + y2) / 2
        )
    )


# =============================================================================
# DRAW TEXT
# =============================================================================

def draw_text(
    frame,
    text,
    position,
    color,
    scale=0.60
):

    x, y = position

    cv2.putText(
        frame,
        text,
        (
            x,
            y
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        BLACK,
        5,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        text,
        (
            x,
            y
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA
    )


# =============================================================================
# DRAW GLOW
# =============================================================================

def draw_glow(
    frame,
    mask,
    color
):

    # -------------------------------------------------------------------------
    # Soft glow
    # -------------------------------------------------------------------------

    glow = np.zeros_like(
        frame
    )

    glow[
        mask > 0
    ] = color

    blurred = cv2.GaussianBlur(
        glow,
        (
            0,
            0
        ),
        18
    )

    frame[:] = cv2.addWeighted(
        frame,
        1.0,
        blurred,
        0.45,
        0
    )


    # -------------------------------------------------------------------------
    # Transparent mask
    # -------------------------------------------------------------------------

    overlay = frame.copy()

    overlay[
        mask > 0
    ] = color

    frame[:] = cv2.addWeighted(
        frame,
        0.78,
        overlay,
        0.22,
        0
    )


    # -------------------------------------------------------------------------
    # Contour
    # -------------------------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        frame,
        contours,
        -1,
        color,
        4,
        cv2.LINE_AA
    )


# =============================================================================
# DRAW TRAIL
# =============================================================================

def draw_trail(
    frame,
    cow_id
):

    if cow_id not in track_history:

        return

    points = track_history[
        cow_id
    ]

    if len(points) < 2:

        return

    for i in range(
        1,
        len(points)
    ):

        cv2.line(
            frame,
            points[i - 1],
            points[i],
            YELLOW,
            2,
            cv2.LINE_AA
        )


# =============================================================================
# OPEN VIDEO
# =============================================================================

print("[INFO] Opening video...")

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    raise RuntimeError(
        f"Could not open video:\n{VIDEO_PATH}"
    )


# =============================================================================
# VIDEO INFORMATION
# =============================================================================

fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:

    fps = 30.0


width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


print(
    f"[INFO] Resolution: "
    f"{width} x {height}"
)

print(
    f"[INFO] FPS: {fps:.2f}"
)

print(
    f"[INFO] Total frames: "
    f"{total_frames}"
)

print(
    f"[INFO] Inference size: "
    f"{IMAGE_SIZE}"
)

print(
    f"[INFO] Confidence: "
    f"{CONFIDENCE_THRESHOLD}"
)

print()


# =============================================================================
# VIDEO WRITER
# =============================================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (
        width,
        height
    )
)

if not writer.isOpened():

    raise RuntimeError(
        "Could not create output video."
    )


# =============================================================================
# FRAME LOOP
# =============================================================================

frame_number = 0


while True:

    ret, frame = cap.read()

    if not ret:

        break

    frame_number += 1

    display = frame.copy()


    # =========================================================================
    # YOLO26 TRACKING
    # =========================================================================

    results = model.track(
        frame,
        persist=TRACK_PERSIST,
        tracker=TRACKER_CONFIG,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
        classes=[COW_CLASS_ID],
        verbose=False
    )


    # =========================================================================
    # NO DETECTION
    # =========================================================================

    if (
        not results
        or
        results[0].boxes is None
        or
        results[0].masks is None
    ):

        draw_text(
            display,
            "NO COW DETECTED",
            (
                20,
                40
            ),
            RED
        )

        writer.write(
            display
        )

        cv2.imshow(
            "Cow Detection V2",
            display
        )

        key = (
            cv2.waitKey(1)
            &
            0xFF
        )

        if key == ord("q"):

            break

        continue


    result = results[0]


    # =========================================================================
    # RAW MASKS
    # =========================================================================

    raw_masks = (
        result.masks.data
        .cpu()
        .numpy()
    )


    # =========================================================================
    # TRACK IDS
    # =========================================================================

    if result.boxes.id is not None:

        track_ids = (
            result.boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )

    else:

        track_ids = np.arange(
            len(raw_masks)
        )


    # =========================================================================
    # CONFIDENCE
    # =========================================================================

    confidences = (
        result.boxes.conf
        .cpu()
        .numpy()
    )


    # =========================================================================
    # CANDIDATE DETECTIONS
    # =========================================================================

    candidates = []


    for index, raw_mask in enumerate(
        raw_masks
    ):

        # ---------------------------------------------------------------------
        # RESIZE
        # ---------------------------------------------------------------------

        mask = cv2.resize(
            raw_mask,
            (
                width,
                height
            ),
            interpolation=cv2.INTER_NEAREST
        )

        mask = (
            mask > 0.5
        ).astype(
            np.uint8
        )


        # ---------------------------------------------------------------------
        # CLEAN
        # ---------------------------------------------------------------------

        mask = clean_mask(
            mask
        )


        # ---------------------------------------------------------------------
        # AREA
        # ---------------------------------------------------------------------

        area = mask_area(
            mask
        )

        if area < MIN_MASK_AREA:

            continue


        # ---------------------------------------------------------------------
        # BBOX
        # ---------------------------------------------------------------------

        box = mask_bbox(
            mask
        )

        if box is None:

            continue


        # ---------------------------------------------------------------------
        # CONFIDENCE
        # ---------------------------------------------------------------------

        confidence = float(
            confidences[index]
        )


        # ---------------------------------------------------------------------
        # TRACK ID
        # ---------------------------------------------------------------------

        cow_id = int(
            track_ids[index]
        )


        # ---------------------------------------------------------------------
        # EDGE
        # ---------------------------------------------------------------------

        edge_touch = touches_edge(
            mask,
            width,
            height
        )


        # ---------------------------------------------------------------------
        # FILL RATIO
        # ---------------------------------------------------------------------

        fill_ratio = mask_fill_ratio(
            mask,
            box
        )


        # ---------------------------------------------------------------------
        # STORE
        # ---------------------------------------------------------------------

        candidates.append(
            {
                "mask": mask,
                "box": box,
                "area": area,
                "confidence": confidence,
                "cow_id": cow_id,
                "edge_touch": edge_touch,
                "fill_ratio": fill_ratio
            }
        )


    # =========================================================================
    # DUPLICATE FILTER
    # =========================================================================

    keep = [
        True
        for _ in candidates
    ]


    for i in range(
        len(candidates)
    ):

        if not keep[i]:

            continue

        for j in range(
            i + 1,
            len(candidates)
        ):

            if not keep[j]:

                continue


            box_i = candidates[i]["box"]

            box_j = candidates[j]["box"]


            iou = calculate_iou(
                box_i,
                box_j
            )


            mask_overlap = calculate_mask_overlap(
                candidates[i]["mask"],
                candidates[j]["mask"]
            )


            # -----------------------------------------------------------------
            # Strong duplicate
            # -----------------------------------------------------------------

            if (
                iou >= MAX_DUPLICATE_IOU
                or
                mask_overlap >= MAX_HEAVY_OVERLAP
            ):

                confidence_i = candidates[i][
                    "confidence"
                ]

                confidence_j = candidates[j][
                    "confidence"
                ]


                if confidence_i >= confidence_j:

                    keep[j] = False

                else:

                    keep[i] = False

                    break


    candidates = [
        candidate
        for index, candidate in enumerate(
            candidates
        )
        if keep[index]
    ]


    # =========================================================================
    # PROCESS COWS
    # =========================================================================

    for candidate in candidates:

        mask = candidate[
            "mask"
        ]

        box = candidate[
            "box"
        ]

        cow_id = candidate[
            "cow_id"
        ]

        confidence = candidate[
            "confidence"
        ]

        edge_touch = candidate[
            "edge_touch"
        ]

        fill_ratio = candidate[
            "fill_ratio"
        ]


        # ---------------------------------------------------------------------
        # DETERMINE QUALITY
        # ---------------------------------------------------------------------

        quality_good = True


        # Very low segmentation fill
        if fill_ratio < 0.18:

            quality_good = False


        # Cow touching edge may be partially visible.
        if edge_touch:

            quality_good = False


        # ---------------------------------------------------------------------
        # COLOR
        # ---------------------------------------------------------------------

        if quality_good:

            color = GREEN

            status = "GOOD"

        else:

            color = RED

            status = "PARTIAL"


        # ---------------------------------------------------------------------
        # GLOW
        # ---------------------------------------------------------------------

        if SHOW_MASK:

            draw_glow(
                display,
                mask,
                color
            )


        # ---------------------------------------------------------------------
        # BOUNDING BOX
        # ---------------------------------------------------------------------

        if SHOW_BOX:

            x1, y1, x2, y2 = box

            cv2.rectangle(
                display,
                (
                    x1,
                    y1
                ),
                (
                    x2,
                    y2
                ),
                color,
                2
            )


        # ---------------------------------------------------------------------
        # CENTER
        # ---------------------------------------------------------------------

        center = get_center(
            box
        )


        # ---------------------------------------------------------------------
        # TRACK HISTORY
        # ---------------------------------------------------------------------

        if cow_id not in track_history:

            track_history[
                cow_id
            ] = deque(
                maxlen=MAX_TRAIL_LENGTH
            )


        track_history[
            cow_id
        ].append(
            center
        )


        if SHOW_TRACK_TRAIL:

            draw_trail(
                display,
                cow_id
            )


        # ---------------------------------------------------------------------
        # ID
        # ---------------------------------------------------------------------

        if SHOW_ID:

            draw_text(
                display,
                f"COW {cow_id}",
                (
                    x1,
                    max(
                        30,
                        y1 - 10
                    )
                ),
                color,
                0.65
            )


        # ---------------------------------------------------------------------
        # STATUS
        # ---------------------------------------------------------------------

        if SHOW_STATUS:

            draw_text(
                display,
                status,
                (
                    x1,
                    min(
                        height - 10,
                        y2 + 25
                    )
                ),
                color,
                0.50
            )


        # ---------------------------------------------------------------------
        # CONFIDENCE
        # ---------------------------------------------------------------------

        draw_text(
            display,
            f"{confidence:.2f}",
            (
                x2 - 55,
                max(
                    25,
                    y1 + 20
                )
            ),
            WHITE,
            0.45
        )


    # =========================================================================
    # GLOBAL INFORMATION
    # =========================================================================

    draw_text(
        display,
        "YOLO26 COW DETECTION V2",
        (
            20,
            35
        ),
        WHITE,
        0.70
    )


    draw_text(
        display,
        f"FRAME {frame_number}/{total_frames}",
        (
            width - 190,
            35
        ),
        WHITE,
        0.50
    )


    # =========================================================================
    # LEGEND
    # =========================================================================

    cv2.circle(
        display,
        (
            25,
            height - 55
        ),
        8,
        GREEN,
        -1
    )

    draw_text(
        display,
        "GOOD",
        (
            42,
            height - 48
        ),
        GREEN,
        0.50
    )


    cv2.circle(
        display,
        (
            115,
            height - 55
        ),
        8,
        RED,
        -1
    )

    draw_text(
        display,
        "PARTIAL",
        (
            132,
            height - 48
        ),
        RED,
        0.50
    )


    # =========================================================================
    # WRITE
    # =========================================================================

    writer.write(
        display
    )


    # =========================================================================
    # SHOW
    # =========================================================================

    cv2.imshow(
        "Cow Detection V2",
        display
    )


    key = (
        cv2.waitKey(1)
        &
        0xFF
    )


    if key == ord("q"):

        break


# =============================================================================
# CLEANUP
# =============================================================================

cap.release()

writer.release()

cv2.destroyAllWindows()


# =============================================================================
# COMPLETE
# =============================================================================

print()
print("=" * 80)
print("DETECTION V2 COMPLETE")
print("=" * 80)
print()

print(
    f"Output video:"
)

print(
    OUTPUT_VIDEO
)

print()