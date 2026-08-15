"""
===============================================================================
COW MORPHOMETRIC MEASUREMENT V3
===============================================================================

YOLO26 SEGMENTATION + TRACKING

GREEN:
    Valid measurement view.
    Cow glows green.
    Length measurement is enabled.

RED:
    Invalid measurement view.
    Cow glows red.
    No measurement line.

Only green cows are measured.

Measurement currently = image-space body-axis length in pixels.

===============================================================================
"""

import cv2
import math
import os
import csv
import numpy as np

from collections import defaultdict, deque

from ultralytics import YOLO


# =============================================================================
# CONFIGURATION
# =============================================================================

VIDEO_PATH = r"D:\cow\videos\cow_video6.mp4"

MODEL_PATH = r"D:\cow\models\yolo26m-seg.pt"

OUTPUT_DIR = r"D:\cow\output\cow_length"

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "cow_length_measurement_v6_1.mp4"
)

CSV_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "cow_length_measurements_v3.csv"
)


# =============================================================================
# YOLO
# =============================================================================

CONFIDENCE_THRESHOLD = 0.40

IOU_THRESHOLD = 0.50

COW_CLASS_ID = 19


# =============================================================================
# VIEW VALIDATION
# =============================================================================

MAX_HORIZONTAL_ANGLE = 15.0

MIN_BODY_ASPECT_RATIO = 1.80


# =============================================================================
# MASK
# =============================================================================

MIN_MASK_AREA = 2500


# =============================================================================
# MEASUREMENT
# =============================================================================

SMOOTHING_WINDOW = 15

MIN_VALID_MEASUREMENTS = 5

MAX_MEASUREMENT_VARIATION = 0.08


# =============================================================================
# VISUALIZATION
# =============================================================================

SHOW_MEASUREMENT_LINE = True

SHOW_ENDPOINTS = True

SHOW_COW_ID = True


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
# LOAD MODEL
# =============================================================================

print()
print("=" * 80)
print("COW MORPHOMETRIC MEASUREMENT V3")
print("=" * 80)
print()

print("[INFO] Loading YOLO26...")
print(MODEL_PATH)

model = YOLO(
    MODEL_PATH
)

print("[OK] YOLO26 loaded.")
print()


# =============================================================================
# HISTORY
# =============================================================================

measurement_history = defaultdict(
    lambda: deque(
        maxlen=SMOOTHING_WINDOW
    )
)

best_measurement = {}

best_measurement_frame = {}

valid_frame_count = defaultdict(int)

invalid_frame_count = defaultdict(int)


# =============================================================================
# ANGLE
# =============================================================================

def normalize_angle(angle):

    while angle > 90:
        angle -= 180

    while angle < -90:
        angle += 180

    return angle


# =============================================================================
# BODY AXIS
# =============================================================================

def calculate_body_axis(mask):

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) < 20:

        return (
            None,
            None,
            None,
            None
        )

    points = np.column_stack(
        (
            xs.astype(np.float32),
            ys.astype(np.float32)
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

    dominant = np.argmax(
        eigenvalues
    )

    direction = eigenvectors[
        :,
        dominant
    ]

    dx = float(
        direction[0]
    )

    dy = float(
        direction[1]
    )

    angle = math.degrees(
        math.atan2(
            dy,
            dx
        )
    )

    angle = normalize_angle(
        angle
    )

    return (
        center,
        direction,
        angle,
        points
    )


# =============================================================================
# BODY ASPECT RATIO
# =============================================================================

def calculate_body_aspect_ratio(
    points,
    direction
):

    if points is None:

        return None

    direction = np.asarray(
        direction,
        dtype=np.float32
    )

    norm = np.linalg.norm(
        direction
    )

    if norm < 1e-6:

        return None

    direction /= norm

    perpendicular = np.array(
        [
            -direction[1],
            direction[0]
        ],
        dtype=np.float32
    )

    center = np.mean(
        points,
        axis=0
    )

    centered = (
        points -
        center
    )

    body_projection = (
        centered @ direction
    )

    width_projection = (
        centered @ perpendicular
    )

    body_length = (
        np.max(body_projection)
        -
        np.min(body_projection)
    )

    body_width = (
        np.max(width_projection)
        -
        np.min(width_projection)
    )

    if body_width <= 1:

        return None

    return float(
        body_length /
        body_width
    )


# =============================================================================
# VIEW VALIDATION
# =============================================================================

def is_valid_view(
    angle,
    aspect_ratio
):

    if angle is None:
        return False

    if aspect_ratio is None:
        return False

    angle_ok = (
        abs(angle)
        <= MAX_HORIZONTAL_ANGLE
    )

    shape_ok = (
        aspect_ratio
        >= MIN_BODY_ASPECT_RATIO
    )

    return (
        angle_ok
        and
        shape_ok
    )


# =============================================================================
# BODY LENGTH
# =============================================================================

def calculate_axis_length(
    points,
    center,
    direction
):

    direction = np.asarray(
        direction,
        dtype=np.float32
    )

    norm = np.linalg.norm(
        direction
    )

    if norm < 1e-6:

        return (
            None,
            None,
            None
        )

    direction /= norm

    centered = (
        points -
        center
    )

    projection = (
        centered @ direction
    )

    min_index = np.argmin(
        projection
    )

    max_index = np.argmax(
        projection
    )

    start_point = points[
        min_index
    ]

    end_point = points[
        max_index
    ]

    length = float(
        projection[max_index]
        -
        projection[min_index]
    )

    return (
        length,
        start_point,
        end_point
    )


# =============================================================================
# SMOOTH
# =============================================================================

def smooth_measurement(
    cow_id,
    value
):

    history = measurement_history[
        cow_id
    ]

    history.append(
        value
    )

    return float(
        np.median(
            np.asarray(
                history,
                dtype=np.float32
            )
        )
    )


# =============================================================================
# STABILITY
# =============================================================================

def measurement_is_stable(
    cow_id
):

    history = measurement_history[
        cow_id
    ]

    if len(history) < MIN_VALID_MEASUREMENTS:

        return False

    values = np.asarray(
        history,
        dtype=np.float32
    )

    median_value = np.median(
        values
    )

    if median_value <= 0:

        return False

    variation = (
        np.max(
            np.abs(
                values -
                median_value
            )
        )
        /
        median_value
    )

    return (
        variation
        <= MAX_MEASUREMENT_VARIATION
    )


# =============================================================================
# UPDATE BEST
# =============================================================================

def update_best(
    cow_id,
    frame_number
):

    if not measurement_is_stable(
        cow_id
    ):

        return

    values = np.asarray(
        measurement_history[
            cow_id
        ],
        dtype=np.float32
    )

    value = float(
        np.median(values)
    )

    if cow_id not in best_measurement:

        best_measurement[
            cow_id
        ] = value

        best_measurement_frame[
            cow_id
        ] = frame_number

        return

    old = best_measurement[
        cow_id
    ]

    difference = (
        abs(
            value - old
        )
        /
        max(
            old,
            1.0
        )
    )

    if difference <= 0.05:

        best_measurement[
            cow_id
        ] = value

        best_measurement_frame[
            cow_id
        ] = frame_number


# =============================================================================
# GLOW EFFECT
# =============================================================================

def apply_cow_glow(
    frame,
    mask,
    color,
    strength=0.45
):

    glow = np.zeros_like(
        frame
    )

    glow[
        mask > 0
    ] = color

    # Blur creates a soft glow around the cow.
    blurred = cv2.GaussianBlur(
        glow,
        (
            0,
            0
        ),
        sigmaX=15,
        sigmaY=15
    )

    # Strong colored interior.
    interior = frame.copy()

    interior[
        mask > 0
    ] = color

    frame[:] = cv2.addWeighted(
        frame,
        1.0,
        blurred,
        strength,
        0
    )

    frame[:] = cv2.addWeighted(
        frame,
        1.0 - strength * 0.35,
        interior,
        strength * 0.35,
        0
    )

    # Clean outline around cow.
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
# ID LABEL
# =============================================================================

def draw_cow_id(
    frame,
    cow_id,
    center,
    color
):

    x = max(
        10,
        int(center[0] - 50)
    )

    y = max(
        30,
        int(center[1] - 20)
    )

    text = f"COW {cow_id}"

    cv2.putText(
        frame,
        text,
        (
            x,
            y
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
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
        0.65,
        color,
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
    f"[INFO] Frames: {total_frames}"
)

print()


# =============================================================================
# WRITER
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
# MAIN LOOP
# =============================================================================

frame_number = 0


while True:

    ret, frame = cap.read()

    if not ret:

        break

    frame_number += 1

    display = frame.copy()


    # =========================================================================
    # YOLO26
    # =========================================================================

    results = model.track(
        frame,
        persist=True,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        classes=[COW_CLASS_ID],
        verbose=False
    )


    if (
        not results
        or
        results[0].boxes is None
        or
        results[0].masks is None
    ):

        writer.write(
            display
        )

        cv2.imshow(
            "Cow Measurement V3",
            display
        )

        if (
            cv2.waitKey(1)
            &
            0xFF
        ) == ord("q"):

            break

        continue


    result = results[0]


    # =========================================================================
    # MASKS
    # =========================================================================

    masks = (
        result.masks.data
        .cpu()
        .numpy()
    )


    # =========================================================================
    # IDS
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
            len(masks)
        )


    # =========================================================================
    # EACH COW
    # =========================================================================

    for index, mask_tensor in enumerate(
        masks
    ):

        # ---------------------------------------------------------------------
        # RESIZE MASK
        # ---------------------------------------------------------------------

        mask = cv2.resize(
            mask_tensor,
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
        # AREA
        # ---------------------------------------------------------------------

        if (
            np.sum(mask)
            <
            MIN_MASK_AREA
        ):

            continue


        # ---------------------------------------------------------------------
        # ID
        # ---------------------------------------------------------------------

        cow_id = int(
            track_ids[index]
        )


        # ---------------------------------------------------------------------
        # BODY AXIS
        # ---------------------------------------------------------------------

        (
            center,
            direction,
            angle,
            points
        ) = calculate_body_axis(
            mask
        )


        if center is None:

            continue


        # ---------------------------------------------------------------------
        # BODY SHAPE
        # ---------------------------------------------------------------------

        aspect_ratio = calculate_body_aspect_ratio(
            points,
            direction
        )


        # ---------------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------------

        valid = is_valid_view(
            angle,
            aspect_ratio
        )


        # =====================================================================
        # GREEN COW
        # =====================================================================

        if valid:

            valid_frame_count[
                cow_id
            ] += 1


            # -----------------------------------------------------------------
            # GREEN GLOW
            # -----------------------------------------------------------------

            apply_cow_glow(
                display,
                mask,
                GREEN,
                strength=0.50
            )


            # -----------------------------------------------------------------
            # LENGTH
            # -----------------------------------------------------------------

            (
                length_px,
                start_point,
                end_point
            ) = calculate_axis_length(
                points,
                center,
                direction
            )


            if length_px is None:

                continue


            # -----------------------------------------------------------------
            # SMOOTH
            # -----------------------------------------------------------------

            smooth_length = smooth_measurement(
                cow_id,
                length_px
            )


            # -----------------------------------------------------------------
            # BEST
            # -----------------------------------------------------------------

            update_best(
                cow_id,
                frame_number
            )


            # -----------------------------------------------------------------
            # MEASUREMENT LINE
            # -----------------------------------------------------------------

            if SHOW_MEASUREMENT_LINE:

                start = (
                    int(start_point[0]),
                    int(start_point[1])
                )

                end = (
                    int(end_point[0]),
                    int(end_point[1])
                )

                cv2.line(
                    display,
                    start,
                    end,
                    YELLOW,
                    4,
                    cv2.LINE_AA
                )


                if SHOW_ENDPOINTS:

                    cv2.circle(
                        display,
                        start,
                        8,
                        YELLOW,
                        -1
                    )

                    cv2.circle(
                        display,
                        end,
                        8,
                        YELLOW,
                        -1
                    )


            # -----------------------------------------------------------------
            # ID
            # -----------------------------------------------------------------

            if SHOW_COW_ID:

                draw_cow_id(
                    display,
                    cow_id,
                    center,
                    GREEN
                )


        # =====================================================================
        # RED COW
        # =====================================================================

        else:

            invalid_frame_count[
                cow_id
            ] += 1


            # -----------------------------------------------------------------
            # CLEAR OLD MEASUREMENT HISTORY
            # -----------------------------------------------------------------

            measurement_history[
                cow_id
            ].clear()


            # -----------------------------------------------------------------
            # RED GLOW
            # -----------------------------------------------------------------

            apply_cow_glow(
                display,
                mask,
                RED,
                strength=0.50
            )


            # -----------------------------------------------------------------
            # NO AXIS
            # NO MEASUREMENT LINE
            # NO ENDPOINTS
            # -----------------------------------------------------------------


            # -----------------------------------------------------------------
            # ID
            # -----------------------------------------------------------------

            if SHOW_COW_ID:

                draw_cow_id(
                    display,
                    cow_id,
                    center,
                    RED
                )


    # =========================================================================
    # MINIMAL GLOBAL STATUS
    # =========================================================================

    # Only a very small status indicator.
    # No angle/body-ratio/weight/BCS wording.

    cv2.putText(
        display,
        "GREEN = MEASURED",
        (
            20,
            height - 45
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        BLACK,
        4,
        cv2.LINE_AA
    )

    cv2.putText(
        display,
        "GREEN = MEASURED",
        (
            20,
            height - 45
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        GREEN,
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        display,
        "RED = NOT MEASURED",
        (
            20,
            height - 18
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        BLACK,
        4,
        cv2.LINE_AA
    )

    cv2.putText(
        display,
        "RED = NOT MEASURED",
        (
            20,
            height - 18
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        RED,
        2,
        cv2.LINE_AA
    )


    # =========================================================================
    # FRAME
    # =========================================================================

    cv2.putText(
        display,
        f"{frame_number}/{total_frames}",
        (
            width - 130,
            35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        WHITE,
        2,
        cv2.LINE_AA
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
        "Cow Measurement V3",
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
# CSV
# =============================================================================

print()
print("=" * 80)
print("FINAL RESULTS")
print("=" * 80)
print()


with open(
    CSV_OUTPUT,
    "w",
    newline=""
) as file:

    csv_writer = csv.writer(
        file
    )

    csv_writer.writerow(
        [
            "cow_id",
            "best_length_pixels",
            "best_frame",
            "valid_frames",
            "invalid_frames",
            "length_cm",
            "weight_kg",
            "bcs"
        ]
    )


    all_ids = sorted(
        set(
            list(
                valid_frame_count.keys()
            )
            +
            list(
                invalid_frame_count.keys()
            )
        )
    )


    for cow_id in all_ids:

        length = best_measurement.get(
            cow_id,
            ""
        )

        best_frame = best_measurement_frame.get(
            cow_id,
            ""
        )

        valid_frames = valid_frame_count.get(
            cow_id,
            0
        )

        invalid_frames = invalid_frame_count.get(
            cow_id,
            0
        )


        if length != "":

            length_text = (
                f"{length:.2f}"
            )

        else:

            length_text = ""


        csv_writer.writerow(
            [
                cow_id,
                length_text,
                best_frame,
                valid_frames,
                invalid_frames,
                "",
                "",
                ""
            ]
        )


        print(
            f"COW {cow_id} | "
            f"BEST LENGTH = "
            f"{length_text} px | "
            f"VALID = {valid_frames} | "
            f"INVALID = {invalid_frames}"
        )


print()
print("=" * 80)
print("DONE")
print("=" * 80)
print()

print(
    f"Video: {OUTPUT_VIDEO}"
)

print(
    f"CSV:   {CSV_OUTPUT}"
)

print()