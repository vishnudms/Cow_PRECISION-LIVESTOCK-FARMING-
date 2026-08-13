import cv2
import os
import time
import numpy as np
from ultralytics import YOLO


# ============================================================
# TEST 5
# DIRECT TOP-ANGLE GOAT DETECTION
#
# IMPORTANT:
# COCO:
#   18 = sheep
#   19 = cow
#
# COCO has NO goat class.
#
# Therefore this test checks whether the COCO sheep/cow
# representations can detect the goats in the video.
#
# NO TRACKING
# NO COUNTING
# NO LINE CROSSING
#
# FIRST SOLVE DETECTION.
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = "videos/cow_video1.mp4"

MODEL_PATH = "yolo11m.pt"

OUTPUT_PATH = "output/test5_goat_top_detection.mp4"


# ============================================================
# GPU
# ============================================================

DEVICE = 0


# ============================================================
# DETECTION
# ============================================================

# Lower confidence because top-angle animals are difficult.

CONFIDENCE = 0.08

# Larger inference resolution.

IMG_SIZE = 1280

# NMS IoU.

IOU = 0.45


# ============================================================
# COCO CLASSES
# ============================================================

SHEEP_CLASS_ID = 18

COW_CLASS_ID = 19

TARGET_CLASSES = [
    SHEEP_CLASS_ID,
    COW_CLASS_ID
]


# ============================================================
# TILING
# ============================================================

# Tiling is important for your top-angle group.

USE_TILING = True

TILE_SIZE = 960

TILE_OVERLAP = 0.25


# ============================================================
# DISPLAY
# ============================================================

DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = 800


# ============================================================
# OUTPUT
# ============================================================

os.makedirs("output", exist_ok=True)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)


if not cap.isOpened():

    print()
    print("=" * 70)
    print("ERROR: Cannot open video")
    print("=" * 70)
    print(VIDEO_PATH)
    print()

    raise SystemExit


video_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

video_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

video_fps = cap.get(
    cv2.CAP_PROP_FPS
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


if video_fps <= 0:

    video_fps = 30.0


# ============================================================
# INFORMATION
# ============================================================

print()
print("=" * 70)
print(" TEST 5 - DIRECT TOP-ANGLE GOAT DETECTION")
print("=" * 70)

print(
    f"Video       : {VIDEO_PATH}"
)

print(
    f"Resolution  : {video_width} x {video_height}"
)

print(
    f"FPS         : {video_fps:.2f}"
)

print(
    f"Frames      : {total_frames}"
)

print(
    f"Model       : {MODEL_PATH}"
)

print(
    f"Image size  : {IMG_SIZE}"
)

print(
    f"Confidence  : {CONFIDENCE}"
)

print(
    f"IoU         : {IOU}"
)

print(
    f"Device      : GPU {DEVICE}"
)

print(
    f"Tiling      : {USE_TILING}"
)

print(
    f"Tile size   : {TILE_SIZE}"
)

print(
    f"Tile overlap: {TILE_OVERLAP}"
)

print()

print("COCO candidate classes:")

print("18 = sheep")

print("19 = cow")

print()

print("IMPORTANT:")

print("COCO does not contain a goat class.")

print(
    "This test uses sheep/cow representations as a first detection test."
)

print("=" * 70)


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


writer = cv2.VideoWriter(

    OUTPUT_PATH,

    fourcc,

    video_fps,

    (
        video_width,
        video_height
    )
)


if not writer.isOpened():

    print("ERROR: Could not create output video.")

    cap.release()

    raise SystemExit


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING YOLO11m")
print("=" * 70)

model = YOLO(
    MODEL_PATH
)

print()

print(
    f"Classes loaded: {len(model.names)}"
)

print(
    f"Class 18: {model.names.get(18, 'unknown')}"
)

print(
    f"Class 19: {model.names.get(19, 'unknown')}"
)

print()

print("MODEL READY")

print("=" * 70)
print()


# ============================================================
# TILE GENERATOR
# ============================================================

def generate_tiles(
    frame,
    tile_size,
    overlap
):

    h, w = frame.shape[:2]

    if (
        w <= tile_size
        and
        h <= tile_size
    ):

        return [
            (
                frame,
                0,
                0
            )
        ]


    step = int(
        tile_size
        *
        (1.0 - overlap)
    )

    if step <= 0:

        step = tile_size // 2


    tiles = []


    # --------------------------------------------------------
    # X positions
    # --------------------------------------------------------

    x_positions = list(
        range(
            0,
            max(
                1,
                w - tile_size + 1
            ),
            step
        )
    )


    # --------------------------------------------------------
    # Ensure right edge is covered
    # --------------------------------------------------------

    if len(x_positions) == 0:

        x_positions = [0]

    elif (
        x_positions[-1]
        + tile_size
        < w
    ):

        x_positions.append(
            w - tile_size
        )


    # --------------------------------------------------------
    # Y positions
    # --------------------------------------------------------

    y_positions = list(
        range(
            0,
            max(
                1,
                h - tile_size + 1
            ),
            step
        )
    )


    # --------------------------------------------------------
    # Ensure bottom edge is covered
    # --------------------------------------------------------

    if len(y_positions) == 0:

        y_positions = [0]

    elif (
        y_positions[-1]
        + tile_size
        < h
    ):

        y_positions.append(
            h - tile_size
        )


    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    for y in y_positions:

        for x in x_positions:

            x1 = max(
                0,
                x
            )

            y1 = max(
                0,
                y
            )

            x2 = min(
                w,
                x1 + tile_size
            )

            y2 = min(
                h,
                y1 + tile_size
            )


            tile = frame[
                y1:y2,
                x1:x2
            ]


            if (
                tile.shape[1] > 10
                and
                tile.shape[0] > 10
            ):

                tiles.append(
                    (
                        tile,
                        x1,
                        y1
                    )
                )


    return tiles


# ============================================================
# BOX IOU
# ============================================================

def box_iou(
    box_a,
    box_b
):

    ax1, ay1, ax2, ay2 = box_a

    bx1, by1, bx2, by2 = box_b


    inter_x1 = max(
        ax1,
        bx1
    )

    inter_y1 = max(
        ay1,
        by1
    )

    inter_x2 = min(
        ax2,
        bx2
    )

    inter_y2 = min(
        ay2,
        by2
    )


    inter_w = max(
        0,
        inter_x2 - inter_x1
    )

    inter_h = max(
        0,
        inter_y2 - inter_y1
    )


    intersection = (
        inter_w
        *
        inter_h
    )


    area_a = (
        max(
            0,
            ax2 - ax1
        )
        *
        max(
            0,
            ay2 - ay1
        )
    )


    area_b = (
        max(
            0,
            bx2 - bx1
        )
        *
        max(
            0,
            by2 - by1
        )
    )


    union = (
        area_a
        +
        area_b
        -
        intersection
    )


    if union <= 0:

        return 0.0


    return intersection / union


# ============================================================
# DUPLICATE REMOVAL
# ============================================================

def remove_duplicates(
    detections,
    iou_threshold=0.50
):

    if len(detections) == 0:

        return []


    # --------------------------------------------------------
    # Sort by confidence
    # --------------------------------------------------------

    detections = sorted(

        detections,

        key=lambda d: d["confidence"],

        reverse=True

    )


    kept = []


    # --------------------------------------------------------
    # NMS
    # --------------------------------------------------------

    for detection in detections:

        keep = True


        for existing in kept:

            # Only compare same class

            if (
                detection["class_id"]
                !=
                existing["class_id"]
            ):

                continue


            overlap = box_iou(

                detection["box"],

                existing["box"]

            )


            if (
                overlap
                >=
                iou_threshold
            ):

                keep = False

                break


        if keep:

            kept.append(
                detection
            )


    return kept


# ============================================================
# DETECTION FUNCTION
# ============================================================

def detect_frame(
    frame
):

    all_detections = []


    # ========================================================
    # NORMAL FULL FRAME
    # ========================================================

    results = model.predict(

        frame,

        classes=TARGET_CLASSES,

        conf=CONFIDENCE,

        iou=IOU,

        imgsz=IMG_SIZE,

        device=DEVICE,

        verbose=False

    )


    result = results[0]


    if (
        result.boxes is not None
        and
        len(result.boxes) > 0
    ):

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        classes = (
            result.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )


        for box, cls, confidence in zip(

            boxes,
            classes,
            confidences

        ):

            x1, y1, x2, y2 = box


            all_detections.append({

                "box": [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2)
                ],

                "class_id": int(cls),

                "confidence": float(
                    confidence
                )

            })


    # ========================================================
    # TILED DETECTION
    # ========================================================

    if USE_TILING:

        tiles = generate_tiles(

            frame,

            TILE_SIZE,

            TILE_OVERLAP

        )


        for tile, offset_x, offset_y in tiles:

            # -----------------------------------------------
            # Avoid running exactly same full-frame image
            # -----------------------------------------------

            if (
                tile.shape[1] == frame.shape[1]
                and
                tile.shape[0] == frame.shape[0]
            ):

                continue


            tile_results = model.predict(

                tile,

                classes=TARGET_CLASSES,

                conf=CONFIDENCE,

                iou=IOU,

                imgsz=IMG_SIZE,

                device=DEVICE,

                verbose=False

            )


            tile_result = tile_results[0]


            if (
                tile_result.boxes is None
                or
                len(tile_result.boxes) == 0
            ):

                continue


            boxes = (
                tile_result.boxes.xyxy
                .cpu()
                .numpy()
            )

            classes = (
                tile_result.boxes.cls
                .cpu()
                .numpy()
                .astype(int)
            )

            confidences = (
                tile_result.boxes.conf
                .cpu()
                .numpy()
            )


            for box, cls, confidence in zip(

                boxes,
                classes,
                confidences

            ):

                x1, y1, x2, y2 = box


                # -------------------------------------------
                # Convert tile coordinates to full frame
                # -------------------------------------------

                x1 += offset_x
                x2 += offset_x

                y1 += offset_y
                y2 += offset_y


                x1 = max(
                    0,
                    min(
                        frame.shape[1] - 1,
                        x1
                    )
                )

                y1 = max(
                    0,
                    min(
                        frame.shape[0] - 1,
                        y1
                    )
                )

                x2 = max(
                    0,
                    min(
                        frame.shape[1] - 1,
                        x2
                    )
                )

                y2 = max(
                    0,
                    min(
                        frame.shape[0] - 1,
                        y2
                    )
                )


                if (
                    x2 <= x1
                    or
                    y2 <= y1
                ):

                    continue


                all_detections.append({

                    "box": [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2)
                    ],

                    "class_id": int(cls),

                    "confidence": float(
                        confidence
                    )

                })


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    final_detections = remove_duplicates(

        all_detections,

        iou_threshold=0.50

    )


    return final_detections


# ============================================================
# FRAME LOOP
# ============================================================

frame_number = 0

total_detections = 0

fps_start = time.time()

fps_frames = 0

display_fps = 0.0


print()
print("=" * 70)
print("STARTING DETECTION")
print("=" * 70)
print()

print(
    "Press Q to stop."
)

print()


while True:

    success, frame = cap.read()


    if not success:

        break


    frame_number += 1


    # ========================================================
    # DETECT
    # ========================================================

    start_time = time.time()


    detections = detect_frame(
        frame
    )


    inference_time = (
        time.time()
        -
        start_time
    )


    # ========================================================
    # COUNTERS
    # ========================================================

    sheep_count = 0

    cow_count = 0


    # ========================================================
    # DRAW DETECTIONS
    # ========================================================

    for index, detection in enumerate(
        detections,
        1
    ):

        x1, y1, x2, y2 = (
            detection["box"]
        )

        class_id = (
            detection["class_id"]
        )

        confidence = (
            detection["confidence"]
        )


        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)


        # ----------------------------------------------------
        # CLASS
        # ----------------------------------------------------

        if class_id == SHEEP_CLASS_ID:

            class_name = "SHEEP"

            sheep_count += 1

            box_color = (
                0,
                255,
                0
            )

        elif class_id == COW_CLASS_ID:

            class_name = "COW"

            cow_count += 1

            box_color = (
                255,
                0,
                255
            )

        else:

            class_name = (
                model.names.get(
                    class_id,
                    "UNKNOWN"
                )
            )

            box_color = (
                0,
                255,
                255
            )


        # ----------------------------------------------------
        # BOX
        # ----------------------------------------------------

        cv2.rectangle(

            frame,

            (
                x1,
                y1
            ),

            (
                x2,
                y2
            ),

            box_color,

            3

        )


        # ----------------------------------------------------
        # CENTER
        # ----------------------------------------------------

        center_x = int(
            (x1 + x2) / 2
        )

        center_y = int(
            (y1 + y2) / 2
        )


        cv2.circle(

            frame,

            (
                center_x,
                center_y
            ),

            5,

            box_color,

            -1

        )


        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        label = (

            f"{class_name} "
            f"{confidence:.2f} "
            f"ID-{index}"

        )


        (tw, th), _ = cv2.getTextSize(

            label,

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            2

        )


        label_y = max(
            25,
            y1
        )


        cv2.rectangle(

            frame,

            (
                x1,
                label_y - th - 10
            ),

            (
                x1 + tw + 8,
                label_y
            ),

            box_color,

            -1

        )


        cv2.putText(

            frame,

            label,

            (
                x1 + 4,
                label_y - 5
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (
                0,
                0,
                0
            ),

            2,

            cv2.LINE_AA

        )


    # ========================================================
    # TOTAL
    # ========================================================

    total_detections = len(
        detections
    )


    # ========================================================
    # FPS
    # ========================================================

    fps_frames += 1

    elapsed = (
        time.time()
        -
        fps_start
    )


    if elapsed >= 1.0:

        display_fps = (
            fps_frames
            /
            elapsed
        )

        fps_frames = 0

        fps_start = time.time()


    # ========================================================
    # INFO PANEL
    # ========================================================

    panel_x = 20
    panel_y = 20
    panel_w = 410
    panel_h = 190


    overlay = frame.copy()


    cv2.rectangle(

        overlay,

        (
            panel_x,
            panel_y
        ),

        (
            panel_x + panel_w,
            panel_y + panel_h
        ),

        (
            0,
            0,
            0
        ),

        -1

    )


    frame = cv2.addWeighted(

        overlay,

        0.70,

        frame,

        0.30,

        0

    )


    cv2.putText(

        frame,

        "TOP-ANGLE ANIMAL DETECTION",

        (
            panel_x + 15,
            panel_y + 30
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (
            255,
            255,
            255
        ),

        2,

        cv2.LINE_AA

    )


    cv2.putText(

        frame,

        f"DETECTIONS : {total_detections}",

        (
            panel_x + 15,
            panel_y + 65
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.60,

        (
            255,
            255,
            255
        ),

        2,

        cv2.LINE_AA

    )


    cv2.putText(

        frame,

        f"SHEEP     : {sheep_count}",

        (
            panel_x + 15,
            panel_y + 100
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.60,

        (
            0,
            255,
            0
        ),

        2,

        cv2.LINE_AA

    )


    cv2.putText(

        frame,

        f"COW       : {cow_count}",

        (
            panel_x + 15,
            panel_y + 135
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.60,

        (
            255,
            0,
            255
        ),

        2,

        cv2.LINE_AA

    )


    cv2.putText(

        frame,

        f"FPS       : {display_fps:.1f}",

        (
            panel_x + 15,
            panel_y + 170
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (
            255,
            255,
            255
        ),

        2,

        cv2.LINE_AA

    )


    # ========================================================
    # TERMINAL
    # ========================================================

    print(

        f"Frame: {frame_number:05d} | "

        f"Detections: {total_detections:2d} | "

        f"Sheep: {sheep_count:2d} | "

        f"Cow: {cow_count:2d} | "

        f"Inference: {inference_time:.2f}s | "

        f"FPS: {display_fps:.1f}"

    )


    # ========================================================
    # SAVE
    # ========================================================

    writer.write(
        frame
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    display_frame = cv2.resize(

        frame,

        (
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT
        ),

        interpolation=cv2.INTER_AREA

    )


    cv2.imshow(

        "TEST 5 - TOP ANGLE GOAT DETECTION",

        display_frame

    )


    # ========================================================
    # KEY
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        print()
        print("Stopping...")
        print()

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

writer.release()

cv2.destroyAllWindows()


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("TEST 5 FINISHED")
print("=" * 70)

print(
    f"Frames processed : {frame_number}"
)

print(
    f"Output video     : {OUTPUT_PATH}"
)

print()

print(
    "Check whether the group of goats receives bounding boxes."
)

print("=" * 70)
