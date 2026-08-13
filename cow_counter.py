"""
============================================================
TOP-ANGLE GOAT/SHEEP DETECTION TEST
============================================================

Purpose:
    Detect goats from a direct/top camera view.

IMPORTANT:
    COCO does NOT contain a goat class.

    COCO:
        18 = sheep
        19 = cow

    Therefore this test uses SHEEP + COW as candidate
    classes. We are testing whether the pretrained model
    can recognize your goats sufficiently from this angle.

This version:
    - YOLO11m
    - GPU
    - 1280 inference
    - low confidence
    - tiled detection
    - overlapping tiles
    - duplicate suppression
    - no ByteTrack
    - no counting
    - no line crossing

FIRST MAKE DETECTION WORK.
THEN WE ADD TRACKING + COUNTING.
============================================================
"""

import cv2
import os
import time
import argparse
import numpy as np

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolo11m.pt"

DEFAULT_SOURCE = "videos/cow_video1.mp4"

DEFAULT_SAVE = "output/goat_detection_test.mp4"


# ============================================================
# GPU
# ============================================================

DEVICE = 0


# ============================================================
# YOLO
# ============================================================

IMG_SIZE = 1280

CONF_THRESHOLD = 0.08

IOU_THRESHOLD = 0.45


# ============================================================
# COCO CLASSES
# ============================================================

SHEEP_CLASS = 18

COW_CLASS = 19

TARGET_CLASSES = [
    SHEEP_CLASS,
    COW_CLASS
]


# ============================================================
# TILED DETECTION
# ============================================================

ENABLE_TILES = True

TILE_SIZE = 960

TILE_OVERLAP = 0.30


# ============================================================
# DUPLICATE REMOVAL
# ============================================================

DUPLICATE_IOU = 0.50


# ============================================================
# DISPLAY
# ============================================================

DISPLAY_WIDTH = 1200

DISPLAY_HEIGHT = 800


# ============================================================
# OUTPUT
# ============================================================

os.makedirs(
    "output",
    exist_ok=True
)


# ============================================================
# PRINT CONFIG
# ============================================================

def print_config(source, save_path):

    print()
    print("=" * 70)
    print(" TOP-ANGLE GOAT DETECTION TEST")
    print("=" * 70)

    print(f"Source          : {source}")
    print(f"Model           : {MODEL_PATH}")
    print(f"GPU             : {DEVICE}")
    print(f"Image size      : {IMG_SIZE}")
    print(f"Confidence      : {CONF_THRESHOLD}")
    print(f"IoU             : {IOU_THRESHOLD}")
    print(f"Tiling          : {ENABLE_TILES}")
    print(f"Tile size       : {TILE_SIZE}")
    print(f"Tile overlap    : {TILE_OVERLAP}")
    print(f"Output          : {save_path}")

    print()
    print("COCO classes:")
    print("18 = sheep")
    print("19 = cow")
    print()
    print("IMPORTANT:")
    print("COCO does not have a goat class.")
    print("SHEEP detection is being used as a goat candidate.")
    print("=" * 70)
    print()


# ============================================================
# BOX IoU
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(
        box1[0],
        box2[0]
    )

    y1 = max(
        box1[1],
        box2[1]
    )

    x2 = min(
        box1[2],
        box2[2]
    )

    y2 = min(
        box1[3],
        box2[3]
    )

    width = max(
        0,
        x2 - x1
    )

    height = max(
        0,
        y2 - y1
    )

    intersection = (
        width * height
    )

    area1 = max(
        0,
        box1[2] - box1[0]
    ) * max(
        0,
        box1[3] - box1[1]
    )

    area2 = max(
        0,
        box2[2] - box2[0]
    ) * max(
        0,
        box2[3] - box2[1]
    )

    union = (
        area1
        +
        area2
        -
        intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# NMS / DUPLICATE REMOVAL
# ============================================================

def remove_duplicates(
    detections
):

    if not detections:
        return []


    detections = sorted(
        detections,
        key=lambda x: x["confidence"],
        reverse=True
    )


    selected = []


    for detection in detections:

        duplicate = False


        for existing in selected:

            if (
                detection["class_id"]
                !=
                existing["class_id"]
            ):
                continue


            iou = calculate_iou(
                detection["box"],
                existing["box"]
            )


            if iou >= DUPLICATE_IOU:

                duplicate = True

                break


        if not duplicate:

            selected.append(
                detection
            )


    return selected


# ============================================================
# GENERATE TILES
# ============================================================

def create_tiles(
    frame
):

    height, width = frame.shape[:2]


    if (
        width <= TILE_SIZE
        and
        height <= TILE_SIZE
    ):

        return [
            (
                frame,
                0,
                0
            )
        ]


    step = int(
        TILE_SIZE
        *
        (
            1.0
            -
            TILE_OVERLAP
        )
    )


    step = max(
        step,
        100
    )


    x_positions = list(
        range(
            0,
            max(
                1,
                width - TILE_SIZE + 1
            ),
            step
        )
    )


    y_positions = list(
        range(
            0,
            max(
                1,
                height - TILE_SIZE + 1
            ),
            step
        )
    )


    if not x_positions:

        x_positions = [0]


    if not y_positions:

        y_positions = [0]


    if (
        x_positions[-1]
        +
        TILE_SIZE
        <
        width
    ):

        x_positions.append(
            max(
                0,
                width - TILE_SIZE
            )
        )


    if (
        y_positions[-1]
        +
        TILE_SIZE
        <
        height
    ):

        y_positions.append(
            max(
                0,
                height - TILE_SIZE
            )
        )


    tiles = []


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
                width,
                x1 + TILE_SIZE
            )

            y2 = min(
                height,
                y1 + TILE_SIZE
            )


            tile = frame[
                y1:y2,
                x1:x2
            ]


            if (
                tile.shape[0] > 20
                and
                tile.shape[1] > 20
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
# RUN YOLO ON ONE IMAGE
# ============================================================

def detect_image(
    model,
    image
):

    results = model.predict(

        image,

        classes=TARGET_CLASSES,

        conf=CONF_THRESHOLD,

        iou=IOU_THRESHOLD,

        imgsz=IMG_SIZE,

        device=DEVICE,

        verbose=False

    )


    result = results[0]


    detections = []


    if (
        result.boxes is None
        or
        len(result.boxes) == 0
    ):

        return detections


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


    for box, cls_id, confidence in zip(
        boxes,
        classes,
        confidences
    ):

        x1, y1, x2, y2 = box


        detections.append({

            "box": [
                float(x1),
                float(y1),
                float(x2),
                float(y2)
            ],

            "class_id": int(
                cls_id
            ),

            "confidence": float(
                confidence
            )

        })


    return detections


# ============================================================
# DETECT FULL FRAME + TILES
# ============================================================

def detect_frame(
    model,
    frame
):

    detections = []


    # ========================================================
    # FULL FRAME
    # ========================================================

    full_detections = detect_image(
        model,
        frame
    )


    detections.extend(
        full_detections
    )


    # ========================================================
    # TILES
    # ========================================================

    if ENABLE_TILES:

        tiles = create_tiles(
            frame
        )


        for tile, offset_x, offset_y in tiles:

            # Don't duplicate the full frame

            if (
                tile.shape[:2]
                ==
                frame.shape[:2]
            ):

                continue


            tile_detections = detect_image(
                model,
                tile
            )


            for detection in tile_detections:

                x1, y1, x2, y2 = (
                    detection["box"]
                )


                # Convert tile coordinates
                # to original frame coordinates.

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


                detections.append({

                    "box": [
                        x1,
                        y1,
                        x2,
                        y2
                    ],

                    "class_id":
                        detection[
                            "class_id"
                        ],

                    "confidence":
                        detection[
                            "confidence"
                        ]

                })


    # ========================================================
    # REMOVE TILE DUPLICATES
    # ========================================================

    detections = remove_duplicates(
        detections
    )


    return detections


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(
    frame,
    detections,
    model,
    fps
):

    sheep_count = 0

    cow_count = 0


    for index, detection in enumerate(
        detections,
        start=1
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


        # ====================================================
        # CLASS
        # ====================================================

        if class_id == SHEEP_CLASS:

            name = "SHEEP"

            sheep_count += 1

            color = (
                0,
                255,
                0
            )

        elif class_id == COW_CLASS:

            name = "COW"

            cow_count += 1

            color = (
                255,
                0,
                255
            )

        else:

            name = model.names.get(
                class_id,
                "ANIMAL"
            )

            color = (
                0,
                255,
                255
            )


        # ====================================================
        # BOX
        # ====================================================

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

            color,

            3

        )


        # ====================================================
        # CENTER
        # ====================================================

        cx = int(
            (
                x1
                +
                x2
            )
            /
            2
        )

        cy = int(
            (
                y1
                +
                y2
            )
            /
            2
        )


        cv2.circle(

            frame,

            (
                cx,
                cy
            ),

            5,

            color,

            -1

        )


        # ====================================================
        # LABEL
        # ====================================================

        label = (
            f"{name} "
            f"{confidence:.2f} "
            f"#{index}"
        )


        cv2.putText(

            frame,

            label,

            (
                x1,
                max(
                    25,
                    y1 - 8
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            color,

            2,

            cv2.LINE_AA

        )


    # ========================================================
    # PANEL
    # ========================================================

    overlay = frame.copy()


    cv2.rectangle(

        overlay,

        (
            10,
            10
        ),

        (
            430,
            205
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

        "TOP-ANGLE DETECTION",

        (
            25,
            40
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.70,

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

        f"DETECTIONS : {len(detections)}",

        (
            25,
            78
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.62,

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
            25,
            113
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.62,

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
            25,
            148
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.62,

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

        f"FPS       : {fps:.1f}",

        (
            25,
            183
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.58,

        (
            255,
            255,
            255
        ),

        2,

        cv2.LINE_AA

    )


    return frame, sheep_count, cow_count


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="video path or webcam index"
    )


    parser.add_argument(
        "--save",
        default=DEFAULT_SAVE,
        help="output video"
    )


    parser.add_argument(
        "--no-show",
        action="store_true"
    )


    args = parser.parse_args()


    source = args.source

    save_path = args.save


    # ========================================================
    # CONFIG
    # ========================================================

    print_config(
        source,
        save_path
    )


    # ========================================================
    # SOURCE
    # ========================================================

    if str(source).isdigit():

        source_for_cv = int(
            source
        )

    else:

        source_for_cv = source


    cap = cv2.VideoCapture(
        source_for_cv
    )


    if not cap.isOpened():

        print()
        print("ERROR: Cannot open source:")
        print(source)
        print()

        return


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

    fps_input = cap.get(
        cv2.CAP_PROP_FPS
    )


    if fps_input <= 0:

        fps_input = 30.0


    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    print(
        f"Video resolution : "
        f"{width} x {height}"
    )

    print(
        f"Input FPS        : "
        f"{fps_input:.2f}"
    )

    print(
        f"Total frames     : "
        f"{total_frames}"
    )

    print()


    # ========================================================
    # MODEL
    # ========================================================

    print(
        "Loading YOLO11m..."
    )


    model = YOLO(
        MODEL_PATH
    )


    print(
        "YOLO11m loaded."
    )

    print()


    # ========================================================
    # WRITER
    # ========================================================

    writer = cv2.VideoWriter(

        save_path,

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        fps_input,

        (
            width,
            height
        )

    )


    if not writer.isOpened():

        print(
            "WARNING: Output video writer failed."
        )

        writer = None


    # ========================================================
    # FPS
    # ========================================================

    previous_time = time.time()

    smooth_fps = 0.0


    frame_number = 0

    max_detections = 0


    # ========================================================
    # LOOP
    # ========================================================

    while True:

        success, frame = cap.read()


        if not success:

            break


        frame_number += 1


        start = time.time()


        # ====================================================
        # DETECT
        # ====================================================

        detections = detect_frame(

            model,

            frame

        )


        # ====================================================
        # FPS
        # ====================================================

        now = time.time()


        instant_fps = 1.0 / max(
            now - previous_time,
            0.0001
        )


        smooth_fps = (

            instant_fps

            if smooth_fps == 0

            else

            (
                0.90
                *
                smooth_fps
                +
                0.10
                *
                instant_fps
            )

        )


        previous_time = now


        # ====================================================
        # DRAW
        # ====================================================

        frame, sheep_count, cow_count = draw_detections(

            frame,

            detections,

            model,

            smooth_fps

        )


        max_detections = max(
            max_detections,
            len(detections)
        )


        # ====================================================
        # TERMINAL
        # ====================================================

        print(

            f"Frame {frame_number:05d} | "

            f"Detections: "
            f"{len(detections):2d} | "

            f"Sheep: "
            f"{sheep_count:2d} | "

            f"Cow: "
            f"{cow_count:2d} | "

            f"FPS: "
            f"{smooth_fps:5.1f}"

        )


        # ====================================================
        # SAVE
        # ====================================================

        if writer is not None:

            writer.write(
                frame
            )


        # ====================================================
        # DISPLAY
        # ====================================================

        if not args.no_show:

            display = cv2.resize(

                frame,

                (
                    DISPLAY_WIDTH,
                    DISPLAY_HEIGHT
                ),

                interpolation=cv2.INTER_AREA

            )


            cv2.imshow(

                "TOP-ANGLE GOAT DETECTION",

                display

            )


            key = cv2.waitKey(1) & 0xFF


            if key == ord("q"):

                print()
                print(
                    "Stopping..."
                )

                break


    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()


    if writer is not None:

        writer.release()


    cv2.destroyAllWindows()


    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print(" DETECTION TEST FINISHED")
    print("=" * 70)

    print(
        f"Frames processed : "
        f"{frame_number}"
    )

    print(
        f"Maximum detections in one frame : "
        f"{max_detections}"
    )

    print(
        f"Output video : "
        f"{save_path}"
    )

    print()
    print(
        "If the goats are still not detected as a group,"
    )

    print(
        "the next step is a CUSTOM GOAT MODEL."
    )

    print("=" * 70)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
