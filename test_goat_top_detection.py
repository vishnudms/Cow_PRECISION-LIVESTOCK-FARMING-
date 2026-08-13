"""
===============================================================================
TOP-VIEW GOAT / SHEEP DETECTION TEST
YOLO11m + OpenCV
===============================================================================

PROJECT:
    D:\COW

VIDEO:
    D:\COW\videos\

MODEL:
    D:\COW\yolo11m.pt

OPTIONAL:
    D:\COW\models\yolo11m.pt

IMPORTANT:
    COCO has:
        class 18 = sheep
        class 19 = cow

    COCO does NOT have a goat class.

    Therefore class 18 is displayed as:
        GOAT/SHEEP

This script is ONLY for testing detection.

ByteTrack and A/B counting can be added after detection works.

===============================================================================
"""

import argparse
import os
import time

import cv2
import torch
from ultralytics import YOLO


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_DIR = r"D:\COW"

VIDEO_DIR = os.path.join(
    PROJECT_DIR,
    "videos"
)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)


# =============================================================================
# MODEL
# =============================================================================

# First look for a model inside D:\COW\models
MODEL_IN_MODELS = os.path.join(
    MODEL_DIR,
    "yolo11m.pt"
)

# Otherwise use D:\COW\yolo11m.pt
MODEL_IN_PROJECT = os.path.join(
    PROJECT_DIR,
    "yolo11m.pt"
)


# =============================================================================
# DETECTION SETTINGS
# =============================================================================

# LOW confidence intentionally used for this difficult top-view test.
CONF_THRESHOLD = 0.10

IOU_THRESHOLD = 0.45

# Larger image can help with small animals.
IMAGE_SIZE = 1280


# =============================================================================
# TARGET CLASSES
# =============================================================================

# COCO:
#
# 18 = sheep
# 19 = cow
#
# There is no goat class in COCO.

TARGET_CLASSES = {
    18: "GOAT",
    19: "COW",
}


# =============================================================================
# DISPLAY SETTINGS
# =============================================================================

WINDOW_NAME = "TOP VIEW GOAT DETECTION"

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720


# =============================================================================
# COLORS
# =============================================================================

# OpenCV uses BGR.

COLOR_BOX = (0, 255, 255)

COLOR_CENTER = (0, 0, 255)

COLOR_TEXT = (255, 255, 255)

COLOR_PANEL = (0, 0, 0)

COLOR_DETECTED = (0, 255, 0)

COLOR_WARNING = (0, 165, 255)


# =============================================================================
# FIND MODEL
# =============================================================================

def find_model():

    if os.path.exists(MODEL_IN_MODELS):

        print(
            f"[MODEL] Found: {MODEL_IN_MODELS}"
        )

        return MODEL_IN_MODELS

    if os.path.exists(MODEL_IN_PROJECT):

        print(
            f"[MODEL] Found: {MODEL_IN_PROJECT}"
        )

        return MODEL_IN_PROJECT

    print()
    print("=" * 70)
    print("[ERROR] YOLO MODEL NOT FOUND")
    print("=" * 70)

    print()
    print("Expected one of:")

    print(
        MODEL_IN_MODELS
    )

    print(
        MODEL_IN_PROJECT
    )

    print()

    return None


# =============================================================================
# CHECK VIDEO
# =============================================================================

def check_video(video_path):

    if not os.path.exists(video_path):

        print()
        print("=" * 70)
        print("[ERROR] VIDEO NOT FOUND")
        print("=" * 70)

        print(
            video_path
        )

        print()

        return None

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        print(
            "[ERROR] OpenCV could not open the video."
        )

        return None

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

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

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = 0

    if fps > 0:

        duration = frame_count / fps

    print()
    print("=" * 70)
    print("VIDEO INFORMATION")
    print("=" * 70)

    print(
        f"Path       : {video_path}"
    )

    print(
        f"Resolution : {width} x {height}"
    )

    print(
        f"FPS        : {fps:.2f}"
    )

    print(
        f"Frames     : {frame_count}"
    )

    print(
        f"Duration   : {duration:.2f} seconds"
    )

    print("=" * 70)

    cap.release()

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frames": frame_count,
    }


# =============================================================================
# DRAW INFORMATION PANEL
# =============================================================================

def draw_panel(
    frame,
    frame_number,
    total_frames,
    fps,
    detection_count,
    sheep_count,
    cow_count,
    total_detected
):

    # -------------------------------------------------------------------------
    # PANEL SIZE
    # -------------------------------------------------------------------------

    panel_width = 390

    panel_height = 300

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (
            panel_width,
            panel_height
        ),
        COLOR_PANEL,
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0
    )

    # -------------------------------------------------------------------------
    # TEXT
    # -------------------------------------------------------------------------

    if total_frames > 0:

        progress = (
            frame_number / total_frames
        ) * 100

    else:

        progress = 0

    lines = [

        (
            "TOP-VIEW GOAT DETECTION",
            COLOR_TEXT
        ),

        (
            f"CURRENT GOATS : {sheep_count}",
            COLOR_DETECTED
        ),

        (
            f"COW (other)   : {cow_count}",
            (255, 180, 0)
        ),

        (
            f"FRAME       : "
            f"{frame_number}/{total_frames}",
            COLOR_TEXT
        ),

        (
            f"PROGRESS    : "
            f"{progress:.1f}%",
            COLOR_TEXT
        ),

        (
            f"FPS         : "
            f"{fps:.1f}",
            COLOR_TEXT
        ),

        (
            f"TOTAL SEEN  : "
            f"{total_detected}",
            COLOR_TEXT
        ),

        (
            "CONFIDENCE  : 0.10",
            COLOR_WARNING
        ),

        (
            "Q = EXIT",
            COLOR_TEXT
        ),
    ]

    y = 38

    for text, color in lines:

        # Make the headline "CURRENT GOATS" stat visually bigger/bolder
        # than the rest of the panel, since it's the key live number.
        is_headline = text.startswith("CURRENT GOATS")
        scale = 0.85 if is_headline else 0.60
        thickness = 2 if is_headline else 1

        cv2.putText(
            frame,
            text,
            (25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA
        )

        y += 30 if is_headline else 24

    return frame


# =============================================================================
# MAIN DETECTION
# =============================================================================

def run_detection(
    video_path,
    model_path,
    save_path=None
):

    # -------------------------------------------------------------------------
    # DEVICE
    # -------------------------------------------------------------------------

    if torch.cuda.is_available():

        device = 0

        print()
        print(
            "[DEVICE] NVIDIA CUDA GPU detected."
        )

        try:

            print(
                "[GPU] "
                + torch.cuda.get_device_name(0)
            )

        except Exception:

            pass

    else:

        device = "cpu"

        print()
        print(
            "[DEVICE] CUDA not available."
        )

        print(
            "[DEVICE] Using CPU."
        )

    # -------------------------------------------------------------------------
    # LOAD MODEL
    # -------------------------------------------------------------------------

    print()
    print(
        "[MODEL] Loading YOLO..."
    )

    model = YOLO(
        model_path
    )

    print(
        "[MODEL] Model loaded."
    )

    # -------------------------------------------------------------------------
    # VIDEO
    # -------------------------------------------------------------------------

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        print(
            "[ERROR] Could not open video."
        )

        return

    fps_input = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps_input <= 0:

        fps_input = 8.0

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

    # -------------------------------------------------------------------------
    # OUTPUT VIDEO
    # -------------------------------------------------------------------------

    writer = None

    if save_path:

        output_dir = os.path.dirname(
            os.path.abspath(
                save_path
            )
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(

            save_path,

            fourcc,

            fps_input,

            (
                width,
                height
            )
        )

        if not writer.isOpened():

            print(
                "[WARNING] Could not create output video."
            )

            writer = None

        else:

            print(
                f"[OUTPUT] {save_path}"
            )

    # -------------------------------------------------------------------------
    # NORMAL RESIZABLE WINDOW
    # -------------------------------------------------------------------------

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        WINDOW_WIDTH,
        WINDOW_HEIGHT
    )

    # -------------------------------------------------------------------------
    # STATISTICS
    # -------------------------------------------------------------------------

    frame_number = 0

    total_detections = 0

    total_goat_sheep_detections = 0

    total_cow_detections = 0

    frames_with_detection = 0

    # -------------------------------------------------------------------------
    # FPS
    # -------------------------------------------------------------------------

    previous_time = time.time()

    fps_smooth = 0.0

    # -------------------------------------------------------------------------
    # START
    # -------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING TOP-VIEW DETECTION")
    print("=" * 70)

    print(
        f"Confidence : {CONF_THRESHOLD}"
    )

    print(
        f"Image size : {IMAGE_SIZE}"
    )

    print(
        f"Device     : {device}"
    )

    print()
    print(
        "Press Q to stop."
    )

    print("=" * 70)

    # -------------------------------------------------------------------------
    # FRAME LOOP
    # -------------------------------------------------------------------------

    while True:

        success, frame = cap.read()

        if not success:

            print()
            print(
                "[VIDEO] End of video."
            )

            break

        frame_number += 1

        # ---------------------------------------------------------------------
        # YOLO
        # ---------------------------------------------------------------------

        results = model.predict(

            source=frame,

            conf=CONF_THRESHOLD,

            iou=IOU_THRESHOLD,

            imgsz=IMAGE_SIZE,

            classes=list(
                TARGET_CLASSES.keys()
            ),

            device=device,

            verbose=False
        )

        result = results[0]

        detection_count = 0

        sheep_count = 0        # this is the CURRENT goat count for this frame

        cow_count = 0

        # ---------------------------------------------------------------------
        # DETECTIONS
        # ---------------------------------------------------------------------

        if result.boxes is not None:

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

            detection_count = len(
                boxes
            )

            if detection_count > 0:

                frames_with_detection += 1

            total_detections += (
                detection_count
            )

            # ---------------------------------------------------------------
            # SPLIT DETECTIONS BY CLASS AND GIVE GOATS A STABLE READING ORDER
            # ---------------------------------------------------------------
            # We sort goat detections top-to-bottom, then left-to-right,
            # so goat #1, #2, #3... land on roughly the same animal frame
            # to frame (as much as is possible without a tracker) instead
            # of the numbers jumping around in raw detection order.
            # ---------------------------------------------------------------

            goat_dets = []
            cow_dets = []

            for box, class_id, confidence in zip(boxes, classes, confidences):

                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                entry = (x1, y1, x2, y2, cx, cy, float(confidence))

                if int(class_id) == 18:
                    goat_dets.append(entry)
                elif int(class_id) == 19:
                    cow_dets.append(entry)

            goat_dets.sort(key=lambda e: (round(e[5] / 40), e[4]))  # row bucket, then x

            sheep_count = len(goat_dets)
            cow_count = len(cow_dets)

            total_goat_sheep_detections += sheep_count
            total_cow_detections += cow_count

            # ---------------------------------------------------------------
            # DRAW GOATS — numbered above each one
            # ---------------------------------------------------------------

            for goat_number, (x1, y1, x2, y2, cx, cy, confidence) in enumerate(
                goat_dets, start=1
            ):

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    COLOR_BOX,
                    2,
                )

                cv2.circle(frame, (cx, cy), 5, COLOR_CENTER, -1)

                # Big, high-contrast number above the goat's box.
                number_label = f"#{goat_number}"
                label_y = max(int(y1) - 14, 30)

                # Black outline behind the text so the number stays
                # readable over light-colored animals.
                cv2.putText(
                    frame, number_label, (int(x1), label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA,
                )
                cv2.putText(
                    frame, number_label, (int(x1), label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_DETECTED, 2, cv2.LINE_AA,
                )

                # Small class/confidence tag below the box, for debugging —
                # separate from the count number so the number stays clean.
                debug_label = f"GOAT {confidence:.2f}"
                cv2.putText(
                    frame, debug_label, (int(x1), int(y2) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BOX, 1, cv2.LINE_AA,
                )

            # ---------------------------------------------------------------
            # DRAW COWS — plain label, not part of the goat count/numbering
            # ---------------------------------------------------------------

            for (x1, y1, x2, y2, cx, cy, confidence) in cow_dets:

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255, 180, 0),
                    2,
                )
                cv2.circle(frame, (cx, cy), 5, COLOR_CENTER, -1)
                label = f"COW {confidence:.2f}"
                label_y = max(int(y1) - 10, 20)
                cv2.putText(
                    frame, label, (int(x1), label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2, cv2.LINE_AA,
                )

        # ---------------------------------------------------------------------
        # FPS
        # ---------------------------------------------------------------------

        now = time.time()

        instant_fps = (
            1.0
            /
            max(
                now - previous_time,
                1e-6
            )
        )

        if fps_smooth == 0:

            fps_smooth = instant_fps

        else:

            fps_smooth = (
                0.9 * fps_smooth
                +
                0.1 * instant_fps
            )

        previous_time = now

        # ---------------------------------------------------------------------
        # PANEL
        # ---------------------------------------------------------------------

        frame = draw_panel(

            frame,

            frame_number,

            total_frames,

            fps_smooth,

            detection_count,

            sheep_count,

            cow_count,

            total_detections
        )

        # ---------------------------------------------------------------------
        # SAVE
        # ---------------------------------------------------------------------

        if writer:

            writer.write(
                frame
            )

        # ---------------------------------------------------------------------
        # DISPLAY
        # ---------------------------------------------------------------------

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        key = (
            cv2.waitKey(1)
            &
            0xFF
        )

        if key == ord("q"):

            print(
                "[USER] Q pressed."
            )

            break

        # ---------------------------------------------------------------------
        # PRINT DETECTION INFO
        # ---------------------------------------------------------------------

        if detection_count > 0:

            print(

                f"[FRAME {frame_number:5d}] "
                f"Detections={detection_count} "
                f"CurrentGoats={sheep_count} "
                f"Cow={cow_count}"
            )

    # -------------------------------------------------------------------------
    # CLEANUP
    # -------------------------------------------------------------------------

    cap.release()

    if writer:

        writer.release()

    cv2.destroyAllWindows()

    # -------------------------------------------------------------------------
    # FINAL STATISTICS
    # -------------------------------------------------------------------------

    detection_rate = 0

    if frame_number > 0:

        detection_rate = (
            frames_with_detection
            /
            frame_number
        ) * 100

    print()
    print("=" * 70)
    print("FINAL DETECTION RESULTS")
    print("=" * 70)

    print(
        f"Frames processed       : "
        f"{frame_number}"
    )

    print(
        f"Frames with detection  : "
        f"{frames_with_detection}"
    )

    print(
        f"Detection frame rate   : "
        f"{detection_rate:.2f}%"
    )

    print(
        f"Total detections       : "
        f"{total_detections}"
    )

    print(
        f"GOAT detections (sum)  : "
        f"{total_goat_sheep_detections}"
    )

    print(
        f"COW detections         : "
        f"{total_cow_detections}"
    )

    if save_path:

        print()
        print(
            f"Saved output           : "
            f"{save_path}"
        )

    print("=" * 70)


# =============================================================================
# COMMAND LINE
# =============================================================================

def main():

    # `global` must appear before ANY use of these names in this function
    # (including as an argparse `default=...` value below) — Python
    # requires the declaration first, otherwise it's a SyntaxError.
    global CONF_THRESHOLD
    global IMAGE_SIZE

    parser = argparse.ArgumentParser(

        description=(
            "YOLO11 top-view goat/sheep "
            "detection test"
        )
    )

    parser.add_argument(

        "--source",

        default=(
            r"D:\COW\videos\cow_video1.mp4"
        ),

        help=(
            "Video path"
        )
    )

    parser.add_argument(

        "--model",

        default=None,

        help=(
            "Optional YOLO model path"
        )
    )

    parser.add_argument(

        "--conf",

        type=float,

        default=CONF_THRESHOLD,

        help=(
            "Detection confidence"
        )
    )

    parser.add_argument(

        "--imgsz",

        type=int,

        default=IMAGE_SIZE,

        help=(
            "YOLO image size"
        )
    )

    parser.add_argument(

        "--save",

        default=None,

        help=(
            "Output video path"
        )
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # UPDATE SETTINGS
    # -------------------------------------------------------------------------

    CONF_THRESHOLD = args.conf

    IMAGE_SIZE = args.imgsz

    # -------------------------------------------------------------------------
    # SOURCE
    # -------------------------------------------------------------------------

    video_path = args.source

    # -------------------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------------------

    if args.model:

        model_path = args.model

    else:

        model_path = find_model()

    if model_path is None:

        return

    # -------------------------------------------------------------------------
    # CHECK VIDEO
    # -------------------------------------------------------------------------

    video_info = check_video(
        video_path
    )

    if video_info is None:

        return

    # -------------------------------------------------------------------------
    # DEFAULT OUTPUT
    # -------------------------------------------------------------------------

    save_path = args.save

    if save_path is None:

        save_path = os.path.join(

            PROJECT_DIR,

            "output",

            "goat_top_detection_test.mp4"
        )

    # -------------------------------------------------------------------------
    # RUN
    # -------------------------------------------------------------------------

    run_detection(

        video_path,

        model_path,

        save_path
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()
