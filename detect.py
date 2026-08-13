import cv2
import time
import argparse
import numpy as np
from pathlib import Path

from ultralytics import YOLOWorld


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolov8s-worldv2.pt"

# Your video
DEFAULT_SOURCE = r"videos\cow_video6.mp4"

# Output video
DEFAULT_OUTPUT = r"output\goat_demo.mp4"

# Goat detector prompt
TARGET_CLASSES = [
    "goat"
]

# Detection confidence
CONFIDENCE = 0.15

# IoU
IOU = 0.50

# YOLO image size
IMAGE_SIZE = 1280

# GPU
DEVICE = 0

# Tracker
TRACKER = "bytetrack.yaml"


# ============================================================
# COUNTING LINE
# ============================================================
#
# IMPORTANT:
# These are NORMALIZED coordinates.
#
# x = 0.0 -> left
# x = 1.0 -> right
# y = 0.0 -> top
# y = 1.0 -> bottom
#
# Current default:
# horizontal line around 60% of image height.
#
# We will adjust this after seeing your actual video.
# ============================================================

LINE_START = (0.10, 0.60)
LINE_END   = (0.90, 0.60)


# ============================================================
# DIRECTION
# ============================================================
#
# If goat crosses:
#
# ABOVE -> BELOW = OUT
# BELOW -> ABOVE = IN
#
# If your real video moves in the opposite direction,
# press "r" during the demo to reverse IN/OUT.
# ============================================================

OUT_DIRECTION = "DOWN"


# ============================================================
# DISPLAY
# ============================================================

SHOW_TRAJECTORY = True
SHOW_CONFIDENCE = True
SHOW_FPS = True


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalized_point(point, width, height):
    """
    Convert normalized coordinate to pixel coordinate.
    """
    x = int(point[0] * width)
    y = int(point[1] * height)
    return x, y


def line_side(point, line_start, line_end):
    """
    Returns which side of the line a point is on.

    Positive and negative values represent opposite sides.
    """

    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def crossed_line(previous_point, current_point, line_start, line_end):
    """
    Detect whether a trajectory crossed the line.
    """

    if previous_point is None:
        return False

    old_side = line_side(
        previous_point,
        line_start,
        line_end
    )

    new_side = line_side(
        current_point,
        line_start,
        line_end
    )

    return (
        old_side < 0 and new_side >= 0
    ) or (
        old_side > 0 and new_side <= 0
    )


def get_direction(
    previous_point,
    current_point,
    line_start,
    line_end
):
    """
    Determine direction of crossing.
    """

    if previous_point is None:
        return None

    old_side = line_side(
        previous_point,
        line_start,
        line_end
    )

    new_side = line_side(
        current_point,
        line_start,
        line_end
    )

    if old_side < 0 and new_side >= 0:
        return "DOWN"

    if old_side > 0 and new_side <= 0:
        return "UP"

    return None


def draw_text(
    frame,
    text,
    position,
    scale=0.7,
    thickness=2
):
    """
    Draw readable text with black background.
    """

    x, y = position

    font = cv2.FONT_HERSHEY_SIMPLEX

    (tw, th), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness
    )

    cv2.rectangle(
        frame,
        (x - 5, y - th - baseline - 5),
        (x + tw + 5, y + 5),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Video file or camera index"
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output video"
    )

    args = parser.parse_args()

    source = args.source
    output_path = args.output

    print()
    print("=" * 70)
    print("        REAL-TIME GOAT DETECTION + TRACKING + COUNTING")
    print("=" * 70)
    print(f"Source       : {source}")
    print(f"Model        : {MODEL_PATH}")
    print(f"Target       : GOAT")
    print(f"Confidence   : {CONFIDENCE}")
    print(f"Image size   : {IMAGE_SIZE}")
    print(f"Device       : GPU {DEVICE}")
    print(f"Tracker      : {TRACKER}")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("Loading YOLO-World...")

    model = YOLOWorld(MODEL_PATH)

    # This is the important part:
    # YOLO-World will look for goats.
    model.set_classes(TARGET_CLASSES)

    print("YOLO-World loaded.")
    print("Prompt:", TARGET_CLASSES)

    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():

        print()
        print("ERROR: Could not open video:")
        print(source)
        print()

        return

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps_source = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print()
    print("VIDEO INFORMATION")
    print("-" * 70)
    print(f"Resolution : {width} x {height}")
    print(f"FPS        : {fps_source:.2f}")
    print(f"Frames     : {total_frames}")
    print("-" * 70)

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # VIDEO WRITER
    # --------------------------------------------------------

    output_fps = fps_source

    if output_fps <= 0:
        output_fps = 30

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        output_fps,
        (width, height)
    )

    # --------------------------------------------------------
    # COUNTING VARIABLES
    # --------------------------------------------------------

    goats_in = 0
    goats_out = 0

    counted_in = set()
    counted_out = set()

    # Track history
    previous_positions = {}

    trajectories = {}

    frame_number = 0

    # FPS calculation
    fps_timer = time.time()
    fps_counter = 0
    display_fps = 0.0

    # Current direction
    out_direction = OUT_DIRECTION

    # --------------------------------------------------------
    # LINE
    # --------------------------------------------------------

    line_start = normalized_point(
        LINE_START,
        width,
        height
    )

    line_end = normalized_point(
        LINE_END,
        width,
        height
    )

    print()
    print("COUNTING LINE")
    print(f"Start: {line_start}")
    print(f"End  : {line_end}")

    print()
    print("CONTROLS")
    print("-" * 70)
    print("R = Reverse IN / OUT direction")
    print("T = Toggle trajectories")
    print("Q = Quit")
    print("-" * 70)
    print()

    # ========================================================
    # FRAME LOOP
    # ========================================================

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # ----------------------------------------------------
        # TRACK GOATS
        # ----------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER,
            conf=CONFIDENCE,
            iou=IOU,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False
        )

        result = results[0]

        # ----------------------------------------------------
        # GET DETECTIONS
        # ----------------------------------------------------

        boxes = result.boxes

        current_ids = set()

        if boxes is not None and len(boxes) > 0:

            xyxy = boxes.xyxy.cpu().numpy()

            confs = boxes.conf.cpu().numpy()

            if boxes.id is not None:

                track_ids = (
                    boxes.id
                    .int()
                    .cpu()
                    .tolist()
                )

            else:

                track_ids = [
                    -1
                ] * len(xyxy)

            # ------------------------------------------------
            # PROCESS EACH GOAT
            # ------------------------------------------------

            for box, confidence, track_id in zip(
                xyxy,
                confs,
                track_ids
            ):

                x1, y1, x2, y2 = map(
                    int,
                    box
                )

                # Center of goat
                cx = int(
                    (x1 + x2) / 2
                )

                cy = int(
                    (y1 + y2) / 2
                )

                current_ids.add(
                    track_id
                )

                # ------------------------------------------------
                # DRAW BOX
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # ------------------------------------------------
                # GOAT ID
                # ------------------------------------------------

                label = f"GOAT {track_id}"

                if SHOW_CONFIDENCE:

                    label += (
                        f" {confidence:.2f}"
                    )

                draw_text(
                    frame,
                    label,
                    (x1, max(30, y1))
                )

                # ------------------------------------------------
                # TRAJECTORY
                # ------------------------------------------------

                if track_id not in trajectories:

                    trajectories[track_id] = []

                trajectories[track_id].append(
                    (cx, cy)
                )

                # Keep last 30 positions
                trajectories[track_id] = (
                    trajectories[track_id][-30:]
                )

                if SHOW_TRAJECTORY:

                    points = trajectories[track_id]

                    for i in range(
                        1,
                        len(points)
                    ):

                        cv2.line(
                            frame,
                            points[i - 1],
                            points[i],
                            (255, 255, 0),
                            2
                        )

                # ------------------------------------------------
                # CROSSING DETECTION
                # ------------------------------------------------

                previous = (
                    previous_positions
                    .get(track_id)
                )

                current = (
                    cx,
                    cy
                )

                if crossed_line(
                    previous,
                    current,
                    line_start,
                    line_end
                ):

                    direction = get_direction(
                        previous,
                        current,
                        line_start,
                        line_end
                    )

                    if direction == out_direction:

                        if track_id not in counted_out:

                            goats_out += 1

                            counted_out.add(
                                track_id
                            )

                            print(
                                f"[OUT] Goat {track_id} "
                                f"| Total OUT = {goats_out}"
                            )

                    else:

                        if track_id not in counted_in:

                            goats_in += 1

                            counted_in.add(
                                track_id
                            )

                            print(
                                f"[IN] Goat {track_id} "
                                f"| Total IN = {goats_in}"
                            )

                previous_positions[
                    track_id
                ] = current

        # ----------------------------------------------------
        # REMOVE OLD TRAJECTORIES
        # ----------------------------------------------------

        active_ids = current_ids

        old_ids = list(
            previous_positions.keys()
        )

        for old_id in old_ids:

            if old_id not in active_ids:

                # Don't delete immediately.
                # ByteTrack may temporarily lose a goat.
                pass

        # ----------------------------------------------------
        # DRAW COUNTING LINE
        # ----------------------------------------------------

        cv2.line(
            frame,
            line_start,
            line_end,
            (0, 0, 255),
            4
        )

        # Label line
        draw_text(
            frame,
            "COUNTING LINE",
            (
                line_start[0] + 10,
                line_start[1] - 10
            ),
            scale=0.65
        )

        # ----------------------------------------------------
        # CALCULATE FPS
        # ----------------------------------------------------

        fps_counter += 1

        elapsed = (
            time.time() - fps_timer
        )

        if elapsed >= 1.0:

            display_fps = (
                fps_counter / elapsed
            )

            fps_counter = 0
            fps_timer = time.time()

        # ----------------------------------------------------
        # CURRENT GOAT COUNT
        # ----------------------------------------------------

        current_goats = len(
            current_ids
        )

        # ----------------------------------------------------
        # DASHBOARD
        # ----------------------------------------------------

        panel_height = 155

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (520, panel_height),
            (0, 0, 0),
            -1
        )

        frame = cv2.addWeighted(
            overlay,
            0.70,
            frame,
            0.30,
            0
        )

        draw_text(
            frame,
            f"GOATS DETECTED : {current_goats}",
            (20, 35),
            scale=0.75
        )

        draw_text(
            frame,
            f"GOATS IN       : {goats_in}",
            (20, 70),
            scale=0.75
        )

        draw_text(
            frame,
            f"GOATS OUT      : {goats_out}",
            (20, 105),
            scale=0.75
        )

        if SHOW_FPS:

            draw_text(
                frame,
                f"FPS            : {display_fps:.1f}",
                (20, 140),
                scale=0.65
            )

        # ----------------------------------------------------
        # WRITE OUTPUT
        # ----------------------------------------------------

        writer.write(frame)

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        cv2.imshow(
            "REAL-TIME GOAT MONITOR",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord("q"):

            break

        # Reverse direction
        elif key == ord("r"):

            if out_direction == "DOWN":

                out_direction = "UP"

            else:

                out_direction = "DOWN"

            print(
                f"Direction changed. "
                f"OUT = {out_direction}"
            )

        # Toggle trajectories
        elif key == ord("t"):

            SHOW_TRAJECTORY = not SHOW_TRAJECTORY

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    writer.release()

    cv2.destroyAllWindows()

    print()
    print("=" * 70)
    print("GOAT DEMO FINISHED")
    print("=" * 70)
    print(f"Frames processed : {frame_number}")
    print(f"GOATS IN         : {goats_in}")
    print(f"GOATS OUT        : {goats_out}")
    print(f"Output video     : {output_path}")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
