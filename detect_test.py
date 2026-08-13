import cv2
import os
import csv
import time
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "videos/cow_video10.mp4"

MODEL_PATH = "yolo11s.pt"

OUTPUT_DIR = "output"

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "cow_detection_test.mp4"
)

CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "cow_detection_test.csv"
)


# ============================================================
# YOLO SETTINGS
# ============================================================

# COCO class ID:
# 19 = cow
COW_CLASS_ID = 19

# Lower confidence so partially occluded cows
# have a better chance of reaching the tracker.
CONFIDENCE = 0.20

# Increased from 512 to 768.
IMG_SIZE = 768

# NVIDIA GPU
DEVICE = 0

# ByteTrack
TRACKER = "bytetrack.yaml"


# ============================================================
# DISPLAY SETTINGS
# ============================================================

DISPLAY_WIDTH = 1100
DISPLAY_HEIGHT = 700


# ============================================================
# TRACKING SETTINGS
# ============================================================

# Number of frames a tracker can survive
# without a strong detection.
TRACK_BUFFER = 30


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    print()
    print("ERROR: Could not open video.")
    print(VIDEO_PATH)
    print()

    raise SystemExit


video_width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

video_height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

video_fps = cap.get(
    cv2.CAP_PROP_FPS
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


if video_fps <= 0:

    video_fps = 30.0


print()
print("=" * 60)
print("VIDEO INFORMATION")
print("=" * 60)

print(
    f"Resolution : {video_width} x {video_height}"
)

print(
    f"FPS        : {video_fps:.2f}"
)

print(
    f"Frames     : {total_frames}"
)

print("=" * 60)
print()


# ============================================================
# READ FIRST FRAME
# ============================================================

ret, first_frame = cap.read()

if not ret:

    print(
        "ERROR: Could not read first frame."
    )

    cap.release()

    raise SystemExit


# ============================================================
# REGION SELECTION
# ============================================================

points = []


selection_scale = min(

    DISPLAY_WIDTH / video_width,

    DISPLAY_HEIGHT / video_height
)


selection_width = int(
    video_width * selection_scale
)

selection_height = int(
    video_height * selection_scale
)


selection_frame = cv2.resize(

    first_frame,

    (
        selection_width,
        selection_height
    ),

    interpolation=cv2.INTER_AREA
)


original_selection_frame = (
    selection_frame.copy()
)


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global selection_frame

    if event != cv2.EVENT_LBUTTONDOWN:

        return


    if len(points) >= 4:

        return


    # --------------------------------------------------------
    # DISPLAY → ORIGINAL VIDEO COORDINATES
    # --------------------------------------------------------

    original_x = int(
        x / selection_scale
    )

    original_y = int(
        y / selection_scale
    )


    original_x = max(
        0,
        min(
            original_x,
            video_width - 1
        )
    )


    original_y = max(
        0,
        min(
            original_y,
            video_height - 1
        )
    )


    points.append(

        (
            original_x,
            original_y
        )

    )


    print(
        f"Point {len(points)}: "
        f"({original_x}, {original_y})"
    )


    # --------------------------------------------------------
    # DRAW POINT
    # --------------------------------------------------------

    cv2.circle(

        selection_frame,

        (
            x,
            y
        ),

        7,

        (
            0,
            255,
            0
        ),

        -1
    )


    # --------------------------------------------------------
    # DRAW CONNECTION
    # --------------------------------------------------------

    if len(points) >= 2:

        p1 = points[-2]

        p2 = points[-1]


        p1_display = (

            int(
                p1[0] * selection_scale
            ),

            int(
                p1[1] * selection_scale
            )

        )


        p2_display = (

            int(
                p2[0] * selection_scale
            ),

            int(
                p2[1] * selection_scale
            )

        )


        cv2.line(

            selection_frame,

            p1_display,

            p2_display,

            (
                0,
                255,
                0
            ),

            3
        )


    # --------------------------------------------------------
    # CLOSE POLYGON
    # --------------------------------------------------------

    if len(points) == 4:

        first = points[0]


        first_display = (

            int(
                first[0] * selection_scale
            ),

            int(
                first[1] * selection_scale
            )

        )


        cv2.line(

            selection_frame,

            (
                x,
                y
            ),

            first_display,

            (
                0,
                255,
                255
            ),

            3
        )


# ============================================================
# REGION WINDOW
# ============================================================

cv2.namedWindow(
    "SELECT COW REGION",
    cv2.WINDOW_NORMAL
)


cv2.resizeWindow(

    "SELECT COW REGION",

    selection_width,

    selection_height
)


cv2.setMouseCallback(

    "SELECT COW REGION",

    mouse_callback
)


print()
print("=" * 60)
print("SELECT MONITORING REGION")
print("=" * 60)
print()
print("Click FOUR points around the cow monitoring area.")
print()
print("ENTER = confirm")
print("R     = reset")
print("Q     = quit")
print()


# ============================================================
# REGION SELECTION LOOP
# ============================================================

while True:

    cv2.imshow(

        "SELECT COW REGION",

        selection_frame
    )


    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    if key == ord("r"):

        points.clear()

        selection_frame = (
            original_selection_frame.copy()
        )

        print(
            "Region reset."
        )


    # --------------------------------------------------------
    # CONFIRM
    # --------------------------------------------------------

    elif key == 13:

        if len(points) == 4:

            break

        print(
            f"Select 4 points. "
            f"Currently selected: {len(points)}"
        )


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    elif key == ord("q"):

        cap.release()

        cv2.destroyAllWindows()

        raise SystemExit


cv2.destroyWindow(
    "SELECT COW REGION"
)


# ============================================================
# CREATE POLYGON
# ============================================================

polygon = np.array(

    points,

    dtype=np.int32
)


print()
print("=" * 60)
print("MONITORING REGION")
print("=" * 60)


for i, point in enumerate(

    points,

    start=1

):

    print(
        f"Point {i}: {point}"
    )


print("=" * 60)
print()


# ============================================================
# RESET VIDEO
# ============================================================

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    0
)


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(

    *"mp4v"
)


writer = cv2.VideoWriter(

    OUTPUT_VIDEO,

    fourcc,

    video_fps,

    (
        video_width,
        video_height
    )
)


if not writer.isOpened():

    print(
        "ERROR: Could not create output video."
    )

    cap.release()

    raise SystemExit


# ============================================================
# LOAD YOLO11s
# ============================================================

print()
print("=" * 60)
print("LOADING YOLO11s")
print("=" * 60)

print(
    f"Model      : {MODEL_PATH}"
)

print(
    f"Confidence : {CONFIDENCE}"
)

print(
    f"Image size : {IMG_SIZE}"
)

print(
    f"Device     : GPU {DEVICE}"
)

print(
    f"Tracker    : {TRACKER}"
)

print("=" * 60)
print()


model = YOLO(
    MODEL_PATH
)


# ============================================================
# TRACKER DATA
# ============================================================

# tracker_id -> display ID

display_ids = {}


next_display_id = 1


# tracker_id -> last seen frame

last_seen = {}


# tracker_id -> last center

last_centers = {}


# tracker_id -> number of consecutive missing frames

missing_frames = {}


# ============================================================
# CSV
# ============================================================

csv_file = open(

    CSV_PATH,

    "w",

    newline="",

    encoding="utf-8"
)


csv_writer = csv.writer(
    csv_file
)


csv_writer.writerow(

    [
        "frame",
        "tracker_id",
        "display_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
        "confidence",
        "inside_region"
    ]

)


# ============================================================
# FPS
# ============================================================

fps_start = time.time()

fps_counter = 0

display_fps = 0.0


# ============================================================
# FRAME COUNTERS
# ============================================================

frame_number = 0

max_detections = 0

max_tracks = 0

total_detection_frames = 0


# ============================================================
# HELPER: DISPLAY ID
# ============================================================

def get_display_id(
    tracker_id
):

    global next_display_id


    if tracker_id not in display_ids:

        display_ids[
            tracker_id
        ] = next_display_id

        next_display_id += 1


    return display_ids[
        tracker_id
    ]


# ============================================================
# HELPER: POINT INSIDE REGION
# ============================================================

def is_inside_region(
    point
):

    result = cv2.pointPolygonTest(

        polygon,

        point,

        False
    )


    return result >= 0


# ============================================================
# START
# ============================================================

print()
print("=" * 60)
print("STARTING DETECTION TEST")
print("=" * 60)
print()
print("This test DOES NOT count IN/OUT.")
print()
print("We are testing:")
print()
print("YOLO11s")
print("ByteTrack")
print("Crowded cows")
print("Occlusion")
print("Stable IDs")
print()
print("Press Q to stop.")
print("=" * 60)
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # READ FRAME
    # --------------------------------------------------------

    success, frame = cap.read()


    if not success:

        break


    frame_number += 1


    # ========================================================
    # TRACKING
    # ========================================================

    results = model.track(

        frame,

        persist=True,

        tracker=TRACKER,

        classes=[
            COW_CLASS_ID
        ],

        conf=CONFIDENCE,

        imgsz=IMG_SIZE,

        device=DEVICE,

        verbose=False
    )


    result = results[0]


    # ========================================================
    # ACTIVE TRACKS
    # ========================================================

    active_track_ids = set()


    detection_count = 0


    # ========================================================
    # PROCESS DETECTIONS
    # ========================================================

    if (

        result.boxes is not None

        and result.boxes.id is not None

    ):


        boxes = (

            result.boxes.xyxy

            .detach()

            .cpu()

            .numpy()
        )


        tracker_ids = (

            result.boxes.id

            .detach()

            .cpu()

            .numpy()

            .astype(int)
        )


        confidences = (

            result.boxes.conf

            .detach()

            .cpu()

            .numpy()
        )


        detection_count = len(
            boxes
        )


        total_detection_frames += 1


        # ====================================================
        # PROCESS EACH COW
        # ====================================================

        for (

            box,

            tracker_id,

            confidence

        ) in zip(

            boxes,

            tracker_ids,

            confidences

        ):


            tracker_id = int(
                tracker_id
            )


            confidence = float(
                confidence
            )


            # ------------------------------------------------
            # ACTIVE TRACK
            # ------------------------------------------------

            active_track_ids.add(
                tracker_id
            )


            # ------------------------------------------------
            # DISPLAY ID
            # ------------------------------------------------

            display_id = get_display_id(
                tracker_id
            )


            # ------------------------------------------------
            # BOX
            # ------------------------------------------------

            x1, y1, x2, y2 = map(

                int,

                box
            )


            # ------------------------------------------------
            # CENTER
            # ------------------------------------------------

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )


            center = (

                center_x,

                center_y

            )


            # ------------------------------------------------
            # BOTTOM CENTER
            # ------------------------------------------------

            bottom_center = (

                center_x,

                y2

            )


            # ------------------------------------------------
            # REGION TEST
            # ------------------------------------------------

            inside = is_inside_region(

                bottom_center
            )


            # ------------------------------------------------
            # TRACK DATA
            # ------------------------------------------------

            last_seen[
                tracker_id
            ] = frame_number


            last_centers[
                tracker_id
            ] = center


            missing_frames[
                tracker_id
            ] = 0


            # ------------------------------------------------
            # CSV
            # ------------------------------------------------

            csv_writer.writerow(

                [

                    frame_number,

                    tracker_id,

                    display_id,

                    x1,

                    y1,

                    x2,

                    y2,

                    center_x,

                    center_y,

                    round(
                        confidence,
                        4
                    ),

                    int(inside)

                ]

            )


            # =================================================
            # COLOR
            # =================================================

            if inside:

                box_color = (

                    0,

                    255,

                    0

                )

            else:

                box_color = (

                    0,

                    255,

                    255

                )


            # =================================================
            # DRAW BOX
            # =================================================

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


            # =================================================
            # LABEL
            # =================================================

            label = (

                f"COW {display_id} "
                f"{confidence:.2f}"

            )


            label_y = max(

                y1 - 10,

                30
            )


            cv2.putText(

                frame,

                label,

                (
                    x1,
                    label_y
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.75,

                box_color,

                2,

                cv2.LINE_AA
            )


            # =================================================
            # CENTER POINT
            # =================================================

            cv2.circle(

                frame,

                center,

                5,

                box_color,

                -1
            )


            # =================================================
            # BOTTOM POINT
            # =================================================

            cv2.circle(

                frame,

                bottom_center,

                5,

                (
                    255,
                    0,
                    255
                ),

                -1
            )


    # ========================================================
    # UPDATE MISSING TRACK COUNTERS
    # ========================================================

    for tracker_id in list(
        last_seen.keys()
    ):

        if tracker_id not in active_track_ids:

            missing_frames[
                tracker_id
            ] = (

                frame_number
                - last_seen[tracker_id]

            )


    # ========================================================
    # MAX STATISTICS
    # ========================================================

    max_detections = max(

        max_detections,

        detection_count
    )


    max_tracks = max(

        max_tracks,

        len(active_track_ids)
    )


    # ========================================================
    # FPS
    # ========================================================

    fps_counter += 1


    elapsed = (

        time.time()
        - fps_start
    )


    if elapsed >= 1.0:

        display_fps = (

            fps_counter
            / elapsed
        )


        fps_counter = 0

        fps_start = time.time()


    # ========================================================
    # DRAW REGION
    # ========================================================

    cv2.polylines(

        frame,

        [
            polygon
        ],

        True,

        (
            255,
            0,
            0
        ),

        4
    )


    # ========================================================
    # REGION LABEL
    # ========================================================

    region_center = np.mean(

        polygon,

        axis=0
    ).astype(int)


    cv2.putText(

        frame,

        "MONITORING REGION",

        (

            int(region_center[0]) - 120,

            int(region_center[1])

        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (
            255,
            0,
            0
        ),

        2,

        cv2.LINE_AA
    )


    # ========================================================
    # INFO PANEL
    # ========================================================

    panel_x = 20

    panel_y = 20

    panel_width = 350

    panel_height = 230


    overlay = frame.copy()


    cv2.rectangle(

        overlay,

        (
            panel_x,
            panel_y
        ),

        (
            panel_x + panel_width,
            panel_y + panel_height
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


    # ========================================================
    # PANEL TEXT
    # ========================================================

    panel_color = (

        255,
        255,
        255

    )


    green = (

        0,
        255,
        0

    )


    cv2.putText(

        frame,

        "COW DETECTION TEST",

        (
            panel_x + 15,
            panel_y + 30
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.70,

        panel_color,

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        frame,

        f"DETECTIONS : {detection_count}",

        (
            panel_x + 15,
            panel_y + 65
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        green,

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        frame,

        f"TRACKS     : {len(active_track_ids)}",

        (
            panel_x + 15,
            panel_y + 95
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        panel_color,

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        frame,

        f"CONF       : {CONFIDENCE:.2f}",

        (
            panel_x + 15,
            panel_y + 125
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.60,

        panel_color,

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        frame,

        f"IMG SIZE   : {IMG_SIZE}",

        (
            panel_x + 15,
            panel_y + 155
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.60,

        panel_color,

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        frame,

        f"FPS        : {display_fps:.1f}",

        (
            panel_x + 15,
            panel_y + 185
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.60,

        panel_color,

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        frame,

        f"FRAME      : {frame_number}/{total_frames}",

        (
            panel_x + 15,
            panel_y + 215
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        panel_color,

        1,

        cv2.LINE_AA
    )


    # ========================================================
    # TERMINAL OUTPUT
    # ========================================================

    print(

        f"Frame: {frame_number:04d} | "

        f"Detections: {detection_count:2d} | "

        f"Tracks: {len(active_track_ids):2d} | "

        f"FPS: {display_fps:4.1f}"

    )


    # ========================================================
    # SAVE FRAME
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

        "YOLO11s + BYTE TRACK - DETECTION TEST",

        display_frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        print()
        print("Stopping test...")

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

writer.release()

csv_file.close()

cv2.destroyAllWindows()


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 60)
print("DETECTION TEST FINISHED")
print("=" * 60)

print(
    f"Frames processed      : {frame_number}"
)

print(
    f"Maximum detections    : {max_detections}"
)

print(
    f"Maximum active tracks : {max_tracks}"
)

print(
    f"Detection frames      : {total_detection_frames}"
)

print(
    f"Model                 : {MODEL_PATH}"
)

print(
    f"Image size            : {IMG_SIZE}"
)

print(
    f"Confidence             : {CONFIDENCE}"
)

print()
print(
    "Output video:"
)

print(
    OUTPUT_VIDEO
)

print()
print(
    "CSV log:"
)

print(
    CSV_PATH
)

print("=" * 60)
