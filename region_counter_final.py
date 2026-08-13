import cv2
import os
import time
import csv
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "videos/cow_video5.mp4"
MODEL_PATH = "yolo11n.pt"

OUTPUT_DIR = "output"
OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "cow_count_final.mp4"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "cow_events.csv"
)

# COCO class ID
# cow = 19
COW_CLASS_ID = 19

# Detection confidence
CONFIDENCE = 0.20

# Inference resolution
IMG_SIZE = 768

# GPU
DEVICE = 0

# ByteTrack
TRACKER = "bytetrack.yaml"

# ------------------------------------------------------------
# Tracking stability
# ------------------------------------------------------------

# Minimum number of frames a new track must survive before
# it is considered reliable.
MIN_TRACK_AGE = 5

# Maximum number of frames a track can disappear before
# being removed from active tracking.
MAX_MISSING_FRAMES = 20

# ------------------------------------------------------------
# Counting line
# ------------------------------------------------------------

# Minimum movement across the line required before counting.
MIN_CROSSING_DISTANCE = 15

# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

DISPLAY_WIDTH = 1100
DISPLAY_HEIGHT = 700


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

    print("ERROR: Cannot open video:")
    print(VIDEO_PATH)

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


print()
print("================================================")
print(" VIDEO INFORMATION")
print("================================================")
print(f"Resolution : {video_width} x {video_height}")
print(f"FPS        : {video_fps:.2f}")
print(f"Frames     : {total_frames}")
print("================================================")
print()


# ============================================================
# READ FIRST FRAME
# ============================================================

ret, first_frame = cap.read()

if not ret:

    print("ERROR: Could not read first frame.")

    cap.release()

    raise SystemExit


# ============================================================
# DISPLAY SCALE
# ============================================================

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


# ============================================================
# REGION SELECTION
# ============================================================

region_points = []

selection_frame = cv2.resize(
    first_frame,
    (
        selection_width,
        selection_height
    ),
    interpolation=cv2.INTER_AREA
)

original_selection_frame = selection_frame.copy()


# ============================================================
# REGION MOUSE CALLBACK
# ============================================================

def region_mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global selection_frame

    if event != cv2.EVENT_LBUTTONDOWN:

        return

    if len(region_points) >= 4:

        return

    original_x = int(
        x / selection_scale
    )

    original_y = int(
        y / selection_scale
    )

    region_points.append(
        (
            original_x,
            original_y
        )
    )

    print(
        f"Region Point {len(region_points)}:"
        f" ({original_x}, {original_y})"
    )

    cv2.circle(
        selection_frame,
        (x, y),
        7,
        (0, 255, 0),
        -1
    )

    if len(region_points) > 1:

        p1 = region_points[-2]

        p2 = region_points[-1]

        p1_display = (
            int(p1[0] * selection_scale),
            int(p1[1] * selection_scale)
        )

        p2_display = (
            int(p2[0] * selection_scale),
            int(p2[1] * selection_scale)
        )

        cv2.line(
            selection_frame,
            p1_display,
            p2_display,
            (0, 255, 0),
            3
        )

    if len(region_points) == 4:

        p1 = region_points[0]

        p2 = region_points[3]

        p1_display = (
            int(p1[0] * selection_scale),
            int(p1[1] * selection_scale)
        )

        p2_display = (
            int(p2[0] * selection_scale),
            int(p2[1] * selection_scale)
        )

        cv2.line(
            selection_frame,
            p1_display,
            p2_display,
            (0, 255, 255),
            3
        )


# ============================================================
# REGION WINDOW
# ============================================================

cv2.namedWindow(
    "SELECT MONITORING REGION",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "SELECT MONITORING REGION",
    selection_width,
    selection_height
)

cv2.setMouseCallback(
    "SELECT MONITORING REGION",
    region_mouse_callback
)


print()
print("================================================")
print(" STEP 1 - SELECT MONITORING REGION")
print("================================================")
print()
print("Click FOUR points around the cow monitoring area.")
print()
print("Order:")
print("1 -> top-left")
print("2 -> top-right")
print("3 -> bottom-right")
print("4 -> bottom-left")
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
        "SELECT MONITORING REGION",
        selection_frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):

        region_points.clear()

        selection_frame = (
            original_selection_frame.copy()
        )

        print("Region reset.")

    elif key == 13:

        if len(region_points) == 4:

            break

        print(
            f"Select 4 points."
            f" Current: {len(region_points)}"
        )

    elif key == ord("q"):

        cap.release()
        cv2.destroyAllWindows()

        raise SystemExit


cv2.destroyWindow(
    "SELECT MONITORING REGION"
)


# ============================================================
# CREATE REGION POLYGON
# ============================================================

region_polygon = np.array(
    region_points,
    dtype=np.int32
)


print()
print("================================================")
print(" MONITORING REGION")
print("================================================")

for i, point in enumerate(
    region_points,
    start=1
):

    print(
        f"Point {i}: {point}"
    )

print("================================================")
print()


# ============================================================
# COUNTING LINE SELECTION
# ============================================================

line_points = []

line_frame = cv2.resize(
    first_frame,
    (
        selection_width,
        selection_height
    ),
    interpolation=cv2.INTER_AREA
)

line_polygon_display = np.array(
    [
        (
            int(x * selection_scale),
            int(y * selection_scale)
        )

        for x, y in region_points
    ],
    dtype=np.int32
)


cv2.polylines(
    line_frame,
    [
        line_polygon_display
    ],
    True,
    (255, 0, 0),
    3
)


# ============================================================
# COUNTING LINE MOUSE CALLBACK
# ============================================================

def line_mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global line_frame

    if event != cv2.EVENT_LBUTTONDOWN:

        return

    if len(line_points) >= 2:

        return

    original_x = int(
        x / selection_scale
    )

    original_y = int(
        y / selection_scale
    )

    line_points.append(
        (
            original_x,
            original_y
        )
    )

    print(
        f"Line Point {len(line_points)}:"
        f" ({original_x}, {original_y})"
    )

    cv2.circle(
        line_frame,
        (x, y),
        7,
        (0, 0, 255),
        -1
    )

    if len(line_points) == 2:

        p1 = line_points[0]

        p2 = line_points[1]

        p1_display = (
            int(p1[0] * selection_scale),
            int(p1[1] * selection_scale)
        )

        p2_display = (
            int(p2[0] * selection_scale),
            int(p2[1] * selection_scale)
        )

        cv2.line(
            line_frame,
            p1_display,
            p2_display,
            (0, 0, 255),
            5
        )


# ============================================================
# COUNTING LINE WINDOW
# ============================================================

cv2.namedWindow(
    "SELECT COUNTING LINE",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "SELECT COUNTING LINE",
    selection_width,
    selection_height
)

cv2.setMouseCallback(
    "SELECT COUNTING LINE",
    line_mouse_callback
)


print()
print("================================================")
print(" STEP 2 - SELECT COUNTING LINE")
print("================================================")
print()
print("Click TWO points to create the counting line.")
print()
print("IMPORTANT:")
print("The FIRST point and SECOND point define direction.")
print()
print("Cows crossing in one direction = IN")
print("Cows crossing in opposite direction = OUT")
print()
print("ENTER = confirm")
print("R     = reset")
print("Q     = quit")
print()


# ============================================================
# LINE SELECTION LOOP
# ============================================================

while True:

    cv2.imshow(
        "SELECT COUNTING LINE",
        line_frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):

        line_points.clear()

        line_frame = cv2.resize(
            first_frame,
            (
                selection_width,
                selection_height
            ),
            interpolation=cv2.INTER_AREA
        )

        cv2.polylines(
            line_frame,
            [
                line_polygon_display
            ],
            True,
            (255, 0, 0),
            3
        )

        print("Counting line reset.")

    elif key == 13:

        if len(line_points) == 2:

            break

        print(
            f"Select 2 points."
            f" Current: {len(line_points)}"
        )

    elif key == ord("q"):

        cap.release()
        cv2.destroyAllWindows()

        raise SystemExit


cv2.destroyWindow(
    "SELECT COUNTING LINE"
)


# ============================================================
# COUNTING LINE
# ============================================================

line_p1 = np.array(
    line_points[0],
    dtype=np.float32
)

line_p2 = np.array(
    line_points[1],
    dtype=np.float32
)


print()
print("================================================")
print(" COUNTING LINE")
print("================================================")

print(
    f"Point 1: {tuple(line_p1.astype(int))}"
)

print(
    f"Point 2: {tuple(line_p2.astype(int))}"
)

print("================================================")
print()


# ============================================================
# SIDE FUNCTION
# ============================================================

def line_side(
    point,
    p1,
    p2
):

    point = np.array(
        point,
        dtype=np.float32
    )

    value = (
        (p2[0] - p1[0])
        * (point[1] - p1[1])
        -
        (p2[1] - p1[1])
        * (point[0] - p1[0])
    )

    return value


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
# LOAD MODEL
# ============================================================

print()
print("================================================")
print(" LOADING YOLO11n")
print("================================================")

model = YOLO(
    MODEL_PATH
)


# ============================================================
# TRACK DATA
# ============================================================

# tracker ID -> display ID
display_ids = {}

next_display_id = 1


# tracker ID -> age
track_age = {}


# tracker ID -> last frame seen
last_seen = {}


# tracker ID -> previous center point
previous_points = {}


# tracker ID -> previous line side
previous_sides = {}


# tracker ID -> last confirmed region state
region_states = {}


# tracker ID -> whether already counted IN
counted_in = set()


# tracker ID -> whether already counted OUT
counted_out = set()


# tracker ID -> event cooldown
last_event_frame = {}


# ============================================================
# COUNTERS
# ============================================================

total_in = 0

total_out = 0

current_cows = 0


# ============================================================
# FPS
# ============================================================

fps_start = time.time()

fps_counter = 0

display_fps = 0.0


# ============================================================
# FRAME
# ============================================================

frame_number = 0


# ============================================================
# CSV
# ============================================================

csv_file = open(
    OUTPUT_CSV,
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
        "display_id",
        "tracker_id",
        "event",
        "total_in",
        "total_out",
        "net"
    ]
)


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
# HELPER: REGION
# ============================================================

def point_inside_region(
    point
):

    result = cv2.pointPolygonTest(
        region_polygon,
        (
            int(point[0]),
            int(point[1])
        ),
        False
    )

    return result >= 0


# ============================================================
# START
# ============================================================

print()
print("================================================")
print(" STARTING FINAL COW COUNTING TEST")
print("================================================")
print()
print("Model       : YOLO11n")
print("Tracker     : ByteTrack")
print("GPU         : CUDA device 0")
print("Image size  :", IMG_SIZE)
print("Confidence  :", CONFIDENCE)
print("Line based  : YES")
print("CSV logging : YES")
print()
print("Press Q to stop.")
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:

        break

    frame_number += 1


    # ========================================================
    # YOLO TRACK
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


    active_track_ids = set()

    active_inside_ids = set()


    # ========================================================
    # PROCESS TRACKS
    # ========================================================

    if (

        result.boxes is not None

        and result.boxes.id is not None

    ):

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        tracker_ids = (
            result.boxes.id
            .cpu()
            .numpy()
            .astype(int)
        )


        for box, tracker_id in zip(
            boxes,
            tracker_ids
        ):


            # ------------------------------------------------
            # ACTIVE
            # ------------------------------------------------

            active_track_ids.add(
                tracker_id
            )

            last_seen[
                tracker_id
            ] = frame_number


            # ------------------------------------------------
            # TRACK AGE
            # ------------------------------------------------

            track_age[
                tracker_id
            ] = track_age.get(
                tracker_id,
                0
            ) + 1


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
            # BOTTOM CENTER
            # ------------------------------------------------

            center = (
                int((x1 + x2) / 2),
                int(y2)
            )


            # =================================================
            # REGION STATE
            # =================================================

            inside = point_inside_region(
                center
            )


            region_states[
                tracker_id
            ] = inside


            if inside:

                active_inside_ids.add(
                    tracker_id
                )


            # =================================================
            # LINE SIDE
            # =================================================

            current_side = line_side(
                center,
                line_p1,
                line_p2
            )


            previous_side = previous_sides.get(
                tracker_id
            )


            previous_point = previous_points.get(
                tracker_id
            )


            # =================================================
            # INITIALIZE
            # =================================================

            if previous_side is None:

                previous_sides[
                    tracker_id
                ] = current_side

                previous_points[
                    tracker_id
                ] = center

            else:

                # =============================================
                # CROSSING TEST
                # =============================================

                crossed = (

                    (
                        previous_side < 0
                        and current_side > 0
                    )

                    or

                    (
                        previous_side > 0
                        and current_side < 0
                    )

                )


                if crossed:

                    # =========================================
                    # CHECK MOVEMENT
                    # =========================================

                    movement = 0

                    if previous_point is not None:

                        movement = np.linalg.norm(
                            np.array(center)
                            -
                            np.array(previous_point)
                        )


                    # =========================================
                    # STABLE TRACK REQUIRED
                    # =========================================

                    stable_track = (
                        track_age[
                            tracker_id
                        ]
                        >= MIN_TRACK_AGE
                    )


                    # =========================================
                    # ONLY COUNT STRONG CROSSINGS
                    # =========================================

                    if (

                        movement
                        >= MIN_CROSSING_DISTANCE

                        and stable_track

                    ):

                        # =====================================
                        # DETERMINE DIRECTION
                        #
                        # We use the region state around
                        # the crossing.
                        # =====================================

                        if inside:

                            event = "IN"

                        else:

                            event = "OUT"


                        # =====================================
                        # EVENT COOLDOWN
                        # =====================================

                        last_event = last_event_frame.get(
                            tracker_id,
                            -9999
                        )


                        if (
                            frame_number
                            -
                            last_event
                            >
                            15
                        ):

                            # =================================
                            # IN
                            # =================================

                            if event == "IN":

                                if tracker_id not in counted_in:

                                    counted_in.add(
                                        tracker_id
                                    )

                                    total_in += 1

                                    last_event_frame[
                                        tracker_id
                                    ] = frame_number


                                    print()
                                    print(
                                        ">>> COW ENTERED"
                                    )

                                    print(
                                        f"Display : "
                                        f"{display_id}"
                                    )

                                    print(
                                        f"Frame   : "
                                        f"{frame_number}"
                                    )

                                    print(
                                        f"IN      : "
                                        f"{total_in}"
                                    )

                                    print(
                                        f"OUT     : "
                                        f"{total_out}"
                                    )


                                    csv_writer.writerow(
                                        [
                                            frame_number,
                                            display_id,
                                            tracker_id,
                                            "IN",
                                            total_in,
                                            total_out,
                                            total_in - total_out
                                        ]
                                    )

                                    csv_file.flush()


                            # =================================
                            # OUT
                            # =================================

                            elif event == "OUT":

                                if tracker_id not in counted_out:

                                    counted_out.add(
                                        tracker_id
                                    )

                                    total_out += 1

                                    last_event_frame[
                                        tracker_id
                                    ] = frame_number


                                    print()
                                    print(
                                        ">>> COW EXITED"
                                    )

                                    print(
                                        f"Display : "
                                        f"{display_id}"
                                    )

                                    print(
                                        f"Frame   : "
                                        f"{frame_number}"
                                    )

                                    print(
                                        f"IN      : "
                                        f"{total_in}"
                                    )

                                    print(
                                        f"OUT     : "
                                        f"{total_out}"
                                    )


                                    csv_writer.writerow(
                                        [
                                            frame_number,
                                            display_id,
                                            tracker_id,
                                            "OUT",
                                            total_in,
                                            total_out,
                                            total_in - total_out
                                        ]
                                    )

                                    csv_file.flush()


                # =============================================
                # UPDATE SIDE
                # =============================================

                previous_sides[
                    tracker_id
                ] = current_side

                previous_points[
                    tracker_id
                ] = center


            # =================================================
            # DRAW BOX
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

                2
            )


            # =================================================
            # DISPLAY NUMBER
            # =================================================

            cv2.putText(

                frame,

                f"COW {display_id}",

                (
                    x1,
                    max(
                        y1 - 10,
                        25
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                box_color,

                2,

                cv2.LINE_AA
            )


            # =================================================
            # BOTTOM CENTER
            # =================================================

            cv2.circle(

                frame,

                center,

                5,

                box_color,

                -1
            )


    # ========================================================
    # LOST TRACK PROTECTION
    # ========================================================

    lost_ids = []

    for tracker_id, last_frame in last_seen.items():

        if (
            frame_number
            -
            last_frame
            >
            MAX_MISSING_FRAMES
        ):

            lost_ids.append(
                tracker_id
            )


    for tracker_id in lost_ids:

        last_seen.pop(
            tracker_id,
            None
        )

        previous_points.pop(
            tracker_id,
            None
        )

        previous_sides.pop(
            tracker_id,
            None
        )

        region_states.pop(
            tracker_id,
            None
        )

        track_age.pop(
            tracker_id,
            None
        )


    # ========================================================
    # CURRENT COWS
    # ========================================================

    current_cows = len(
        active_inside_ids
    )


    # ========================================================
    # NET
    # ========================================================

    net_count = (
        total_in
        -
        total_out
    )


    # ========================================================
    # FPS
    # ========================================================

    fps_counter += 1

    elapsed = (
        time.time()
        -
        fps_start
    )


    if elapsed >= 1.0:

        display_fps = (
            fps_counter
            /
            elapsed
        )

        fps_counter = 0

        fps_start = time.time()


    # ========================================================
    # DRAW REGION
    # ========================================================

    cv2.polylines(

        frame,

        [
            region_polygon
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
    # DRAW COUNTING LINE
    # ========================================================

    cv2.line(

        frame,

        tuple(
            line_p1.astype(int)
        ),

        tuple(
            line_p2.astype(int)
        ),

        (
            0,
            0,
            255
        ),

        5
    )


    # ========================================================
    # LINE LABEL
    # ========================================================

    line_mid_x = int(
        (line_p1[0] + line_p2[0]) / 2
    )

    line_mid_y = int(
        (line_p1[1] + line_p2[1]) / 2
    )


    cv2.putText(

        frame,

        "COUNT LINE",

        (
            line_mid_x + 10,
            line_mid_y - 10
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (
            0,
            0,
            255
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

    panel_height = 205


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

    cv2.putText(

        frame,

        "COW MONITORING",

        (
            panel_x + 15,
            panel_y + 30
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

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

        f"CURRENT : {current_cows}",

        (
            panel_x + 15,
            panel_y + 65
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

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

        f"IN      : {total_in}",

        (
            panel_x + 15,
            panel_y + 95
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

        f"OUT     : {total_out}",

        (
            panel_x + 15,
            panel_y + 125
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

        f"NET     : {net_count}",

        (
            panel_x + 15,
            panel_y + 155
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

        f"FPS     : {display_fps:.1f}",

        (
            panel_x + 15,
            panel_y + 185
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

        f"FRAME : {frame_number}/{total_frames}",

        (
            panel_x + 185,
            panel_y + 185
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.50,

        (
            255,
            255,
            255
        ),

        1,

        cv2.LINE_AA
    )


    # ========================================================
    # TERMINAL
    # ========================================================

    print(

        f"Frame: {frame_number:04d} | "
        f"Current: {current_cows} | "
        f"IN: {total_in} | "
        f"OUT: {total_out} | "
        f"Net: {net_count} | "
        f"FPS: {display_fps:.1f}"

    )


    # ========================================================
    # SAVE OUTPUT
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

        "COW COUNTING - FINAL TEST",

        display_frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        print()
        print("Stopping...")

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
print("================================================")
print(" FINAL COW COUNTING REPORT")
print("================================================")

print(
    f"Frames processed : {frame_number}"
)

print(
    f"Total IN         : {total_in}"
)

print(
    f"Total OUT        : {total_out}"
)

print(
    f"Net count        : {net_count}"
)

print(
    f"Current cows     : {current_cows}"
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
    "Event CSV:"
)

print(
    OUTPUT_CSV
)

print("================================================")
