import cv2
import os
import time
import csv
import numpy as np
from ultralytics import YOLO


# ============================================================
# TEST 5
# PREDEFINED LINES
#
# YOLO11m
# ByteTrack
# GPU
# 768 inference
# Fixed monitoring region
# Fixed counting lines
# Directional IN / OUT
# Lost-track protection
# CSV event log
# ============================================================


# ============================================================
# 1. VIDEO / MODEL
# ============================================================

VIDEO_PATH = "videos/cow_video10.mp4"

MODEL_PATH = "yolo11m.pt"

OUTPUT_PATH = "output/test5_predefined_lines.mp4"

CSV_PATH = "output/test5_events.csv"


# ============================================================
# 2. YOLO SETTINGS
# ============================================================

COW_CLASS_ID = 19

CONFIDENCE = 0.10

IMG_SIZE = 768

DEVICE = 0

TRACKER = "bytetrack.yaml"


# ============================================================
# 3. TRACKING SETTINGS
# ============================================================

MAX_MISSING_FRAMES = 60

MIN_MOVEMENT = 3

LINE_SEQUENCE_TIMEOUT = 180


# ============================================================
# 4. ORIGINAL VIDEO COORDINATES
#
# Video:
# 1920 x 1080
#
# These coordinates are designed for the fixed camera
# shown in your screenshot.
# ============================================================


# ------------------------------------------------------------
# MONITORING REGION
# ------------------------------------------------------------

REGION_POINTS = np.array(
    [
        (190, 300),
        (1300, 330),
        (1300, 600),
        (180, 540)
    ],
    dtype=np.int32
)


# ------------------------------------------------------------
# LINE A
#
# Upper counting line
# ------------------------------------------------------------

LINE_A_P1 = (200, 350)

LINE_A_P2 = (1300, 385)


# ------------------------------------------------------------
# LINE B
#
# Lower counting line
# ------------------------------------------------------------

LINE_B_P1 = (185, 465)

LINE_B_P2 = (1300, 515)


# ============================================================
# DISPLAY
# ============================================================

DISPLAY_WIDTH = 1100

DISPLAY_HEIGHT = 700


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    "output",
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
    print("ERROR: Could not open video:")
    print(VIDEO_PATH)
    print()

    exit()


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
print("=" * 65)
print(" TEST 5 - PREDEFINED LINES")
print("=" * 65)

print(
    f"Video       : {VIDEO_PATH}"
)

print(
    f"Resolution  : {video_width} x {video_height}"
)

print(
    f"Video FPS   : {video_fps:.2f}"
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
    f"Device      : GPU {DEVICE}"
)

print(
    f"Tracker     : {TRACKER}"
)

print("=" * 65)

print()


# ============================================================
# IMPORTANT:
# SCALE PREDEFINED COORDINATES IF VIDEO IS NOT 1920x1080
# ============================================================

BASE_WIDTH = 1920.0

BASE_HEIGHT = 1080.0


scale_x = video_width / BASE_WIDTH

scale_y = video_height / BASE_HEIGHT


def scale_point(point):

    return (
        int(point[0] * scale_x),
        int(point[1] * scale_y)
    )


# ============================================================
# SCALED REGION
# ============================================================

polygon = np.array(
    [
        scale_point(
            tuple(point)
        )
        for point in REGION_POINTS
    ],
    dtype=np.int32
)


# ============================================================
# SCALED LINES
# ============================================================

line_a_p1 = np.array(
    scale_point(
        LINE_A_P1
    ),
    dtype=np.float32
)

line_a_p2 = np.array(
    scale_point(
        LINE_A_P2
    ),
    dtype=np.float32
)


line_b_p1 = np.array(
    scale_point(
        LINE_B_P1
    ),
    dtype=np.float32
)

line_b_p2 = np.array(
    scale_point(
        LINE_B_P2
    ),
    dtype=np.float32
)


# ============================================================
# PRINT GEOMETRY
# ============================================================

print()
print("=" * 65)
print(" PREDEFINED GEOMETRY")
print("=" * 65)

print()
print("Monitoring region:")

for i, point in enumerate(
    polygon,
    1
):

    print(
        f"R{i}: {tuple(point)}"
    )


print()

print(
    "LINE A:"
)

print(
    f"{tuple(line_a_p1.astype(int))}"
    f" -> "
    f"{tuple(line_a_p2.astype(int))}"
)


print()

print(
    "LINE B:"
)

print(
    f"{tuple(line_b_p1.astype(int))}"
    f" -> "
    f"{tuple(line_b_p2.astype(int))}"
)


print()

print(
    "Expected direction:"
)

print(
    "LINE A -> LINE B = IN"
)

print(
    "LINE B -> LINE A = OUT"
)

print("=" * 65)

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
    OUTPUT_PATH,
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

    exit()


# ============================================================
# LOAD YOLO11m
# ============================================================

print()
print("=" * 65)
print(" LOADING YOLO11m")
print("=" * 65)
print()


model = YOLO(
    MODEL_PATH
)


print(
    "YOLO11m loaded successfully."
)

print()


# ============================================================
# TRACK DATA
# ============================================================


# Last known bottom-center point
last_points = {}


# Last frame where ID was seen
last_seen = {}


# Trajectory history
trajectories = {}


# ============================================================
# LINE SIDE STATES
# ============================================================

last_side_a = {}

last_side_b = {}


# ============================================================
# CROSSING SEQUENCE
#
# None
# A
# B
#
# A means cow has crossed LINE A first.
# B means cow has crossed LINE B first.
# ============================================================

first_line = {}


# Frame where first line was crossed
first_line_frame = {}


# ============================================================
# COUNTED IDs
# ============================================================

counted_ids = set()


# ============================================================
# REGION STATE
# ============================================================

inside_ids = set()


# ============================================================
# COUNTERS
# ============================================================

total_in = 0

total_out = 0


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
        "event",
        "direction"
    ]
)


# ============================================================
# FPS
# ============================================================

fps_start = time.time()

fps_frames = 0

display_fps = 0.0


# ============================================================
# FRAME
# ============================================================

frame_number = 0


# ============================================================
# GEOMETRY
# ============================================================


def line_side(
    point,
    p1,
    p2
):

    return (

        (p2[0] - p1[0])
        * (point[1] - p1[1])

        -

        (p2[1] - p1[1])
        * (point[0] - p1[0])

    )


# ============================================================
# CROSSING DETECTION
# ============================================================


def crossed_line(
    previous_point,
    current_point,
    p1,
    p2
):

    old_side = line_side(
        previous_point,
        p1,
        p2
    )

    new_side = line_side(
        current_point,
        p1,
        p2
    )


    # Same side
    if (
        old_side == 0
        or new_side == 0
    ):

        return False


    return (
        old_side * new_side < 0
    )


# ============================================================
# REGION
# ============================================================


def point_inside_region(
    point
):

    result = cv2.pointPolygonTest(

        polygon,

        (
            int(point[0]),
            int(point[1])
        ),

        False
    )


    return result >= 0


# ============================================================
# MAIN LOOP
# ============================================================

print()
print("=" * 65)
print(" STARTING TEST 5")
print("=" * 65)
print()

print(
    "Predefined lines enabled."
)

print(
    "No manual point selection required."
)

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
    # ACTIVE IDS THIS FRAME
    # ========================================================

    active_ids = set()


    # ========================================================
    # YOLO + BYTETRACK
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


    detections = 0


    # ========================================================
    # DETECTIONS
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


        ids = (

            result.boxes.id
            .cpu()
            .numpy()
            .astype(int)

        )


        detections = len(
            ids
        )


        # ====================================================
        # PROCESS EVERY COW
        # ====================================================

        for box, tracker_id in zip(
            boxes,
            ids
        ):


            tracker_id = int(
                tracker_id
            )


            active_ids.add(
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

            current_point = (

                int(
                    (x1 + x2) / 2
                ),

                int(
                    y2
                )

            )


            # ------------------------------------------------
            # PREVIOUS POINT
            # ------------------------------------------------

            previous_point = last_points.get(
                tracker_id
            )


            # ------------------------------------------------
            # SAVE POINT
            # ------------------------------------------------

            last_points[
                tracker_id
            ] = current_point


            last_seen[
                tracker_id
            ] = frame_number


            # ------------------------------------------------
            # TRAJECTORY
            # ------------------------------------------------

            if tracker_id not in trajectories:

                trajectories[
                    tracker_id
                ] = []


            trajectories[
                tracker_id
            ].append(
                current_point
            )


            if len(
                trajectories[
                    tracker_id
                ]
            ) > 25:

                trajectories[
                    tracker_id
                ] = trajectories[
                    tracker_id
                ][-25:]


            # =================================================
            # CURRENT COW
            # =================================================

            if point_inside_region(
                current_point
            ):

                inside_ids.add(
                    tracker_id
                )


            # =================================================
            # LINE PROCESSING
            # =================================================

            if previous_point is not None:


                movement = (

                    abs(
                        current_point[0]
                        - previous_point[0]
                    )

                    +

                    abs(
                        current_point[1]
                        - previous_point[1]
                    )

                )


                if movement >= MIN_MOVEMENT:


                    # =========================================
                    # CURRENT LINE SIDES
                    # =========================================

                    current_a_side = line_side(

                        current_point,

                        line_a_p1,

                        line_a_p2

                    )


                    current_b_side = line_side(

                        current_point,

                        line_b_p1,

                        line_b_p2

                    )


                    # =========================================
                    # PREVIOUS LINE SIDES
                    # =========================================

                    previous_a_side = last_side_a.get(
                        tracker_id
                    )


                    previous_b_side = last_side_b.get(
                        tracker_id
                    )


                    # =========================================
                    # FIRST FRAME
                    # =========================================

                    if previous_a_side is None:

                        last_side_a[
                            tracker_id
                        ] = current_a_side

                        last_side_b[
                            tracker_id
                        ] = current_b_side

                    else:


                        # =====================================
                        # LINE A CROSSING
                        # =====================================

                        crossed_a = (

                            previous_a_side
                            * current_a_side
                            < 0

                        )


                        # =====================================
                        # LINE B CROSSING
                        # =====================================

                        crossed_b = (

                            previous_b_side
                            * current_b_side
                            < 0

                        )


                        # =====================================
                        # A CROSSED
                        # =====================================

                        if crossed_a:


                            # ---------------------------------
                            # If B was crossed first:
                            #
                            # B -> A = OUT
                            # ---------------------------------

                            if (

                                first_line.get(
                                    tracker_id
                                )
                                == "B"

                                and tracker_id
                                not in counted_ids

                            ):


                                elapsed_frames = (

                                    frame_number
                                    -

                                    first_line_frame.get(
                                        tracker_id,
                                        frame_number
                                    )

                                )


                                if (
                                    elapsed_frames
                                    <= LINE_SEQUENCE_TIMEOUT
                                ):


                                    total_out += 1


                                    counted_ids.add(
                                        tracker_id
                                    )


                                    csv_writer.writerow(
                                        [
                                            frame_number,
                                            tracker_id,
                                            "OUT",
                                            "B_TO_A"
                                        ]
                                    )


                                    csv_file.flush()


                                    print()
                                    print(
                                        ">>> COW OUT"
                                    )

                                    print(
                                        f"Frame     : "
                                        f"{frame_number}"
                                    )

                                    print(
                                        f"Track     : "
                                        f"{tracker_id}"
                                    )

                                    print(
                                        f"IN        : "
                                        f"{total_in}"
                                    )

                                    print(
                                        f"OUT       : "
                                        f"{total_out}"
                                    )


                                    # Reset
                                    first_line.pop(
                                        tracker_id,
                                        None
                                    )

                                    first_line_frame.pop(
                                        tracker_id,
                                        None
                                    )


                            # ---------------------------------
                            # Otherwise A is first
                            # ---------------------------------

                            elif (

                                first_line.get(
                                    tracker_id
                                )
                                is None

                            ):


                                first_line[
                                    tracker_id
                                ] = "A"


                                first_line_frame[
                                    tracker_id
                                ] = frame_number


                        # =====================================
                        # B CROSSED
                        # =====================================

                        if crossed_b:


                            # ---------------------------------
                            # If A was crossed first:
                            #
                            # A -> B = IN
                            # ---------------------------------

                            if (

                                first_line.get(
                                    tracker_id
                                )
                                == "A"

                                and tracker_id
                                not in counted_ids

                            ):


                                elapsed_frames = (

                                    frame_number
                                    -

                                    first_line_frame.get(
                                        tracker_id,
                                        frame_number
                                    )

                                )


                                if (
                                    elapsed_frames
                                    <= LINE_SEQUENCE_TIMEOUT
                                ):


                                    total_in += 1


                                    counted_ids.add(
                                        tracker_id
                                    )


                                    csv_writer.writerow(
                                        [
                                            frame_number,
                                            tracker_id,
                                            "IN",
                                            "A_TO_B"
                                        ]
                                    )


                                    csv_file.flush()


                                    print()
                                    print(
                                        ">>> COW IN"
                                    )

                                    print(
                                        f"Frame     : "
                                        f"{frame_number}"
                                    )

                                    print(
                                        f"Track     : "
                                        f"{tracker_id}"
                                    )

                                    print(
                                        f"IN        : "
                                        f"{total_in}"
                                    )

                                    print(
                                        f"OUT       : "
                                        f"{total_out}"
                                    )


                                    # Reset
                                    first_line.pop(
                                        tracker_id,
                                        None
                                    )

                                    first_line_frame.pop(
                                        tracker_id,
                                        None
                                    )


                            # ---------------------------------
                            # Otherwise B is first
                            # ---------------------------------

                            elif (

                                first_line.get(
                                    tracker_id
                                )
                                is None

                            ):


                                first_line[
                                    tracker_id
                                ] = "B"


                                first_line_frame[
                                    tracker_id
                                ] = frame_number


                        # =====================================
                        # UPDATE SIDES
                        # =====================================

                        last_side_a[
                            tracker_id
                        ] = current_a_side


                        last_side_b[
                            tracker_id
                        ] = current_b_side


            # =================================================
            # DRAW BOX
            # =================================================

            if tracker_id in inside_ids:

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

                3
            )


            # =================================================
            # BOTTOM CENTER POINT
            # =================================================

            cv2.circle(

                frame,

                current_point,

                5,

                box_color,

                -1
            )


            # =================================================
            # TRAJECTORY
            # =================================================

            trajectory = trajectories[
                tracker_id
            ]


            if len(
                trajectory
            ) > 1:


                for i in range(
                    1,
                    len(trajectory)
                ):


                    cv2.line(

                        frame,

                        trajectory[i - 1],

                        trajectory[i],

                        (
                            255,
                            255,
                            0
                        ),

                        2
                    )


    # ========================================================
    # REMOVE OLD TRACK DATA
    #
    # IMPORTANT:
    # We wait 60 frames before deleting an ID.
    # ========================================================

    for tracker_id in list(
        last_seen.keys()
    ):


        missing_frames = (

            frame_number
            -

            last_seen[
                tracker_id
            ]

        )


        if missing_frames > MAX_MISSING_FRAMES:


            last_points.pop(
                tracker_id,
                None
            )


            last_seen.pop(
                tracker_id,
                None
            )


            trajectories.pop(
                tracker_id,
                None
            )


            last_side_a.pop(
                tracker_id,
                None
            )


            last_side_b.pop(
                tracker_id,
                None
            )


            first_line.pop(
                tracker_id,
                None
            )


            first_line_frame.pop(
                tracker_id,
                None
            )


    # ========================================================
    # CLEAR EXPIRED SEQUENCES
    # ========================================================

    for tracker_id in list(
        first_line_frame.keys()
    ):


        if (

            frame_number
            -

            first_line_frame[
                tracker_id
            ]

            >

            LINE_SEQUENCE_TIMEOUT

        ):


            first_line.pop(
                tracker_id,
                None
            )


            first_line_frame.pop(
                tracker_id,
                None
            )


    # ========================================================
    # CURRENT COW COUNT
    # ========================================================

    current_cows = len(
        inside_ids
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
    # DRAW LINE A
    # ========================================================

    cv2.line(

        frame,

        tuple(
            line_a_p1.astype(int)
        ),

        tuple(
            line_a_p2.astype(int)
        ),

        (
            0,
            0,
            255
        ),

        5
    )


    # ========================================================
    # DRAW LINE B
    # ========================================================

    cv2.line(

        frame,

        tuple(
            line_b_p1.astype(int)
        ),

        tuple(
            line_b_p2.astype(int)
        ),

        (
            255,
            0,
            255
        ),

        5
    )


    # ========================================================
    # LABEL LINE A
    # ========================================================

    line_a_center = (

        int(
            (
                line_a_p1[0]
                +
                line_a_p2[0]
            )
            / 2
        ),

        int(
            (
                line_a_p1[1]
                +
                line_a_p2[1]
            )
            / 2
        )

    )


    cv2.putText(

        frame,

        "LINE A",

        line_a_center,

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (
            0,
            0,
            255
        ),

        3,

        cv2.LINE_AA
    )


    # ========================================================
    # LABEL LINE B
    # ========================================================

    line_b_center = (

        int(
            (
                line_b_p1[0]
                +
                line_b_p2[0]
            )
            / 2
        ),

        int(
            (
                line_b_p1[1]
                +
                line_b_p2[1]
            )
            / 2
        )

    )


    cv2.putText(

        frame,

        "LINE B",

        line_b_center,

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (
            255,
            0,
            255
        ),

        3,

        cv2.LINE_AA
    )


    # ========================================================
    # INFO PANEL
    # ========================================================

    panel_x = 20

    panel_y = 20

    panel_width = 390

    panel_height = 220


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

        0.72,

        frame,

        0.28,

        0
    )


    # ========================================================
    # PANEL TITLE
    # ========================================================

    cv2.putText(

        frame,

        "COW COUNTING - TEST 5",

        (
            panel_x + 15,
            panel_y + 30
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.68,

        (
            255,
            255,
            255
        ),

        2,

        cv2.LINE_AA
    )


    # ========================================================
    # CURRENT
    # ========================================================

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


    # ========================================================
    # IN
    # ========================================================

    cv2.putText(

        frame,

        f"IN      : {total_in}",

        (
            panel_x + 15,
            panel_y + 100
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


    # ========================================================
    # OUT
    # ========================================================

    cv2.putText(

        frame,

        f"OUT     : {total_out}",

        (
            panel_x + 15,
            panel_y + 135
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


    # ========================================================
    # NET
    # ========================================================

    cv2.putText(

        frame,

        f"NET     : {net_count}",

        (
            panel_x + 15,
            panel_y + 170
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


    # ========================================================
    # FPS
    # ========================================================

    cv2.putText(

        frame,

        f"FPS     : {display_fps:.1f}",

        (
            panel_x + 15,
            panel_y + 205
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


    # ========================================================
    # DETECTIONS
    # ========================================================

    cv2.putText(

        frame,

        f"DET     : {detections}",

        (
            panel_x + 210,
            panel_y + 205
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


    # ========================================================
    # TERMINAL OUTPUT
    # ========================================================

    print(

        f"Frame: {frame_number:04d} | "
        f"Detections: {detections:2d} | "
        f"Current: {current_cows:2d} | "
        f"IN: {total_in:2d} | "
        f"OUT: {total_out:2d} | "
        f"NET: {net_count:3d} | "
        f"FPS: {display_fps:.1f}"

    )


    # ========================================================
    # WRITE OUTPUT
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

        "TEST 5 - PREDEFINED LINES",

        display_frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        print()
        print(
            "Stopping Test 5..."
        )

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
print("=" * 65)
print(" TEST 5 FINISHED")
print("=" * 65)

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
    OUTPUT_PATH
)

print()

print(
    "CSV event log:"
)

print(
    CSV_PATH
)

print("=" * 65)
