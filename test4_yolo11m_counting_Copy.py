import cv2
import os
import time
import csv
import numpy as np
from ultralytics import YOLO


# ============================================================
# TEST 4
# YOLO11m + 768 + ByteTrack + 2-LINE COUNTING
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

# CHANGE THIS TO YOUR NEW TEST VIDEO
VIDEO_PATH = "videos/cow_video1.mp4"

MODEL_PATH = "yolo11m.pt"

OUTPUT_PATH = "output/test4_cow_counting.mp4"

CSV_PATH = "output/test4_events.csv"


# COCO cow class
COW_CLASS_ID = 19


# ------------------------------------------------------------
# Detection
# ------------------------------------------------------------

CONFIDENCE = 0.10

IMG_SIZE = 768

DEVICE = 0


# ------------------------------------------------------------
# Tracker
# ------------------------------------------------------------

TRACKER = "bytetrack.yaml"


# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

DISPLAY_WIDTH = 1100
DISPLAY_HEIGHT = 700


# ------------------------------------------------------------
# Region protection
# ------------------------------------------------------------

BOUNDARY_MARGIN = 20


# ------------------------------------------------------------
# State confirmation
# ------------------------------------------------------------

STATE_CONFIRM_FRAMES = 3


# ------------------------------------------------------------
# Lost track protection
# ------------------------------------------------------------

MAX_MISSING_FRAMES = 60


# ------------------------------------------------------------
# Minimum movement required for crossing
# ------------------------------------------------------------

MIN_LINE_MOVEMENT = 3


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs("output", exist_ok=True)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print()
    print("ERROR: Could not open video")
    print(VIDEO_PATH)
    print()

    exit()


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
print("=" * 60)
print(" TEST 4 - YOLO11m + 2-LINE COUNTING")
print("=" * 60)
print(f"Video       : {VIDEO_PATH}")
print(f"Resolution  : {video_width} x {video_height}")
print(f"Video FPS   : {video_fps:.2f}")
print(f"Frames      : {total_frames}")
print(f"Model       : {MODEL_PATH}")
print(f"Image size  : {IMG_SIZE}")
print(f"Confidence  : {CONFIDENCE}")
print(f"Device      : GPU {DEVICE}")
print(f"Tracker     : {TRACKER}")
print("=" * 60)
print()


# ============================================================
# READ FIRST FRAME
# ============================================================

ret, first_frame = cap.read()

if not ret:

    print("ERROR: Could not read first frame.")

    cap.release()

    exit()


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
# REGION POINTS
# ============================================================

region_points = []


# ============================================================
# COUNTING LINE POINTS
#
# User selects:
#
# Point 1 + Point 2 = LINE A
# Point 3 + Point 4 = LINE B
#
# These are separate from the 4-point monitoring region.
# ============================================================

line_points = []


# ============================================================
# SELECTION MODE
#
# First select 4 region points.
# Then select 4 line points.
# ============================================================

selection_mode = "region"


# ============================================================
# DRAW SELECTION
# ============================================================

def draw_selection():

    global selection_frame

    selection_frame = original_selection_frame.copy()


    # --------------------------------------------------------
    # REGION
    # --------------------------------------------------------

    if len(region_points) > 0:

        for i, p in enumerate(region_points):

            x = int(p[0] * selection_scale)
            y = int(p[1] * selection_scale)

            cv2.circle(
                selection_frame,
                (x, y),
                7,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                selection_frame,
                f"R{i + 1}",
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )


        if len(region_points) > 1:

            for i in range(
                len(region_points) - 1
            ):

                p1 = region_points[i]
                p2 = region_points[i + 1]

                cv2.line(
                    selection_frame,
                    (
                        int(p1[0] * selection_scale),
                        int(p1[1] * selection_scale)
                    ),
                    (
                        int(p2[0] * selection_scale),
                        int(p2[1] * selection_scale)
                    ),
                    (0, 255, 0),
                    3
                )


        if len(region_points) == 4:

            p1 = region_points[0]
            p2 = region_points[-1]

            cv2.line(
                selection_frame,
                (
                    int(p1[0] * selection_scale),
                    int(p1[1] * selection_scale)
                ),
                (
                    int(p2[0] * selection_scale),
                    int(p2[1] * selection_scale)
                ),
                (0, 255, 255),
                3
            )


    # --------------------------------------------------------
    # COUNTING LINES
    # --------------------------------------------------------

    for i, p in enumerate(line_points):

        x = int(p[0] * selection_scale)
        y = int(p[1] * selection_scale)

        cv2.circle(
            selection_frame,
            (x, y),
            7,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            selection_frame,
            f"L{i + 1}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )


    if len(line_points) >= 2:

        cv2.line(
            selection_frame,
            (
                int(line_points[0][0] * selection_scale),
                int(line_points[0][1] * selection_scale)
            ),
            (
                int(line_points[1][0] * selection_scale),
                int(line_points[1][1] * selection_scale)
            ),
            (0, 0, 255),
            4
        )


    if len(line_points) == 4:

        cv2.line(
            selection_frame,
            (
                int(line_points[2][0] * selection_scale),
                int(line_points[2][1] * selection_scale)
            ),
            (
                int(line_points[3][0] * selection_scale),
                int(line_points[3][1] * selection_scale)
            ),
            (255, 0, 255),
            4
        )


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global selection_mode

    if event != cv2.EVENT_LBUTTONDOWN:
        return


    original_x = int(
        x / selection_scale
    )

    original_y = int(
        y / selection_scale
    )


    # ========================================================
    # REGION
    # ========================================================

    if selection_mode == "region":

        if len(region_points) >= 4:
            return

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

        draw_selection()


        if len(region_points) == 4:

            print()
            print(
                "4 region points selected."
            )

            print(
                "Press ENTER to continue to counting lines."
            )


    # ========================================================
    # COUNTING LINES
    # ========================================================

    elif selection_mode == "lines":

        if len(line_points) >= 4:
            return

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

        draw_selection()


# ============================================================
# SELECTION WINDOW
# ============================================================

cv2.namedWindow(
    "TEST 4 - SELECT REGION",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "TEST 4 - SELECT REGION",
    selection_width,
    selection_height
)

cv2.setMouseCallback(
    "TEST 4 - SELECT REGION",
    mouse_callback
)


print()
print("=" * 60)
print(" REGION SELECTION")
print("=" * 60)
print()
print("Select FOUR points around the monitoring area.")
print()
print("R = reset")
print("ENTER = continue")
print("Q = quit")
print()


# ============================================================
# REGION SELECTION
# ============================================================

while True:

    cv2.imshow(
        "TEST 4 - SELECT REGION",
        selection_frame
    )

    key = cv2.waitKey(1) & 0xFF


    if key == ord("r"):

        region_points.clear()
        line_points.clear()

        selection_mode = "region"

        draw_selection()

        print("Selection reset.")


    elif key == 13:

        if (
            selection_mode == "region"
            and len(region_points) == 4
        ):

            selection_mode = "lines"

            print()
            print("=" * 60)
            print(" COUNTING LINE SELECTION")
            print("=" * 60)
            print()
            print("Select LINE A using 2 points.")
            print("Then select LINE B using 2 points.")
            print()
            print("L1 + L2 = LINE A")
            print("L3 + L4 = LINE B")
            print()
            print("ENTER = confirm")
            print("R = reset")
            print("Q = quit")
            print()


        elif (
            selection_mode == "lines"
            and len(line_points) == 4
        ):

            break


    elif key == ord("q"):

        cap.release()
        cv2.destroyAllWindows()

        exit()


cv2.destroyAllWindows()


# ============================================================
# CREATE REGION
# ============================================================

polygon = np.array(
    region_points,
    dtype=np.int32
)


# ============================================================
# CREATE COUNTING LINES
# ============================================================

line_a_p1 = np.array(
    line_points[0],
    dtype=np.float32
)

line_a_p2 = np.array(
    line_points[1],
    dtype=np.float32
)

line_b_p1 = np.array(
    line_points[2],
    dtype=np.float32
)

line_b_p2 = np.array(
    line_points[3],
    dtype=np.float32
)


print()
print("=" * 60)
print(" REGION")
print("=" * 60)

for i, point in enumerate(
    region_points,
    1
):

    print(
        f"R{i}: {point}"
    )


print()
print("LINE A:")

print(
    f"{tuple(line_points[0])}"
    f" -> "
    f"{tuple(line_points[1])}"
)


print()
print("LINE B:")

print(
    f"{tuple(line_points[2])}"
    f" -> "
    f"{tuple(line_points[3])}"
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
# LOAD MODEL
# ============================================================

print()
print("=" * 60)
print(" LOADING YOLO11m")
print("=" * 60)
print()


model = YOLO(
    MODEL_PATH
)


# ============================================================
# TRACK DATA
# ============================================================

# tracker_id -> history of points
trajectories = {}


# tracker_id -> last frame
last_seen = {}


# tracker_id -> last point
last_points = {}


# tracker_id -> confirmed region state
region_states = {}


# tracker_id -> candidate region state
candidate_states = {}


# tracker_id -> candidate frame count
candidate_counts = {}


# tracker_id -> line A crossed
crossed_a = set()


# tracker_id -> line B crossed
crossed_b = set()


# tracker_id -> counting direction stage
#
# "none"
# "a_to_b"
# "b_to_a"
#
crossing_stage = {}


# IDs already counted
counted_ids = set()


# ============================================================
# COUNTERS
# ============================================================

total_in = 0
total_out = 0


# ============================================================
# CURRENT ACTIVE COWS
# ============================================================

current_inside_ids = set()


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
# FRAME NUMBER
# ============================================================

frame_number = 0


# ============================================================
# GEOMETRY FUNCTIONS
# ============================================================

def point_side(
    point,
    line_p1,
    line_p2
):

    return (

        (line_p2[0] - line_p1[0])
        * (point[1] - line_p1[1])

        -

        (line_p2[1] - line_p1[1])
        * (point[0] - line_p1[0])

    )


# ============================================================
# LINE CROSSING
# ============================================================

def crossed_line(
    previous_point,
    current_point,
    line_p1,
    line_p2
):

    previous_side = point_side(
        previous_point,
        line_p1,
        line_p2
    )

    current_side = point_side(
        current_point,
        line_p1,
        line_p2
    )


    # Same side
    if (
        previous_side == 0
        or current_side == 0
    ):

        return False


    return (
        previous_side * current_side < 0
    )


# ============================================================
# REGION STATE
# ============================================================

def get_region_state(point):

    distance = cv2.pointPolygonTest(
        polygon,
        (
            int(point[0]),
            int(point[1])
        ),
        True
    )


    if distance > BOUNDARY_MARGIN:
        return "inside"


    if distance < -BOUNDARY_MARGIN:
        return "outside"


    return "boundary"


# ============================================================
# START
# ============================================================

print()
print("=" * 60)
print(" STARTING TEST 4")
print("=" * 60)
print()
print("YOLO11m")
print("ByteTrack")
print("768 inference")
print("GPU 0")
print("2-line directional counting")
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


    active_track_ids = set()

    current_inside_ids = set()


    # ========================================================
    # YOLO TRACKING
    # ========================================================

    results = model.track(

        frame,

        persist=True,

        tracker=TRACKER,

        classes=[COW_CLASS_ID],

        conf=CONFIDENCE,

        imgsz=IMG_SIZE,

        device=DEVICE,

        verbose=False
    )


    result = results[0]


    detections = 0


    # ========================================================
    # PROCESS DETECTIONS
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


        detections = len(
            tracker_ids
        )


        for box, tracker_id in zip(
            boxes,
            tracker_ids
        ):


            tracker_id = int(
                tracker_id
            )


            active_track_ids.add(
                tracker_id
            )


            x1, y1, x2, y2 = map(
                int,
                box
            )


            # ------------------------------------------------
            # BOTTOM CENTER
            # ------------------------------------------------

            point = (
                int((x1 + x2) / 2),
                int(y2)
            )


            # ------------------------------------------------
            # LAST DATA
            # ------------------------------------------------

            previous_point = last_points.get(
                tracker_id
            )


            last_points[
                tracker_id
            ] = point


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
                point
            )


            # Keep last 30 points

            if len(
                trajectories[
                    tracker_id
                ]
            ) > 30:

                trajectories[
                    tracker_id
                ] = trajectories[
                    tracker_id
                ][-30:]


            # =================================================
            # REGION STATE
            # =================================================

            raw_state = get_region_state(
                point
            )


            if tracker_id not in region_states:

                if raw_state == "boundary":

                    region_states[
                        tracker_id
                    ] = "outside"

                else:

                    region_states[
                        tracker_id
                    ] = raw_state


                candidate_states[
                    tracker_id
                ] = None


                candidate_counts[
                    tracker_id
                ] = 0


            previous_region = region_states[
                tracker_id
            ]


            # -------------------------------------------------
            # STABILIZE REGION
            # -------------------------------------------------

            if raw_state == "boundary":

                stable_region = previous_region

                candidate_states[
                    tracker_id
                ] = None

                candidate_counts[
                    tracker_id
                ] = 0


            elif raw_state == previous_region:

                stable_region = previous_region

                candidate_states[
                    tracker_id
                ] = None

                candidate_counts[
                    tracker_id
                ] = 0


            else:

                if (
                    candidate_states.get(
                        tracker_id
                    )
                    != raw_state
                ):

                    candidate_states[
                        tracker_id
                    ] = raw_state

                    candidate_counts[
                        tracker_id
                    ] = 1

                else:

                    candidate_counts[
                        tracker_id
                    ] += 1


                if (
                    candidate_counts[
                        tracker_id
                    ]
                    >= STATE_CONFIRM_FRAMES
                ):

                    stable_region = raw_state

                    candidate_states[
                        tracker_id
                    ] = None

                    candidate_counts[
                        tracker_id
                    ] = 0

                else:

                    stable_region = previous_region


            region_states[
                tracker_id
            ] = stable_region


            # =================================================
            # CURRENT INSIDE
            # =================================================

            if stable_region == "inside":

                current_inside_ids.add(
                    tracker_id
                )


            # =================================================
            # TWO-LINE CROSSING
            # =================================================

            if previous_point is not None:

                moved = (

                    abs(
                        point[0]
                        - previous_point[0]
                    )

                    +

                    abs(
                        point[1]
                        - previous_point[1]
                    )

                )


                if moved >= MIN_LINE_MOVEMENT:

                    # =========================================
                    # LINE A
                    # =========================================

                    if crossed_line(

                        previous_point,
                        point,
                        line_a_p1,
                        line_a_p2

                    ):

                        crossed_a.add(
                            tracker_id
                        )


                        # -------------------------------------
                        # If B was already crossed,
                        # direction is B -> A
                        # -------------------------------------

                        if tracker_id in crossed_b:

                            crossing_stage[
                                tracker_id
                            ] = "b_to_a"

                        else:

                            crossing_stage[
                                tracker_id
                            ] = "a_to_b"


                    # =========================================
                    # LINE B
                    # =========================================

                    if crossed_line(

                        previous_point,
                        point,
                        line_b_p1,
                        line_b_p2

                    ):

                        crossed_b.add(
                            tracker_id
                        )


                        # -------------------------------------
                        # If A was already crossed,
                        # direction is A -> B
                        # -------------------------------------

                        if tracker_id in crossed_a:

                            crossing_stage[
                                tracker_id
                            ] = "a_to_b"

                        else:

                            crossing_stage[
                                tracker_id
                            ] = "b_to_a"


            # =================================================
            # COUNT COMPLETE CROSSING
            # =================================================

            stage = crossing_stage.get(
                tracker_id
            )


            if (

                stage is not None

                and tracker_id not in counted_ids

                and tracker_id in crossed_a

                and tracker_id in crossed_b

            ):


                # =============================================
                # A -> B
                # =============================================

                if stage == "a_to_b":

                    total_in += 1

                    event = "IN"

                    direction = "A_TO_B"


                # =============================================
                # B -> A
                # =============================================

                else:

                    total_out += 1

                    event = "OUT"

                    direction = "B_TO_A"


                counted_ids.add(
                    tracker_id
                )


                csv_writer.writerow(
                    [
                        frame_number,
                        tracker_id,
                        event,
                        direction
                    ]
                )


                csv_file.flush()


                print()
                print(
                    ">>> COW COUNTED"
                )

                print(
                    f"Event     : {event}"
                )

                print(
                    f"Direction : {direction}"
                )

                print(
                    f"Frame     : {frame_number}"
                )

                print(
                    f"IN        : {total_in}"
                )

                print(
                    f"OUT       : {total_out}"
                )


            # =================================================
            # DRAW BOX
            # =================================================

            if stable_region == "inside":

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

                (x1, y1),

                (x2, y2),

                box_color,

                3
            )


            # -------------------------------------------------
            # NO TRACKER ID DISPLAYED
            # -------------------------------------------------

            # Only draw a small center/bottom point

            cv2.circle(

                frame,

                point,

                5,

                box_color,

                -1
            )


            # =================================================
            # DRAW TRAJECTORY
            # =================================================

            trajectory = trajectories[
                tracker_id
            ]


            if len(trajectory) >= 2:

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
    # LOST TRACK PROTECTION
    #
    # We DON'T immediately delete old tracking data.
    # ========================================================

    for tracker_id in list(
        last_seen.keys()
    ):

        missing = (
            frame_number
            - last_seen[
                tracker_id
            ]
        )


        if missing > MAX_MISSING_FRAMES:

            trajectories.pop(
                tracker_id,
                None
            )

            last_points.pop(
                tracker_id,
                None
            )

            last_seen.pop(
                tracker_id,
                None
            )

            region_states.pop(
                tracker_id,
                None
            )

            candidate_states.pop(
                tracker_id,
                None
            )

            candidate_counts.pop(
                tracker_id,
                None
            )


    # ========================================================
    # CURRENT COUNT
    # ========================================================

    current_cows = len(
        current_inside_ids
    )


    # ========================================================
    # NET
    # ========================================================

    net_count = (
        total_in
        - total_out
    )


    # ========================================================
    # FPS
    # ========================================================

    fps_frames += 1

    elapsed = (
        time.time()
        - fps_start
    )


    if elapsed >= 1.0:

        display_fps = (
            fps_frames
            / elapsed
        )

        fps_frames = 0

        fps_start = time.time()


    # ========================================================
    # DRAW MONITORING REGION
    # ========================================================

    cv2.polylines(

        frame,

        [polygon],

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
    # LINE LABELS
    # ========================================================

    line_a_center = (

        int(
            (line_a_p1[0] + line_a_p2[0])
            / 2
        ),

        int(
            (line_a_p1[1] + line_a_p2[1])
            / 2
        )

    )


    line_b_center = (

        int(
            (line_b_p1[0] + line_b_p2[0])
            / 2
        ),

        int(
            (line_b_p1[1] + line_b_p2[1])
            / 2
        )

    )


    cv2.putText(

        frame,

        "LINE A",

        line_a_center,

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (
            0,
            0,
            255
        ),

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        frame,

        "LINE B",

        line_b_center,

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (
            255,
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

        0.72,

        frame,

        0.28,

        0
    )


    # ========================================================
    # PANEL
    # ========================================================

    cv2.putText(

        frame,

        "COW COUNTING - TEST 4",

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

        f"DET : {detections}",

        (
            panel_x + 190,
            panel_y + 185
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

        f"Frame: {frame_number:04d} | "
        f"Detections: {detections:2d} | "
        f"Current: {current_cows:2d} | "
        f"IN: {total_in:2d} | "
        f"OUT: {total_out:2d} | "
        f"NET: {net_count:3d} | "
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

        "TEST 4 - COW COUNTING",

        display_frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        print()
        print("Stopping Test 4...")

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
print(" TEST 4 FINISHED")
print("=" * 60)

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

print("=" * 60)
