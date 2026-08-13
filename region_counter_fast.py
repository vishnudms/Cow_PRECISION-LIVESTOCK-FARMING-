import cv2
import os
import time
import numpy as np
from ultralytics import YOLO


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = "videos/cow_video10.mp4"
OUTPUT_PATH = "output/cow_in_out_final.mp4"
MODEL_PATH = "yolo11n.pt"

# COCO class ID
# 19 = cow
COW_CLASS_ID = 19

# Detection confidence
CONFIDENCE = 0.30

# Inference resolution
# 512 gives better FPS than 640
IMG_SIZE = 512

# GPU
DEVICE = 0

# DO NOT USE half=True
# New Ultralytics versions warn about it.
USE_HALF = False

# Tracker
TRACKER = "bytetrack.yaml"

# ============================================================
# DISPLAY
# ============================================================

DISPLAY_WIDTH = 1100
DISPLAY_HEIGHT = 700

# ============================================================
# REGION SETTINGS
# ============================================================

# Distance from boundary where we consider the cow
# to be "on the boundary".
BOUNDARY_MARGIN = 35

# Number of consecutive frames required before
# changing inside <-> outside.
STATE_CONFIRM_FRAMES = 5

# ============================================================
# LOST TRACK PROTECTION
# ============================================================

# If YOLO temporarily loses a cow, keep its box/state
# for this many frames.
MAX_MISSING_FRAMES = 20


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs("output", exist_ok=True)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

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


original_selection_frame = selection_frame.copy()


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global selection_frame

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if len(points) >= 4:
        return


    # --------------------------------------------------------
    # Convert display coordinates back to ORIGINAL
    # 1920 x 1080 coordinates
    # --------------------------------------------------------

    original_x = int(
        x / selection_scale
    )

    original_y = int(
        y / selection_scale
    )


    points.append(
        (
            original_x,
            original_y
        )
    )


    print(
        f"Point {len(points)} : "
        f"({original_x}, {original_y})"
    )


    # --------------------------------------------------------
    # Draw selected point
    # --------------------------------------------------------

    cv2.circle(
        selection_frame,
        (x, y),
        6,
        (0, 255, 0),
        -1
    )


    # --------------------------------------------------------
    # Draw connecting line
    # --------------------------------------------------------

    if len(points) > 1:

        previous = points[-2]

        previous_display = (
            int(previous[0] * selection_scale),
            int(previous[1] * selection_scale)
        )

        cv2.line(
            selection_frame,
            previous_display,
            (x, y),
            (0, 255, 0),
            3
        )


    # --------------------------------------------------------
    # Close polygon
    # --------------------------------------------------------

    if len(points) == 4:

        first = points[0]

        first_display = (
            int(first[0] * selection_scale),
            int(first[1] * selection_scale)
        )

        cv2.line(
            selection_frame,
            (x, y),
            first_display,
            (0, 255, 255),
            3
        )


# ============================================================
# CREATE SELECTION WINDOW
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
print("================================================")
print(" SELECT COW MONITORING REGION")
print("================================================")
print()
print("Click FOUR points around the monitoring area.")
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

        print("Region reset.")


    # --------------------------------------------------------
    # CONFIRM
    # --------------------------------------------------------

    elif key == 13:

        if len(points) == 4:
            break

        print(
            f"Please select 4 points."
            f" Selected: {len(points)}"
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
print("================================================")
print(" SELECTED REGION")
print("================================================")

for i, point in enumerate(
    points,
    start=1
):

    print(
        f"Point {i}: {point}"
    )

print("================================================")
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

    raise SystemExit


# ============================================================
# LOAD YOLO
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

# Hidden tracker ID -> visible sequential number
display_ids = {}

next_display_id = 1


# Tracker ID -> confirmed state
#
# "inside"
# "outside"
cow_states = {}


# Tracker ID -> candidate state
candidate_states = {}


# Tracker ID -> candidate frame count
candidate_counts = {}


# Tracker ID -> last bounding box
last_boxes = {}


# Tracker ID -> last state point
last_points = {}


# Tracker ID -> last frame seen
last_seen_frame = {}


# Tracker ID -> last detection confidence
last_confidences = {}


# ============================================================
# COUNTING
# ============================================================

counted_in_ids = set()

counted_out_ids = set()

total_in = 0

total_out = 0


# ============================================================
# FPS
# ============================================================

fps_start_time = time.time()

fps_frame_count = 0

display_fps = 0.0


# ============================================================
# FRAME
# ============================================================

frame_number = 0

current_cows = 0


# ============================================================
# DISPLAY ID FUNCTION
# ============================================================

def get_display_id(tracker_id):

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
# REGION STATE
# ============================================================

def get_region_state(point):

    distance = cv2.pointPolygonTest(
        polygon,
        point,
        True
    )


    # Clearly inside
    if distance > BOUNDARY_MARGIN:

        return "inside"


    # Clearly outside
    if distance < -BOUNDARY_MARGIN:

        return "outside"


    # Close to boundary
    return "boundary"


# ============================================================
# BOX COLOR
# ============================================================

def get_color(state):

    if state == "inside":

        # GREEN
        return (
            0,
            255,
            0
        )

    else:

        # YELLOW
        return (
            0,
            255,
            255
        )


# ============================================================
# START
# ============================================================

print()
print("================================================")
print(" STARTING FINAL COW COUNTING")
print("================================================")
print()
print("Model       :", MODEL_PATH)
print("GPU         :", DEVICE)
print("Image size  :", IMG_SIZE)
print("Confidence  :", CONFIDENCE)
print("Half        :", USE_HALF)
print("Boundary    :", BOUNDARY_MARGIN)
print("Confirm     :", STATE_CONFIRM_FRAMES)
print("Lost frames :", MAX_MISSING_FRAMES)
print()
print("Press Q to stop.")
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # ========================================================
    # READ FRAME
    # ========================================================

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1


    # ========================================================
    # ACTIVE TRACKS IN THIS FRAME
    # ========================================================

    active_track_ids = set()


    # ========================================================
    # YOLO TRACKING
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


        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )


        # ====================================================
        # EACH COW
        # ====================================================

        for box, tracker_id, confidence in zip(
            boxes,
            tracker_ids,
            confidences
        ):


            x1, y1, x2, y2 = map(
                int,
                box
            )


            # ------------------------------------------------
            # ACTIVE TRACK
            # ------------------------------------------------

            active_track_ids.add(
                tracker_id
            )


            # ------------------------------------------------
            # DISPLAY NUMBER
            # ------------------------------------------------

            display_id = get_display_id(
                tracker_id
            )


            # ------------------------------------------------
            # STATE POINT
            #
            # Instead of using the absolute bottom edge,
            # use a point slightly above the bottom.
            #
            # This prevents a cow's leg/hoof touching the
            # boundary from immediately changing state.
            # ------------------------------------------------

            box_height = y2 - y1

            state_point = (
                int(
                    (x1 + x2) / 2
                ),
                int(
                    y2
                    - box_height * 0.12
                )
            )


            # ------------------------------------------------
            # SAVE TRACK DATA
            # ------------------------------------------------

            last_boxes[
                tracker_id
            ] = (
                x1,
                y1,
                x2,
                y2
            )


            last_points[
                tracker_id
            ] = state_point


            last_seen_frame[
                tracker_id
            ] = frame_number


            last_confidences[
                tracker_id
            ] = float(
                confidence
            )


            # =================================================
            # INITIALIZE NEW COW
            # =================================================

            if tracker_id not in cow_states:

                initial_state = get_region_state(
                    state_point
                )


                # If starting directly on the boundary,
                # assume outside until it clearly enters.
                if initial_state == "boundary":

                    initial_state = "outside"


                cow_states[
                    tracker_id
                ] = initial_state


                candidate_states[
                    tracker_id
                ] = None


                candidate_counts[
                    tracker_id
                ] = 0


            # =================================================
            # PREVIOUS STATE
            # =================================================

            previous_state = cow_states[
                tracker_id
            ]


            # =================================================
            # CURRENT RAW STATE
            # =================================================

            raw_state = get_region_state(
                state_point
            )


            # =================================================
            # STATE STABILIZATION
            # =================================================

            if raw_state == "boundary":

                # ------------------------------------------------
                # IMPORTANT:
                #
                # Do absolutely nothing while the cow is close
                # to the boundary.
                # ------------------------------------------------

                stable_state = previous_state


                candidate_states[
                    tracker_id
                ] = None


                candidate_counts[
                    tracker_id
                ] = 0


            elif raw_state == previous_state:

                # No state change

                stable_state = previous_state


                candidate_states[
                    tracker_id
                ] = None


                candidate_counts[
                    tracker_id
                ] = 0


            else:

                # ------------------------------------------------
                # Potential state change
                # ------------------------------------------------

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


                # ------------------------------------------------
                # Confirm only after several frames
                # ------------------------------------------------

                if (
                    candidate_counts[
                        tracker_id
                    ]
                    >= STATE_CONFIRM_FRAMES
                ):

                    stable_state = raw_state


                    candidate_states[
                        tracker_id
                    ] = None


                    candidate_counts[
                        tracker_id
                    ] = 0


                else:

                    stable_state = previous_state


            # =================================================
            # STATE CHANGE
            # =================================================

            if stable_state != previous_state:

                cow_states[
                    tracker_id
                ] = stable_state


                # =================================================
                # COW ENTERED
                # =================================================

                if (

                    previous_state == "outside"

                    and stable_state == "inside"

                ):

                    if tracker_id not in counted_in_ids:

                        counted_in_ids.add(
                            tracker_id
                        )

                        total_in += 1


                        print()
                        print(
                            ">>> COW ENTERED"
                        )

                        print(
                            f"Display ID : {display_id}"
                        )

                        print(
                            f"IN         : {total_in}"
                        )

                        print(
                            f"OUT        : {total_out}"
                        )


                # =================================================
                # COW EXITED
                # =================================================

                elif (

                    previous_state == "inside"

                    and stable_state == "outside"

                ):

                    if tracker_id not in counted_out_ids:

                        counted_out_ids.add(
                            tracker_id
                        )

                        total_out += 1


                        print()
                        print(
                            ">>> COW EXITED"
                        )

                        print(
                            f"Display ID : {display_id}"
                        )

                        print(
                            f"IN         : {total_in}"
                        )

                        print(
                            f"OUT        : {total_out}"
                        )


            # =================================================
            # BOX COLOR
            # =================================================

            color = get_color(
                cow_states[
                    tracker_id
                ]
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

                color,

                3
            )


            # =================================================
            # DRAW DISPLAY NUMBER
            # =================================================

            number_text = str(
                display_id
            )


            text_x = x1

            text_y = max(
                y1 - 12,
                30
            )


            cv2.putText(

                frame,

                number_text,

                (
                    text_x,
                    text_y
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                1.0,

                color,

                3,

                cv2.LINE_AA
            )


            # =================================================
            # DRAW STATE POINT
            # =================================================

            cv2.circle(

                frame,

                state_point,

                6,

                color,

                -1
            )


    # ========================================================
    # KEEP TEMPORARILY LOST COWS
    # ========================================================

    # These cows were detected recently but YOLO missed them
    # in the current frame.
    #
    # We continue displaying their last box and state.

    for tracker_id in list(
        last_boxes.keys()
    ):

        if tracker_id in active_track_ids:
            continue


        missing_frames = (
            frame_number
            - last_seen_frame.get(
                tracker_id,
                frame_number
            )
        )


        if missing_frames > MAX_MISSING_FRAMES:
            continue


        # ----------------------------------------------------
        # Last known box
        # ----------------------------------------------------

        x1, y1, x2, y2 = last_boxes[
            tracker_id
        ]


        # ----------------------------------------------------
        # Last known point
        # ----------------------------------------------------

        state_point = last_points[
            tracker_id
        ]


        # ----------------------------------------------------
        # Display number
        # ----------------------------------------------------

        display_id = get_display_id(
            tracker_id
        )


        # ----------------------------------------------------
        # Last known state
        # ----------------------------------------------------

        state = cow_states.get(
            tracker_id,
            "outside"
        )


        color = get_color(
            state
        )


        # ----------------------------------------------------
        # Draw faded-looking persistent box
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

            color,

            2
        )


        # ----------------------------------------------------
        # Draw number
        # ----------------------------------------------------

        cv2.putText(

            frame,

            str(display_id),

            (
                x1,
                max(
                    y1 - 12,
                    30
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.0,

            color,

            3,

            cv2.LINE_AA
        )


    # ========================================================
    # CURRENT COW COUNT
    # ========================================================

    current_inside_ids = set()


    for tracker_id in last_boxes.keys():

        last_seen = last_seen_frame.get(
            tracker_id,
            -999999
        )


        missing_frames = (
            frame_number
            - last_seen
        )


        # Only recently active tracks
        if missing_frames <= MAX_MISSING_FRAMES:

            if cow_states.get(
                tracker_id
            ) == "inside":

                current_inside_ids.add(
                    tracker_id
                )


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

    fps_frame_count += 1

    elapsed = (
        time.time()
        - fps_start_time
    )


    if elapsed >= 1.0:

        display_fps = (
            fps_frame_count
            / elapsed
        )


        fps_frame_count = 0

        fps_start_time = time.time()


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
    # DRAW REGION LABEL
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
    # INFORMATION PANEL
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
    # PANEL TITLE
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


    # ========================================================
    # CURRENT
    # ========================================================

    cv2.putText(

        frame,

        f"CURRENT : {current_cows}",

        (
            panel_x + 15,
            panel_y + 67
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.68,

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
            panel_y + 99
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
    # OUT
    # ========================================================

    cv2.putText(

        frame,

        f"OUT     : {total_out}",

        (
            panel_x + 15,
            panel_y + 131
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
    # NET
    # ========================================================

    cv2.putText(

        frame,

        f"NET     : {net_count}",

        (
            panel_x + 15,
            panel_y + 163
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
    # FPS
    # ========================================================

    cv2.putText(

        frame,

        f"FPS     : {display_fps:.1f}",

        (
            panel_x + 185,
            panel_y + 163
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
    # FRAME NUMBER
    # ========================================================

    cv2.putText(

        frame,

        f"FRAME   : {frame_number}/{total_frames}",

        (
            panel_x + 15,
            panel_y + 193
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (
            200,
            200,
            200
        ),

        1,

        cv2.LINE_AA
    )


    # ========================================================
    # TERMINAL OUTPUT
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
    # SAVE FULL-RESOLUTION VIDEO
    # ========================================================

    writer.write(
        frame
    )


    # ========================================================
    # SMALL DISPLAY
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

        "COW COUNTING - REAL TIME",

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
    "Output saved to:"
)

print(
    OUTPUT_PATH
)

print("================================================")
