"""
===============================================================================
TEST GOAT TOP-DETECTION
YOLO11 + ByteTrack + A/B LINE COUNTING
===============================================================================

Purpose:
    Test top-angle goat/sheep detection and tracking from cow_video1.mp4.

Detection:
    YOLO11m COCO model

Tracking:
    ByteTrack

COCO classes:
    18 = sheep
    19 = cow

IMPORTANT:
    COCO does NOT contain a goat class.
    Class 18 is therefore displayed as GOAT/SHEEP.

Counting:
    LINE A -> LINE B = IN
    LINE B -> LINE A = OUT

Example:

    python goat_top_test.py --source cow_video1.mp4

    python goat_top_test.py \
        --source cow_video1.mp4 \
        --save goat_test_output.mp4

    python goat_top_test.py \
        --source cow_video1.mp4 \
        --conf 0.25

    python goat_top_test.py \
        --source cow_video1.mp4 \
        --pick-lines

===============================================================================
"""

import argparse
import os
import time
from collections import defaultdict, deque

import cv2
from ultralytics import YOLO


# =============================================================================
# CONFIGURATION
# =============================================================================

# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------

MODEL_PATH = "yolo11m.pt"

# ---------------------------------------------------------------------------
# TARGET CLASSES
# ---------------------------------------------------------------------------
#
# COCO:
#   18 = sheep
#   19 = cow
#
# There is NO goat class in COCO.
#
# For your goat top-view test:
#   18 -> GOAT/SHEEP
#
# If you only want goat/sheep:
#       TARGET_CLASSES = {18: "GOAT/SHEEP"}
#
# If you also want cows:
#       TARGET_CLASSES = {
#           18: "GOAT/SHEEP",
#           19: "COW"
#       }
#

TARGET_CLASSES = {
    18: "GOAT/SHEEP",
    19: "COW",
}

# ---------------------------------------------------------------------------
# YOLO SETTINGS
# ---------------------------------------------------------------------------

CONF_THRESH = 0.25
IOU_THRESH = 0.45

# Image size used by YOLO.
# 1280 can help with small animals but uses more GPU memory.
IMGSZ = 1280

# ---------------------------------------------------------------------------
# BYTETRACK SETTINGS
# ---------------------------------------------------------------------------

TRACK_BUFFER = 30

TRACK_HIGH_THRESH = 0.40
TRACK_LOW_THRESH = 0.05
NEW_TRACK_THRESH = 0.40
MATCH_THRESH = 0.70

# ---------------------------------------------------------------------------
# LINE SETTINGS
# ---------------------------------------------------------------------------
#
# These are for your 1920x1080 cow_video1.mp4.
#
# You MUST verify them visually.
#

LINE_A = (400, 180, 400, 640)
LINE_B = (1150, 200, 1150, 620)

# ---------------------------------------------------------------------------
# TRACK HISTORY
# ---------------------------------------------------------------------------

TRAIL_LEN = 40

# Minimum age before counting.
MIN_TRACK_AGE_FOR_COUNT = 2

# ---------------------------------------------------------------------------
# LOST TRACK HANDLING
# ---------------------------------------------------------------------------
#
# Do not immediately delete a track when ByteTrack temporarily loses it.
#

MAX_MISSING_FRAMES = 30


# =============================================================================
# GEOMETRY
# =============================================================================

def orientation(a, b, c):
    """
    Orientation test used by segment intersection.
    """

    value = (
        (b[1] - a[1]) * (c[0] - b[0])
        - (b[0] - a[0]) * (c[1] - b[1])
    )

    if abs(value) < 1e-9:
        return 0

    return 1 if value > 0 else 2


def on_segment(a, b, c):
    """
    Checks whether point b lies on segment a-c.
    """

    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
        and
        min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def segments_intersect(p1, p2, p3, p4):
    """
    Checks whether two line segments intersect.

    p1 -> p2
        animal movement

    p3 -> p4
        counting line
    """

    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, p3, p2):
        return True

    if o2 == 0 and on_segment(p1, p4, p2):
        return True

    if o3 == 0 and on_segment(p3, p1, p4):
        return True

    if o4 == 0 and on_segment(p3, p2, p4):
        return True

    return False


# =============================================================================
# TRACK STATE
# =============================================================================

class TrackState:

    def __init__(self, cls_name):

        self.history = deque(maxlen=TRAIL_LEN)

        self.age = 0

        self.missing_frames = 0

        self.hit_A = None
        self.hit_B = None

        self.counted = False

        self.cls_name = cls_name

        self.last_frame = -1


# =============================================================================
# LINE COUNTER
# =============================================================================

class LineCounter:

    def __init__(self):

        self.tracks = {}

        self.count_in = 0
        self.count_out = 0

        self.per_class = defaultdict(
            lambda: {
                "in": 0,
                "out": 0
            }
        )

        # IMPORTANT:
        # This is incremented ONCE per VIDEO FRAME.
        self.frame_idx = 0

        self.events = []

    # -------------------------------------------------------------------------
    # FRAME UPDATE
    # -------------------------------------------------------------------------

    def new_frame(self):

        self.frame_idx += 1

    # -------------------------------------------------------------------------
    # TRACK UPDATE
    # -------------------------------------------------------------------------

    def update(
        self,
        track_id,
        cx,
        cy,
        cls_name
    ):

        if track_id not in self.tracks:

            self.tracks[track_id] = TrackState(
                cls_name
            )

        st = self.tracks[track_id]

        st.age += 1

        st.missing_frames = 0

        st.last_frame = self.frame_idx

        previous_point = (
            st.history[-1]
            if len(st.history) > 0
            else None
        )

        current_point = (
            int(cx),
            int(cy)
        )

        st.history.append(current_point)

        # Need two points to calculate movement.
        if previous_point is None:
            return None

        # Don't count extremely young tracks.
        if st.age < MIN_TRACK_AGE_FOR_COUNT:
            return None

        # Already counted.
        if st.counted:
            return None

        event = None

        # ---------------------------------------------------------------------
        # CHECK LINE A
        # ---------------------------------------------------------------------

        if st.hit_A is None:

            crossed_A = segments_intersect(
                previous_point,
                current_point,
                (LINE_A[0], LINE_A[1]),
                (LINE_A[2], LINE_A[3])
            )

            if crossed_A:

                st.hit_A = self.frame_idx

        # ---------------------------------------------------------------------
        # CHECK LINE B
        # ---------------------------------------------------------------------

        if st.hit_B is None:

            crossed_B = segments_intersect(
                previous_point,
                current_point,
                (LINE_B[0], LINE_B[1]),
                (LINE_B[2], LINE_B[3])
            )

            if crossed_B:

                st.hit_B = self.frame_idx

        # ---------------------------------------------------------------------
        # DETERMINE DIRECTION
        # ---------------------------------------------------------------------

        if (
            st.hit_A is not None
            and
            st.hit_B is not None
            and
            not st.counted
        ):

            if st.hit_A < st.hit_B:

                self.count_in += 1

                self.per_class[cls_name]["in"] += 1

                event = "IN"

            elif st.hit_B < st.hit_A:

                self.count_out += 1

                self.per_class[cls_name]["out"] += 1

                event = "OUT"

            if event:

                st.counted = True

                self.events.append(
                    {
                        "frame": self.frame_idx,
                        "track_id": track_id,
                        "class": cls_name,
                        "event": event
                    }
                )

        return event

    # -------------------------------------------------------------------------
    # MISSING TRACK MANAGEMENT
    # -------------------------------------------------------------------------

    def update_missing_tracks(self, active_ids):

        for track_id, state in self.tracks.items():

            if track_id not in active_ids:

                state.missing_frames += 1

    # -------------------------------------------------------------------------
    # CLEANUP
    # -------------------------------------------------------------------------

    def cleanup(self):

        remove_ids = []

        for track_id, state in self.tracks.items():

            if state.missing_frames > MAX_MISSING_FRAMES:

                remove_ids.append(track_id)

        for track_id in remove_ids:

            del self.tracks[track_id]


# =============================================================================
# DRAW TRACK TRAIL
# =============================================================================

def draw_trail(frame, state):

    points = list(state.history)

    if len(points) < 2:
        return

    for i in range(1, len(points)):

        p1 = points[i - 1]
        p2 = points[i]

        cv2.line(
            frame,
            p1,
            p2,
            (255, 255, 0),
            2
        )


# =============================================================================
# DRAW OVERLAY
# =============================================================================

def draw_overlay(
    frame,
    counter,
    fps,
    detection_count,
    active_count
):

    h, w = frame.shape[:2]

    # -------------------------------------------------------------------------
    # LINE A
    # -------------------------------------------------------------------------

    cv2.line(
        frame,
        LINE_A[:2],
        LINE_A[2:],
        (0, 0, 255),
        3
    )

    cv2.putText(
        frame,
        "LINE A",
        (
            LINE_A[0] + 10,
            LINE_A[1] + 25
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    # -------------------------------------------------------------------------
    # LINE B
    # -------------------------------------------------------------------------

    cv2.line(
        frame,
        LINE_B[:2],
        LINE_B[2:],
        (255, 0, 255),
        3
    )

    cv2.putText(
        frame,
        "LINE B",
        (
            LINE_B[0] + 10,
            LINE_B[1] + 25
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2
    )

    # -------------------------------------------------------------------------
    # PANEL
    # -------------------------------------------------------------------------

    panel_w = 330
    panel_h = 250

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (10 + panel_w, 10 + panel_h),
        (0, 0, 0),
        -1
    )

    frame[:] = cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0
    )

    # -------------------------------------------------------------------------
    # CURRENT TRACKS
    # -------------------------------------------------------------------------

    current_inside = 0

    for state in counter.tracks.values():

        if state.counted:
            continue

        if (
            (state.hit_A is not None)
            !=
            (state.hit_B is not None)
        ):

            current_inside += 1

    # -------------------------------------------------------------------------
    # TEXT
    # -------------------------------------------------------------------------

    lines = [

        (
            "GOAT TOP DETECTION",
            (255, 255, 255)
        ),

        (
            f"ACTIVE TRACKS : {active_count}",
            (255, 255, 255)
        ),

        (
            f"CURRENT       : {current_inside}",
            (0, 255, 0)
        ),

        (
            f"IN            : {counter.count_in}",
            (0, 255, 0)
        ),

        (
            f"OUT           : {counter.count_out}",
            (0, 0, 255)
        ),

        (
            f"NET           : "
            f"{counter.count_in - counter.count_out}",
            (255, 255, 255)
        ),

        (
            f"DETECTIONS    : {detection_count}",
            (255, 255, 255)
        ),

        (
            f"FPS           : {fps:.1f}",
            (255, 255, 255)
        ),

        (
            "A -> B = IN",
            (200, 200, 200)
        ),

        (
            "B -> A = OUT",
            (200, 200, 200)
        ),
    ]

    y = 38

    for text, color in lines:

        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            1,
            cv2.LINE_AA
        )

        y += 23

    return frame


# =============================================================================
# WRITE BYTETRACK CONFIG
# =============================================================================

def write_tracker_yaml():

    config = f"""
tracker_type: bytetrack

track_high_thresh: {TRACK_HIGH_THRESH}

track_low_thresh: {TRACK_LOW_THRESH}

new_track_thresh: {NEW_TRACK_THRESH}

track_buffer: {TRACK_BUFFER}

match_thresh: {MATCH_THRESH}

fuse_score: True
"""

    path = "bytetrack_goat.yaml"

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(config)

    return path


# =============================================================================
# PICK LINES
# =============================================================================

def pick_lines_interactively(source):

    if str(source).isdigit():

        source = int(source)

    cap = cv2.VideoCapture(source)

    ok, frame = cap.read()

    cap.release()

    if not ok:

        print(
            "[ERROR] Could not read first frame."
        )

        return

    points = []

    window_name = (
        "Click: "
        "A-start, A-end, "
        "B-start, B-end"
    )

    def mouse_callback(
        event,
        x,
        y,
        flags,
        param
    ):

        if (
            event == cv2.EVENT_LBUTTONDOWN
            and len(points) < 4
        ):

            points.append(
                (x, y)
            )

            print(
                f"Point {len(points)}: "
                f"({x}, {y})"
            )

    cv2.namedWindow(window_name)

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    while True:

        display = frame.copy()

        # Draw selected points.

        for i, point in enumerate(points):

            cv2.circle(
                display,
                point,
                6,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                display,
                str(i + 1),
                (
                    point[0] + 10,
                    point[1]
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Line A

        if len(points) >= 2:

            cv2.line(
                display,
                points[0],
                points[1],
                (0, 0, 255),
                3
            )

        # Line B

        if len(points) >= 4:

            cv2.line(
                display,
                points[2],
                points[3],
                (255, 0, 255),
                3
            )

        cv2.imshow(
            window_name,
            display
        )

        key = cv2.waitKey(20) & 0xFF

        if key == 27:

            break

        if len(points) == 4:

            cv2.waitKey(1000)

            break

    cv2.destroyAllWindows()

    if len(points) == 4:

        a1, a2, b1, b2 = points

        print()
        print("=" * 70)
        print("COPY THESE INTO THE SCRIPT")
        print("=" * 70)

        print(
            f"LINE_A = "
            f"({a1[0]}, {a1[1]}, "
            f"{a2[0]}, {a2[1]})"
        )

        print(
            f"LINE_B = "
            f"({b1[0]}, {b1[1]}, "
            f"{b2[0]}, {b2[1]})"
        )

        print("=" * 70)


# =============================================================================
# MAIN RUN FUNCTION
# =============================================================================

def run(
    source,
    save_path=None,
    show=True
):

    print()
    print("=" * 70)
    print("GOAT TOP DETECTION TEST")
    print("=" * 70)

    print(
        f"[MODEL] {MODEL_PATH}"
    )

    print(
        f"[SOURCE] {source}"
    )

    print(
        f"[CONF] {CONF_THRESH}"
    )

    print(
        f"[IMAGE SIZE] {IMGSZ}"
    )

    print("=" * 70)

    # -------------------------------------------------------------------------
    # CHECK SOURCE
    # -------------------------------------------------------------------------

    source_for_cv = source

    if str(source).isdigit():

        source_for_cv = int(source)

    elif not str(source).startswith(
        ("rtsp://", "http://", "https://")
    ):

        if not os.path.exists(source):

            print()
            print(
                "[ERROR] Video file does not exist:"
            )

            print(
                source
            )

            return

    # -------------------------------------------------------------------------
    # LOAD MODEL
    # -------------------------------------------------------------------------

    print(
        "[MODEL] Loading YOLO..."
    )

    model = YOLO(
        MODEL_PATH
    )

    print(
        "[MODEL] Loaded."
    )

    # -------------------------------------------------------------------------
    # SOURCE INFORMATION
    # -------------------------------------------------------------------------

    cap = cv2.VideoCapture(
        source_for_cv
    )

    if not cap.isOpened():

        print(
            "[ERROR] Could not open source."
        )

        return

    fps_input = (
        cap.get(cv2.CAP_PROP_FPS)
        or 8.0
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

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()

    print()
    print(
        f"[VIDEO] Resolution : "
        f"{width} x {height}"
    )

    print(
        f"[VIDEO] FPS        : "
        f"{fps_input:.2f}"
    )

    print(
        f"[VIDEO] Frames     : "
        f"{total_frames}"
    )

    # -------------------------------------------------------------------------
    # OUTPUT WRITER
    # -------------------------------------------------------------------------

    writer = None

    if save_path:

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
                "[ERROR] Could not create output video."
            )

            writer = None

    # -------------------------------------------------------------------------
    # TRACKER
    # -------------------------------------------------------------------------

    tracker_yaml = write_tracker_yaml()

    print(
        f"[TRACKER] {tracker_yaml}"
    )

    # -------------------------------------------------------------------------
    # COUNTER
    # -------------------------------------------------------------------------

    counter = LineCounter()

    # -------------------------------------------------------------------------
    # YOLO TRACK
    # -------------------------------------------------------------------------

    print()
    print(
        "[START] Detection + ByteTrack..."
    )

    results = model.track(

        source=source_for_cv,

        stream=True,

        persist=True,

        tracker=tracker_yaml,

        conf=CONF_THRESH,

        iou=IOU_THRESH,

        imgsz=IMGSZ,

        classes=list(
            TARGET_CLASSES.keys()
        ),

        verbose=False,

    )

    previous_time = time.time()

    fps_smooth = 0.0

    frame_number = 0

    # -------------------------------------------------------------------------
    # FRAME LOOP
    # -------------------------------------------------------------------------

    for result in results:

        frame_number += 1

        # IMPORTANT:
        # Advance frame index ONCE.
        counter.new_frame()

        frame = result.orig_img.copy()

        active_ids = set()

        detection_count = 0

        # ---------------------------------------------------------------------
        # DETECTIONS
        # ---------------------------------------------------------------------

        if (
            result.boxes is not None
            and
            result.boxes.id is not None
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

            # ---------------------------------------------------------------
            # EACH TRACK
            # ---------------------------------------------------------------

            for (
                box,
                track_id,
                class_id,
                confidence
            ) in zip(
                boxes,
                ids,
                classes,
                confidences
            ):

                x1, y1, x2, y2 = box

                # Center point.

                cx = int(
                    (x1 + x2) / 2
                )

                cy = int(
                    (y1 + y2) / 2
                )

                class_name = (
                    TARGET_CLASSES.get(
                        class_id,
                        "ANIMAL"
                    )
                )

                active_ids.add(
                    track_id
                )

                # -----------------------------------------------------------
                # UPDATE COUNTER
                # -----------------------------------------------------------

                event = counter.update(

                    track_id,

                    cx,
                    cy,

                    class_name
                )

                state = counter.tracks[
                    track_id
                ]

                # -----------------------------------------------------------
                # DRAW TRAIL
                # -----------------------------------------------------------

                draw_trail(
                    frame,
                    state
                )

                # -----------------------------------------------------------
                # DRAW BOX
                # -----------------------------------------------------------

                box_color = (
                    0,
                    200,
                    255
                )

                cv2.rectangle(

                    frame,

                    (
                        int(x1),
                        int(y1)
                    ),

                    (
                        int(x2),
                        int(y2)
                    ),

                    box_color,

                    2
                )

                # -----------------------------------------------------------
                # DRAW CENTER
                # -----------------------------------------------------------

                cv2.circle(

                    frame,

                    (
                        cx,
                        cy
                    ),

                    5,

                    (0, 0, 255),

                    -1
                )

                # -----------------------------------------------------------
                # LABEL
                # -----------------------------------------------------------

                label = (

                    f"{class_name} "
                    f"ID:{track_id} "
                    f"{confidence:.2f}"
                )

                cv2.putText(

                    frame,

                    label,

                    (
                        int(x1),
                        max(
                            int(y1) - 8,
                            20
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.48,

                    box_color,

                    1,

                    cv2.LINE_AA
                )

                # -----------------------------------------------------------
                # EVENT
                # -----------------------------------------------------------

                if event:

                    event_color = (

                        (0, 255, 0)
                        if event == "IN"
                        else
                        (0, 0, 255)
                    )

                    cv2.putText(

                        frame,

                        event,

                        (
                            cx - 25,
                            cy - 20
                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        1.0,

                        event_color,

                        3,

                        cv2.LINE_AA
                    )

                    print(

                        f"[EVENT] "
                        f"Frame={frame_number} "
                        f"ID={track_id} "
                        f"{class_name} "
                        f"-> {event}"
                    )

        # ---------------------------------------------------------------------
        # LOST TRACK MANAGEMENT
        # ---------------------------------------------------------------------

        counter.update_missing_tracks(
            active_ids
        )

        counter.cleanup()

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
        # OVERLAY
        # ---------------------------------------------------------------------

        frame = draw_overlay(

            frame,

            counter,

            fps_smooth,

            detection_count,

            len(active_ids)
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

        if show:

            cv2.imshow(
                "GOAT TOP DETECTION",
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

    # -------------------------------------------------------------------------
    # CLEANUP
    # -------------------------------------------------------------------------

    if writer:

        writer.release()

    if show:

        cv2.destroyAllWindows()

    # -------------------------------------------------------------------------
    # FINAL RESULTS
    # -------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"IN  : {counter.count_in}"
    )

    print(
        f"OUT : {counter.count_out}"
    )

    print(
        f"NET : "
        f"{counter.count_in - counter.count_out}"
    )

    print()
    print("PER CLASS:")

    for class_name, counts in (
        counter.per_class.items()
    ):

        print(
            f"{class_name}: "
            f"IN={counts['in']} "
            f"OUT={counts['out']}"
        )

    print()
    print(
        f"TOTAL EVENTS: "
        f"{len(counter.events)}"
    )

    if save_path:

        print(
            f"OUTPUT: "
            f"{save_path}"
        )

    print("=" * 70)


# =============================================================================
# COMMAND LINE
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(

        description=(
            "YOLO11 + ByteTrack "
            "top-angle goat/sheep detection"
        )
    )

    parser.add_argument(

        "--source",

        default="cow_video1.mp4",

        help=(
            "Video path, webcam index, "
            "or RTSP URL"
        )
    )

    parser.add_argument(

        "--model",

        default=MODEL_PATH,

        help=(
            "YOLO model path"
        )
    )

    parser.add_argument(

        "--save",

        default=None,

        help=(
            "Save annotated output "
            "to this video path"
        )
    )

    parser.add_argument(

        "--conf",

        type=float,

        default=CONF_THRESH,

        help=(
            "Detection confidence "
            "(default: 0.25)"
        )
    )

    parser.add_argument(

        "--imgsz",

        type=int,

        default=IMGSZ,

        help=(
            "YOLO inference image size"
        )
    )

    parser.add_argument(

        "--no-show",

        action="store_true",

        help=(
            "Disable live preview"
        )
    )

    parser.add_argument(

        "--pick-lines",

        action="store_true",

        help=(
            "Click Line A and Line B "
            "on the first video frame"
        )
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # UPDATE GLOBAL SETTINGS
    # -------------------------------------------------------------------------

    MODEL_PATH = args.model

    CONF_THRESH = args.conf

    IMGSZ = args.imgsz

    # -------------------------------------------------------------------------
    # PICK LINES
    # -------------------------------------------------------------------------

    if args.pick_lines:

        pick_lines_interactively(
            args.source
        )

    # -------------------------------------------------------------------------
    # RUN
    # -------------------------------------------------------------------------

    else:

        run(

            source=args.source,

            save_path=args.save,

            show=not args.no_show

        )
