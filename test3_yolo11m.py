import cv2
import os
import csv
import time
from ultralytics import YOLO


# ============================================================
# TEST 3
# YOLO11m + ByteTrack
# DETECTION / TRACKING DIAGNOSTIC
# ============================================================

VIDEO_PATH = "videos/cow_video10.mp4"

MODEL_PATH = "yolo11m.pt"

OUTPUT_PATH = "output/cow_yolo11m_test.mp4"

CSV_PATH = "output/cow_yolo11m_test.csv"


# ============================================================
# SETTINGS
# ============================================================

COW_CLASS_ID = 19

CONFIDENCE = 0.15

IMG_SIZE = 640

DEVICE = 0

TRACKER = "bytetrack.yaml"


# ============================================================
# TRACKER SETTINGS
# ============================================================

# We use ByteTrack's normal configuration first.
#
# Do NOT change the tracker aggressively yet.
#
# This test is to determine whether YOLO11m itself
# can separate the cows.

TRACK_BUFFER = 60


# ============================================================
# DISPLAY
# ============================================================

DISPLAY_WIDTH = 1100

DISPLAY_HEIGHT = 700


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
    print("ERROR: Could not open video.")
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
# VIDEO INFORMATION
# ============================================================

print()
print("=" * 60)
print(" TEST 3 - YOLO11m + ByteTrack")
print("=" * 60)

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

print(
    f"Track buffer: {TRACK_BUFFER}"
)

print("=" * 60)
print()


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


print(
    "Model loaded successfully."
)

print()


# ============================================================
# TRACKER DISPLAY IDs
# ============================================================

display_ids = {}

next_display_id = 1


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
# STATISTICS
# ============================================================

frame_number = 0

max_detections = 0

max_tracks = 0

detection_frames = 0

tracking_frames = 0


total_detection_count = 0

total_track_count = 0


# ============================================================
# FPS
# ============================================================

fps_start = time.time()

fps_frames = 0

display_fps = 0.0


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
        "detections",
        "tracks",
        "tracker_id",
        "display_id",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y"
    ]
)


# ============================================================
# START PROCESSING
# ============================================================

print("=" * 60)
print(" STARTING TEST 3")
print("=" * 60)
print()
print("Press Q to stop.")
print()


while True:

    # ========================================================
    # READ FRAME
    # ========================================================

    success, frame = cap.read()


    if not success:

        break


    frame_number += 1


    # ========================================================
    # YOLO + BYTE TRACK
    #
    # IMPORTANT:
    #
    # We intentionally DO NOT pass half=True.
    #
    # This avoids the deprecated 'half' warning.
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
    # DETECTIONS
    # ========================================================

    detections = 0

    tracks = 0


    boxes = None

    confidences = None

    tracker_ids = None


    if result.boxes is not None:

        # ----------------------------------------------------
        # Detection count
        # ----------------------------------------------------

        detections = len(
            result.boxes
        )


        # ----------------------------------------------------
        # Bounding boxes
        # ----------------------------------------------------

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # Tracker IDs
        # ----------------------------------------------------

        if result.boxes.id is not None:

            tracker_ids = (
                result.boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )

            tracks = len(
                tracker_ids
            )


    # ========================================================
    # STATISTICS
    # ========================================================

    if detections > 0:

        detection_frames += 1


    if tracks > 0:

        tracking_frames += 1


    total_detection_count += detections

    total_track_count += tracks


    max_detections = max(
        max_detections,
        detections
    )


    max_tracks = max(
        max_tracks,
        tracks
    )


    # ========================================================
    # DRAW DETECTIONS
    # ========================================================

    if boxes is not None:

        for index, box in enumerate(
            boxes
        ):

            x1, y1, x2, y2 = map(
                int,
                box
            )


            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            confidence = 0.0

            if (
                confidences is not None
                and index < len(confidences)
            ):

                confidence = float(
                    confidences[index]
                )


            # ------------------------------------------------
            # Tracker ID
            # ------------------------------------------------

            tracker_id = None

            display_id = None


            if (
                tracker_ids is not None
                and index < len(tracker_ids)
            ):

                tracker_id = int(
                    tracker_ids[index]
                )

                display_id = get_display_id(
                    tracker_id
                )


            # ------------------------------------------------
            # Center
            # ------------------------------------------------

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
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

                (
                    0,
                    255,
                    0
                ),

                3
            )


            # =================================================
            # LABEL
            # =================================================

            if display_id is not None:

                label = (
                    f"COW {display_id} "
                    f"{confidence:.2f}"
                )

            else:

                label = (
                    f"COW "
                    f"{confidence:.2f}"
                )


            text_y = max(
                y1 - 10,
                25
            )


            cv2.putText(

                frame,

                label,

                (
                    x1,
                    text_y
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


            # =================================================
            # CENTER POINT
            # =================================================

            cv2.circle(

                frame,

                (
                    center_x,
                    center_y
                ),

                5,

                (
                    0,
                    255,
                    255
                ),

                -1
            )


            # =================================================
            # CSV
            # =================================================

            csv_writer.writerow(

                [
                    frame_number,
                    detections,
                    tracks,
                    tracker_id,
                    display_id,
                    round(
                        confidence,
                        4
                    ),
                    x1,
                    y1,
                    x2,
                    y2,
                    center_x,
                    center_y
                ]

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
    # INFORMATION PANEL
    # ========================================================

    overlay = frame.copy()


    cv2.rectangle(

        overlay,

        (
            15,
            15
        ),

        (
            370,
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


    # ========================================================
    # PANEL TITLE
    # ========================================================

    cv2.putText(

        frame,

        "YOLO11m + BYTETRACK TEST",

        (
            30,
            45
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
    # PANEL DATA
    # ========================================================

    cv2.putText(

        frame,

        f"DETECTIONS : {detections}",

        (
            30,
            80
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (
            0,
            255,
            255
        ),

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        frame,

        f"TRACKS     : {tracks}",

        (
            30,
            110
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

        f"MAX DET    : {max_detections}",

        (
            30,
            140
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

        f"MAX TRACK  : {max_tracks}",

        (
            30,
            170
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

        f"FPS : {display_fps:.1f}",

        (
            250,
            140
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
            250,
            170
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
    # TERMINAL OUTPUT
    # ========================================================

    print(

        f"Frame: {frame_number:04d} | "
        f"Detections: {detections:2d} | "
        f"Tracks: {tracks:2d} | "
        f"FPS: {display_fps:4.1f}"

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

        "TEST 3 - YOLO11m + ByteTrack",

        display_frame
    )


    # ========================================================
    # KEY
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

average_detections = 0.0

average_tracks = 0.0


if frame_number > 0:

    average_detections = (
        total_detection_count
        / frame_number
    )

    average_tracks = (
        total_track_count
        / frame_number
    )


print()
print("=" * 60)
print(" TEST 3 FINISHED")
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
    f"Average detections    : {average_detections:.2f}"
)

print(
    f"Average tracks        : {average_tracks:.2f}"
)

print(
    f"Detection frames      : {detection_frames}"
)

print(
    f"Tracking frames       : {tracking_frames}"
)

print(
    f"Model                 : YOLO11m"
)

print(
    f"Image size            : {IMG_SIZE}"
)

print(
    f"Confidence             : {CONFIDENCE}"
)

print()
print("Output video:")
print(OUTPUT_PATH)

print()
print("CSV log:")
print(CSV_PATH)

print("=" * 60)
print()
