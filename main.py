"""
===============================================================================
COW PLF - MAIN APPLICATION
===============================================================================

Single production entry point.

Example:

    python main.py --video videos/cow_video6.mp4

Outputs:

    output/
        annotated_*.mp4
        measurements_*.csv

The pipeline contains:

    YOLO26m-seg
    ByteTrack
    V5 permanent identity
    Global cross-day identity
    Morphometrics
===============================================================================
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2


PROJECT_ROOT = (
    Path(__file__).resolve().parent
)

SRC_PATH = (
    PROJECT_ROOT / "src"
)

if str(SRC_PATH) not in sys.path:

    sys.path.insert(
        0,
        str(SRC_PATH)
    )


from cow_plf.pipeline import (
    CowPLFPipeline
)


# =============================================================================
# DISPLAY
# =============================================================================

SHOW_MASK = True

SHOW_BOX = True

SHOW_LOCAL_ID = True

SHOW_GLOBAL_ID = True

SHOW_MEASUREMENTS = True

MASK_ALPHA = 0.25


# =============================================================================
# DRAW TEXT
# =============================================================================

def draw_text(
    frame,
    text,
    position,
    scale=0.55,
    thickness=2,
    color=(255, 255, 255)
):

    x, y = position

    font = cv2.FONT_HERSHEY_SIMPLEX

    (tw, th), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness
    )

    x1 = max(
        0,
        x - 5
    )

    y1 = max(
        0,
        y - th - baseline - 5
    )

    x2 = min(
        frame.shape[1] - 1,
        x + tw + 5
    )

    y2 = min(
        frame.shape[0] - 1,
        y + 5
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    global SHOW_MASK
    global SHOW_BOX

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True,
        help="Input video path"
    )

    args = parser.parse_args()

    video_path = Path(
        args.video
    )

    if not video_path.is_absolute():

        video_path = (
            PROJECT_ROOT
            / video_path
        )

    video_path = video_path.resolve()

    if not video_path.exists():

        print(
            f"[ERROR] Video not found: "
            f"{video_path}"
        )

        return 1

    video_name = (
        video_path.stem
    )

    # =========================================================================
    # OUTPUT
    # =========================================================================

    output_dir = (
        PROJECT_ROOT
        / "output"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_video = (
        output_dir
        /
        f"annotated_{video_name}.mp4"
    )

    output_csv = (
        output_dir
        /
        f"measurements_{video_name}.csv"
    )

    # =========================================================================
    # PIPELINE
    # =========================================================================

    pipeline = CowPLFPipeline(
        project_root=PROJECT_ROOT
    )

    pipeline.start_video(
        video_path.name
    )

    # =========================================================================
    # VIDEO
    # =========================================================================

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            f"[ERROR] Could not open "
            f"{video_path}"
        )

        pipeline.close()

        return 1

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

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        fps = 30.0

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # =========================================================================
    # OUTPUT WRITER
    # =========================================================================

    writer = cv2.VideoWriter(

        str(output_video),

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        fps,

        (
            width,
            height
        )
    )

    if not writer.isOpened():

        print(
            "[ERROR] Could not create output video."
        )

        cap.release()

        pipeline.close()

        return 1

    # =========================================================================
    # CSV
    # =========================================================================

    csv_file = open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8"
    )

    csv_fields = [

    "timestamp",
    "video",
    "frame",
    "tracker_id",
    "local_cow_id",
    "global_cow_id",
    "confidence",
    "global_score",

    "mask_area_px",
    "contour_area_px",
    "perimeter_px",

    "bbox_x",
    "bbox_y",
    "bbox_width_px",
    "bbox_height_px",
    "bbox_area_px",

    "body_length_px",
    "body_width_px",
    "length_width_ratio",

    "pca_length_px",
    "pca_width_px",
    "pca_length_width_ratio",

    "convex_hull_area_px",
    "solidity",
    "extent",
    "circularity",
    "convexity",

    "detector_bbox_width_px",
    "detector_bbox_height_px",
    "detector_bbox_area_px",

    "mask_to_bbox_ratio",
    "normalized_area",

    "centroid_x",
    "centroid_y",

    "orientation_deg",
    "pca_angle_deg",
     ]

    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=csv_fields
    )

    csv_writer.writeheader()

    # =========================================================================
    # RUNTIME
    # =========================================================================

    frame_number = 0

    fps_counter = 0

    fps_timer = time.time()

    display_fps = 0.0

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            frame_number += 1

            results = pipeline.process_frame(
                frame,
                csv_writer=csv_writer
            )

            # =================================================================
            # DRAW RESULTS
            # =================================================================

            for result in results:

                x1, y1, x2, y2 = (
                    result["box"]
                )

                local_id = (
                    result["local_cow_id"]
                )

                global_id = (
                    result["global_cow_id"]
                )

                confidence = (
                    result["confidence"]
                )

                mask = (
                    result["mask"]
                )

                features = (
                    result["morphometrics"]
                )

                # -------------------------------------------------------------
                # MASK
                # -------------------------------------------------------------

                if (
                    SHOW_MASK
                    and
                    mask is not None
                ):

                    overlay = (
                        frame.copy()
                    )

                    overlay[
                        mask
                    ] = (
                        0,
                        255,
                        0
                    )

                    frame = cv2.addWeighted(

                        overlay,

                        MASK_ALPHA,

                        frame,

                        1.0 -
                        MASK_ALPHA,

                        0
                    )

                # -------------------------------------------------------------
                # BOX
                # -------------------------------------------------------------

                if SHOW_BOX:

                    cv2.rectangle(

                        frame,

                        (x1, y1),

                        (x2, y2),

                        (0, 255, 0),

                        2
                    )

                # -------------------------------------------------------------
                # MAIN LABEL
                # -------------------------------------------------------------

                if (
                    SHOW_GLOBAL_ID
                    and
                    global_id is not None
                ):

                    label = (
                        str(global_id)
                    )

                else:

                    label = (
                        "GLOBAL PENDING"
                    )

                label += (
                    f" {confidence:.2f}"
                )

                draw_text(

                    frame,

                    label,

                    (
                        x1,
                        max(
                            25,
                            y1
                        )
                    ),

                    0.62,

                    2,

                    (
                        0,
                        255,
                        255
                    )
                )

                info_y = (
                    max(
                        25,
                        y1
                    )
                    + 23
                )

                # -------------------------------------------------------------
                # LOCAL ID
                # -------------------------------------------------------------

                if SHOW_LOCAL_ID:

                    draw_text(

                        frame,

                        f"LOCAL COW {local_id}",

                        (
                            x1,
                            info_y
                        ),

                        0.48,

                        1
                    )

                    info_y += 19

                # -------------------------------------------------------------
                # MORPHOMETRICS
                # -------------------------------------------------------------

                if (
                    SHOW_MEASUREMENTS
                    and
                    features is not None
                ):

                    draw_text(

                        frame,

                        f"L:"
                        f"{features['body_length_px']:.0f}px "
                        f"W:"
                        f"{features['body_width_px']:.0f}px",

                        (
                            x1,
                            info_y
                        ),

                        0.47,

                        1,

                        (
                            0,
                            255,
                            0
                        )
                    )

                    info_y += 19

                    draw_text(

                        frame,

                        f"Area:"
                        f"{features['mask_area_px']:.0f} "
                        f"L/W:"
                        f"{features['length_width_ratio']:.2f}",

                        (
                            x1,
                            info_y
                        ),

                        0.47,

                        1,

                        (
                            0,
                            255,
                            0
                        )
                    )

            # =================================================================
            # FPS
            # =================================================================

            fps_counter += 1

            elapsed = (
                time.time()
                -
                fps_timer
            )

            if elapsed >= 1.0:

                display_fps = (
                    fps_counter
                    /
                    elapsed
                )

                fps_counter = 0

                fps_timer = time.time()

            draw_text(

                frame,

                (
                    f"FPS {display_fps:.1f} | "
                    f"FRAME {frame_number}/"
                    f"{total_frames}"
                ),

                (
                    15,
                    30
                ),

                0.56,
                2
            )

            draw_text(

                frame,

                (
                    f"LOCAL COWS "
                    f"{len(pipeline.profiles)} | "
                    f"GLOBAL "
                    f"{len(pipeline.local_to_global)}"
                ),

                (
                    15,
                    58
                ),

                0.50,
                2
            )

            # =================================================================
            # WRITE
            # =================================================================

            writer.write(
                frame
            )

            csv_file.flush()

            # =================================================================
            # DISPLAY
            # =================================================================

            cv2.imshow(
                "COW PLF - Production",
                frame
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if key == ord("q"):
                break

            if key == ord("m"):

                SHOW_MASK = (
                    not SHOW_MASK
                )

            if key == ord("b"):

                SHOW_BOX = (
                    not SHOW_BOX
                )

    finally:

        cap.release()

        writer.release()

        csv_file.close()

        cv2.destroyAllWindows()

        pipeline.close()

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print()
    print("=" * 80)
    print("COW PLF PROCESSING COMPLETE")
    print("=" * 80)

    print(
        "Video:",
        video_path
    )

    print(
        "Frames:",
        frame_number
    )

    print(
        "Output:",
        output_video
    )

    print(
        "CSV:",
        output_csv
    )

    print()

    print(
        pipeline.summary()
    )

    print("=" * 80)

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )