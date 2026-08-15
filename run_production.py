from pathlib import Path
import csv
import cv2

from src.cow_plf.pipeline import CowPLFPipeline


PROJECT_ROOT = Path(r"D:\cow")

VIDEO_PATH = PROJECT_ROOT / "videos" / "cow_video6.mp4"

OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "production_cow_video6.mp4"
OUTPUT_CSV = OUTPUT_DIR / "production_cow_video6.csv"


def main():

    print("=" * 80)
    print("COW PLF - PRODUCTION PIPELINE")
    print("=" * 80)

    print()
    print("[INFO] Video:", VIDEO_PATH)

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    # ------------------------------------------------------------------
    # VIDEO
    # ------------------------------------------------------------------

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open input video."
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print(
        f"[INFO] Resolution: {width} x {height}"
    )

    print(
        f"[INFO] FPS: {fps:.2f}"
    )

    print(
        f"[INFO] Frames: {total_frames}"
    )

    # ------------------------------------------------------------------
    # OUTPUT VIDEO
    # ------------------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps if fps > 0 else 30.0,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            "Could not create output video."
        )

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    csv_file = open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    )

    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
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
            "orientation_deg",

            "pca_length_px",
            "pca_width_px",
            "pca_angle_deg",
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

            "centroid_x",
            "centroid_y",
            "normalized_area",
        ]
    )

    csv_writer.writeheader()

    # ------------------------------------------------------------------
    # PIPELINE
    # ------------------------------------------------------------------

    pipeline = CowPLFPipeline(
        project_root=PROJECT_ROOT,

        model_path="models/yolo26m-seg.pt",

        global_database=
            "database/global_cows.json",

        confidence=0.20,

        iou=0.50,

        image_size=1280,

        device=0,

        tracker="bytetrack.yaml",

        appearance_interval=3,

        min_crop_size=30,

        global_similarity_threshold=0.82,

        global_min_observations=3,

        global_max_embeddings=20,

        morphometric_interval=3,

        v5_max_distance=220,

        v5_max_missed_frames=600,

        v5_appearance_threshold=0.70,

        v5_strong_rejection_threshold=0.40,
    )

    pipeline.start_video(
        VIDEO_PATH.name
    )

    # ------------------------------------------------------------------
    # PROCESS VIDEO
    # ------------------------------------------------------------------

    print()
    print("[INFO] Starting production processing...")
    print()

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            results = pipeline.process_frame(
                frame,
                csv_writer=csv_writer
            )

            # ----------------------------------------------------------
            # DRAW RESULTS
            # ----------------------------------------------------------

            for result in results:

                x1, y1, x2, y2 = result["box"]

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                local_id = result[
                    "local_cow_id"
                ]

                global_id = result[
                    "global_cow_id"
                ]

                if global_id is None:
                    global_id = "PENDING"

                label = (
                    f"Local:{local_id} "
                    f"Global:{global_id}"
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

            writer.write(frame)

            # ----------------------------------------------------------
            # PROGRESS
            # ----------------------------------------------------------

            if pipeline.frame_number % 30 == 0:

                print(
                    f"[FRAME] "
                    f"{pipeline.frame_number}/"
                    f"{total_frames} | "
                    f"Local cows: "
                    f"{len(pipeline.profiles)} | "
                    f"Global: "
                    f"{len(pipeline.local_to_global)}"
                )

    finally:

        cap.release()

        writer.release()

        csv_file.close()

        pipeline.close()

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("PRODUCTION PIPELINE COMPLETE")
    print("=" * 80)

    print()

    summary = pipeline.summary()

    for key, value in summary.items():

        print(
            f"{key:25s}: {value}"
        )

    print()

    print(
        "[OUTPUT VIDEO]",
        OUTPUT_VIDEO
    )

    print(
        "[OUTPUT CSV]",
        OUTPUT_CSV
    )

    print()

    print("=" * 80)


if __name__ == "__main__":
    main()